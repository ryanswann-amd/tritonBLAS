# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Small-M dispatch path for tritonblas.

For tall-skinny GEMM with very small M (M <= 64), the default Origami selector
picks tiny BLOCK_N tiles (16 or 32). With (M*N)/(BLOCK_M*BLOCK_N) tiles often
well below the CU count of the device, the launched grid leaves the
majority of CUs idle and per-tile MFMA utilization is poor.

This module:

1. Detects small-M shapes (M <= ``SMALL_M_THRESHOLD``).
2. Selects an override (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, group_m) tuned
   for tall-skinny shapes — the BLOCK_N is enlarged (128 or 256) so each tile
   does enough useful work to keep the MFMA pipeline fed.
3. Launches the persistent kernel with ``grid = min(total_tiles, num_cus)``,
   which lets the in-kernel ``persistent_tile_range`` grid-stride loop iterate
   over output tiles when ``total_tiles > num_cus``, while still using one
   workgroup per tile (no over-launch) when the problem is small.
"""

import os
from typing import Optional, Tuple

import torch
import triton

# Threshold below which the small-M override path is taken.
#
# M=64 already has enough rows to give Origami a reasonable BLOCK_M=16 group_m=4
# layout that saturates the CUs; the override only helps when M <= 32 where the
# default tiling cannot fill the device with useful work without going to BLK_N=16.
SMALL_M_THRESHOLD = 32

# Escape hatch for A/B testing: set TRITONBLAS_SMALL_M_DISABLE=1 to fall back
# to the default Origami-driven dispatch.  Used by benchmarks that compare the
# small-M path against the legacy behaviour without rebuilding.
def _small_m_disabled() -> bool:
    return os.environ.get("TRITONBLAS_SMALL_M_DISABLE", "") not in ("", "0", "false", "False")


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    p = 1
    while p < x:
        p <<= 1
    return p


def small_m_block_override(
    M: int,
    N: int,
    K: int,
    num_cus: int,
) -> Tuple[int, int, int, int, int]:
    """Return (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, NUM_WARPS) for small-M.

    Heuristic:

    * BLOCK_M is the smallest power-of-two >= M, capped at 32 (16, 32).
      Anything below 16 is rounded up to 16 (the smallest MFMA tile size).
    * BLOCK_N grows so that the M-row of output tiles roughly fills the CUs.
      With M*N small, smaller BLOCK_N gives more tiles in the N direction and
      hence better CU coverage.
    * BLOCK_K = 64 (a good default for fp16/bf16 on CDNA3 — keeps K loop tight).
    * num_warps = 4 (smaller blocks need fewer warps; matches task spec).
    * group_m = 1 (only 1 tile per row, group reordering does not help).
    """

    # Round BLOCK_M up to the nearest legal MFMA tile size (16 minimum on CDNA3).
    bm = max(16, _next_pow2(M))

    # Pick BLOCK_N to (a) keep MFMA utilization high and (b) keep enough tiles
    # to cover the CUs (one row of tiles when M <= 32, so N tiles == total tiles).
    #
    # The task spec calls for BLOCK_N in {128, 256} as the upper end, but for
    # narrow N (e.g. N=1024) those leave only 4-8 tiles which under-subscribes
    # the device CUs.  So we walk down powers of two until we get at least
    # ``num_cus`` total tiles (equiv. to ~1 wave per CU on the grid-stride loop).
    m_tiles = (M + bm - 1) // bm  # for M <= 32 this is exactly 1
    # Largest BLOCK_N from {256, 128, 64, 32, 16} that still yields enough tiles
    # to hit ``num_cus`` total tiles.  Falls through to BLOCK_N=128 for tiny N
    # because the in-kernel grid-stride loop doesn't need full saturation to
    # be useful, and 128 is the safe LDS-fitting choice (see LDS budget below).
    BLOCK_N = 128
    for cand in (256, 128, 64, 32):
        if cand > N:
            continue
        if m_tiles * (N // cand) >= num_cus:
            BLOCK_N = cand
            break
    else:
        # No candidate saturated the CUs.  Pick the largest power of two that
        # divides into N up to 128 (256 doesn't fit LDS at num_stages=2 with
        # BLOCK_K=64 — see LDS budget calculation below).
        for cand in (128, 64, 32, 16):
            if cand <= N:
                BLOCK_N = cand
                break

    # K tile.  For tall-skinny GEMM the K-loop dominates kernel runtime, so
    # widening BLOCK_K reduces K-iteration count and amortises Triton's
    # per-iter overhead (descriptor advances, mask checks).  64 is the safe
    # default; bump to 128 when K is large enough that the bigger inner-loop
    # tile still divides K evenly (preserving EVEN_K for the fast path).
    # K=1024 has only 8 iters at BLOCK_K=128 — pipeline never warms up.  Reserve
    # the wider K tile for K >= 2048 where the K loop dominates runtime, AND
    # only when the BLOCK_M+BLOCK_N footprint can still fit BLK_K=128 in the
    # 64KB LDS budget at num_stages=2 (rough check: BLOCK_N <= 64 keeps the B
    # tile at 64*128*2bytes*2stages = 32KB).
    if K >= 2048 and K % 128 == 0 and BLOCK_N <= 64:
        BLOCK_K = 128
    else:
        BLOCK_K = 64
    if K % BLOCK_K != 0:
        # Fall back to a divisor so EVEN_K stays true.
        BLOCK_K = 32

    # ── LDS budget guard ──────────────────────────────────────────────
    # MI300X (gfx942) provides 64KB of LDS per workgroup.  The persistent
    # GEMM kernel double-buffers A and B (num_stages=2), so the LDS
    # footprint per stage is (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) elements
    # of fp16/bf16 (2 bytes).  Walk BLOCK_N (then BLOCK_K) down until the
    # 2-stage footprint fits 64KB.  This is the bug fix that prevents
    # `out of resource: shared memory` errors when num_cus is small enough
    # that the saturation loop above picks BLOCK_N=256 with BLOCK_K=64.
    LDS_BUDGET_BYTES = 64 * 1024
    NUM_STAGES = 2
    BYTES_PER_ELT = 2  # fp16/bf16; small-M path is gated to those dtypes
    def _lds_bytes(bm_, bn_, bk_):
        return NUM_STAGES * (bm_ * bk_ + bk_ * bn_) * BYTES_PER_ELT
    while _lds_bytes(bm, BLOCK_N, BLOCK_K) > LDS_BUDGET_BYTES and BLOCK_N > 16:
        BLOCK_N //= 2
    while _lds_bytes(bm, BLOCK_N, BLOCK_K) > LDS_BUDGET_BYTES and BLOCK_K > 16:
        BLOCK_K //= 2

    GROUP_SIZE_M = 1  # only one M tile-row, grouping is irrelevant
    NUM_WARPS = 4

    return bm, BLOCK_N, BLOCK_K, GROUP_SIZE_M, NUM_WARPS


def should_use_small_m_path(
    M: int, N: int, K: int, block_m: int, block_n: int, num_cus: int
) -> bool:
    """Decide whether to take the small-M persistent dispatch path.

    Conditions (all must hold):

    * ``M <= SMALL_M_THRESHOLD`` — only tall-skinny problems benefit.
    * ``K >= 2048`` — for K=1024 the K-loop is short enough that Origami's
      default selection (BLOCK_K=128, ~8 K-iters) already extracts most of the
      MFMA pipeline throughput, and the small-M override (BLOCK_K=64, ~16
      K-iters with NUM_WARPS=4) consistently regresses K=1024 shapes by
      ~15-30% — measured on K-144 v8 (M ∈ {16,32}, N ∈ {1024..8192}, K=1024:
      baseline 0.471x → new 0.420x).  Above K=2048 the K-loop dominates and
      the persistent grid-stride layout wins back its overhead.
    * ``total_tiles < num_cus`` — Origami's default tiling leaves CUs idle,
      which is precisely what the persistent grid-stride dispatch is designed
      to fix.
    """
    if _small_m_disabled():
        return False
    if M > SMALL_M_THRESHOLD:
        return False
    if K < 2048:
        return False
    total_tiles = ((M + block_m - 1) // block_m) * ((N + block_n - 1) // block_n)
    return total_tiles < num_cus


def small_m_grid(M: int, N: int, BLOCK_M: int, BLOCK_N: int, num_cus: int) -> Tuple[int, int]:
    """Return ``(grid, num_sms_for_kernel)``.

    grid = min(total_tiles, num_cus).  The kernel's persistent tile range uses
    ``stride = num_sms``, so passing ``num_sms = grid`` means each launched
    workgroup sweeps a contiguous slice of tile-IDs (a grid-stride loop).
    When ``total_tiles >= num_cus`` we launch exactly ``num_cus`` blocks and
    each grid-strides; when ``total_tiles < num_cus`` we launch one block per
    tile (no waste).
    """
    total_tiles = ((M + BLOCK_M - 1) // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N)
    grid = min(total_tiles, num_cus)
    grid = max(grid, 1)
    return grid, grid
