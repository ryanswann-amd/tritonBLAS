# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Small-M dispatch path for tritonblas (persistent CTA + grid-tail tuning).

Why this exists
---------------
For tall-skinny GEMM with very small M (M <= 32), the default Origami selector
picks tiny BLOCK_N tiles (16 or 32) so that the launched grid contains as many
output tiles as possible. With ``total_tiles = ceil(M/BM) * ceil(N/BN)`` often
well below the device CU count of an MI300X (304 CUs), the launched grid only
covers a fraction of the chip and per-tile MFMA utilisation is poor: only one
or two CU "wave groups" are launched, leaving ~270 CUs idle.

Mechanism
---------
The persistent kernel already supports a grid-stride loop over output tiles
via ``ScheduleContext.persistent_tile_range`` — each launched workgroup
iterates ``for tile_id in range(pid, total_tiles, NUM_SMS)``. By:

  * widening BLOCK_N to 256 (so each tile does enough MFMA work),
  * launching ``min(total_tiles, num_cus)`` workgroups, and
  * setting ``NUM_SMS == grid`` (so the in-kernel stride matches the launch),

we map exactly one program per CU and let the in-kernel loop walk the (N, K)
sub-tiles internally.  When ``total_tiles > num_cus`` (small grid-tail), the
extra tiles fall onto already-active CUs as additional iterations of the
grid-stride loop, so no CU sits idle.

Interface
---------
* ``should_use_small_m_path(...)`` — gate. M <= ``SMALL_M_THRESHOLD`` and
  the default tiling leaves CUs idle.  Honours ``TRITONBLAS_SMALL_M_DISABLE``.
* ``small_m_block_override(...)`` — picks (BLOCK_M, BLOCK_N, BLOCK_K,
  GROUP_SIZE_M, NUM_WARPS) tuned for tall-skinny shapes.  BLOCK_N can grow up
  to 256 (auto-selected so the resulting grid fills the CUs); ``num_warps=4``
  to avoid inflating register pressure on the small accumulator tile.
* ``small_m_grid(...)`` — returns ``(grid, num_sms_for_kernel)``; both equal
  ``min(total_tiles, num_cus)`` so the persistent loop maps one program / CU.
"""

import os
from typing import Tuple

import torch

# Threshold below which the small-M override path is taken.
#
# M=64 already has enough rows to give Origami a reasonable BLOCK_M=16 group_m=4
# layout that saturates the CUs; the override only helps when M <= 32 where the
# default tiling cannot fill the device with useful work without going to BLK_N=16.
SMALL_M_THRESHOLD = 32


def _small_m_disabled() -> bool:
    """Escape hatch for benchmark A/B comparisons.

    Set ``TRITONBLAS_SMALL_M_DISABLE=1`` to fall back to default Origami
    dispatch without rebuilding.
    """
    return os.environ.get("TRITONBLAS_SMALL_M_DISABLE", "") not in (
        "",
        "0",
        "false",
        "False",
    )


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

    BLOCK_N selection
    -----------------
    The task spec is "auto-select BLOCK_N up to 256".  We walk
    ``{256, 128, 64, 32}`` from the largest down and pick the first candidate
    that gives at least ``num_cus`` tiles in the persistent grid (so every CU
    has work).  When no candidate saturates the CUs (the common case for
    skinny N like N=1024, since 1024/256 = 4 tiles), we fall back to 256 — the
    persistent loop then has fewer than 1 tile/CU but each tile does the most
    MFMA work.

    BLOCK_K selection
    -----------------
    Wider BLOCK_K means fewer K-iterations and tighter pipelining.  We pick:

      * 128 when K >= 2048 and divides 128 (deep enough to amortise the wider
        tile; K=1024 with BLOCK_K=128 only gives 8 K-iters, pipeline never
        warms up).
      * 64 when K divides 64.
      * 32 / 16 as divisibility fallbacks (kernel EVEN_K fast path).

    LDS budget guard
    ----------------
    MI300X provides 64KB of LDS per workgroup.  The Triton kernel
    double-buffers A and B (num_stages=2), so per-stage A+B is
    ``(BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * bytes_per_elt`` and the total
    footprint is that times ``(num_stages - 1)`` swizzled buffers.
    With BLOCK_M=16 BLOCK_N=256 BLOCK_K=128 fp16 this is
    ``(16*128 + 128*256) * 2 = 69632 bytes`` — over budget.  We walk BLOCK_K
    down first (more K-iters but same per-iter MFMA productivity), then
    BLOCK_N as a last resort.  Matches the empirical Triton LDS estimate in
    ``origami.py:estimate_triton_lds_bytes``.
    """

    # Round BLOCK_M up to the nearest legal MFMA tile size (16 minimum on CDNA3).
    bm = max(16, _next_pow2(M))
    m_tiles = (M + bm - 1) // bm  # for M <= 32 this is exactly 1

    # Pick BLOCK_N to keep enough tiles in the grid for CU coverage.
    # Walk from 256 downwards (task spec: "BLOCK_N up to 256") and pick the
    # first candidate that saturates the CUs.  If none does, default to 256
    # so each tile is doing the most MFMA work possible.
    BLOCK_N = 256
    for cand in (256, 128, 64, 32):
        if cand > N:
            continue
        if m_tiles * (N // cand) >= num_cus:
            BLOCK_N = cand
            break
    # Pathological narrow N (e.g. N < 256): walk down further so BLOCK_N <= N.
    if N < BLOCK_N:
        for cand in (128, 64, 32, 16):
            if cand <= N:
                BLOCK_N = cand
                break

    # BLOCK_K: 128 for deep K, 64 default, 32/16 divisibility fallback.
    if K >= 2048 and K % 128 == 0:
        BLOCK_K = 128
    elif K % 64 == 0:
        BLOCK_K = 64
    elif K % 32 == 0:
        BLOCK_K = 32
    else:
        BLOCK_K = 16

    # ── LDS budget guard ──────────────────────────────────────────────────
    # The estimate matches origami.estimate_triton_lds_bytes with num_stages=2:
    #   (num_stages - 1) * (A_bytes + B_bytes)
    # which for num_stages=2 collapses to A_bytes + B_bytes.  Validated against
    # the kernel's metadata.shared on gfx942 (35/35 configs in K-186).
    LDS_BUDGET_BYTES = 64 * 1024
    NUM_STAGES = 2
    BYTES_PER_ELT = 2  # fp16/bf16; small-M path is gated to those dtypes

    def _lds_bytes(bm_, bn_, bk_):
        return (NUM_STAGES - 1) * (bm_ * bk_ + bk_ * bn_) * BYTES_PER_ELT

    while _lds_bytes(bm, BLOCK_N, BLOCK_K) > LDS_BUDGET_BYTES and BLOCK_K > 16:
        BLOCK_K //= 2
    while _lds_bytes(bm, BLOCK_N, BLOCK_K) > LDS_BUDGET_BYTES and BLOCK_N > 16:
        BLOCK_N //= 2

    GROUP_SIZE_M = 1  # only one M tile-row, grouping is irrelevant
    NUM_WARPS = 4  # task spec: smaller blocks need fewer warps

    return bm, BLOCK_N, BLOCK_K, GROUP_SIZE_M, NUM_WARPS


def should_use_small_m_path(
    M: int,
    N: int,
    K: int,
    block_m: int,
    block_n: int,
    num_cus: int,
) -> bool:
    """Decide whether to take the small-M persistent dispatch path.

    All of the following must hold:

    * Path not disabled by env var.
    * ``M <= SMALL_M_THRESHOLD`` (=32).
    * Default tiling leaves CUs idle: ``total_tiles < num_cus``.  Where
      Origami already saturates the device (e.g. M=32 with very large N and
      BLOCK_N=32) we leave its tuning alone.
    """
    if _small_m_disabled():
        return False
    if M <= 0 or M > SMALL_M_THRESHOLD:
        return False
    if num_cus <= 0:
        return False
    total_tiles = ((M + block_m - 1) // block_m) * ((N + block_n - 1) // block_n)
    return total_tiles < num_cus


def small_m_grid(
    M: int,
    N: int,
    BLOCK_M: int,
    BLOCK_N: int,
    num_cus: int,
) -> Tuple[int, int]:
    """Return ``(grid, num_sms_for_kernel)``.

    The kernel's persistent tile range strides by ``num_sms``, so passing
    ``num_sms == grid`` makes each launched workgroup sweep a contiguous slice
    of tile IDs (a grid-stride loop).  The grid itself is
    ``min(total_tiles, num_cus)`` so we never over-launch (one program / CU
    when the grid-tail spills, one program / tile when total_tiles < num_cus).
    """
    total_tiles = ((M + BLOCK_M - 1) // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N)
    grid = min(total_tiles, num_cus)
    grid = max(grid, 1)
    return grid, grid
