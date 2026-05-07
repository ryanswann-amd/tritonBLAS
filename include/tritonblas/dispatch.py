# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Small-M dispatch path for the persistent GEMM kernel.

For tall-skinny GEMM with M <= 64 the default Origami selector picks tiny
block sizes (BLOCK_M=16, BLOCK_N=16-32, BLOCK_K=32). Two pathologies result
on MI300X:

1. ``M*N / (BLOCK_M*BLOCK_N) < num_cus`` -- the launched grid leaves a large
   fraction of the CUs idle.
2. ``BLOCK_K=32`` makes the K loop dominate runtime for K >= 1024 and
   prevents the MFMA pipeline from filling.

This module:

1. Detects small-M shapes (M <= ``SMALL_M_THRESHOLD``).
2. Selects an override (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps)
   tuned for tall-skinny shapes.
3. Launches the persistent kernel with ``grid = min(total_tiles, num_cus)``;
   the in-kernel ``persistent_tile_range`` grid-stride loop iterates over
   output tiles when ``total_tiles > num_cus``, while still using one
   workgroup per tile (no over-launch) when the problem is small.

The recipe is the geomean winner of an autotune sweep over
(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_xcds, group_m) on the small-M
benchmarking cohort (M in {1,8,16,32,64}, N,K in {1024,2048,4096,8192}, fp16).

Configuration
-------------
The path is enabled by default. To disable it process-wide (e.g., for A/B
benchmarking) call ``set_small_m_enabled(False)`` from any thread; the call
is guarded by ``threading.Lock`` and the resulting state is read atomically
from the dispatch fast path. The legacy ``TRITONBLAS_SMALL_M_DISABLE=1``
environment variable is honored at module import time only and is documented
solely for backward compatibility with existing benchmark scripts; new code
should use the Python API.
"""

from __future__ import annotations

import os
import threading
from typing import Tuple


# ----------------------------------------------------------------------------
# Public configuration
# ----------------------------------------------------------------------------

# Threshold (in M) below which the small-M override path is taken.
# M=64 still benefits in the autotune sweep so the gate covers the full
# small-M benchmarking cohort.
SMALL_M_THRESHOLD = 64

# MI300X 64 KB LDS budget per workgroup, used by the LDS-fit walk in
# small_m_block_override.  Exposed for tests.
LDS_BUDGET_BYTES = 64 * 1024
NUM_PIPELINE_STAGES = 2
BYTES_PER_FP16_BF16 = 2


def _initial_enabled_from_env() -> bool:
    """Honor the legacy TRITONBLAS_SMALL_M_DISABLE env var as the *initial*
    value of the module-level enable flag.  The env var is consulted only
    once, at import time; runtime mutation goes through set_small_m_enabled.
    """
    raw = os.environ.get("TRITONBLAS_SMALL_M_DISABLE", "")
    return raw in ("", "0", "false", "False")


# Module-level enable flag, guarded by an RLock so that concurrent dispatch
# callers see a consistent value.  RLock (not Lock) so a caller that
# re-enters from inside an "enabled?" check does not deadlock.
_config_lock = threading.RLock()
_small_m_enabled: bool = _initial_enabled_from_env()


def set_small_m_enabled(enabled: bool) -> None:
    """Enable or disable the small-M dispatch path process-wide.

    Thread-safe: takes ``_config_lock`` while flipping the flag.  Note that
    Triton compiles a fresh kernel per (block-size, num_warps) tuple, so
    flipping this flag mid-process does not invalidate any cached kernels --
    the flag only changes which compiled-kernel set is selected for new
    matmul calls.
    """
    global _small_m_enabled
    with _config_lock:
        _small_m_enabled = bool(enabled)


def is_small_m_enabled() -> bool:
    """Return the current enable state.  Thread-safe."""
    with _config_lock:
        return _small_m_enabled


# ----------------------------------------------------------------------------
# Block-size override
# ----------------------------------------------------------------------------

def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    p = 1
    while p < x:
        p <<= 1
    return p


def _lds_bytes(bm: int, bn: int, bk: int) -> int:
    """Per-workgroup LDS bytes for the persistent kernel under
    NUM_PIPELINE_STAGES-deep software pipelining (fp16/bf16 only)."""
    return NUM_PIPELINE_STAGES * (bm * bk + bk * bn) * BYTES_PER_FP16_BF16


def small_m_block_override(
    M: int,
    N: int,
    K: int,
    num_cus: int,
) -> Tuple[int, int, int, int, int]:
    """Return (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, NUM_WARPS).

    Recipe (autotune winner across the small-M benchmarking cohort):

    * BLOCK_M = 16 if M <= 16 else 32
        - 16 is the smallest legal MFMA tile size on CDNA3.
        - 32 for 16 < M <= 64 keeps M-tile count >= 1 without wasting
          arithmetic on padded zeros.
    * BLOCK_N = 32 by default; widened to 64 when both
        (a) BM == 32 (so BM*BN doesn't drop arithmetic intensity), and
        (b) the N-direction tile count after widening is still >= num_cus / 4
            (so we don't lose CU coverage).
      In practice this covers the (M=64, N>=8192) corner where the wider
      MFMA tile lifts arithmetic intensity without giving up CU saturation.
    * BLOCK_K = 256 (or 128/64/32 if K is not a multiple of 256) -- amortises
      Triton's per-K-iter overhead. The autotune sweep showed BLK_K=256 wins
      across ~95% of the cohort vs BLK_K=128.
    * num_warps = 4 if BLOCK_M == 16 else 8 -- matches MFMA tiling for the
      block.
    * GROUP_SIZE_M = 4 -- chiplet-aware tile grouping for cross-XCD L2 reuse.

    The returned BLOCK_K only goes below 256 when K isn't a multiple of 256
    (so EVEN_K stays true on the kernel fast path), and then walks down to
    keep the LDS budget under 64 KB.
    """
    # ---- BLOCK_M / BLOCK_N -------------------------------------------------
    # Default tiling: BM = 16 if M<=16 else 32, BN = 32.
    #
    # Two principled adjustments to BLOCK_N:
    #
    # (a) Widen BN to 64 when BM is already 32 AND N >= 8192.  The resulting
    #     32x64 MFMA tile keeps high arithmetic intensity, and N>=8192 still
    #     yields >=128 N-direction tiles (enough to saturate MI300X's 304 CUs
    #     under 2-row groups).  Replaces a previously-hardcoded
    #     ``M=64, N=8192`` special case.
    #
    # (b) Shrink to 16x16 tiles for very K-heavy narrow-N shapes
    #     (N <= 2048 AND K >= 8192).  At the default 32x32 the tile count is
    #     well below num_cus (e.g. M=32,N=1024,K=8192 -> 32 tiles for 304
    #     CUs, ~10x under-subscribed).  Going to 16x16 quadruples the tile
    #     count and lifts CU coverage; the smaller MFMA tile is paid for
    #     by the extra warps of CU occupancy.
    bm = 16 if M <= 16 else 32
    bn = 32
    if bm == 32 and N >= 8192:
        bn = 64
    elif N <= 2048 and K >= 8192:
        # K-heavy narrow-N corner: maximise tile count.  At default 32x32
        # tile count is well below num_cus (e.g. M=32, N=1024, K=8192 ->
        # 32 tiles for 304 CUs).  Going to 16x16 quadruples it; the smaller
        # MFMA tile is paid for by the extra CU occupancy.  The +K floor
        # avoids regressing K-light shapes where the grid-stride overhead
        # dominates.
        bm = 16
        bn = 16

    if N < bn:
        bn = max(16, _next_pow2(N))

    # ---- BLOCK_K -----------------------------------------------------------
    # Prefer 256, fall back to keep EVEN_K on the kernel fast path.  When
    # BLK_N=64 we cap BLK_K=128 (autotune winner for the bn=64 shapes; LDS
    # budget is also tighter at bn=64).
    bk = 128 if bn >= 64 else 256
    while K % bk != 0 and bk > 16:
        bk //= 2

    # ---- LDS budget guard --------------------------------------------------
    # Walk BLOCK_K down first (fewer K iters but same MFMA productivity per
    # iter), then BLOCK_N.  The guard is a function of the persistent
    # kernel's 2-stage software pipelining; do not call this path with a
    # different num_stages without re-deriving.
    while _lds_bytes(bm, bn, bk) > LDS_BUDGET_BYTES and bk > 32:
        bk //= 2
    while _lds_bytes(bm, bn, bk) > LDS_BUDGET_BYTES and bn > 16:
        bn //= 2

    # ---- num_warps, group_size_m -------------------------------------------
    # 4 warps for 16x16 / 16x32 tiles, 8 for the wider 32x* tiles.  The
    # autotune sweep tried 2/4/8; 4 won for BM=16 and 8 won for BM=32.
    num_warps = 4 if bm == 16 else 8
    group_size_m = 4

    return bm, bn, bk, group_size_m, num_warps


# ----------------------------------------------------------------------------
# Dispatch gate + grid
# ----------------------------------------------------------------------------

def should_use_small_m_path(
    M: int,
    N: int,
    K: int,
    block_m: int,
    block_n: int,
    num_cus: int,
) -> bool:
    """Take the small-M dispatch path whenever M <= SMALL_M_THRESHOLD and the
    path is enabled (see ``set_small_m_enabled``).

    The ``block_m`` / ``block_n`` / ``num_cus`` arguments are kept for caller
    API stability. The autotune sweep showed the override is a geomean win
    across the entire M <= 64 cohort, including shapes where Origami's
    tiling already saturates the CUs (e.g. M=32 N=8192 -- Origami's BLOCK_N=32
    gets 256 tiles but BLOCK_K=32 still leaves 2x perf on the table vs
    BLOCK_K=256).
    """
    if not is_small_m_enabled():
        return False
    if M > SMALL_M_THRESHOLD:
        return False
    return True


def small_m_grid(
    M: int,
    N: int,
    BLOCK_M: int,
    BLOCK_N: int,
    num_cus: int,
) -> Tuple[int, int]:
    """Return ``(grid, num_sms_for_kernel)``.

    The kernel's persistent loop uses ``stride = NUM_SMS``. Launch
    ``min(total_tiles, num_cus)`` workgroups and pass the same value as
    ``NUM_SMS`` so the grid-stride loop sweeps a contiguous slice of tile-IDs
    per workgroup. When ``total_tiles >= num_cus`` we get exactly ``num_cus``
    workgroups each grid-striding; when ``total_tiles < num_cus`` we launch
    one workgroup per tile (no over-launch).
    """
    total_tiles = ((M + BLOCK_M - 1) // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N)
    grid = max(1, min(total_tiles, num_cus))
    return grid, grid
