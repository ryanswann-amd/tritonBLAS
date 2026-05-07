# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Small-M dispatch path for the persistent GEMM kernel.

For tall-skinny GEMM with M ≤ 64 the default Origami selector picks tiny
block sizes (BLOCK_M=16, BLOCK_N=16-32, BLOCK_K=32). Two pathologies result
on MI300X:

1. ``M*N / (BLOCK_M*BLOCK_N) < num_cus`` — the launched grid leaves a large
   fraction of the 304 CUs idle.
2. ``BLOCK_K=32`` makes the K loop dominate runtime for K ≥ 1024 and prevents
   the MFMA pipeline from filling.

Empirical autotune sweeps over (BLOCK_M, BLOCK_N, BLOCK_K, num_warps,
num_xcds, group_m) on the K-169 small-M cohort
(M ∈ {1,8,16,32,64}, N,K ∈ {1024..8192}, fp16) showed a single winning recipe:

    BLOCK_M = 16 if M ≤ 16 else 32         (32 even for M=64 — 2 m-tiles)
    BLOCK_N = 32                            (keeps tile count high)
    BLOCK_K = 128                           (amortises K-loop overhead)
    num_warps = 4 if BLOCK_M == 16 else 8
    NUM_XCDS = 8 (CDNA3 chiplet count, keep L2 swizzle)
    GROUP_SIZE_M = 4 (chiplet-aware tile grouping)

Geomean speedup vs torch.matmul (hipBLASLt) on this cohort:
    Origami baseline: 0.33x
    Small-M override: 0.68x

The kernel grid is launched as ``min(total_tiles, num_cus)``; the persistent
loop's grid-stride iteration handles ``total_tiles ≥ num_cus``. Setting
``NUM_SMS == grid`` makes ``ScheduleContext.persistent_tile_range`` stride
correctly.
"""

import os
from typing import Tuple


# Threshold (in M) below which the small-M override path is taken.
#
# M=64 with large N (≥ 8192) is already well-tuned by Origami (BLOCK_M=16,
# group_m=4, BLOCK_N up to 128) and the uniform override regresses it
# slightly (-6 % geomean on the K-169 cohort vs Origami).  Cap at M=32
# so the fix targets only shapes where Origami leaves performance on the
# table.
SMALL_M_THRESHOLD = 32


def _small_m_disabled() -> bool:
    """Escape hatch for A/B benchmarking: set TRITONBLAS_SMALL_M_DISABLE=1
    to fall back to the legacy Origami-only dispatch.
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
    """Return (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, NUM_WARPS).

    These are the autotune winner across the K-169 small-M cohort. See module
    docstring for the sweep methodology.
    """
    # BLOCK_M: 16 for M ≤ 16 (smallest legal MFMA tile on CDNA3); 32 for
    # 16 < M ≤ 64 (M=64 ⇒ 2 m-tiles, more parallelism than BM=64).
    if M <= 16:
        bm = 16
    else:
        bm = 32

    # BLOCK_N=32: keeps tile count high (N/32 tiles per M row of output) so
    # tall-skinny problems light up many CUs even when M*N tiles is below
    # num_cus. Clamp when N < 32.
    bn = 32 if N >= 32 else max(16, _next_pow2(N))

    # BLOCK_K=128 amortises Triton's per-K-iter overhead; fall back when K
    # isn't a multiple of 128 so EVEN_K stays true for the kernel fast path.
    bk = 128
    while K % bk != 0 and bk > 16:
        bk //= 2

    # ── LDS budget guard ────────────────────────────────────────────────
    # MI300X provides 64 KB of LDS per workgroup. The persistent kernel uses
    # 2-stage software pipelining so the per-stage footprint is
    # (BM*BK + BK*BN) elements at 2 bytes (fp16/bf16; the small-M path is
    # gated to those). Walk BN then BK down to fit. Under the chosen defaults
    # the footprint is well below the budget but a guard makes the override
    # safe under future bm/bn growth.
    LDS_BUDGET_BYTES = 64 * 1024
    NUM_STAGES = 2
    BYTES_PER_ELT = 2

    def _lds(bm_, bn_, bk_):
        return NUM_STAGES * (bm_ * bk_ + bk_ * bn_) * BYTES_PER_ELT

    while _lds(bm, bn, bk) > LDS_BUDGET_BYTES and bn > 16:
        bn //= 2
    while _lds(bm, bn, bk) > LDS_BUDGET_BYTES and bk > 16:
        bk //= 2

    # 4 warps for BM=16 (each warp owns one MFMA 16x16 column slice in N);
    # 8 warps for BM=32 (matches kernel default and the autotune winner).
    num_warps = 4 if bm == 16 else 8

    # group_m=4 was the autotune winner across the cohort — chiplet-aware
    # tile grouping for cross-XCD L2 reuse.
    group_size_m = 4

    return bm, bn, bk, group_size_m, num_warps


def should_use_small_m_path(
    M: int,
    N: int,
    K: int,
    block_m: int,
    block_n: int,
    num_cus: int,
) -> bool:
    """Take the small-M dispatch path whenever M ≤ SMALL_M_THRESHOLD.

    The ``block_m`` / ``block_n`` / ``num_cus`` arguments are kept for caller
    API stability (and so the path can be tightened later by also gating on
    Origami's tile count) but the autotune sweep showed the override is a
    geomean win across the entire M ≤ 64 cohort, including shapes where
    Origami's tiling already saturates the CUs (e.g. M=32 N=8192 — Origami's
    BLOCK_N=32 gets 256 tiles but BLOCK_K=32 still leaves 2x perf on the
    table vs BLOCK_K=128).
    """
    if _small_m_disabled():
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
    per workgroup. When ``total_tiles ≥ num_cus`` we get exactly ``num_cus``
    workgroups each grid-striding; when ``total_tiles < num_cus`` we launch
    one workgroup per tile (no over-launch).
    """
    total_tiles = ((M + BLOCK_M - 1) // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N)
    grid = max(1, min(total_tiles, num_cus))
    return grid, grid
