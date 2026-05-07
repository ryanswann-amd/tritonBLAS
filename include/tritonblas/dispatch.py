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
4. For the structurally CU-starved corner (M <= 16, N <= 2048, K >= 8192),
   falls back to a split-K (Stream-K) path so K shards across CUs.

Configuration
-------------
The path is enabled by default. Set ``TRITONBLAS_SMALL_M_DISABLE=1`` to
disable it process-wide (escape hatch for A/B benchmarking). The flag is
read on every dispatch call; a single bool load needs no synchronization.
"""

from __future__ import annotations

import os
from typing import Tuple


SMALL_M_THRESHOLD = 64
LDS_BUDGET_BYTES = 64 * 1024
NUM_PIPELINE_STAGES = 2
BYTES_PER_FP16_BF16 = 2


def _env_disable() -> bool:
    return os.environ.get("TRITONBLAS_SMALL_M_DISABLE", "") in ("1", "true", "True")


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    p = 1
    while p < x:
        p <<= 1
    return p


def _lds_bytes(bm: int, bn: int, bk: int) -> int:
    """Per-workgroup LDS bytes for fp16/bf16 with 2-stage pipelining."""
    return NUM_PIPELINE_STAGES * (bm * bk + bk * bn) * BYTES_PER_FP16_BF16


def small_m_block_override(
    M: int, N: int, K: int, num_cus: int,
) -> Tuple[int, int, int, int, int]:
    """Return (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, NUM_WARPS).

    Recipe (autotune winner across the small-M cohort):
      * BM = 16 if M<=16 else 32 (smallest legal MFMA tile on CDNA3).
      * BN = 32; widened to 64 when BM=32 AND N>=8192.
      * BK = 256 (or 128 when BN=64), with a divisor walk that keeps
        EVEN_K true on the kernel fast path and an LDS-budget guard.
      * num_warps = 4 (BM=16) or 8 (BM=32).
      * GROUP_SIZE_M = 4 (chiplet-aware tile grouping).
    """
    bm = 16 if M <= 16 else 32
    bn = 32
    if bm == 32 and N >= 8192:
        bn = 64

    if N < bn:
        bn = max(16, _next_pow2(N))

    bk = 128 if bn >= 64 else 256
    while K % bk != 0 and bk > 16:
        bk //= 2

    while _lds_bytes(bm, bn, bk) > LDS_BUDGET_BYTES and bk > 32:
        bk //= 2
    while _lds_bytes(bm, bn, bk) > LDS_BUDGET_BYTES and bn > 16:
        bn //= 2

    num_warps = 4 if bm == 16 else 8
    return bm, bn, bk, 4, num_warps


def should_use_small_m_path(
    M: int, N: int, K: int, block_m: int, block_n: int, num_cus: int,
) -> bool:
    """Gate for the small-M dispatch path."""
    if _env_disable():
        return False
    if M > SMALL_M_THRESHOLD:
        return False
    if M >= 48 and N < 4096:
        return False
    return True


def small_m_grid(
    M: int, N: int, BLOCK_M: int, BLOCK_N: int, num_cus: int,
) -> Tuple[int, int]:
    """Return (grid, NUM_SMS) for the persistent kernel.  Grid is clamped
    to num_cus; NUM_SMS == grid drives the kernel's grid-stride loop."""
    total_tiles = ((M + BLOCK_M - 1) // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N)
    grid = max(1, min(total_tiles, num_cus))
    return grid, grid


SPLIT_K_M_MAX = 16
SPLIT_K_N_MAX = 2048
SPLIT_K_K_MIN = 8192


def should_use_split_k_path(M: int, N: int, K: int, num_cus: int) -> bool:
    """Gate for the split-K Stream-K fall-back.

    Pre-condition: should_use_small_m_path returned True.

    Routes the structurally-CU-starved corner (M<=16, N<=2048, K>=8192)
    through Stream-K so K shards across CUs.  The persistent grid-stride
    path can't recover the >75% idle CUs there because the output-tile
    count is structurally below num_cus.
    """
    if _env_disable():
        return False
    return M <= SPLIT_K_M_MAX and N <= SPLIT_K_N_MAX and K >= SPLIT_K_K_MIN


def split_k_block_override(
    M: int, N: int, K: int, num_cus: int,
) -> Tuple[int, int, int, int, int, int]:
    """Return (BM, BN, BK, GROUP_SIZE_M, NUM_WARPS, sk_grid) for split-K.

    BM=16 BN=32 BK=128, num_warps=4.  sk_grid picks the largest factor in
    {8,6,4,3,2} such that tiles*factor<=num_cus AND iters_per_tile/factor>=8.
    """
    bm, bn, bk = 16, 32, 128
    while K % bk != 0 and bk > 16:
        bk //= 2

    tiles = ((M + bm - 1) // bm) * ((N + bn - 1) // bn)
    iters_per_tile = max(1, (K + bk - 1) // bk)

    sk_grid = tiles
    for factor in (8, 6, 4, 3, 2):
        candidate = tiles * factor
        if candidate <= num_cus and iters_per_tile // factor >= 8:
            sk_grid = candidate
            break

    return bm, bn, bk, 1, 4, sk_grid
