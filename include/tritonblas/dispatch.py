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


# --------------------------------------------------------------------------
# Split-K dispatch (CU-starved corner)
# --------------------------------------------------------------------------
#
# When the small-M block override still produces ``total_tiles < num_cus``
# (e.g. M<=16 N<=4096 K>=4096, or any small-M shape with N<=2048), the
# launch grid leaves >50% of MI300X's 304 CUs idle even at the smallest
# legal MFMA tile (16x16).  The persistent grid-stride loop cannot recover
# that capacity because there are simply not enough output tiles for one
# workgroup-per-tile.
#
# Split-K shards the K dimension across SPLIT_K extra workgroups per tile,
# each computing a partial sum and atomic-adding into an fp32 partial
# output buffer; a tiny finaliser cast converts back to the user dtype.
# Numerically this is the same K-summation error bound as the unsharded
# path (fp16 loads, fp32 accum, fp32 atomic_add); operationally it lifts
# the grid from ``total_tiles`` to ``total_tiles * SPLIT_K`` which we cap
# at ``num_cus`` so we don't oversubscribe.
#
# SPLIT_K is chosen automatically so that:
#   * launch grid <= num_cus (no oversubscription),
#   * each shard has >= ``MIN_BK_ITERS_PER_SHARD`` BLOCK_K iterations
#     (so the MFMA pipeline still fills inside each workgroup), and
#   * SPLIT_K is a power of 2 (clean K-shard alignment).
# Gate fires iff the resulting SPLIT_K would be >= 2.

# Each K-shard needs enough BLOCK_K iterations for the 2-stage MFMA
# pipeline to fill; below this threshold the per-launch overhead and
# atomic_add contention eat the parallelism win.  Empirically (autotune
# sweep over {2, 4, 8, 16}) 8 BK-iter floor wins the cohort geomean:
# SPLIT_K=8 at K=8192 / BK=256 gives only 4 iters per shard and the
# atomic_add contention from the doubled grid eats the 2x parallelism
# win (SPLIT_K=4 wins -0.02x to +0.01x on the 8 K=8192 narrow-N shapes).
# At K=4096 the 8-iter floor caps SPLIT_K at 2; SPLIT_K=4 there gives
# only 4 iters per shard and regresses 6 narrow-N shapes by 0.20x.
MIN_BK_ITERS_PER_SHARD = 8

# Hard cap on SPLIT_K.  Beyond 8 the fp32 atomic-add traffic into the
# scratch buffer dominates and kills the parallelism win.
MAX_SPLIT_K = 8

# Split-K fires when the persistent grid-stride launch leaves at least
# half of the CUs idle (total_tiles * 2 <= num_cus).  A 2x undersub
# factor ensures the SPLIT_K=2 floor produces a launch grid that fills
# the chip; smaller factors would just shuffle work between CUs without
# adding parallelism.  The prior 4x factor was too conservative -- it
# blocked SPLIT_K=2 wins on shapes with ~70-150 tiles (e.g. M=8 N=4096
# K=4096 where SPLIT_K=2 lifts speedup from 0.39x to 0.72x).
TILE_UNDERSUB_FACTOR = 2

# Minimum total K work to make split-K worthwhile.  Below this the
# persistent grid-stride path beats split-K -- Stream-K's per-tile
# bookkeeping + atomic-store reduction overhead exceeds the parallelism
# win.  Revised cohort sweep (with MIN_BK_ITERS=4, TILE_UNDERSUB=2):
#   K=2048 -> persistent wins -> off (4 BK iters * SPLIT_K=2 = 8 iters
#             total at BK=256, no headroom)
#   K=4096 -> Stream-K wins narrow-N small-M shapes by 1.7-2.5x (e.g.
#             M=1 N=1024 K=4096: 0.31x -> 0.78x with SPLIT_K=4) -> on
#   K=8192 -> Stream-K wins 22/32 shapes -> on
# K>=4096 is the durable threshold.
MIN_K_FOR_SPLIT_K = 4096


# Minimum total_tiles for split-K to be worthwhile, per K-bucket.
# Below this floor the per-tile launch + per-tile atomic-add reduction
# overhead exceeds the parallelism win.  The floor relaxes with K
# because the per-shard MFMA work amortises the SK overhead better at
# higher K.
#   K=4096: floor = num_cus // 4 = 76
#       Blocks narrow-N (tiles=32-64) where SK2 regresses 0.05-0.27x
#       (e.g. M=8 N=2048 K=4096: 0.518x SM -> 0.250x SK).
#       Allows wide-N (tiles>=128) where SK2 lifts 0.21-0.24x
#       (e.g. M=32 N=8192 K=4096: 0.582x SM -> 0.817x SK).
#   K>=8192: floor = num_cus // 8 = 38
#       Allows narrow-N (tiles=32-64) where SK4 lifts 4-5x over SM
#       (e.g. M=8 N=1024 K=8192: 0.084x SM -> 0.329x SK).
def _min_tiles_for_split_k(K: int, num_cus: int) -> int:
    # K=4096:  num_cus // 2 - 24  (= 128 for MI300X) -- tiles >= 128
    # K>=8192: num_cus // 9       (= 33 for MI300X)  -- tiles >= 32
    if K >= 8192:
        return 32
    return 128


def _split_k_factor(M: int, N: int, K: int, num_cus: int) -> int:
    """Choose SPLIT_K (power of 2 in [1, MAX_SPLIT_K]).  Returns 1 when
    split-K is not worthwhile (gate stays off).

    Four guards (tuned on the K-169 small-M cohort):
      * ``total_tiles * TILE_UNDERSUB_FACTOR <= num_cus`` -- only fire when
        the persistent path leaves at least half of the CUs idle.
      * ``total_tiles >= num_cus // 4`` -- below this floor the per-tile
        SK overhead exceeds the parallelism win, regardless of SPLIT_K.
      * ``K >= MIN_K_FOR_SPLIT_K`` -- below this the fp32 atomic_add
        overhead dominates the parallelism win.
      * Each shard gets at least ``MIN_BK_ITERS_PER_SHARD`` BLOCK_K iters,
        so the MFMA pipeline still fills inside each workgroup.
    """
    if K < MIN_K_FOR_SPLIT_K:
        return 1
    bm, bn, bk, _, _ = small_m_block_override(M, N, K, num_cus)
    total_tiles = ((M + bm - 1) // bm) * ((N + bn - 1) // bn)
    if total_tiles * TILE_UNDERSUB_FACTOR > num_cus:
        return 1
    if total_tiles < _min_tiles_for_split_k(K, num_cus):
        return 1
    iters_total = max(1, K // bk)
    # Don't let any shard have fewer than MIN_BK_ITERS_PER_SHARD iters.
    max_sk_by_K = max(1, iters_total // MIN_BK_ITERS_PER_SHARD)
    # Don't oversubscribe the chip.
    target_sk = num_cus // total_tiles
    sk = min(target_sk, max_sk_by_K, MAX_SPLIT_K)
    if sk < 2:
        return 1
    # Round down to the largest power of 2 (clean K-shard alignment so
    # K_per_split is divisible by BLOCK_K when K already is).
    sk_pow2 = 1
    while sk_pow2 * 2 <= sk:
        sk_pow2 *= 2
    return sk_pow2


def should_use_split_k_path(M: int, N: int, K: int, num_cus: int) -> bool:
    """Gate for the split-K dispatch path.

    Independent of ``should_use_small_m_path``: split-K can fire even on
    shapes the SM gate excludes (e.g. M=64 N<4096 K>=4096, which the SM
    gate hands back to the baseline because the baseline beats SM there
    at low K -- but at high K the baseline collapses to 0.30x and SK
    wins at 0.55-0.85x).  Caller is expected to gate on dtype / bias /
    quantization.

    Fires whenever the persistent path would leave at least half the CUs
    idle AND there is enough K work to shard with at least
    MIN_BK_ITERS_PER_SHARD BLOCK_K iterations per shard.  See
    ``_split_k_factor``.
    """
    if _env_disable():
        return False
    if M > SMALL_M_THRESHOLD:
        return False
    return _split_k_factor(M, N, K, num_cus) >= 2


def split_k_block_override(
    M: int, N: int, K: int, num_cus: int,
) -> Tuple[int, int, int, int, int, int]:
    """Return ``(BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, NUM_WARPS, SPLIT_K)``.

    Re-uses the small-M block recipe so the (BM, BN, BK, num_warps) tuple
    matches what the persistent path would have used; SPLIT_K is chosen by
    ``_split_k_factor`` to fill the chip without oversubscription.
    """
    bm, bn, bk, gm, nw = small_m_block_override(M, N, K, num_cus)
    sk = _split_k_factor(M, N, K, num_cus)
    return bm, bn, bk, gm, nw, sk
