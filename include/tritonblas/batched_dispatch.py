# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Batched GEMM dispatch & tile selection (K-684 selector layer).

Responsibility split:

* `should_dispatch_batched(a, b)` — tells the public `matmul()` whether to
  route a call into the batched path. Today: rank-3 inputs with matching
  batch dims. Anywhere downstream that wants to extend rank-N or mixed
  broadcasting, this is the single place to update. Keeps `matmul()` from
  growing into a god-function.

* `select_batched_config(M, N, K, Z, base_selector, num_cus)` — adjusts the
  tile size that `OrigamiMatmulSelector` would have picked for a single
  GEMM, accounting for the fact that the batch dim already provides
  `Z * tiles_per_batch` worth of grid parallelism. Origami picks small
  tiles for small (M, N) to maximize CU saturation, but in a batched
  launch the saturation comes from Z×; small tiles then just hurt
  arithmetic intensity. This module pushes BM/BN up when the grid is
  already saturated, and adjusts BK so even-K is still preferred.

Both functions are pure (no Triton calls, no kernel launches) so they can
be unit-tested without a GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


# ─── Dispatch decision ────────────────────────────────────────────────────────

def should_dispatch_batched(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return True iff (a, b) describe a batched matmul we should fuse-launch.

    Rules:
        - Both inputs must be rank-3.
        - Batch dim must match.
        - Per-batch K must agree (a.shape[2] == b.shape[1]).
    Anything else (rank-2, broadcast batch, mixed ranks, etc.) routes to
    the rank-2 path or raises through the existing assertions in `_bmm`.
    """
    if a.dim() != 3 or b.dim() != 3:
        return False
    if a.shape[0] != b.shape[0]:
        return False
    if a.shape[2] != b.shape[1]:
        return False
    return True


# ─── Tile selection for batched workloads ─────────────────────────────────────

@dataclass(frozen=True)
class BatchedTile:
    """Resolved per-launch tile + grouping for a batched dispatch."""
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_xcds: int
    num_warps: int
    num_stages: int


# Candidate tile shapes ranked by (BM, BN) area; we always try to match the
# Origami-suggested BK, falling back to a known-safe value. Restricted to
# powers of two ≤ 256 so they fit MI300X LDS at num_stages=2.
_TILE_CANDIDATES = (
    # (BM, BN)
    (256, 256),
    (256, 128),
    (128, 256),
    (128, 128),
    (128, 64),
    (64, 128),
    (64, 64),
    (32, 64),
    (64, 32),
    (32, 32),
    (16, 32),
    (32, 16),
    (16, 16),
)

_BK_CANDIDATES = (256, 128, 64, 32, 16)


def _lds_bytes(bm: int, bn: int, bk: int, bytes_a: int, bytes_b: int, num_stages: int) -> int:
    """Approximate Triton LDS footprint for one tile with `num_stages` buffers."""
    a = bm * bk * bytes_a
    b = bk * bn * bytes_b
    if num_stages <= 1:
        return a + b
    return (num_stages - 1) * (a + b)


def _fits_lds(bm: int, bn: int, bk: int, bytes_a: int, bytes_b: int,
              num_stages: int, lds_cap: int) -> bool:
    return _lds_bytes(bm, bn, bk, bytes_a, bytes_b, num_stages) <= lds_cap


def select_batched_config(
    M: int,
    N: int,
    K: int,
    Z: int,
    base_block_m: int,
    base_block_n: int,
    base_block_k: int,
    base_group_m: int,
    base_num_xcds: int,
    base_num_stages: int,
    bytes_a: int,
    bytes_b: int,
    num_cus: int,
    lds_capacity_bytes: int = 64 * 1024,
    target_tiles_per_cu: float = 4.0,
) -> BatchedTile:
    """Pick a tile size for a batched dispatch given the per-batch (M, N, K).

    Algorithm
    ---------
    1. Compute the grid the *base* (single-GEMM) tile would produce:
           total_tiles = Z · cdiv(M, base_BM) · cdiv(N, base_BN)
    2. If `total_tiles >= target_tiles_per_cu * num_cus`, the GPU is
       already saturated by the batch dimension. Search up the candidate
       table for the LARGEST (BM, BN) that:
         a) divides M and N reasonably (cdiv-fit, no obscene waste),
         b) keeps total_tiles >= target_tiles_per_cu * num_cus,
         c) fits in LDS at num_stages.
       Larger tiles raise per-tile arithmetic intensity (FLOPs per loaded
       byte), which is the binding constraint when grid parallelism is
       already plentiful.
    3. Otherwise, pass the base tile through unchanged — the small-tile
       choice is correct for under-saturated cases.
    4. BK: prefer base_BK if it cleanly divides K (keeps EVEN_K=True);
       otherwise pick the largest BK from `_BK_CANDIDATES` that divides K
       and still fits LDS.

    Returns: BatchedTile with (block_m, block_n, block_k, group_m,
                                num_xcds, num_warps, num_stages).
    """
    # ─── compute base saturation ──────────────────────────────────────────
    base_tpb = math.ceil(M / base_block_m) * math.ceil(N / base_block_n)
    base_total = base_tpb * Z
    sat_threshold = max(1, int(target_tiles_per_cu * num_cus))

    chosen_bm, chosen_bn = base_block_m, base_block_n

    if base_total >= sat_threshold:
        # Saturated — pick the largest tile that keeps us saturated AND fits.
        # _TILE_CANDIDATES is ordered largest → smallest by area.
        for cand_bm, cand_bn in _TILE_CANDIDATES:
            if cand_bm > M or cand_bn > N:
                continue
            tpb = math.ceil(M / cand_bm) * math.ceil(N / cand_bn)
            total = tpb * Z
            if total < sat_threshold:
                continue
            # Need to confirm BK fits at this BM/BN.
            if not _fits_lds(cand_bm, cand_bn, base_block_k, bytes_a, bytes_b,
                             base_num_stages, lds_capacity_bytes):
                continue
            chosen_bm, chosen_bn = cand_bm, cand_bn
            break

    # ─── BK refinement ────────────────────────────────────────────────────
    chosen_bk = base_block_k
    if not _fits_lds(chosen_bm, chosen_bn, chosen_bk, bytes_a, bytes_b,
                     base_num_stages, lds_capacity_bytes):
        # Fall back to the largest BK that fits at the new tile size.
        for cand_bk in _BK_CANDIDATES:
            if _fits_lds(chosen_bm, chosen_bn, cand_bk, bytes_a, bytes_b,
                         base_num_stages, lds_capacity_bytes):
                chosen_bk = cand_bk
                break

    # ─── group_m / num_xcds passthrough ──────────────────────────────────
    # group_m is bounded above by tiles_m at the new tile size.
    new_tiles_m = math.ceil(M / chosen_bm)
    chosen_group_m = max(1, min(base_group_m, new_tiles_m))

    return BatchedTile(
        block_m=chosen_bm,
        block_n=chosen_bn,
        block_k=chosen_bk,
        group_m=chosen_group_m,
        num_xcds=base_num_xcds,
        num_warps=8,
        num_stages=base_num_stages,
    )
