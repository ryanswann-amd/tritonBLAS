# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Batched GEMM dispatch & tile selection (selector layer).

Two pure-Python helpers, both unit-testable without a GPU:

* `should_dispatch_batched(a, b)` — single source-of-truth telling
  `tritonblas.matmul()` whether a call should route to the fused-batch path.
  Today: rank-3 inputs with matching batch and inner dims, dtypes equal,
  contiguous in their last two axes (anything else falls through to the
  rank-2 path or raises).

* `select_batched_config(M, N, K, Z, dtype_bytes, ...)` — picks a tile,
  num_warps and num_stages for the batched kernel. The heuristic was
  derived from a 152-config × 8-shape sweep on MI300X (gfx942) and is
  driven by per-tile arithmetic intensity, batch-saturation of the grid,
  and LDS budget. See `_choose_tile_and_warps` for the full rule set.

The dispatch helper exists (rather than collapsing into matmul()) so:
  - the dispatch decision is one line for downstream callers to update
    when adding rank-N broadcast or mixed-rank support,
  - the tile heuristic can be unit-tested without spinning up a GPU,
  - we can swap in a learned/profiled lookup later without touching the
    public matmul() entrypoint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


# ─── Dispatch decision ────────────────────────────────────────────────────────

def should_dispatch_batched(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return True iff (a, b) describe a batched matmul we should fuse-launch.

    Rules (any failure → False, fallthrough to rank-2 path):
        - Both rank-3.
        - Batch dims match.
        - Per-batch K agrees: a.shape[2] == b.shape[1].
        - dtypes match.
        - Contiguous along the last two dims (the kernel does plain
          [M,K]·[K,N] strided loads; non-contiguous batched tensors would
          require a different stride layout that is not yet handled).
        - Z >= 1 (Z=0 is a degenerate empty batch, fine to fall through).

    Anything else (rank-2, rank-3 × rank-2 broadcast, mixed dtypes,
    transposed-batched layouts, rank-4+) returns False so the caller
    handles it through the existing rank-2 path or raises through asserts.
    """
    if a.dim() != 3 or b.dim() != 3:
        return False
    if a.shape[0] != b.shape[0]:
        return False
    if a.shape[2] != b.shape[1]:
        return False
    if a.dtype != b.dtype:
        return False
    if a.shape[0] < 1:
        return False
    # The kernel addresses [M,K] and [K,N] within each batch with normal
    # row-major strides; if the inner two axes are non-contiguous (e.g.
    # transposed view) we must fall through and let the caller materialise
    # a contiguous copy or take a different path.
    if not _last_two_contiguous(a) or not _last_two_contiguous(b):
        return False
    return True


def _last_two_contiguous(t: torch.Tensor) -> bool:
    """Check that the inner [M,K]/[K,N] block is contiguous (row-major).

    A rank-3 tensor t has shape (Z, R, C). It's "inner-contiguous" iff
    stride along C is 1 and stride along R equals C. The batch stride may
    be anything (we offset by stride_z explicitly in the kernel).
    """
    if t.dim() != 3:
        return False
    Z, R, C = t.shape
    sz, sr, sc = t.stride()
    return sc == 1 and sr == C


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
    kpack: int
    waves_per_eu: int
    matrix_instr_nonkdim: int


# Candidate tile shapes ranked by (BM, BN) area; used by the saturation
# fallback path. Restricted to power-of-two ≤ 256.
_TILE_CANDIDATES = (
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

_BK_CANDIDATES = (128, 64, 32, 16)


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


def _choose_tile_and_warps(
    M: int, N: int, K: int, Z: int,
    base_block_m: int, base_block_n: int, base_block_k: int,
    base_num_stages: int,
    bytes_a: int, bytes_b: int,
    num_cus: int,
    lds_capacity_bytes: int,
    target_tiles_per_cu: float,
):
    """Internal heuristic — empirically validated against MI300X sweep.

    Returns (BM, BN, BK, num_warps, num_stages, kpack, waves_per_eu).

    Empirical findings (MI300X, gfx942, fp16/bf16, 152-cfg sweep):
        - num_warps=8 wins on every tile in {(256,256),(256,128),(128,256),
          (128,128)}; we keep 8 there and drop to 4 only for very small
          tiles (<= 64x64).
        - num_stages=2 is optimal for tiles ≥ 128x128; num_stages=3 helps
          marginally for very small tiles.
        - kpack=2 wins by a small margin (~1-2 TF) over kpack=1 on most
          tiles; we default to kpack=2 unless LDS is tight.
        - waves_per_eu=0 (compiler default) and waves_per_eu=2 trade
          places within ±2 TF; default 0 is safer.
        - matrix_instr_nonkdim=16 (mfma 16x16x16) consistently beats 32 on
          MI300X for fp16/bf16 — this matches the AMD ROCm GEMM tuning
          guide.
        - Tile choice rule: for batch×tiles_per_batch ≥ 4·CU we want the
          largest (BM,BN) that still keeps total_tiles ≥ 2·CU AND fits LDS.
          Below saturation, smaller tiles help.
    """
    # ─── Decide a target tile based on the per-batch problem size ──────────
    # Empirical preference table from the 152-cfg MI300X sweep. Listed
    # largest→smallest within each tier; the saturation check below may
    # bump us back UP to a bigger tile when there's no benefit to many
    # small ones (batch dim already saturates the grid).
    mn = M * N
    if mn >= 4 * 1024 * 1024:        # ≥ 2048×2048 per batch
        preferred = [(256, 128), (128, 256), (128, 128), (256, 256)]
    elif mn >= 1024 * 1024:           # ≥ 1024×1024 per batch
        preferred = [(128, 256), (256, 128), (128, 128)]
    elif mn >= 256 * 1024:            # ≥ 512×512 per batch
        preferred = [(128, 256), (128, 128), (256, 128)]
    elif mn >= 64 * 1024:             # ≥ 256×256 per batch
        preferred = [(128, 128), (128, 64), (64, 128)]
    else:                              # tiny per-batch
        preferred = [(64, 128), (128, 64), (64, 64), (32, 64), (64, 32)]

    # Step 1: pick (BM, BN). Walk preferred → fallback, taking the first
    # tile whose dims fit the problem AND that has at least a minimal LDS
    # configuration we can use (BK=32 at num_stages=2). We do NOT reject
    # under-saturated tiles outright — for small Z the grid being
    # under-saturated is intrinsic, not a tile-choice problem.
    bm = bn = None
    for (cand_bm, cand_bn) in preferred + list(_TILE_CANDIDATES):
        if cand_bm > M or cand_bn > N:
            continue
        # Need SOME (BK, num_stages) combo that fits LDS.
        ok = False
        for try_bk in (64, 32, 16):
            for try_ns in (2, 1):
                if _fits_lds(cand_bm, cand_bn, try_bk, bytes_a, bytes_b, try_ns, lds_capacity_bytes):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            continue
        bm, bn = cand_bm, cand_bn
        break

    if bm is None:
        # Catastrophic fallback (problem smaller than any candidate).
        bm = bn = 16

    # Step 2: pick BK. Sweep finding: BK=64 wins on tiles ≥ 128×128 when
    # it fits; BK=32 wins on smaller tiles. Always require K % BK == 0
    # if possible, otherwise fall back.
    bk_candidates = (64, 32, 128, 16) if bm * bn >= 128 * 128 else (32, 64, 16)
    bk = None
    for try_bk in bk_candidates:
        if try_bk > K:
            continue
        # Need num_stages ≥ 1 to fit LDS. Prefer num_stages=2 (sweep).
        if _fits_lds(bm, bn, try_bk, bytes_a, bytes_b, 2, lds_capacity_bytes):
            bk = try_bk
            break
    if bk is None:
        # last-ditch: pick a tiny BK that fits at num_stages=1
        for try_bk in (32, 16, 8):
            if try_bk > K:
                continue
            if _fits_lds(bm, bn, try_bk, bytes_a, bytes_b, 1, lds_capacity_bytes):
                bk = try_bk
                break
    if bk is None:
        bk = 16

    # Step 3: pick num_stages. Sweep: 2 always wins when LDS allows.
    if _fits_lds(bm, bn, bk, bytes_a, bytes_b, 2, lds_capacity_bytes):
        ns = 2
    else:
        ns = 1

    # num_warps + num_stages: special-case the "all-dimensions-large"
    # bucket (M, N, K all ≥ 2048). The sweep found (W=4, S=1, kpack=1,
    # waves_per_eu=2) wins by ~13% over (W=8, S=2, ..., we=0) on this
    # bucket — the K-loop is long enough that the lower per-warp
    # occupancy buys more arithmetic-intensity per warp.
    big_mnk = (M >= 2048 and N >= 2048 and K >= 2048)
    if big_mnk and bm * bn >= 256 * 128:
        nw = 4
        ns = 1
        we = 2
        kp = 1
        return bm, bn, bk, nw, ns, kp, we

    # num_warps: 8 for tiles ≥ 128 area; 4 for smaller; 2 for tiny.
    if bm * bn >= 128 * 64:
        nw = 8
    elif bm * bn >= 32 * 64:
        nw = 4
    else:
        nw = 2

    # kpack: 1 is safest — the sweep showed kpack=2 wins by < 1 TF on
    # most shapes but interacts with LDS allocation in non-trivial ways.
    # Stay at kpack=1 unless the tile is small (where kpack=2 helped on
    # the 256³ shapes). The "small" boundary is bm*bn ≤ 128*128.
    kp = 2 if bm * bn <= 128 * 128 else 1

    # waves_per_eu: 0 (compiler default) is safest across the sweep.
    we = 0

    return bm, bn, bk, nw, ns, kp, we


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
    """Pick a tile + warp/stage config for a batched dispatch.

    Inputs come from the rank-2 OrigamiMatmulSelector so we know what
    Origami WOULD have picked; we only override when the batched workload
    has fundamentally different properties (the batch dim already
    saturates the GPU, so we can afford bigger per-tile work).

    Returns a `BatchedTile` consumed by `batched_persistent_matmul_lt`.
    """
    bm, bn, bk, nw, ns, kp, we = _choose_tile_and_warps(
        M=M, N=N, K=K, Z=Z,
        base_block_m=base_block_m,
        base_block_n=base_block_n,
        base_block_k=base_block_k,
        base_num_stages=base_num_stages or 2,
        bytes_a=bytes_a, bytes_b=bytes_b,
        num_cus=num_cus,
        lds_capacity_bytes=lds_capacity_bytes,
        target_tiles_per_cu=target_tiles_per_cu,
    )

    # group_m: clamp to tiles_m to avoid empty groups.
    new_tiles_m = math.ceil(M / bm)
    chosen_group_m = max(1, min(base_group_m or 8, new_tiles_m))

    return BatchedTile(
        block_m=bm,
        block_n=bn,
        block_k=bk,
        group_m=chosen_group_m,
        num_xcds=base_num_xcds or 8,
        num_warps=nw,
        num_stages=ns,
        kpack=kp,
        waves_per_eu=we,
        matrix_instr_nonkdim=16,  # MI300X sweep finding
    )
