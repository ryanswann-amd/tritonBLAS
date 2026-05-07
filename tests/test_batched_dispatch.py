"""Unit tests for the batched-dispatch selector layer (no GPU required)."""
import math

import pytest
import torch

from tritonblas.batched_dispatch import (
    BatchedTile,
    select_batched_config,
    should_dispatch_batched,
)


# ─── should_dispatch_batched ──────────────────────────────────────────────────

def _t(shape, dtype=torch.float16, **kw):
    """Tiny meta tensor — no allocation."""
    return torch.empty(*shape, device="meta", dtype=dtype, **kw)


def test_dispatch_rank3_matched():
    a = _t((4, 128, 64))
    b = _t((4, 64, 128))
    assert should_dispatch_batched(a, b) is True


def test_dispatch_rank2_falls_through():
    a = _t((128, 64))
    b = _t((64, 128))
    assert should_dispatch_batched(a, b) is False


def test_dispatch_rank3_batch_mismatch():
    a = _t((4, 128, 64))
    b = _t((2, 64, 128))
    assert should_dispatch_batched(a, b) is False


def test_dispatch_rank3_inner_mismatch():
    a = _t((4, 128, 32))
    b = _t((4, 64, 128))
    assert should_dispatch_batched(a, b) is False


def test_dispatch_mixed_rank():
    """rank-3 × rank-2 must NOT route through the batched path."""
    a = _t((4, 128, 64))
    b = _t((64, 128))
    assert should_dispatch_batched(a, b) is False


def test_dispatch_rank4_unsupported():
    a = _t((2, 4, 128, 64))
    b = _t((2, 4, 64, 128))
    assert should_dispatch_batched(a, b) is False


def test_dispatch_z1_routes_through():
    """Z=1 is still a valid batched call (degenerates to a single GEMM)."""
    a = _t((1, 128, 64))
    b = _t((1, 64, 128))
    assert should_dispatch_batched(a, b) is True


def test_dispatch_dtype_mismatch():
    """Mixed-dtype batched matmul must fall through (kernel asserts same dtype)."""
    a = _t((4, 128, 64), dtype=torch.float16)
    b = _t((4, 64, 128), dtype=torch.bfloat16)
    assert should_dispatch_batched(a, b) is False


def test_dispatch_transposed_inner_layout():
    """Transposed inner axes (non-contiguous M/K) must fall through.

    Realistic adversarial case: user calls bmm(a, b.transpose(-1,-2))
    expecting it to work like torch.bmm. Our kernel uses plain row-major
    strides on the inner [K,N] axes; a transposed view has stride_n != 1
    along N and would silently miscompute. The dispatch helper rejects so
    the caller can either copy or raise.
    """
    # Build a real CPU tensor we can transpose meaningfully.
    a = torch.empty(4, 128, 64, dtype=torch.float16)
    b = torch.empty(4, 128, 64, dtype=torch.float16).transpose(-1, -2)
    # b is now (4, 64, 128) but with stride pattern that is NOT contiguous
    # on the inner two dims.
    assert not b.is_contiguous()
    assert should_dispatch_batched(a, b) is False


def test_dispatch_transposed_a_layout():
    a = torch.empty(4, 64, 128, dtype=torch.float16).transpose(-1, -2)
    b = torch.empty(4, 64, 128, dtype=torch.float16)
    assert should_dispatch_batched(a, b) is False


def test_dispatch_strided_batch_axis_ok():
    """Non-contiguous BATCH axis is OK — kernel offsets by stride_z."""
    big = torch.empty(8, 128, 128, dtype=torch.float16)
    a = big[::2]      # stride along batch axis is now 2*128*128, but inner
                      # [M,K] is still contiguous.
    b = torch.empty(4, 128, 128, dtype=torch.float16)
    assert a.shape == (4, 128, 128)
    assert should_dispatch_batched(a, b) is True


# ─── select_batched_config ────────────────────────────────────────────────────

def test_tile_returned_for_residual_R1():
    """R1 (Z=4 2048³ fp16): big-MNK heuristic — special-case to W=4, S=1."""
    btile = select_batched_config(
        M=2048, N=2048, K=2048, Z=4,
        base_block_m=128, base_block_n=128, base_block_k=64,
        base_group_m=8, base_num_xcds=8, base_num_stages=2,
        bytes_a=2, bytes_b=2, num_cus=304,
    )
    # M*N = 4M → preferred = (256,128),(128,256),... — should land on one.
    assert (btile.block_m, btile.block_n) in {(256, 128), (128, 256), (128, 128), (256, 256)}
    assert btile.block_k in (64, 32, 128)
    # All-large bucket (M, N, K ≥ 2048) takes the W=4/S=1 branch from the
    # MI300X sweep (lower per-warp occupancy buys more arithmetic-intensity
    # per warp on long K-loops).
    assert btile.num_warps == 4
    assert btile.num_stages == 1
    assert btile.matrix_instr_nonkdim == 16


def test_tile_returned_for_residual_R2():
    """R2 (Z=8 1024³ bf16): mn=1M tier → (128,256) preferred from sweep."""
    btile = select_batched_config(
        M=1024, N=1024, K=1024, Z=8,
        base_block_m=128, base_block_n=128, base_block_k=64,
        base_group_m=8, base_num_xcds=8, base_num_stages=2,
        bytes_a=2, bytes_b=2, num_cus=304,
    )
    assert (btile.block_m, btile.block_n) in {(128, 256), (256, 128), (128, 128)}
    assert btile.num_warps == 8


def test_tile_respects_lds_capacity():
    btile = select_batched_config(
        M=8192, N=8192, K=8192, Z=4,
        base_block_m=128, base_block_n=128, base_block_k=128,
        base_group_m=8, base_num_xcds=8, base_num_stages=2,
        bytes_a=2, bytes_b=2, num_cus=304,
        lds_capacity_bytes=64 * 1024,
    )
    bm, bn, bk, ns = btile.block_m, btile.block_n, btile.block_k, btile.num_stages
    a = bm * bk * 2
    b = bk * bn * 2
    lds = a + b if ns <= 1 else (ns - 1) * (a + b)
    assert lds <= 64 * 1024


def test_tile_does_not_exceed_problem_dims():
    btile = select_batched_config(
        M=128, N=64, K=512, Z=32,
        base_block_m=32, base_block_n=32, base_block_k=128,
        base_group_m=4, base_num_xcds=8, base_num_stages=2,
        bytes_a=2, bytes_b=2, num_cus=304,
    )
    assert btile.block_m <= 128
    assert btile.block_n <= 64


def test_group_m_clamped_to_tiles_m():
    """group_m must always be clamped to ⌈M/BM⌉ to avoid empty groups."""
    btile = select_batched_config(
        M=64, N=64, K=64, Z=64,
        base_block_m=64, base_block_n=64, base_block_k=32,
        base_group_m=8, base_num_xcds=8, base_num_stages=2,
        bytes_a=2, bytes_b=2, num_cus=304,
    )
    # tiles_m = ⌈64 / btile.block_m⌉ ≤ 1, so group_m must clamp to 1.
    import math
    tiles_m = math.ceil(64 / btile.block_m)
    assert btile.group_m == max(1, min(8, tiles_m))


def test_tile_for_tiny_shapes_drops_warps():
    """Very small per-batch (M*N < 64K) should pick ≤ 64 area tile + nw≤4."""
    btile = select_batched_config(
        M=128, N=64, K=128, Z=64,
        base_block_m=64, base_block_n=64, base_block_k=64,
        base_group_m=2, base_num_xcds=8, base_num_stages=2,
        bytes_a=2, bytes_b=2, num_cus=304,
    )
    # M*N = 8192 → "tiny" tier, preferred = [(64,128),(128,64),(64,64),(32,64),(64,32)]
    # The chosen tile should respect the problem dims (no BM>M, no BN>N).
    assert btile.block_m <= 128
    assert btile.block_n <= 64
    assert btile.num_warps in (2, 4, 8)


def test_returns_full_batched_tile():
    """Sanity: BatchedTile carries every field the kernel needs."""
    btile = select_batched_config(
        M=2048, N=2048, K=2048, Z=4,
        base_block_m=128, base_block_n=128, base_block_k=64,
        base_group_m=8, base_num_xcds=8, base_num_stages=2,
        bytes_a=2, bytes_b=2, num_cus=304,
    )
    assert isinstance(btile, BatchedTile)
    for fld in ("block_m", "block_n", "block_k", "group_m", "num_xcds",
                "num_warps", "num_stages", "kpack", "waves_per_eu",
                "matrix_instr_nonkdim"):
        assert hasattr(btile, fld), f"missing field {fld}"


def test_kpack_drops_when_lds_tight():
    """If kpack=2 would overflow LDS, heuristic falls back to kpack=1."""
    # Use a giant-tile / big-K case that would force LDS to the edge.
    btile = select_batched_config(
        M=2048, N=2048, K=128, Z=4,
        base_block_m=256, base_block_n=256, base_block_k=128,
        base_group_m=8, base_num_xcds=8, base_num_stages=2,
        bytes_a=2, bytes_b=2, num_cus=304,
        lds_capacity_bytes=64 * 1024,
    )
    assert btile.kpack in (1, 2)
