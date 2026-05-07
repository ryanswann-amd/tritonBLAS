"""Unit tests for the batched-dispatch selector layer (no GPU required)."""
import math

import pytest
import torch

from tritonblas.batched_dispatch import (
    select_batched_config,
    should_dispatch_batched,
)


# ─── should_dispatch_batched ──────────────────────────────────────────────────

def _t(shape):
    """Tiny meta tensor — no allocation."""
    return torch.empty(*shape, device="meta")


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
    a = _t((4, 128, 64))
    b = _t((64, 128))
    assert should_dispatch_batched(a, b) is False


def test_dispatch_rank4_unsupported():
    a = _t((2, 4, 128, 64))
    b = _t((2, 4, 64, 128))
    assert should_dispatch_batched(a, b) is False


# ─── select_batched_config ────────────────────────────────────────────────────

def test_tile_unchanged_for_undersaturated():
    """When the base tile already produces a small grid, do nothing."""
    btile = select_batched_config(
        M=2048, N=2048, K=2048, Z=4,
        base_block_m=128, base_block_n=128, base_block_k=128,
        base_group_m=8, base_num_xcds=8, base_num_stages=2,
        bytes_a=2, bytes_b=2, num_cus=304,
    )
    # Z=4, 16x16 tiles per batch = 256 tpb, 1024 total < 4*304=1216 threshold.
    # Tile is left unchanged.
    assert (btile.block_m, btile.block_n) == (128, 128)


def test_tile_grows_when_saturated_small():
    """For Z=64, M=N=256 (Origami picks 16x16) we should grow the tile."""
    btile = select_batched_config(
        M=256, N=256, K=256, Z=64,
        base_block_m=16, base_block_n=16, base_block_k=128,
        base_group_m=8, base_num_xcds=8, base_num_stages=2,
        bytes_a=2, bytes_b=2, num_cus=304,
    )
    # 64 batches × (256/16)² = 16384 tiles >> 1216 threshold.
    # Should grow up; specifically still keep total >= 4*304.
    assert btile.block_m >= 16
    assert btile.block_n >= 16
    # And actually pick something larger than the original 16x16.
    assert (btile.block_m * btile.block_n) > (16 * 16)


def test_tile_respects_lds_capacity():
    """Should never pick a tile whose LDS footprint exceeds capacity."""
    btile = select_batched_config(
        M=8192, N=8192, K=8192, Z=4,
        base_block_m=128, base_block_n=128, base_block_k=128,
        base_group_m=8, base_num_xcds=8, base_num_stages=2,
        bytes_a=2, bytes_b=2, num_cus=304,
        lds_capacity_bytes=64 * 1024,
    )
    bm, bn, bk = btile.block_m, btile.block_n, btile.block_k
    lds = (2 - 1) * (bm * bk * 2 + bk * bn * 2)  # num_stages=2
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
    btile = select_batched_config(
        M=128, N=128, K=128, Z=64,
        base_block_m=128, base_block_n=128, base_block_k=128,
        base_group_m=8, base_num_xcds=8, base_num_stages=2,
        bytes_a=2, bytes_b=2, num_cus=304,
    )
    # tiles_m = 128/128 = 1, so group_m must clamp down to 1.
    assert btile.group_m == 1


def test_bk_kept_when_divides_K_evenly():
    """If base BK already divides K, keep it (preserves EVEN_K=True)."""
    btile = select_batched_config(
        M=2048, N=2048, K=2048, Z=4,
        base_block_m=128, base_block_n=128, base_block_k=64,
        base_group_m=8, base_num_xcds=8, base_num_stages=2,
        bytes_a=2, bytes_b=2, num_cus=304,
    )
    # BK=64 fits at 128x128 with num_stages=2: (1)*(128*64*2 + 64*128*2)=32KiB ≤ 64KiB
    assert btile.block_k == 64
