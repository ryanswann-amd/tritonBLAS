"""Unit tests for the large-K square fp16/bf16 StreamK auto-enable gate.

The gate is a pure boolean predicate that decides whether to override
Origami's persistent scheduler with StreamK. The cohort is enumerated from
a per-shape A/B sweep, so its membership must stay strictly bounded.

These tests pin the inclusion list and at least one negative case per axis
(dtype, square, K threshold, listed-cell membership) so a future edit cannot
silently broaden or narrow the cohort without an explicit test diff.
"""
import pytest
import torch  # type: ignore

from tritonblas.matmul import (  # type: ignore
    _LARGEK_SQUARE_STREAMK_CELLS,
    _largeK_square_streamk_gate,
)


GATED_CELLS = [
    (1024, 4096),
    (2048, 4096),
    (2048, 8192),
    (4096, 4096),
    (4096, 8192),
    (4096, 16384),
]


def test_cell_set_matches_documented_inclusion_list():
    """The frozenset must contain exactly the enumerated winning cells —
    no more, no fewer. Any change here requires a corresponding test diff."""
    assert _LARGEK_SQUARE_STREAMK_CELLS == frozenset(GATED_CELLS)


@pytest.mark.parametrize("M, K", GATED_CELLS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gate_fires_on_each_listed_cell(M, K, dtype):
    """Every (M, K, dtype) tuple in the inclusion list must return True."""
    assert _largeK_square_streamk_gate(M, M, K, dtype) is True


@pytest.mark.parametrize("M, K", GATED_CELLS)
def test_gate_excludes_fp32(M, K):
    """fp32 is unaffected by the LDS-bound regression; gate must skip it."""
    assert _largeK_square_streamk_gate(M, M, K, torch.float32) is False


@pytest.mark.parametrize("M, K", GATED_CELLS)
def test_gate_excludes_rectangular(M, K):
    """Non-square (M != N) shapes are out of cohort; gate must skip them."""
    assert _largeK_square_streamk_gate(M, M * 2, K, torch.float16) is False
    assert _largeK_square_streamk_gate(M * 2, M, K, torch.float16) is False


@pytest.mark.parametrize(
    "M, K",
    [
        # K below the gated cells for each M-tier (small-K guard cohort).
        (1024, 512),
        (1024, 1024),
        (1024, 2048),
        (2048, 512),
        (2048, 1024),
        (2048, 2048),
        (4096, 512),
        (4096, 1024),
        (4096, 2048),
        # Listed M-tiers but K above any verified-winning cell.
        (1024, 8192),
        (1024, 16384),
        (2048, 16384),
        # M-tiers outside the cohort entirely.
        (512, 4096),
        (3072, 4096),
        (8192, 4096),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gate_excludes_unlisted_shapes(M, K, dtype):
    """Adjacent and out-of-cohort shapes must NOT trigger the gate."""
    assert _largeK_square_streamk_gate(M, M, K, dtype) is False


def test_gate_excludes_int_and_fp64_dtypes():
    """Only fp16/bf16 are gated; other dtypes must be skipped."""
    for dtype in (torch.float32, torch.float64, torch.int32, torch.int8):
        for (M, K) in GATED_CELLS:
            assert _largeK_square_streamk_gate(M, M, K, dtype) is False
