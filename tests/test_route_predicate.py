"""Unit tests for the bf16 hipBLASLt route-OUT predicate stack.

These tests are torch-free and exercise the data-only predicate module
directly: they assert that known anchor cells inside each layer of the
stacked envelope route to hipBLASLt, and that representative off-envelope
shapes stay on tritonBLAS. They protect against typos in the predicate
table (a single bad tuple silently mis-routes a cohort) and against
accidental loosening of the closed-form envelope boundaries.
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    K971_ROUTE_TABLE,
    P8_ROUTE_OUT,
    R_E1_route_to_hbl,
    R_P5_route_to_hbl,
    _p8_route_out,
)


BF16 = "torch.bfloat16"
FP16 = "torch.float16"
FP32 = "torch.float32"


# ---------------------------------------------------------------------------
# Layer 1: P8 strict-equality stacked union (36-cell bf16 union).
# ---------------------------------------------------------------------------

# One representative anchor cell per sub-cohort. If any of these stops
# routing, the corresponding sub-cohort's measurement campaign has been
# silently broken.
P8_HAPPY_PATH_ANCHORS = [
    (20352, 2048, 1024),  # A    S32 anchor               (paired hbl/tb ~1.25x)
    (20352, 1024, 1024),  # NB   S32 N/2 neighbour        (paired hbl/tb 1.657x, max)
    (10112, 2048, 1024),  # E2   M-axis-extension admit   (paired hbl/tb 1.759x)
    (14208,  128, 1024),  # N128 extreme-aspect admit     (paired hbl/tb 3.254x, max)
]


@pytest.mark.parametrize("M,N,K", P8_HAPPY_PATH_ANCHORS)
def test_p8_anchor_routes_to_hbl(M, N, K):
    """A representative cell from each P8 sub-cohort must route-OUT."""
    assert _p8_route_out(M, N, K, BF16) is True


def test_p8_total_cell_count():
    """P8 union is exactly the 36-cell stacked envelope.

    13 anchors + 12 neighbours + 3 axis-extension + 8 N=128 admits = 36.
    Any drift here means a cell was added or dropped without updating the
    documented attribution.
    """
    assert len(P8_ROUTE_OUT) == 36


def test_p8_carve_out_M4480_N128_K768_stays_on_tb():
    """The (4480, 128, 768, bf16) cell is deliberately OMITTED.

    It measured 0.967x at the M-floor of the anchor M-set (worse on
    hipBLASLt than on tritonBLAS) and is held out of the union by
    construction. If a future edit re-admits it, this test fails.
    """
    assert _p8_route_out(4480, 128, 768, BF16) is False


def test_p8_dtype_gating_fp16():
    """P8 is bf16-only — the same shape under fp16 must NOT route-OUT."""
    assert _p8_route_out(20352, 2048, 1024, FP16) is False


def test_p8_off_envelope_stays_on_tb():
    """Shapes that don't appear in the P8 table stay on tritonBLAS."""
    assert _p8_route_out(64, 64, 64, BF16) is False
    assert _p8_route_out(20352, 2048, 1023, BF16) is False  # K off by one
    assert _p8_route_out(20353, 2048, 1024, BF16) is False  # M off by one


# ---------------------------------------------------------------------------
# Layer 2: E1 closed-form axis-aligned envelope.
#   bf16  ∧  N ∈ {1792, 2048, 3072}  ∧  K ∈ {256, 768, 1024}
#   ∧  M >= 4480  ∧  K >= 256
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "M,N,K",
    [
        (4480, 1792, 256),   # M-floor + K-floor + N-floor of envelope
        (4480, 2048, 768),   # mid-envelope cell
        (8192, 3072, 1024),  # interior cell
        (65536, 2048, 256),  # large-M extreme of envelope
    ],
)
def test_e1_inside_envelope_routes_to_hbl(M, N, K):
    assert R_E1_route_to_hbl(M, N, K, BF16) is True


@pytest.mark.parametrize(
    "M,N,K,reason",
    [
        (4479, 2048, 1024, "M one below floor"),
        (4480, 2048, 128,  "K below floor — Triton-favoured per cross-arch"),
        (4480, 2048, 512,  "K not in {256, 768, 1024}"),
        (4480, 4096, 1024, "N not in {1792, 2048, 3072}"),
    ],
)
def test_e1_outside_envelope_stays_on_tb(M, N, K, reason):
    assert R_E1_route_to_hbl(M, N, K, BF16) is False, reason


def test_e1_dtype_gating_fp16():
    """E1 is bf16-only."""
    assert R_E1_route_to_hbl(8192, 2048, 1024, FP16) is False
    assert R_E1_route_to_hbl(8192, 2048, 1024, FP32) is False


# ---------------------------------------------------------------------------
# Layer 3: P5 closed-form structural pathology predicate.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "M,N,K,clause",
    [
        (1024, 2048, 2048, "clause-1 over-tile mid-rect"),
        (8192, 2048,  256, "clause-2 hbl-strong large-M skinny"),
        (16,   1600, 4096, "clause-3 extreme aspect / skinny long-K"),
        (8192, 1792,  768, "clause-4 N=1792 shallow-K dual"),
    ],
)
def test_p5_clauses_route_to_hbl(M, N, K, clause):
    assert R_P5_route_to_hbl(M, N, K, BF16) is True, clause


def test_p5_off_envelope_stays_on_tb():
    """A benign square shape with no pathology trait stays on tritonBLAS."""
    assert R_P5_route_to_hbl(512, 512, 512, BF16) is False


def test_p5_dtype_gating_fp16():
    assert R_P5_route_to_hbl(8192, 2048, 256, FP16) is False


# ---------------------------------------------------------------------------
# Layer 4: legacy K971 strict-equality anchors (fp16 + bf16 mid-square long-K).
# ---------------------------------------------------------------------------

def test_k971_legacy_anchor_present():
    """The legacy mid-square long-K anchors stay in the table."""
    assert (1024, 1024, 16384, BF16) in K971_ROUTE_TABLE
    assert (1024, 1024, 16384, FP16) in K971_ROUTE_TABLE
    assert (2048, 2048, 32768, BF16) in K971_ROUTE_TABLE
    assert (2048, 2048, 32768, FP16) in K971_ROUTE_TABLE


def test_k971_table_size_unchanged():
    """K971 holds exactly 8 cells (4 mid-square shapes × 2 dtypes)."""
    assert len(K971_ROUTE_TABLE) == 8
