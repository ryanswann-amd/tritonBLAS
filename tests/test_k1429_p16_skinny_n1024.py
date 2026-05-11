"""K-1429 (S-002): pin tests for the P16 `skinny_N1024` K-COMPLEMENT
29-cell route-OUT frozenset (10th-position).

Pin coverage (analogous to K-1409, K-1417 productionisation tests):

* 29 admit cells × 1 dispatch trace each (positive admit verification)
* 1 reject cell (boundary) — verifies the strict 1.05 floor correctly
  excludes the (2048, 1024, 4096, bf16) cell at r=1.015
* 5 boundary controls — N just above/below 1024 cells must NOT route
* Frozenset-shape pin (29 cells exactly, anchor enumeration)
* Disjointness pins vs P8 / K971 / P12 / K-1367 P13 / K-1397 P13 / K-1409 P15
"""
from __future__ import annotations

import pytest
import torch

from tritonblas._route_predicate import (
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29,
    _k1429_p16_skinny_n1024_routeout,
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4,
    K971_ROUTE_TABLE,
    _P8_MFMA_ISSUE_STALL_ROUTEOUT,
)


# ---------------------------------------------------------------------------
# Frozenset shape pins (regression — must match K-1429 paired n=30 admit set)
# ---------------------------------------------------------------------------

EXPECTED_ADMIT_29 = frozenset({
    # BASE region (K ∈ {4096, 8192, 16384}) — 17 admit cells
    (2048, 1024,  4096, "torch.float16"),
    (2048, 1024,  8192, "torch.bfloat16"),
    (2048, 1024,  8192, "torch.float16"),
    (2048, 1024, 16384, "torch.bfloat16"),
    (2048, 1024, 16384, "torch.float16"),
    (4096, 1024,  4096, "torch.bfloat16"),
    (4096, 1024,  4096, "torch.float16"),
    (4096, 1024,  8192, "torch.bfloat16"),
    (4096, 1024,  8192, "torch.float16"),
    (4096, 1024, 16384, "torch.bfloat16"),
    (4096, 1024, 16384, "torch.float16"),
    (8192, 1024,  4096, "torch.bfloat16"),
    (8192, 1024,  4096, "torch.float16"),
    (8192, 1024,  8192, "torch.bfloat16"),
    (8192, 1024,  8192, "torch.float16"),
    (8192, 1024, 16384, "torch.bfloat16"),
    (8192, 1024, 16384, "torch.float16"),
    # EXTENSION region (K ∈ {2048, 32768}) — 12 admit cells
    (2048, 1024,  2048, "torch.bfloat16"),
    (2048, 1024,  2048, "torch.float16"),
    (2048, 1024, 32768, "torch.bfloat16"),
    (2048, 1024, 32768, "torch.float16"),
    (4096, 1024,  2048, "torch.bfloat16"),
    (4096, 1024,  2048, "torch.float16"),
    (4096, 1024, 32768, "torch.bfloat16"),
    (4096, 1024, 32768, "torch.float16"),
    (8192, 1024,  2048, "torch.bfloat16"),
    (8192, 1024,  2048, "torch.float16"),
    (8192, 1024, 32768, "torch.bfloat16"),
    (8192, 1024, 32768, "torch.float16"),
})

REJECT_CELL = (2048, 1024, 4096, "torch.bfloat16")  # r=1.015, fails 1.05 floor


def test_frozenset_size_is_exactly_29():
    assert len(_K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29) == 29


def test_frozenset_matches_expected_admit_set():
    assert _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29 == EXPECTED_ADMIT_29


def test_reject_cell_is_excluded():
    assert REJECT_CELL not in _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29


# ---------------------------------------------------------------------------
# Predicate behavior — admit cells
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", sorted(EXPECTED_ADMIT_29))
def test_predicate_returns_true_for_admit_cell(M, N, K, dtype_str):
    dtype = torch.bfloat16 if dtype_str == "torch.bfloat16" else torch.float16
    assert _k1429_p16_skinny_n1024_routeout(M, N, K, dtype) is True


def test_predicate_returns_false_for_reject_cell():
    M, N, K, dtype_str = REJECT_CELL
    dtype = torch.bfloat16
    assert _k1429_p16_skinny_n1024_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# Boundary controls — N just outside the bucket must NOT route via P16
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", [
    (2048,  512, 8192, "torch.bfloat16"),    # adjacent N=512 — covered by P15
    (2048, 2048, 8192, "torch.bfloat16"),    # adjacent N=2048 — out of bucket
    (1024, 1024, 8192, "torch.bfloat16"),    # M=1024 — out of bucket
    (2048, 1024, 8192, "torch.float32"),     # dtype not in {bf16,fp16}
    (2048, 1024, 1024, "torch.bfloat16"),    # K=1024 — out of bucket
])
def test_predicate_returns_false_outside_bucket(M, N, K, dtype_str):
    dtype = (torch.bfloat16 if dtype_str == "torch.bfloat16"
             else torch.float16 if dtype_str == "torch.float16"
             else torch.float32)
    assert _k1429_p16_skinny_n1024_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# Disjointness firewalls vs the prior 9-predicate stack (sibling-N).
# ---------------------------------------------------------------------------

def test_disjoint_from_p8_mfma_envelope():
    assert _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29.isdisjoint(
        _P8_MFMA_ISSUE_STALL_ROUTEOUT)


def test_disjoint_from_k971_route_table():
    assert _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29.isdisjoint(K971_ROUTE_TABLE)


def test_disjoint_from_k1361_p12_square_mid():
    assert _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29.isdisjoint(
        _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)


def test_disjoint_from_k1367_p13_skinny_n128():
    assert _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29.isdisjoint(
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)


def test_disjoint_from_k1397_p13_skinny_n256():
    assert _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29.isdisjoint(
        _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12)


def test_disjoint_from_k1409_p15_skinny_n512():
    assert _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29.isdisjoint(
        _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT)


# ---------------------------------------------------------------------------
# Cohort scoping — every admit cell must satisfy the K-1429 sweep grid.
# ---------------------------------------------------------------------------

def test_every_admit_cell_in_sweep_grid():
    for M, N, K, dtype_str in _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29:
        assert M in (2048, 4096, 8192)
        assert N == 1024
        assert K in (2048, 4096, 8192, 16384, 32768)
        assert dtype_str in ("torch.bfloat16", "torch.float16")
