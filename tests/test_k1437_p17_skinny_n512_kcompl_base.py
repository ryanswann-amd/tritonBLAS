"""K-1437 (S-002 sibling): pin tests for the P17 `skinny_N512` K-COMPLEMENT
BASE 18-cell route-OUT frozenset (11th-position).

Pin coverage (analogous to K-1409 / K-1417 / K-1429 productionisation tests):

* 18 admit cells × 1 dispatch trace each (positive admit verification).  All
  18 are at realistic production tiles (BK ∈ {128, 256} — NOT a synthetic
  BK=64 — per K-539 lesson REVISE).  Cells are M ∈ {2048, 4096, 8192} ×
  N=512 × K ∈ {4096, 8192, 16384} × dtype ∈ {bf16, fp16}.
* 5 boundary controls — N just above/below 512 cells must NOT route via P17.
* Frozenset-shape pin (18 cells exactly, anchor enumeration).
* Disjointness pins vs P8 / K971 / P12 / K-1367 P13 / K-1397 P13 / K-1409 P15
  / K-1429 P16.  In particular asserts the BASE ⨄ EXTREMES partition of the
  N=512 K-COMPLEMENT region (BASE K ∈ {4096, 8192, 16384} disjoint from
  EXTREMES K ∈ {2048, 32768}).
* Full-stack ``k971_route_decision`` integration test: every admit cell
  routes via the 11-deep predicate chain (with streamk=False,
  work_stealing=False, matched dtypes).
"""
from __future__ import annotations

import pytest
import torch

from tritonblas._route_predicate import (
    _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_18,
    _k1437_p17_skinny_n512_kcompl_base_routeout,
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29,
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4,
    K971_ROUTE_TABLE,
    _P8_MFMA_ISSUE_STALL_ROUTEOUT,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# Frozenset shape pins (regression — must match K-1409 paired n=30 BASE
# admit set; bucket-rule enumeration M ∈ {2048,4096,8192} × N=512 ×
# K ∈ {4096,8192,16384} × {bf16,fp16}).
# ---------------------------------------------------------------------------

EXPECTED_BASE_ADMIT_18 = frozenset({
    # M=2048 row × K ∈ {4096, 8192, 16384} × {bf16, fp16}
    (2048, 512,  4096, "torch.bfloat16"),  # P5-overlap, included for
                                           # frozenset completeness
    (2048, 512,  4096, "torch.float16"),
    (2048, 512,  8192, "torch.bfloat16"),
    (2048, 512,  8192, "torch.float16"),
    (2048, 512, 16384, "torch.bfloat16"),
    (2048, 512, 16384, "torch.float16"),
    # M=4096 row × K ∈ {4096, 8192, 16384} × {bf16, fp16}
    (4096, 512,  4096, "torch.bfloat16"),
    (4096, 512,  4096, "torch.float16"),
    (4096, 512,  8192, "torch.bfloat16"),
    (4096, 512,  8192, "torch.float16"),
    (4096, 512, 16384, "torch.bfloat16"),
    (4096, 512, 16384, "torch.float16"),
    # M=8192 row × K ∈ {4096, 8192, 16384} × {bf16, fp16}
    (8192, 512,  4096, "torch.bfloat16"),
    (8192, 512,  4096, "torch.float16"),
    (8192, 512,  8192, "torch.bfloat16"),
    (8192, 512,  8192, "torch.float16"),
    (8192, 512, 16384, "torch.bfloat16"),
    (8192, 512, 16384, "torch.float16"),
})


def test_frozenset_size_is_exactly_18():
    assert len(_K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_18) == 18


def test_frozenset_matches_expected_admit_set():
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_18 == EXPECTED_BASE_ADMIT_18


# ---------------------------------------------------------------------------
# Predicate behavior — admit cells (18 / 18 must return True).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", sorted(EXPECTED_BASE_ADMIT_18))
def test_predicate_returns_true_for_admit_cell(M, N, K, dtype_str):
    dtype = torch.bfloat16 if dtype_str == "torch.bfloat16" else torch.float16
    assert _k1437_p17_skinny_n512_kcompl_base_routeout(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# Boundary controls — must NOT route via P17 (other predicates may catch
# them — these checks are P17-local).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", [
    (2048,  512, 2048, "torch.bfloat16"),    # K=2048 — P15 EXTREMES territory
    (2048,  512, 32768, "torch.bfloat16"),   # K=32768 — P15 EXTREMES territory
    (2048,  256, 8192, "torch.bfloat16"),    # adjacent N=256 — P13 N=256 territory
    (2048, 1024, 8192, "torch.bfloat16"),    # adjacent N=1024 — P16 N=1024 territory
    (1024,  512, 8192, "torch.bfloat16"),    # M=1024 — out of bucket
    (2048,  512, 8192, "torch.float32"),     # dtype not in {bf16,fp16}
    (2048,  512, 1024, "torch.bfloat16"),    # K=1024 — out of bucket
])
def test_predicate_returns_false_outside_bucket(M, N, K, dtype_str):
    dtype = (torch.bfloat16 if dtype_str == "torch.bfloat16"
             else torch.float16 if dtype_str == "torch.float16"
             else torch.float32)
    assert _k1437_p17_skinny_n512_kcompl_base_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# Disjointness firewalls vs the prior 10-predicate stack (sibling-N + N=512
# BASE ⨄ EXTREMES partition).
# ---------------------------------------------------------------------------

def test_disjoint_from_p8_mfma_envelope():
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_18.isdisjoint(
        _P8_MFMA_ISSUE_STALL_ROUTEOUT)


def test_disjoint_from_k971_route_table():
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_18.isdisjoint(K971_ROUTE_TABLE)


def test_disjoint_from_k1361_p12_square_mid():
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_18.isdisjoint(
        _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)


def test_disjoint_from_k1367_p13_skinny_n128():
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_18.isdisjoint(
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)


def test_disjoint_from_k1397_p13_skinny_n256():
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_18.isdisjoint(
        _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12)


def test_disjoint_from_k1409_p15_skinny_n512_extremes():
    """BASE K ∈ {4096, 8192, 16384} ⨄ EXTREMES K ∈ {2048, 32768}; the
    union is the full N=512 K-COMPL region at the K-1308 sweep grid."""
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_18.isdisjoint(
        _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT)


def test_disjoint_from_k1429_p16_skinny_n1024():
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_18.isdisjoint(
        _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29)


# ---------------------------------------------------------------------------
# Cohort scoping — every admit cell must satisfy the K-1437 BASE bucket rule.
# ---------------------------------------------------------------------------

def test_every_admit_cell_in_base_sweep_grid():
    for M, N, K, dtype_str in _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_18:
        assert M in (2048, 4096, 8192)
        assert N == 512
        assert K in (4096, 8192, 16384)
        assert dtype_str in ("torch.bfloat16", "torch.float16")


def test_n512_kcompl_base_union_extremes_is_full_30_cell_grid():
    """K-1437 P17 BASE (18) ⨄ K-1417 P15 EXTREMES (12) = full 30-cell N=512
    K-COMPL grid at the K-1308 sweep (K ∈ {2048, 4096, 8192, 16384, 32768}).
    """
    union = (
        _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_18
        | _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT
    )
    assert len(union) == 30
    expected = frozenset(
        (M, 512, K, dt)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert union == expected


# ---------------------------------------------------------------------------
# Full-stack k971_route_decision integration — the 11-deep dispatch chain
# must route every BASE admit cell to hipBLASLt with realistic production
# tile dispatch flags (streamk=False, work_stealing=False, matched dtypes).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", sorted(EXPECTED_BASE_ADMIT_18))
def test_full_dispatch_chain_routes_admit_cell_to_hbl(M, N, K, dtype_str):
    dtype = torch.bfloat16 if dtype_str == "torch.bfloat16" else torch.float16
    assert k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False,
    ) is True
