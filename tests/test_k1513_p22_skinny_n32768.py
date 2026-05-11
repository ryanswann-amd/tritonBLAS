"""Pin tests for the P22 ``skinny_N32768`` K-COMPLEMENT 30-cell strict-equality
route-OUT frozenset (14th-position).

Coverage:

* Frozenset shape (size + exact admit-set match against the 3 × 1 × 5 × 2
  bucket grid: M ∈ {2048,4096,8192} × N=32768 × K ∈ {2048,4096,8192,16384,
  32768} × {bf16,fp16}).
* Predicate behaviour on every admit cell (30/30 must return True).
* Boundary controls — N just above/below 32768 and K outside the K-grid
  must NOT route via P22.
* Disjointness firewalls vs the prior 13-predicate stack (P8 / K971 /
  P12 / P13(N=128) / P13(N=256) / P15 / P16 / P17 / P19 / P21).
* Full-stack ``k971_route_decision`` integration — every P22 admit cell
  routes True through the 14-deep dispatch chain.
"""
from __future__ import annotations

import pytest
import torch

from tritonblas._route_predicate import (
    _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30,
    _k1513_p22_skinny_n32768_routeout,
    _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT,
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30,
    _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17,
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
# Frozenset shape pin (regression — must match the 3 × 1 × 5 × 2 bucket grid
# exactly; 30/30 cells admit at the strict ratio_median ≥ 1.05 ∧ p(<1.05)
# < 0.01 gate).
# ---------------------------------------------------------------------------

EXPECTED_ADMIT_30 = frozenset(
    (M, 32768, K, dt)
    for M in (2048, 4096, 8192)
    for K in (2048, 4096, 8192, 16384, 32768)
    for dt in ("torch.bfloat16", "torch.float16")
)


def test_frozenset_size_is_exactly_30():
    assert len(_K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30) == 30


def test_frozenset_matches_expected_admit_set():
    assert _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30 == EXPECTED_ADMIT_30


def test_every_admit_cell_in_sweep_grid():
    for M, N, K, dtype_str in _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30:
        assert M in (2048, 4096, 8192)
        assert N == 32768
        assert K in (2048, 4096, 8192, 16384, 32768)
        assert dtype_str in ("torch.bfloat16", "torch.float16")


# ---------------------------------------------------------------------------
# Predicate behavior — admit cells (30 / 30 must return True).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", sorted(EXPECTED_ADMIT_30))
def test_predicate_returns_true_for_admit_cell(M, N, K, dtype_str):
    dtype = torch.bfloat16 if dtype_str == "torch.bfloat16" else torch.float16
    assert _k1513_p22_skinny_n32768_routeout(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# Boundary controls — must NOT route via P22 (other predicates may catch
# them — these checks are P22-local).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", [
    (2048, 16384,  8192, "torch.bfloat16"),    # adjacent N=16384 — P19 territory
    (2048,   256,  8192, "torch.bfloat16"),    # adjacent N=256 — P13 / P21 territory
    (2048, 32768,  1024, "torch.bfloat16"),    # K=1024 — out of bucket
    (2048, 32768, 65536, "torch.bfloat16"),    # K=65536 — out of bucket
    (1024, 32768,  8192, "torch.bfloat16"),    # M=1024 — out of bucket
    (16384, 32768, 8192, "torch.bfloat16"),    # M=16384 — out of bucket
    (2048, 32768,  8192, "torch.float32"),     # dtype not in {bf16,fp16}
])
def test_predicate_returns_false_outside_bucket(M, N, K, dtype_str):
    dtype = (torch.bfloat16 if dtype_str == "torch.bfloat16"
             else torch.float16 if dtype_str == "torch.float16"
             else torch.float32)
    assert _k1513_p22_skinny_n32768_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# Disjointness firewalls vs the prior 13-predicate stack (sibling-N firewall:
# every prior K-COMPLEMENT predicate uses N ∈ {128,256,512,1024,16384}; P12
# is bounded to M=N=K ∈ {2048,4096}; P8/K971 cap at N ≤ 2048).  These are
# also asserted at module load — duplicated here as belt-and-braces per
# the K-1437 minimalist convention.
# ---------------------------------------------------------------------------

def test_disjoint_from_p8_mfma_envelope():
    assert _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30.isdisjoint(
        _P8_MFMA_ISSUE_STALL_ROUTEOUT)


def test_disjoint_from_k971_route_table():
    assert _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30.isdisjoint(K971_ROUTE_TABLE)


def test_disjoint_from_k1361_p12_square_mid():
    assert _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)


def test_disjoint_from_k1367_p13_skinny_n128():
    assert _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)


def test_disjoint_from_k1397_p13_skinny_n256():
    assert _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12)


def test_disjoint_from_k1409_p15_skinny_n512_extremes():
    assert _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT)


def test_disjoint_from_k1429_p16_skinny_n1024():
    assert _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29)


def test_disjoint_from_k1437_p17_skinny_n512_kcompl_base():
    assert _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17)


def test_disjoint_from_k1478_p19_skinny_n16384():
    """Sibling N-bucket on the K-COMPLEMENT N-ladder; different N column."""
    assert _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30)


def test_disjoint_from_k1503_p21_skinny_n256_kmid():
    """Sibling N-bucket on the K-COMPLEMENT N-ladder; P21 lives at N=256."""
    assert _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT)


# ---------------------------------------------------------------------------
# K-COMPLEMENT N-ladder coverage closure pin.  Combined with the prior
# K-COMPLEMENT predicates, the N-axis now covers
# N ∈ {128, 256, 512, 1024, 16384, 32768}.
# ---------------------------------------------------------------------------

def test_n_ladder_top_rung_covered():
    """Every (M, K, dtype) of the K-1513 sweep grid must be admitted by P22
    (top rung: N=32768 column on full K-grid, 30/30 admits)."""
    for M in (2048, 4096, 8192):
        for K in (2048, 4096, 8192, 16384, 32768):
            for dt_str in ("torch.bfloat16", "torch.float16"):
                assert (M, 32768, K, dt_str) in _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30


# ---------------------------------------------------------------------------
# Full-stack k971_route_decision integration — the 14-deep dispatch chain
# must route every P22 admit cell to hipBLASLt with realistic production
# tile dispatch flags (streamk=False, work_stealing=False, matched dtypes).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", sorted(EXPECTED_ADMIT_30))
def test_full_dispatch_chain_routes_admit_cell_to_hbl(M, N, K, dtype_str):
    dtype = torch.bfloat16 if dtype_str == "torch.bfloat16" else torch.float16
    assert k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False,
    ) is True


# ---------------------------------------------------------------------------
# Streamk / work_stealing / dtype-mismatch carve-outs — the route-OUT must
# defer to the in-kernel path when any of the carve-out conditions hold.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("enable_streamk,work_stealing", [
    (True, False),
    (False, True),
    (True, True),
])
def test_full_dispatch_returns_false_with_streamk_or_work_stealing(
    enable_streamk, work_stealing
):
    # Pick any admit cell.
    dtype = torch.bfloat16
    assert k971_route_decision(
        4096, 32768, 8192, dtype, dtype,
        enable_streamk=enable_streamk, work_stealing=work_stealing,
    ) is False


def test_full_dispatch_returns_false_with_mismatched_dtypes():
    assert k971_route_decision(
        4096, 32768, 8192, torch.bfloat16, torch.float16,
        enable_streamk=False, work_stealing=False,
    ) is False
