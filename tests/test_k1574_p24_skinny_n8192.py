"""Pin tests for the P24 ``skinny_N8192`` K-COMPLEMENT 30-cell strict-equality
route-OUT frozenset (16th-position).

Coverage:

* Frozenset shape (size + exact admit-set match against the 3 × 1 × 5 × 2
  bucket grid: M ∈ {2048,4096,8192} × N=8192 × K ∈ {2048,4096,8192,16384,
  32768} × {bf16,fp16}).
* Predicate behaviour on every admit cell (30/30 must return True).
* Boundary controls — N just above/below 8192 and (M, K) outside the bucket
  must NOT route via P24.
* Sibling-N firewall vs every prior K-COMPLEMENT predicate (P13(N=128),
  P13(N=256), P15(N=512), P16(N=1024), P17(N=512), P19(N=16384),
  P21(N=256-mid), P22(N=32768), P23(N=512 alias-stack)).  N=8192 is a
  previously-uncovered K-COMPLEMENT column so all sibling-N firewall
  asserts must pass — NO intentional aliases (unlike K-1538 P23 which
  aliases P15+P17).
* Non-K-COMPLEMENT disjointness vs P8 / K971 / P12 (anchored at low-N or
  square M=N=K, no overlap with N=8192).
* Full-stack ``k971_route_decision`` integration — every P24 admit cell
  routes True through the 16-deep dispatch chain (LOAD-BEARING — P24
  itself catches every cell because no upstream predicate covers N=8192).
* Streamk / work_stealing / dtype-mismatch carve-outs.
"""
from __future__ import annotations

import pytest
import torch

from tritonblas._route_predicate import (
    _K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30,
    _k1574_p24_skinny_n8192_routeout,
    _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30,
    _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30,
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
# exactly; 30/30 strict-admits = 30 total).
# ---------------------------------------------------------------------------

EXPECTED_ADMIT_30 = frozenset(
    (M, 8192, K, dt)
    for M in (2048, 4096, 8192)
    for K in (2048, 4096, 8192, 16384, 32768)
    for dt in ("torch.bfloat16", "torch.float16")
)


def test_frozenset_size_is_exactly_30():
    assert len(_K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30) == 30


def test_frozenset_matches_expected_admit_set():
    assert _K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30 == EXPECTED_ADMIT_30


def test_every_admit_cell_in_sweep_grid():
    for M, N, K, dtype_str in _K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30:
        assert M in (2048, 4096, 8192)
        assert N == 8192
        assert K in (2048, 4096, 8192, 16384, 32768)
        assert dtype_str in ("torch.bfloat16", "torch.float16")


# ---------------------------------------------------------------------------
# Predicate behavior — admit cells (30 / 30 must return True).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", sorted(EXPECTED_ADMIT_30))
def test_predicate_returns_true_for_admit_cell(M, N, K, dtype_str):
    dtype = torch.bfloat16 if dtype_str == "torch.bfloat16" else torch.float16
    assert _k1574_p24_skinny_n8192_routeout(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# Boundary controls — must NOT route via P24 (other predicates may catch
# them — these checks are P24-local).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", [
    (2048, 4096,  8192, "torch.bfloat16"),    # adjacent N=4096 — uncovered (no predicate)
    (2048, 16384, 8192, "torch.bfloat16"),    # adjacent N=16384 — P19 territory
    (2048,  512,  4096, "torch.float16"),     # N=512 — P15/P17/P23 territory
    (2048, 8191,  4096, "torch.float16"),     # off-by-one N=8191
    (2048, 8193,  4096, "torch.float16"),     # off-by-one N=8193
    (2048, 8192,  1024, "torch.bfloat16"),    # K=1024 — out of bucket
    (2048, 8192, 65536, "torch.bfloat16"),    # K=65536 — out of bucket
    (1024, 8192,  8192, "torch.bfloat16"),    # M=1024 — out of bucket
    (16384, 8192, 8192, "torch.bfloat16"),    # M=16384 — out of bucket
    (2048, 8192,  8192, "torch.float32"),     # dtype not in {bf16,fp16}
])
def test_predicate_returns_false_outside_bucket(M, N, K, dtype_str):
    dtype = (torch.bfloat16 if dtype_str == "torch.bfloat16"
             else torch.float16 if dtype_str == "torch.float16"
             else torch.float32)
    assert _k1574_p24_skinny_n8192_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# Sibling-N firewall vs the prior K-COMPLEMENT predicates.  N=8192 is a
# previously-uncovered K-COMPLEMENT column — every prior frozenset uses
# N ∈ {128, 256, 512, 1024, 16384, 32768}; NO intentional aliases (unlike
# K-1538 P23 which aliases P15+P17 over N=512).
# ---------------------------------------------------------------------------

def test_disjoint_from_p8_mfma_envelope():
    assert _K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30.isdisjoint(
        _P8_MFMA_ISSUE_STALL_ROUTEOUT)


def test_disjoint_from_k971_route_table():
    assert _K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30.isdisjoint(K971_ROUTE_TABLE)


def test_disjoint_from_k1361_p12_square_mid():
    assert _K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)


def test_disjoint_from_k1367_p13_skinny_n128():
    assert _K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)


def test_disjoint_from_k1397_p13_skinny_n256():
    assert _K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12)


def test_disjoint_from_k1409_p15_skinny_n512_extremes():
    assert _K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT)


def test_disjoint_from_k1429_p16_skinny_n1024():
    assert _K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29)


def test_disjoint_from_k1437_p17_skinny_n512_base():
    assert _K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17)


def test_disjoint_from_k1478_p19_skinny_n16384():
    assert _K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30)


def test_disjoint_from_k1503_p21_skinny_n256_kmid():
    assert _K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT)


def test_disjoint_from_k1513_p22_skinny_n32768():
    assert _K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30)


def test_disjoint_from_k1538_p23_skinny_n512_alias():
    assert _K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30)


# ---------------------------------------------------------------------------
# K-COMPLEMENT N-ladder coverage pin.  With P24 in place the K-COMPLEMENT
# axis covers N ∈ {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
# 65536} — the N=8192 middle-bucket gap is now closed.  Every (M, K, dtype)
# of the K-1574 sweep grid must be admitted by P24.
# ---------------------------------------------------------------------------

def test_n8192_bucket_fully_covered():
    for M in (2048, 4096, 8192):
        for K in (2048, 4096, 8192, 16384, 32768):
            for dt_str in ("torch.bfloat16", "torch.float16"):
                assert (M, 8192, K, dt_str) in _K1574_P24_SKINNY_N8192_KCOMPL_ALIASSTACK_30


# ---------------------------------------------------------------------------
# Full-stack k971_route_decision integration — the 16-deep dispatch chain
# must route every P24 admit cell to hipBLASLt with realistic production
# tile dispatch flags (streamk=False, work_stealing=False, matched dtypes).
# Unlike K-1538 P23 (which is unreachable behind P15+P17), P24 is
# LOAD-BEARING — the chain reaches position 16 because no upstream predicate
# covers N=8192.
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
    dtype = torch.bfloat16
    assert k971_route_decision(
        4096, 8192, 8192, dtype, dtype,
        enable_streamk=enable_streamk, work_stealing=work_stealing,
    ) is False


def test_full_dispatch_returns_false_with_mismatched_dtypes():
    assert k971_route_decision(
        4096, 8192, 8192, torch.bfloat16, torch.float16,
        enable_streamk=False, work_stealing=False,
    ) is False


def test_disable_env_set_short_circuit():
    """``disable_env_set=True`` overrides everything and returns False."""
    assert k971_route_decision(
        4096, 8192, 8192, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False,
        disable_env_set=True,
    ) is False
