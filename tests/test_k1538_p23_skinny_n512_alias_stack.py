"""Pin tests for the P23 ``skinny_N512`` K-COMPLEMENT 30-cell strict-equality
alias-stack route-OUT frozenset (15th-position).

Coverage:

* Frozenset shape (size + exact admit-set match against the 3 × 1 × 5 × 2
  bucket grid: M ∈ {2048,4096,8192} × N=512 × K ∈ {2048,4096,8192,16384,
  32768} × {bf16,fp16}).
* Predicate behaviour on every admit cell (30/30 must return True).
* Boundary controls — N just above/below 512 and (M, K) outside the bucket
  must NOT route via P23.
* Sibling-N firewall vs the prior K-COMPLEMENT predicates that own
  *different* N columns (P13(N=128), P13(N=256), P16(N=1024), P19(N=16384),
  P21(N=256-mid), P22(N=32768)).
* Alias-stack pin — P23 must be a strict cover of P15 (K-1409 N=512
  EXTREMES) ⨄ P17 (K-1437 N=512 BASE).  These are intentional aliases per
  the K-1493 P20 alias-stack convention; the ``isdisjoint`` checks would
  fail by construction and are NOT run.
* Non-K-COMPLEMENT disjointness vs P8 / K971 / P12 (anchored at low-N or
  square M=N=K, no overlap with N=512).
* Full-stack ``k971_route_decision`` integration — every P23 admit cell
  routes True through the 15-deep dispatch chain (caught upstream by
  P15+P17, but the regression pin still asserts route-OUT semantics).
* Streamk / work_stealing / dtype-mismatch carve-outs.
"""
from __future__ import annotations

import pytest
import torch

from tritonblas._route_predicate import (
    _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30,
    _k1538_p23_skinny_n512_alias_routeout,
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
# exactly; 29/30 strict-admits + 1 preserved-by-P15 reject = 30 total).
# ---------------------------------------------------------------------------

EXPECTED_ADMIT_30 = frozenset(
    (M, 512, K, dt)
    for M in (2048, 4096, 8192)
    for K in (2048, 4096, 8192, 16384, 32768)
    for dt in ("torch.bfloat16", "torch.float16")
)


def test_frozenset_size_is_exactly_30():
    assert len(_K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30) == 30


def test_frozenset_matches_expected_admit_set():
    assert _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30 == EXPECTED_ADMIT_30


def test_every_admit_cell_in_sweep_grid():
    for M, N, K, dtype_str in _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30:
        assert M in (2048, 4096, 8192)
        assert N == 512
        assert K in (2048, 4096, 8192, 16384, 32768)
        assert dtype_str in ("torch.bfloat16", "torch.float16")


# ---------------------------------------------------------------------------
# Predicate behavior — admit cells (30 / 30 must return True).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", sorted(EXPECTED_ADMIT_30))
def test_predicate_returns_true_for_admit_cell(M, N, K, dtype_str):
    dtype = torch.bfloat16 if dtype_str == "torch.bfloat16" else torch.float16
    assert _k1538_p23_skinny_n512_alias_routeout(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# Boundary controls — must NOT route via P23 (other predicates may catch
# them — these checks are P23-local).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", [
    (2048, 1024,  8192, "torch.bfloat16"),    # adjacent N=1024 — P16 territory
    (2048,  256,  8192, "torch.bfloat16"),    # adjacent N=256 — P13 / P21 territory
    (2048,  511,  4096, "torch.float16"),     # off-by-one N=511
    (2048,  513,  4096, "torch.float16"),     # off-by-one N=513
    (2048,  512,  1024, "torch.bfloat16"),    # K=1024 — out of bucket
    (2048,  512, 65536, "torch.bfloat16"),    # K=65536 — out of bucket
    (1024,  512,  8192, "torch.bfloat16"),    # M=1024 — out of bucket
    (16384, 512,  8192, "torch.bfloat16"),    # M=16384 — out of bucket
    (2048,  512,  8192, "torch.float32"),     # dtype not in {bf16,fp16}
])
def test_predicate_returns_false_outside_bucket(M, N, K, dtype_str):
    dtype = (torch.bfloat16 if dtype_str == "torch.bfloat16"
             else torch.float16 if dtype_str == "torch.float16"
             else torch.float32)
    assert _k1538_p23_skinny_n512_alias_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# Alias-stack pin — P23 is a strict cover of P15 ⨄ P17 over the N=512
# K-COMPLEMENT cohort.  The ``isdisjoint`` check vs P15/P17 is INTENTIONALLY
# omitted (alias-stack convention per K-1493 P20).  Replaced with positive
# ``issubset`` cover-pins so a future audit cannot silently break the
# load-bearing-fallback property.
# ---------------------------------------------------------------------------

def test_alias_stack_covers_p15_n512_extremes():
    """P23 must contain every cell of K-1409 P15 N=512 EXTREMES
    (K ∈ {2048, 32768} corner of the N=512 cohort)."""
    assert _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT.issubset(
        _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30)


def test_alias_stack_covers_p17_n512_base():
    """P23 must contain every cell of K-1437 P17 N=512 BASE
    (K ∈ {4096, 8192, 16384} band of the N=512 cohort)."""
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17.issubset(
        _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30)


# ---------------------------------------------------------------------------
# Sibling-N firewall vs the prior K-COMPLEMENT predicates that own
# *different* N columns.  These are also asserted at module load —
# duplicated here as belt-and-braces per the K-1437 minimalist convention.
# ---------------------------------------------------------------------------

def test_disjoint_from_p8_mfma_envelope():
    assert _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30.isdisjoint(
        _P8_MFMA_ISSUE_STALL_ROUTEOUT)


def test_disjoint_from_k971_route_table():
    assert _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30.isdisjoint(K971_ROUTE_TABLE)


def test_disjoint_from_k1361_p12_square_mid():
    assert _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)


def test_disjoint_from_k1367_p13_skinny_n128():
    assert _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)


def test_disjoint_from_k1397_p13_skinny_n256():
    assert _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12)


def test_disjoint_from_k1429_p16_skinny_n1024():
    assert _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29)


def test_disjoint_from_k1478_p19_skinny_n16384():
    assert _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30)


def test_disjoint_from_k1503_p21_skinny_n256_kmid():
    assert _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT)


def test_disjoint_from_k1513_p22_skinny_n32768():
    assert _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30)


# ---------------------------------------------------------------------------
# K-COMPLEMENT N-ladder coverage pin.  With P23 in place the K-COMPLEMENT
# axis covers N ∈ {128, 256, 512, 1024, 16384, 32768} (N=2048 still
# pending K-1525 productionisation).  Every (M, K, dtype) of the K-1538
# sweep grid must be admitted by P23.
# ---------------------------------------------------------------------------

def test_n512_alias_rung_covered():
    for M in (2048, 4096, 8192):
        for K in (2048, 4096, 8192, 16384, 32768):
            for dt_str in ("torch.bfloat16", "torch.float16"):
                assert (M, 512, K, dt_str) in _K1538_P23_SKINNY_N512_KCOMPL_ROUTEOUT_30


# ---------------------------------------------------------------------------
# Full-stack k971_route_decision integration — the 15-deep dispatch chain
# must route every P23 admit cell to hipBLASLt with realistic production
# tile dispatch flags (streamk=False, work_stealing=False, matched dtypes).
# (P15+P17 catch every cell upstream; this pins the route-OUT semantics
# end-to-end so an ablation of either P15 or P17 still yields True via
# the P23 fallback.)
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
        4096, 512, 8192, dtype, dtype,
        enable_streamk=enable_streamk, work_stealing=work_stealing,
    ) is False


def test_full_dispatch_returns_false_with_mismatched_dtypes():
    assert k971_route_decision(
        4096, 512, 8192, torch.bfloat16, torch.float16,
        enable_streamk=False, work_stealing=False,
    ) is False


def test_disable_env_set_short_circuit():
    """``disable_env_set=True`` overrides everything and returns False."""
    assert k971_route_decision(
        4096, 512, 8192, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False,
        disable_env_set=True,
    ) is False


# ---------------------------------------------------------------------------
# Ablation pin — REVIEWER-MANDATED.  The 15th-position P23 slot is
# unreachable under normal operation because every cell is short-circuited
# by P15 (9th) and P17 (11th) upstream.  This block silences the upstream
# K-COMPLEMENT layers (P5, E1, P15, P17) and proves P23 carries the full
# 30-cell N=512 envelope on its own — i.e. the slot is not dead code; it
# is a load-bearing fallback if any upstream layer is ever ablated.
# Without this test the alias-stack convention has zero runtime coverage.
# ---------------------------------------------------------------------------

@pytest.fixture
def upstream_n512_silenced(monkeypatch):
    """Monkey-patch every upstream predicate that could short-circuit a
    P23 N=512 admit cell.  P5 (closed-form) and E1 (axis-aligned envelope)
    can catch isolated N=512 cells; P15 (EXTREMES) and P17 (BASE) cover
    the entire P23 cohort by construction.  Silencing all four guarantees
    that any N=512 admit reaching the dispatch chain is routed *only*
    through the P23 slot — proving slot-15 reachability."""
    import tritonblas._route_predicate as rp
    monkeypatch.setattr(rp, "_k1409_p15_skinny_n512_routeout",
                        lambda M, N, K, dt: False)
    monkeypatch.setattr(rp, "_k1437_p17_skinny_n512_kcompl_base_routeout",
                        lambda M, N, K, dt: False)
    monkeypatch.setattr(rp, "R_K979_P5_route_to_hbl",
                        lambda M, N, K, dt: False)
    monkeypatch.setattr(rp, "R_K1142_E1_route_to_hbl",
                        lambda M, N, K, dt: False)
    yield


@pytest.mark.parametrize("M,N,K,dtype_str", sorted(EXPECTED_ADMIT_30))
def test_ablation_p23_carries_full_envelope_when_upstream_silenced(
    upstream_n512_silenced, M, N, K, dtype_str
):
    """With P5+E1+P15+P17 silenced, every one of the 30 P23 admit cells
    must still route True via the 15th-position slot.  Proves P23 is the
    load-bearing fallback — not dead code."""
    dtype = torch.bfloat16 if dtype_str == "torch.bfloat16" else torch.float16
    assert k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False,
    ) is True


def test_ablation_p23_alone_routes_strict_reject_cell(upstream_n512_silenced):
    """The single K-1538 strict-reject cell (2048, 512, 2048, bf16) is
    preserved by P15 EXTREMES upstream in production.  When P15 is
    silenced, P23 alias-stack still routes it (frozenset membership) —
    confirming the 'preserved-by-upstream-routing' commentary in the
    P23 docstring is observably true under ablation."""
    assert k971_route_decision(
        2048, 512, 2048, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False,
    ) is True


def test_ablation_p23_does_not_overshoot_outside_bucket(upstream_n512_silenced):
    """Negative ablation pin — silencing the upstream layers must NOT
    cause P23 to admit cells outside its 30-element strict-equality
    frozenset.  N=1024 / N=256 / off-grid K must still return False."""
    for (M, N, K, dt) in (
        (2048, 1024,  8192, torch.bfloat16),
        (2048,  256,  8192, torch.bfloat16),
        (2048,  512,  1024, torch.bfloat16),
        (2048,  512, 65536, torch.bfloat16),
        (1024,  512,  8192, torch.bfloat16),
    ):
        # k971_route_decision may still route via OTHER predicates — but
        # the P23-direct check must match the bucket exactly.
        from tritonblas._route_predicate import (
            _k1538_p23_skinny_n512_alias_routeout as p23,
        )
        assert p23(M, N, K, dt) is False
