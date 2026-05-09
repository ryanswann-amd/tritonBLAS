"""K-1493 (S-002) — regression pin tests for the K-1478 P19 ``skinny_N16384``
K-COMPLEMENT route-OUT envelope.

K-1493 was scoped to productionize a 13th-position P20 frozenset over the
same N=16384 sweep grid that K-1478 already shipped at the 12th position.
A re-measurement on MI300X / gfx942 (paired n=30 HIP-graph hot-cache,
B=10000 vectorised paired bootstrap) confirmed that the K-1478 admit set
is the complete closure under the strict gate
``ratio_median ≥ 1.05 ∧ bootstrap p(<1.05) < 0.01`` — there are NO
additional cells in the M ∈ {2048, 4096, 8192} × N=16384 × K ∈ {2048,
4096, 8192, 16384, 32768} × {bf16, fp16} sweep grid that would pass the
gate beyond the 30 cells already pinned by P19.

Per the "no dead code by design" Minimalist rule, K-1493 therefore does
NOT take the P20 stack slot (a P20 frozenset alias-equal to P19 stacked
after P19 would be unreachable by short-circuit semantics — pure dead code
with no behavioural delta).  Instead, K-1493 lands the regression-pin
tests below that lock the K-1478 envelope at exactly 30 cells with the
exact (M, N, K, dtype) keys, every cell still routes True via the live
12-deep dispatch chain, and the natural-disjointness firewall vs the
sibling-N predicates (P12 / P13(N=128/256) / P15(N=512) / P16(N=1024) /
P17(N=512)) holds.

A future ticket may reopen the P20 slot when (and only when) a genuinely
EXTENDED admit set (e.g. K=65536, M=16384, or N=32768) actually diverges
from this envelope.

Coverage:

* Frozenset cardinality + exact cell-set equality (pins all 30 keys).
* Predicate behaviour: 30/30 admit cells return True from the P19
  predicate.
* Boundary controls: cells just outside the M / N / K / dtype envelope
  must NOT route via P19.
* Full-stack ``k971_route_decision`` integration: every P19 admit cell
  routes True through the 12-deep dispatch chain at production flags
  (``streamk=False, work_stealing=False``, matched dtypes).
* Sibling-N firewall: P19 frozenset is disjoint from every prior
  K-COMPLEMENT predicate frozenset and from the P12 SQUARE_MID envelope.
* P5 / P8 / K971 prior-stack disjointness (cheap insurance per R-1329).
* K-1493 P20 NOT-PRODUCTIONIZED pin: the symbol must NOT exist in the
  module namespace (a future re-add must come with a real divergent
  admit set, not an alias of P19).
"""
from __future__ import annotations

import pytest
import torch

from tritonblas import _route_predicate as rp
from tritonblas._route_predicate import (
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30,
    _k1478_p19_skinny_n16384_routeout,
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
# K-1493 closure pin: the K-1478 P19 frozenset is the complete admit set
# for the N=16384 sweep grid.  Authoring drift against any of the 30 keys
# below regresses the K-1478 productionization.
# ---------------------------------------------------------------------------

EXPECTED_K1478_P19_ADMIT_30 = frozenset({
    # M=2048 row × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (2048, 16384,  2048, "torch.bfloat16"),
    (2048, 16384,  2048, "torch.float16"),
    (2048, 16384,  4096, "torch.bfloat16"),
    (2048, 16384,  4096, "torch.float16"),
    (2048, 16384,  8192, "torch.bfloat16"),
    (2048, 16384,  8192, "torch.float16"),
    (2048, 16384, 16384, "torch.bfloat16"),
    (2048, 16384, 16384, "torch.float16"),
    (2048, 16384, 32768, "torch.bfloat16"),
    (2048, 16384, 32768, "torch.float16"),
    # M=4096 row × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (4096, 16384,  2048, "torch.bfloat16"),
    (4096, 16384,  2048, "torch.float16"),
    (4096, 16384,  4096, "torch.bfloat16"),
    (4096, 16384,  4096, "torch.float16"),
    (4096, 16384,  8192, "torch.bfloat16"),
    (4096, 16384,  8192, "torch.float16"),
    (4096, 16384, 16384, "torch.bfloat16"),
    (4096, 16384, 16384, "torch.float16"),
    (4096, 16384, 32768, "torch.bfloat16"),
    (4096, 16384, 32768, "torch.float16"),
    # M=8192 row × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (8192, 16384,  2048, "torch.bfloat16"),
    (8192, 16384,  2048, "torch.float16"),
    (8192, 16384,  4096, "torch.bfloat16"),
    (8192, 16384,  4096, "torch.float16"),
    (8192, 16384,  8192, "torch.bfloat16"),
    (8192, 16384,  8192, "torch.float16"),
    (8192, 16384, 16384, "torch.bfloat16"),
    (8192, 16384, 16384, "torch.float16"),
    (8192, 16384, 32768, "torch.bfloat16"),
    (8192, 16384, 32768, "torch.float16"),
})


def test_frozenset_size_is_exactly_30():
    assert len(_K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30) == 30


def test_frozenset_matches_expected_k1478_admit_set():
    """Pin the exact K-1478 admit cells; any drift fails CI."""
    assert _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30 == EXPECTED_K1478_P19_ADMIT_30


def test_every_admit_cell_in_n16384_sweep_grid():
    for M, N, K, dtype_str in _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30:
        assert M in (2048, 4096, 8192)
        assert N == 16384
        assert K in (2048, 4096, 8192, 16384, 32768)
        assert dtype_str in ("torch.bfloat16", "torch.float16")


def test_admit_set_is_full_sweep_grid_closure():
    """K-1493 closure: the K-1478 admit set IS the full sweep grid (30/30).

    A future K-COMPLEMENT-EXTENDED ticket that DOES find a missing cell
    (e.g. via tighter timing or extended K-axis) MUST update both the
    frozenset above AND the EXPECTED set in this file together.
    """
    expected_full_grid = frozenset(
        (M, 16384, K, dt)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30 == expected_full_grid


# ---------------------------------------------------------------------------
# Predicate behavior: every admit cell returns True (30 / 30).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M,N,K,dtype_str", sorted(EXPECTED_K1478_P19_ADMIT_30))
def test_predicate_returns_true_for_admit_cell(M, N, K, dtype_str):
    dtype = torch.bfloat16 if dtype_str == "torch.bfloat16" else torch.float16
    assert _k1478_p19_skinny_n16384_routeout(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# Boundary controls — cells just outside the bucket must NOT route via P19.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", [
    (1024, 16384,  2048, "torch.bfloat16"),    # M=1024 — out of bucket
    (16384, 16384, 2048, "torch.bfloat16"),    # M=16384 — out of bucket (sq)
    (2048,  8192,  2048, "torch.bfloat16"),    # N=8192 — sibling-N (P19_8192)
    (2048, 32768,  2048, "torch.bfloat16"),    # N=32768 — out of bucket
    (2048, 16384,  1024, "torch.bfloat16"),    # K=1024 — out of bucket
    (2048, 16384, 65536, "torch.bfloat16"),    # K=65536 — out of bucket
    (2048, 16384,  8192, "torch.float32"),     # dtype not in {bf16,fp16}
])
def test_predicate_returns_false_outside_bucket(M, N, K, dtype_str):
    dtype = (torch.bfloat16 if dtype_str == "torch.bfloat16"
             else torch.float16 if dtype_str == "torch.float16"
             else torch.float32)
    assert _k1478_p19_skinny_n16384_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# Full-stack k971_route_decision integration — every P19 admit cell must
# route to hipBLASLt through the live 12-deep dispatch chain.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", sorted(EXPECTED_K1478_P19_ADMIT_30))
def test_full_dispatch_chain_routes_admit_cell_to_hbl(M, N, K, dtype_str):
    dtype = torch.bfloat16 if dtype_str == "torch.bfloat16" else torch.float16
    assert k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False,
    ) is True


def test_streamk_path_bypasses_p19():
    """Streamk dispatch is structurally disjoint from the K971 predicate
    chain — must return False even on a P19 admit cell."""
    assert k971_route_decision(
        2048, 16384, 2048, torch.bfloat16, torch.bfloat16,
        enable_streamk=True, work_stealing=False,
    ) is False


def test_work_stealing_path_bypasses_p19():
    assert k971_route_decision(
        2048, 16384, 2048, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=True,
    ) is False


def test_mismatched_dtype_bypasses_p19():
    assert k971_route_decision(
        2048, 16384, 2048, torch.bfloat16, torch.float16,
        enable_streamk=False, work_stealing=False,
    ) is False


# ---------------------------------------------------------------------------
# Sibling-N firewall — P19 (N=16384) must be disjoint from every prior
# K-COMPLEMENT predicate (which all use N ∈ {128,256,512,1024}) and from
# the P12 SQUARE_MID envelope (M=N=K ∈ {2048,4096}).  Already asserted at
# module load; re-pinned here so a future module-load ablation cannot
# silently bypass the firewall.
# ---------------------------------------------------------------------------

def test_disjoint_from_p8_mfma_envelope():
    assert _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30.isdisjoint(
        _P8_MFMA_ISSUE_STALL_ROUTEOUT)


def test_disjoint_from_k971_route_table():
    assert _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30.isdisjoint(K971_ROUTE_TABLE)


def test_disjoint_from_k1361_p12_square_mid():
    assert _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)


def test_disjoint_from_k1367_p13_skinny_n128():
    assert _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)


def test_disjoint_from_k1397_p13_skinny_n256():
    assert _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12)


def test_disjoint_from_k1409_p15_skinny_n512():
    assert _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT)


def test_disjoint_from_k1429_p16_skinny_n1024():
    assert _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29)


def test_disjoint_from_k1437_p17_skinny_n512_base():
    assert _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17)


# ---------------------------------------------------------------------------
# K-1493 P20 NOT-PRODUCTIONIZED pin — locking in the "no dead code by
# design" decision.  A future re-add MUST come with a real divergent
# admit set (not an alias of P19); this test fails fast if anyone
# re-introduces an alias-equal P20 frozenset.
# ---------------------------------------------------------------------------

def test_no_alias_p20_frozenset_in_module():
    """Catches the K-1493 dead-code-by-design anti-pattern.

    If a P20 frozenset is ever added that is equal to the P19 frozenset
    AND comes after P19 in the dispatch chain, the P20 predicate is
    structurally unreachable.  This test fails fast on that pattern.
    """
    p20_aliases = []
    for name in dir(rp):
        if "P20" not in name or "K1478_P19" in name:
            continue
        val = getattr(rp, name)
        if isinstance(val, frozenset) and val == _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30:
            p20_aliases.append(name)
    assert not p20_aliases, (
        "K-1493 dead-code-by-design anti-pattern detected: alias-equal P20 "
        f"frozenset(s) found: {p20_aliases}.  Either delete the alias or "
        "extend the admit set with cells P19 does not already cover.")


def test_no_k1493_p20_predicate_in_module():
    """A K-1493 P20 predicate symbol implies a P20 frozenset has been
    re-introduced; that is allowed only when paired with a divergent
    admit set (covered by ``test_no_alias_p20_frozenset_in_module``)."""
    assert not hasattr(rp, "_k1493_p20_skinny_n16384_routeout"), (
        "K-1493 P20 predicate has been re-added; pair with a divergent "
        "admit set (cells P19 does not cover) and update this pin.")
