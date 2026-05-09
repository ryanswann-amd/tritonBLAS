"""K-1513 / K-1543 P21 — skinny_N32768 K-COMPLEMENT 30-cell route-OUT unit tests.

Pytest invariants for the 14th-position frozenset
``_K1513_P21_SKINNY_N32768_KCOMPL_ROUTEOUT_30``:

  1. Cardinality / shape — exactly 30 cells, full M×K×dtype Cartesian product
     at N=32768.
  2. Membership — predicate returns True for every admit cell and False for
     a representative set of negative pins (sibling-N firewall, dtype carve-
     outs, M outside the admit set).
  3. Disjointness — single consolidated isdisjoint assert against the *union*
     of every prior predicate frozenset that lives in `_route_predicate`
     (catches any future predicate-chain edit that violates the sibling-N
     firewall, without a per-frozenset assert wall in module-load code).
"""
from __future__ import annotations

import itertools
import pytest

from tritonblas import _route_predicate as rp


P21 = rp._K1513_P21_SKINNY_N32768_KCOMPL_ROUTEOUT_30


# ----------------------------------------------------------------------------
# (1) Cardinality / shape
# ----------------------------------------------------------------------------

def test_p21_cardinality_is_30():
    assert len(P21) == 30


def test_p21_is_full_M_K_dtype_product_at_N32768():
    expected = {
        (M, 32768, K, dt)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dt in ("torch.bfloat16", "torch.float16")
    }
    assert P21 == expected


# ----------------------------------------------------------------------------
# (2) Membership — predicate function returns True/False as expected
# ----------------------------------------------------------------------------

class _DT:
    """Lightweight dtype stub whose `str()` matches `str(torch.bfloat16)`."""
    def __init__(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return self._name


@pytest.mark.parametrize("M,N,K,dt", sorted(P21))
def test_p21_predicate_admits_every_cell(M, N, K, dt):
    assert rp._k1513_p21_skinny_n32768_routeout(M, N, K, _DT(dt)) is True


@pytest.mark.parametrize(
    "M,N,K,dt_name",
    [
        # Sibling-N firewall negatives — same M/K/dtype, wrong N.
        (2048, 16384, 2048, "torch.bfloat16"),   # P19/P20 territory
        (4096, 65536, 4096, "torch.bfloat16"),   # un-swept frontier
        (8192,  1024, 2048, "torch.float16"),    # P16 territory
        # Out-of-set M
        (1024, 32768, 2048, "torch.bfloat16"),
        (16384, 32768, 8192, "torch.float16"),
        # Out-of-set K
        (2048, 32768,  1024, "torch.bfloat16"),
        (4096, 32768, 65536, "torch.float16"),
        # Out-of-set dtype (fp32 / fp8 carve-outs)
        (2048, 32768, 2048, "torch.float32"),
        (4096, 32768, 8192, "torch.float8_e4m3fnuz"),
    ],
)
def test_p21_predicate_rejects_negatives(M, N, K, dt_name):
    assert rp._k1513_p21_skinny_n32768_routeout(M, N, K, _DT(dt_name)) is False


# ----------------------------------------------------------------------------
# (3) Disjointness — single consolidated assert against the union of priors
# ----------------------------------------------------------------------------

# All prior strict-equality frozensets the P21 stack must avoid intersecting.
# A future predicate-chain edit that accidentally widens any of these to
# include an N=32768 cell will trip this single test.
_PRIOR_FROZENSETS = (
    "_P8_MFMA_ISSUE_STALL_ROUTEOUT",
    "K971_ROUTE_TABLE",
    "_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4",
    "_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18",
    "_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12",
    "_K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT",
    "_K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29",
    "_K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17",
    "_K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30",
    "_K1478_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_N",
)


def _prior_union():
    union = set()
    for name in _PRIOR_FROZENSETS:
        fs = getattr(rp, name, None)
        if fs is None:
            continue   # forward-compat: name renamed in a future refactor
        union |= set(fs)
    return union


def test_p21_disjoint_with_union_of_prior_predicates():
    """One consolidated disjointness assert — replaces 10 per-frozenset
    asserts.  Any future edit that lets a prior predicate admit an N=32768
    cell will trip this test (and only this test), pointing reviewers at
    the sibling-N firewall invariant."""
    overlap = P21 & _prior_union()
    assert overlap == set(), (
        f"P21 N=32768 envelope overlaps prior-predicate union at {len(overlap)} "
        f"cell(s): {sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''}. "
        f"Sibling-N firewall violated — a prior frozenset has been widened "
        f"to include N=32768."
    )
