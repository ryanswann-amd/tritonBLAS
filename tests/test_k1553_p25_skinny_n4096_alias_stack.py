"""Unit fixture — K-1553 P25 skinny_N4096 K-COMPLEMENT 30-cell alias handle.

After the K-1581 minimalist refactor (per The Minimalist's REVISE feedback
on the prior duplicate-frozenset attempt, and following the K-1489
reviewer-consensus precedent for unreachable alias slots), the K-1553-named
17th-slot handle is exposed in `_route_predicate.py` as a single
module-level alias of the K-1566 P24 frozenset rather than as a duplicate
frozenset + asserts + predicate function:

    _K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30 = _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30

This fixture verifies the alias relationship and the transitive routing
contract — i.e. that every K-1553 admit cell still dispatches to hipBLASLt
via the load-bearing P24 predicate (so the alias remains a faithful audit
handle for the K-1553 30-cell envelope), and that the alias is
identity-equal (not merely value-equal) so any future reassignment of
either symbol is caught at module load.

Source measurement (still-of-record): K-1553 paired n=30 HIP-graph
hot-cache on MI300X / gfx942 (combined with K-1559 60-cell mid-band
N ∈ {4096, 8192} confirmation), TRITONBLAS_DISABLE_K971=1, B=10000
vectorised paired bootstrap against the live post-K-1532 routing oracle
(HEAD 95e2c47); 30/30 admit at strict ratio_median ≥ 1.05 ∧ p(<1.05) <
0.01 gate; cohort geomean tb/hbl = 1.234×, range 1.114×-1.501×, 0
regressions.  K-1566 P24 productionised the same envelope as the
load-bearing 16th-slot route-OUT.
"""
import pytest

from tritonblas._route_predicate import (
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4,
    _K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30,
    _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# (a) ALIAS IDENTITY — `_K1553_P25_...` is the same Python object as
#     `_K1566_P24_...`, not merely a value-equal copy.  Identity (rather
#     than equality) catches a future maintainer accidentally rebinding
#     P25 to a different frozenset literal that happens to be value-equal
#     today but could drift apart later.
# ---------------------------------------------------------------------------
def test_p25_is_identity_alias_of_p24():
    assert (
        _K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30
        is _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30
    ), (
        "K-1553 P25 17th-slot handle must be a Python identity alias of "
        "K-1566 P24 (same frozenset object), not a value-equal copy.  "
        "Per the K-1581 minimalist refactor, the K-1553-named handle is a "
        "single module-level rebinding of P24's frozenset; if this "
        "identity check fails, P25 has been re-authored as a duplicate "
        "literal — either restore the alias or document why P25 has "
        "diverged from P24 with paired n=30 evidence."
    )


# ---------------------------------------------------------------------------
# (b) ALIAS CARDINALITY — transitively inherited from P24 but pinned here
#     so a P24 contraction (which would also contract the K-1553 admit
#     handle) trips a K-1553-named test failure as well as a K-1566 one.
# ---------------------------------------------------------------------------
def test_p25_alias_resolves_to_thirty_cells():
    assert len(_K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30) == 30


def test_p25_alias_covers_full_n4096_kcompl_grid():
    expected = {
        (M, 4096, K, dtype)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dtype in ("torch.bfloat16", "torch.float16")
    }
    assert _K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30 == expected


# ---------------------------------------------------------------------------
# (c) ROUTING CONTRACT — every K-1553 admit cell still dispatches to
#     hipBLASLt via the public `k971_route_decision`, carried by the
#     load-bearing P24 predicate at the 16th slot.  This is what the
#     K-1553 audit handle ultimately asserts about runtime behaviour.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30)
)
def test_p25_admit_cell_routes_to_hbl(cell):
    M, N, K, dtype = cell
    assert (
        k971_route_decision(
            M, N, K, dtype, dtype,
            enable_streamk=False, work_stealing=False,
            disable_env_set=False,
        )
        is True
    ), f"K-1553 admit cell {cell} failed to dispatch to hipBLASLt via P24."


# ---------------------------------------------------------------------------
# (d) ALIAS-COVERAGE INVARIANT — every K-1553 cell must be in P24 ⨄ P12
#     (trivially true while the alias holds, but pinned explicitly so
#     reviewer-relevant invariants stay self-documenting).
# ---------------------------------------------------------------------------
def test_p25_cells_covered_by_p24_union_p12():
    union = (
        _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30
        | _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4
    )
    uncovered = _K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30 - union
    assert not uncovered, (
        f"K-1553 P25 alias has {len(uncovered)} cells not covered by "
        f"P24 ⨄ P12: {sorted(uncovered)}"
    )


def test_p25_p12_alias_overlap_is_exactly_two_cells():
    overlap = (
        _K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30
        & _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4
    )
    assert overlap == frozenset({
        (4096, 4096, 4096, "torch.bfloat16"),
        (4096, 4096, 4096, "torch.float16"),
    }), (
        "K-1553 P25 ∩ P12 must be exactly the (4096, 4096, 4096, "
        "{bf16, fp16}) diagonal pair (K-1295 P12 PMC-square-mid alias); "
        f"observed overlap: {sorted(overlap)}"
    )


# ---------------------------------------------------------------------------
# (e) CARVE-OUT NEGATIVES — streamk / work_stealing / dtype-mismatch must
#     short-circuit routing OFF even on K-1553 admit cells.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable_streamk": True,  "work_stealing": False},
        {"enable_streamk": False, "work_stealing": True},
    ],
    ids=["streamk_on", "work_stealing_on"],
)
def test_p25_admit_cell_carved_out_by_flags(kwargs):
    cell = (2048, 4096, 4096, "torch.bfloat16")  # canonical K-1553 admit
    M, N, K, dtype = cell
    assert k971_route_decision(
        M, N, K, dtype, dtype, disable_env_set=False, **kwargs,
    ) is False


def test_p25_admit_cell_carved_out_by_dtype_mismatch():
    M, N, K = 2048, 4096, 4096
    assert k971_route_decision(
        M, N, K, "torch.bfloat16", "torch.float16",
        enable_streamk=False, work_stealing=False, disable_env_set=False,
    ) is False
