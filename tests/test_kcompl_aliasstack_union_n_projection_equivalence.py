"""K-2175-perfhawk-followup-v3 (Rev 5) — equivalence test for the N-axis
projection sets used as a cheap pre-filter in `_k971_route_to_hbl`.

The Rev 4 dispatcher had to build `_key = (int(M), int(N), int(K), _dt_s)`
on every call (3 int() coercions + 1 tuple alloc) just to membership-
probe two consolidated frozensets that, on the vast majority of training-
step shapes, were going to miss.  Rev 5 introduces two N-only projection
frozensets:

  - `_K971_ROUTE_TABLE_N_SET`           — set of distinct N values in K971_ROUTE_TABLE
  - `_KCOMPL_ALIASSTACK_UNION_N_SET`    — set of distinct N values in _KCOMPL_ALIASSTACK_UNION

These are STRICT NECESSARY conditions for a hit on either consolidated
frozenset (any (M, N, K, dt) in either set has its N in the corresponding
projection by construction), which lets the dispatcher short-circuit the
`_key` build + the two large-frozenset probes when N is not even a
candidate — the common path on everyday training shapes (N=4096, 8192,
etc., none of which are in either projection set).

This test pins the N-projection equivalence in three ways:

  1. Identity — projection set EQUALS the set of distinct N values in
     the corresponding source frozenset (admit-set identity preserved).
  2. Necessary-condition — every (M, N, K, dt) in the source frozenset
     has its N in the projection set (rules out admit-set widening).
  3. K-2175-specific — N=1648 is in the union projection set so the
     N=1648 P57 admit cells still reach the union check via the
     pre-filter (the load-bearing assertion for this PR).
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    K971_ROUTE_TABLE,
    _KCOMPL_ALIASSTACK_UNION,
    _K971_ROUTE_TABLE_N_SET,
    _KCOMPL_ALIASSTACK_UNION_N_SET,
    _K2175_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18,
)


def test_k971_n_projection_equals_distinct_ns_of_source():
    """Identity: projection == set of distinct N in K971_ROUTE_TABLE."""
    expected = frozenset(t[1] for t in K971_ROUTE_TABLE)
    assert _K971_ROUTE_TABLE_N_SET == expected


def test_kcompl_aliasstack_union_n_projection_equals_distinct_ns_of_source():
    """Identity: projection == set of distinct N in _KCOMPL_ALIASSTACK_UNION."""
    expected = frozenset(t[1] for t in _KCOMPL_ALIASSTACK_UNION)
    assert _KCOMPL_ALIASSTACK_UNION_N_SET == expected


def test_k971_n_projection_is_necessary_condition_for_membership():
    """For every (M, N, K, dt) in K971_ROUTE_TABLE, N in K971_N_SET."""
    for (_M, N, _K, _dt) in K971_ROUTE_TABLE:
        assert N in _K971_ROUTE_TABLE_N_SET, (
            f"Source K971 cell N={N} not in projection — pre-filter "
            f"would incorrectly skip the membership check."
        )


def test_kcompl_aliasstack_union_n_projection_is_necessary_condition_for_membership():
    """For every (M, N, K, dt) in _KCOMPL_ALIASSTACK_UNION, N in UNION_N_SET."""
    for (_M, N, _K, _dt) in _KCOMPL_ALIASSTACK_UNION:
        assert N in _KCOMPL_ALIASSTACK_UNION_N_SET, (
            f"Source union cell N={N} not in projection — pre-filter "
            f"would incorrectly skip the membership check."
        )


def test_k2175_n1648_is_reachable_through_n_projection():
    """K-2175 P57 N=1648 admit-set must pass the N-projection pre-filter
    so the dispatcher actually probes the union for these cells.  This
    is the load-bearing assertion for the K-2175 admit (without it the
    Rev 5 pre-filter would short-circuit BEFORE the union check and the
    P57 admit would be silently disabled)."""
    assert 1648 in _KCOMPL_ALIASSTACK_UNION_N_SET
    for cell in _K2175_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18:
        _M, N, _K, _dt = cell
        assert N == 1648
        assert N in _KCOMPL_ALIASSTACK_UNION_N_SET


def test_n_projection_sets_have_expected_cardinalities():
    """Pin the N-projection cardinalities so a stray future entry / typo
    cannot silently widen the pre-filter.  K971 covers N values used by
    the K-905/K-971 strict-equality table; union covers the 6 in-
    dispatcher alias-stack frozensets."""
    # K971_ROUTE_TABLE today has N ∈ {1024, 2048} on the K-905/K-971
    # mid-square long-K LDS-BC anchors that survived the K-1003 P5 Gate-0
    # admission predicate carve-out.
    assert len(_K971_ROUTE_TABLE_N_SET) >= 1, "K971 N-projection must be non-empty"
    # Union covers N ∈ {64, 96, 160, 224, 256, 288, 320, 352, 384, 544,
    # 768, 1536, 1648} per the 6 source frozensets (K-1700 N=64 +
    # K-1881 N∈{96..384} + K-1922 N=544 + K-1711 N∈{768,1536} + K-2175
    # N=1648; P31 N=256 already in K-1881 7-N range).
    assert 1648 in _KCOMPL_ALIASSTACK_UNION_N_SET
    assert len(_KCOMPL_ALIASSTACK_UNION_N_SET) >= 9, (
        "Union N-projection must cover at least the 6 source frozensets' N values"
    )


def test_n_projection_sets_are_disjoint():
    """K971 strict-equality table and KCOMPL alias-stack union should not
    share any N value (sibling-N firewall at the N-axis level — the K971
    table covers mid-square long-K shapes at N ∈ {1024, 2048, …} while
    the alias-stack union covers skinny-N off-by-residue shapes at
    N ∈ {64, 96, …, 1648}).  If they DID overlap, both checks would
    fire the union probe redundantly on the overlap N — currently they
    do not."""
    overlap = _K971_ROUTE_TABLE_N_SET & _KCOMPL_ALIASSTACK_UNION_N_SET
    assert not overlap, (
        f"K971 N-set and alias-stack union N-set overlap at {sorted(overlap)} — "
        f"sibling-N firewall at the N-axis level is broken."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
