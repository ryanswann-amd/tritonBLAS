"""K-2175-perfhawk-followup — equivalence test for the consolidated O(1)
`_KCOMPL_ALIASSTACK_UNION` frozenset that replaces the prior O(rungs)
chain of independent membership-checks in `_k971_route_to_hbl`.

Per K-2175-PR Performance-Hawk feedback: the chained dispatcher pattern
of one `(int(M), int(N), int(K), str(a_dtype)) in _K..._..._...: return
True` line per admitted rung is linear-in-rungs on every GEMM call and
rebuilds the lookup tuple on every line.  This consolidation collapses
the 6 in-dispatcher membership checks (K-1700 N=64 + K-1711 N∈{384,768,
1536} + P31 N=256 + K-1881 N∈{96..384} + K-1922 N=544 + K-2175 N=1648)
into one O(1) union frozenset.

This test pins the equivalence in three orthogonal ways:

  1. Cardinality — `len(union) == sum(len(part) for part in parts)` —
     proves the parts are pairwise disjoint AND nothing got dropped.

  2. Pairwise disjointness — `parts[i].isdisjoint(parts[j])` for every
     i < j — proves no part double-counted into the union.

  3. Identity — `union == parts[0] | parts[1] | ... | parts[N-1]` —
     proves the consolidated literal exactly matches the disjoint union
     of the parts.

Together (1)+(3) imply admit-set identity is preserved by the dispatcher
refactor (chained-checks → single union check), so the consolidation is
a strict refactor with zero observable behavior change.

The K-1908 closed-form predicate (`_is_k1908_offby48_kcompl_admit`) is
NOT a member of this union: it uses a closed-form mod-based admit
(1 modulo + 2 set lookups) and stays as a separate dispatcher entry
ahead of the union.  Disjointness with K-1908's admit-N {1520, 1584}
is naturally maintained by sibling-N firewall (no member of the union
has N ∈ {1520, 1584}); a regression test asserts this below.
"""
from __future__ import annotations

import itertools

import pytest

from tritonblas._route_predicate import (
    _KCOMPL_ALIASSTACK_UNION,
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29,
    _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
    _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120,
    _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
    _K2175_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18,
    _K1908_OFFBY48_ADMIT_N,
)


PARTS = (
    ("K-1700_P29_N64", _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29),
    ("K-1711_P30_NMID", _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34),
    ("P31_N256", _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28),
    ("K-1881_P32-P38", _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120),
    ("K-1922_P40_N544", _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18),
    ("K-2175_P57_N1648", _K2175_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18),
)
EXPECTED_TOTAL = 29 + 34 + 28 + 120 + 18 + 18  # 247


def test_union_equals_disjoint_union_of_parts():
    """Identity: the consolidated union literal exactly equals the |-of-parts."""
    expected = frozenset()
    for _, part in PARTS:
        expected = expected | part
    assert _KCOMPL_ALIASSTACK_UNION == expected


def test_union_cardinality_matches_sum_of_parts():
    """Cardinality + disjointness one-shot: |union| == Σ|parts|."""
    assert len(_KCOMPL_ALIASSTACK_UNION) == EXPECTED_TOTAL
    assert sum(len(part) for _, part in PARTS) == EXPECTED_TOTAL


def test_pairwise_disjointness_of_parts():
    """No two source frozensets share any (M, N, K, dtype) cell."""
    for (name_i, part_i), (name_j, part_j) in itertools.combinations(PARTS, 2):
        assert part_i.isdisjoint(part_j), (
            f"Alias-stack parts {name_i!r} and {name_j!r} share "
            f"{len(part_i & part_j)} cell(s); union dispatch would "
            f"double-count.  Sample overlap: {sorted(part_i & part_j)[:3]}"
        )


def test_union_is_disjoint_with_k1908_admit_n():
    """K-1908 closed-form (admit-N ∈ {1520, 1584}) and the union must be
    sibling-N disjoint so the dispatcher's K-1908-first-then-union ordering
    preserves admit-set identity."""
    union_ns = {N for (_, N, _, _) in _KCOMPL_ALIASSTACK_UNION}
    overlap = union_ns & _K1908_OFFBY48_ADMIT_N
    assert not overlap, (
        f"Alias-stack union N-set overlaps K-1908 admit-N at {sorted(overlap)}; "
        f"sibling-N firewall is broken — dispatcher routing semantics changed."
    )


def test_every_part_is_a_subset_of_union():
    """Trivially follows from identity; checked anyway for diagnostic surface."""
    for name, part in PARTS:
        assert part <= _KCOMPL_ALIASSTACK_UNION, (
            f"Part {name!r} ({len(part)} cells) not fully covered by the union; "
            f"dispatcher consolidation dropped {len(part - _KCOMPL_ALIASSTACK_UNION)} cell(s)."
        )


def test_k2175_n1648_is_in_the_union():
    """Specifically pin that the K-2175 P57 N=1648 admit-set is reached by
    the union check (the pre-consolidation chained dispatcher line for
    K-2175 was deleted; it is now reachable via the union only)."""
    assert _K2175_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18 <= _KCOMPL_ALIASSTACK_UNION
    # And every cell has N=1648
    for cell in _K2175_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18:
        assert cell in _KCOMPL_ALIASSTACK_UNION
        M, N, K, dt = cell
        assert N == 1648


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
