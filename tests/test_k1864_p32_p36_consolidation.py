"""K-1864 (S-002) — P32–P36 alias-stack frozenset consolidation tests.

Audit context: with 5 consecutive K-COMPLEMENT route-out promotions landing in
``_k971_route_to_hbl()`` (P32 N=160 / P33 N=224 / P34 N=96 / P35 N=288 / P36
N=320), the dispatcher walked 5 separate ``frozenset`` membership checks
back-to-back — one tuple build and 5 hash lookups per dispatch in the worst
case.  K-1864 collapses the 5 chained checks into ONE membership probe over a
single 89-cell frozenset
(``_K1864_P32_P36_SKINNY_KCOMPL_ROUTEOUT_89``) whose admit cells are inlined
as literal tuples in ``_route_predicate.py`` (single source of truth).

PRD enumerated the audit cohort as "P32 N=160, P33 N=192, P34 N=128/N=288/N=96
variants, P35 N=288, P36 N=320 totaling ~106+ cells".  The actual production
roster as merged at fix/K-1850 HEAD is **89 cells** at N ∈ {96, 160, 224, 288,
320}; the PRD's earlier figure was an upper-bound estimate before each
slot's K-1794/K-1818/K-1832/K-1843 paired-n30 sub-cohort enumeration trimmed
the grid (e.g. the K-1810 PR landed N=160 not N=192; K-1843 P36 dropped
(2048, 320, 4096, bf16) as upstream-aliased).  The full production roster is
enumerated cell-by-cell as ``ALL_PROMOTED_CELLS`` below — the source of truth
for the bit-identical-routing replay against the legacy 5-chain reference
oracle.

Invariants pinned by this file (per the minimalist split convention
R-1532 / R-1720 / R-1775 — source holds data, tests hold structural
invariants):

  1. The canonical 89-cell roster is exactly the literal cells enumerated
     here (any silent expansion, contraction, or tuple-shape drift fires).
  2. The 5 per-slot N-axis projection views derived in ``_route_predicate.py``
     each return the expected sub-cardinality (P32=18 / P33=18 / P34=18 /
     P35=18 / P36=17).
  3. Pairwise disjointness is empirical (not a relabelled set): the actual
     N-axis projection of the canonical roster is exactly the 5 distinct
     integers {96, 160, 224, 288, 320}, so per-slot views cannot overlap.
  4. **Bit-identical routing replay**: for every cell in ``ALL_PROMOTED_CELLS``
     AND every cell in a hostile-shape control sample (cells that should
     NOT route via P32–P36), the consolidated single-frozenset lookup
     returns the SAME verdict as the legacy 5-chain reference oracle
     ``_chained_5frozenset_lookup_PRE_K1864``.  This is the load-bearing
     contract: dispatcher behaviour must be byte-for-byte preserved across
     the consolidation.
  5. The single upstream-aliased exclusion at (2048, 320, 4096, bf16) is
     preserved (per R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST).
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K1864_P32_P36_SKINNY_KCOMPL_ROUTEOUT_89,
    _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18,
    _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18,
    _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18,
    _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18,
    _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17,
)


CONSOLIDATED = _K1864_P32_P36_SKINNY_KCOMPL_ROUTEOUT_89

# Literal enumeration of the FULL production roster of cells promoted by
# P32–P36 at fix/K-1850 HEAD.  Sorted by (N, M, K, dtype).  This is the
# source of truth for the bit-identical-routing replay below — the test
# does NOT derive this set from the consolidated frozenset (which would
# tautologically pass).
ALL_PROMOTED_CELLS = (
    # ---- P34 N=96 (18 cells, K-1818 sub-cohort) ----
    (2048,  96,  4096, "torch.bfloat16"),
    (2048,  96,  4096, "torch.float16"),
    (2048,  96,  8192, "torch.bfloat16"),
    (2048,  96,  8192, "torch.float16"),
    (2048,  96, 16384, "torch.bfloat16"),
    (2048,  96, 16384, "torch.float16"),
    (4096,  96,  4096, "torch.bfloat16"),
    (4096,  96,  4096, "torch.float16"),
    (4096,  96,  8192, "torch.bfloat16"),
    (4096,  96,  8192, "torch.float16"),
    (4096,  96, 16384, "torch.bfloat16"),
    (4096,  96, 16384, "torch.float16"),
    (8192,  96,  4096, "torch.bfloat16"),
    (8192,  96,  4096, "torch.float16"),
    (8192,  96,  8192, "torch.bfloat16"),
    (8192,  96,  8192, "torch.float16"),
    (8192,  96, 16384, "torch.bfloat16"),
    (8192,  96, 16384, "torch.float16"),
    # ---- P32 N=160 (18 cells, K-1794 sub-cohort) ----
    (2048, 160,  4096, "torch.bfloat16"),
    (2048, 160,  4096, "torch.float16"),
    (2048, 160,  8192, "torch.bfloat16"),
    (2048, 160,  8192, "torch.float16"),
    (2048, 160, 16384, "torch.bfloat16"),
    (2048, 160, 16384, "torch.float16"),
    (4096, 160,  4096, "torch.bfloat16"),
    (4096, 160,  4096, "torch.float16"),
    (4096, 160,  8192, "torch.bfloat16"),
    (4096, 160,  8192, "torch.float16"),
    (4096, 160, 16384, "torch.bfloat16"),
    (4096, 160, 16384, "torch.float16"),
    (8192, 160,  4096, "torch.bfloat16"),
    (8192, 160,  4096, "torch.float16"),
    (8192, 160,  8192, "torch.bfloat16"),
    (8192, 160,  8192, "torch.float16"),
    (8192, 160, 16384, "torch.bfloat16"),
    (8192, 160, 16384, "torch.float16"),
    # ---- P33 N=224 (18 cells, K-1794 sub-cohort) ----
    (2048, 224,  4096, "torch.bfloat16"),
    (2048, 224,  4096, "torch.float16"),
    (2048, 224,  8192, "torch.bfloat16"),
    (2048, 224,  8192, "torch.float16"),
    (2048, 224, 16384, "torch.bfloat16"),
    (2048, 224, 16384, "torch.float16"),
    (4096, 224,  4096, "torch.bfloat16"),
    (4096, 224,  4096, "torch.float16"),
    (4096, 224,  8192, "torch.bfloat16"),
    (4096, 224,  8192, "torch.float16"),
    (4096, 224, 16384, "torch.bfloat16"),
    (4096, 224, 16384, "torch.float16"),
    (8192, 224,  4096, "torch.bfloat16"),
    (8192, 224,  4096, "torch.float16"),
    (8192, 224,  8192, "torch.bfloat16"),
    (8192, 224,  8192, "torch.float16"),
    (8192, 224, 16384, "torch.bfloat16"),
    (8192, 224, 16384, "torch.float16"),
    # ---- P35 N=288 (18 cells, K-1832 sub-cohort) ----
    (2048, 288,  4096, "torch.bfloat16"),
    (2048, 288,  4096, "torch.float16"),
    (2048, 288,  8192, "torch.bfloat16"),
    (2048, 288,  8192, "torch.float16"),
    (2048, 288, 16384, "torch.bfloat16"),
    (2048, 288, 16384, "torch.float16"),
    (4096, 288,  4096, "torch.bfloat16"),
    (4096, 288,  4096, "torch.float16"),
    (4096, 288,  8192, "torch.bfloat16"),
    (4096, 288,  8192, "torch.float16"),
    (4096, 288, 16384, "torch.bfloat16"),
    (4096, 288, 16384, "torch.float16"),
    (8192, 288,  4096, "torch.bfloat16"),
    (8192, 288,  4096, "torch.float16"),
    (8192, 288,  8192, "torch.bfloat16"),
    (8192, 288,  8192, "torch.float16"),
    (8192, 288, 16384, "torch.bfloat16"),
    (8192, 288, 16384, "torch.float16"),
    # ---- P36 N=320 (17 cells, K-1843 full grid MINUS the
    #      (2048, 320, 4096, bf16) upstream-aliased exclusion) ----
    (2048, 320,  4096, "torch.float16"),
    (2048, 320,  8192, "torch.bfloat16"),
    (2048, 320,  8192, "torch.float16"),
    (2048, 320, 16384, "torch.bfloat16"),
    (2048, 320, 16384, "torch.float16"),
    (4096, 320,  4096, "torch.bfloat16"),
    (4096, 320,  4096, "torch.float16"),
    (4096, 320,  8192, "torch.bfloat16"),
    (4096, 320,  8192, "torch.float16"),
    (4096, 320, 16384, "torch.bfloat16"),
    (4096, 320, 16384, "torch.float16"),
    (8192, 320,  4096, "torch.bfloat16"),
    (8192, 320,  4096, "torch.float16"),
    (8192, 320,  8192, "torch.bfloat16"),
    (8192, 320,  8192, "torch.float16"),
    (8192, 320, 16384, "torch.bfloat16"),
    (8192, 320, 16384, "torch.float16"),
)


def _chained_5frozenset_lookup_PRE_K1864(cell):
    """Reference oracle: faithful replay of the pre-K-1864 dispatch path
    (5 chained ``cell in <frozenset>`` membership checks in P32→P33→P34→P35
    →P36 order, short-circuiting via ``or``).  Used to assert that the
    K-1864 single-frozenset consolidation is byte-for-byte equivalent to
    the legacy chain on the full ``ALL_PROMOTED_CELLS`` roster AND on
    hostile-shape control cells."""
    return (
        cell in _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18
        or cell in _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18
        or cell in _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18
        or cell in _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18
        or cell in _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17
    )


def _consolidated_lookup(cell):
    """The K-1864 single-frozenset dispatch path."""
    return cell in CONSOLIDATED


# ---------------------------------------------------------------------------
# 1. Roster pinning — the canonical 89-cell set is exactly the literal
#    enumeration above.  Any silent expansion, contraction, or tuple-shape
#    drift fires here at module load.
# ---------------------------------------------------------------------------

def test_canonical_cardinality_is_89():
    assert len(CONSOLIDATED) == 89


def test_promoted_cell_roster_has_89_unique_cells():
    """``ALL_PROMOTED_CELLS`` must enumerate the full 89-cell roster (no
    duplicates, no missing cells, no extras)."""
    assert len(ALL_PROMOTED_CELLS) == 89
    assert len(set(ALL_PROMOTED_CELLS)) == 89, (
        "ALL_PROMOTED_CELLS must contain 89 UNIQUE cells (sorted by N,M,K,dtype)"
    )


def test_canonical_equals_promoted_cell_roster():
    """The canonical inlined frozenset MUST be exactly the literal roster
    enumerated in ``ALL_PROMOTED_CELLS`` — symmetric difference is empty."""
    canonical_minus_roster = CONSOLIDATED - frozenset(ALL_PROMOTED_CELLS)
    roster_minus_canonical = frozenset(ALL_PROMOTED_CELLS) - CONSOLIDATED
    assert canonical_minus_roster == frozenset(), (
        f"K-1864 canonical frozenset has cells NOT in ALL_PROMOTED_CELLS "
        f"(silent expansion?): {sorted(canonical_minus_roster)}"
    )
    assert roster_minus_canonical == frozenset(), (
        f"ALL_PROMOTED_CELLS has cells NOT in K-1864 canonical frozenset "
        f"(silent contraction?): {sorted(roster_minus_canonical)}"
    )


# ---------------------------------------------------------------------------
# 2. Per-slot derived view cardinalities — the dispatcher does not consult
#    these, but they back the existing per-slot tests
#    (tests/test_p3{2..6}_*_alias_stack.py).
# ---------------------------------------------------------------------------

def test_per_slot_view_cardinalities():
    assert len(_P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18) == 18
    assert len(_P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18) == 18
    assert len(_P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18) == 18
    assert len(_P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18) == 18
    assert len(_P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17) == 17


def test_per_slot_views_partition_canonical():
    """The 5 per-slot N-axis projections MUST exactly partition the
    canonical 89-cell roster (their union recovers it; pairwise empty)."""
    union = (
        _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18
        | _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18
        | _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18
        | _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18
        | _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17
    )
    assert union == CONSOLIDATED


# ---------------------------------------------------------------------------
# 3. Empirical disjointness — the actual N-axis projection of the canonical
#    roster is exactly the 5 distinct integers {96, 160, 224, 288, 320},
#    so per-slot N-axis projections cannot overlap.  This is checked
#    against the actual literal roster, not against a relabelled set.
# ---------------------------------------------------------------------------

def test_canonical_n_axis_projection_is_exactly_five_distinct_rungs():
    n_axis = sorted({c[1] for c in CONSOLIDATED})
    assert n_axis == [96, 160, 224, 288, 320], (
        f"Canonical roster's N-axis projection must be exactly the 5 "
        f"distinct K-COMPLEMENT-promoted rungs {{96,160,224,288,320}}; "
        f"observed: {n_axis}.  Per-slot views' pairwise disjointness "
        f"depends on this — any new N rung MUST land via a separate slot "
        f"OR the consolidation invariant must be re-derived."
    )


def test_per_slot_views_are_pairwise_disjoint():
    """Empirical pairwise-disjointness check on the actual derived views
    (NOT a tautology over a relabelled set): every constituent slot's
    N-axis projection on the canonical roster shares zero cells with
    every other slot."""
    constituents = (
        ("P32(N=160)", _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18),
        ("P33(N=224)", _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18),
        ("P34(N=96)",  _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18),
        ("P35(N=288)", _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18),
        ("P36(N=320)", _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17),
    )
    for i, (name_i, s_i) in enumerate(constituents):
        for j, (name_j, s_j) in enumerate(constituents):
            if i >= j:
                continue
            overlap = s_i & s_j
            assert overlap == frozenset(), (
                f"K-1864 consolidation requires pairwise-disjoint per-slot "
                f"views; {name_i} ∩ {name_j} = {sorted(overlap)}"
            )


# ---------------------------------------------------------------------------
# 4. Bit-identical routing replay — THE load-bearing contract.  Every
#    promoted cell AND every hostile-shape control cell must yield the
#    SAME verdict from the K-1864 single-frozenset lookup as from the
#    legacy 5-chain reference oracle.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell", ALL_PROMOTED_CELLS)
def test_bit_identical_routing_for_every_promoted_cell(cell):
    """For every cell in the full 89-cell promoted-cell roster, the
    K-1864 single-frozenset dispatch MUST return the same verdict (True,
    in this case) as the legacy 5-chain reference oracle.  Parametrised
    so any drift fires at the cell granularity, not as a generic test
    failure."""
    chained = _chained_5frozenset_lookup_PRE_K1864(cell)
    consolidated = _consolidated_lookup(cell)
    assert chained is True, (
        f"K-1864 reference-oracle invariant: every cell in "
        f"ALL_PROMOTED_CELLS must route True under the legacy 5-chain "
        f"path; {cell} routed {chained}"
    )
    assert consolidated == chained, (
        f"K-1864 consolidation drift on promoted cell {cell}: "
        f"chained={chained}, consolidated={consolidated}"
    )


# Hostile-shape control sample — cells that should NOT route via P32–P36.
# Every hostile cell is at an N-axis value OUTSIDE the 5 productionised
# rungs {96, 160, 224, 288, 320}, plus the upstream-aliased P36 exclusion
# cell that was deliberately removed from the roster.  Any drift between
# chained and consolidated lookups on these cells indicates a silent
# routing-set expansion — both must return False.
def _hostile_control_cells():
    """Generate the hostile-shape control sample.  Uses N values that span
    the gaps BETWEEN the 5 productionised rungs and the rungs adjacent to
    them (so any one-off / off-by-32 / off-by-64 drift would surface)."""
    cells = []
    hostile_n = (
        # Below the N=96 rung
        16, 32, 48, 64, 80,
        # Between N=96 and N=160 (off-by-32 / off-by-64)
        112, 128, 144,
        # Between N=160 and N=224 (off-by-32 / off-by-64)
        176, 192, 208,
        # Between N=224 and N=288 (off-by-32 / off-by-64)
        240, 256, 272,
        # Between N=288 and N=320 (off-by-32)
        304,
        # Above N=320 (adjacent rungs that are NOT in P32-P36)
        336, 352, 384, 512, 768, 1024, 1280, 1536, 2048, 4096,
    )
    for M in (2048, 4096, 8192):
        for N in hostile_n:
            for K in (2048, 4096, 8192, 16384, 32768):
                for dt in ("torch.bfloat16", "torch.float16"):
                    cells.append((M, N, K, dt))
    # Plus the single upstream-aliased P36 exclusion cell — must be False
    # under both paths (it was deliberately omitted from the canonical
    # roster per R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST).
    cells.append((2048, 320, 4096, "torch.bfloat16"))
    return tuple(cells)


HOSTILE_CONTROL_CELLS = _hostile_control_cells()


def test_hostile_control_sample_is_substantive():
    """Sanity check: the hostile sample must be at least an order of
    magnitude larger than the promoted-cell roster so the bit-identical
    test exercises the False branch densely."""
    assert len(HOSTILE_CONTROL_CELLS) >= 750, (
        f"hostile sample too small: {len(HOSTILE_CONTROL_CELLS)} cells "
        f"(want >= 750 — ~8x the 89-cell promoted roster — for "
        f"substantive negative-case coverage)"
    )


@pytest.mark.parametrize("cell", HOSTILE_CONTROL_CELLS)
def test_bit_identical_routing_for_every_hostile_cell(cell):
    """For every hostile-shape control cell, the K-1864 single-frozenset
    dispatch MUST return the same verdict (False, in every case) as the
    legacy 5-chain reference oracle.  Parametrised so any drift fires at
    the cell granularity — protects against silent routing-set expansion
    when future PMC slots land at adjacent N values."""
    chained = _chained_5frozenset_lookup_PRE_K1864(cell)
    consolidated = _consolidated_lookup(cell)
    assert chained is False, (
        f"K-1864 hostile-control invariant: hostile cell {cell} routed "
        f"True under legacy 5-chain path (test sample must exclude "
        f"productionised cells)"
    )
    assert consolidated == chained, (
        f"K-1864 consolidation drift on hostile cell {cell}: "
        f"chained={chained}, consolidated={consolidated}"
    )


# ---------------------------------------------------------------------------
# 5. Upstream-aliased exclusion — the one cell deliberately dropped from
#    P36 to avoid duplicate routing must remain absent from the canonical
#    roster.
# ---------------------------------------------------------------------------

def test_p36_upstream_aliased_exclusion_is_preserved():
    """The (2048, 320, 4096, bf16) cell is excluded from P36 per
    R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST (already routed by an
    upstream alias-stack slot at paired-n30 ratio_TB/HBL=1.0001, p=0.293
    — within parity band).  It MUST remain absent from the consolidated
    roster, otherwise consolidation would silently re-introduce duplicate
    routing for that single cell."""
    assert (2048, 320, 4096, "torch.bfloat16") not in CONSOLIDATED
    assert (2048, 320, 4096, "torch.bfloat16") not in ALL_PROMOTED_CELLS
