"""K-1881 (S-002) — P32–P38 alias-stack frozenset consolidation tests.

Extends K-1864 in-place to absorb P37 (N=352, K-1868) + P38 (N=384, K-1880).
ONE 120-cell frozenset replaces the would-be 7-probe chain.

Roster: N ∈ {96, 160, 224, 288, 320, 352, 384} = 18+18+18+18+17+17+14 = 120.
The "missing" cells (vs the PRD's ~125 estimate) are the 6 upstream-aliased
exclusions enforced by ``test_upstream_aliased_exclusions_are_preserved``.

Invariants:
  1. Canonical roster equals literal ``ALL_PROMOTED_CELLS`` (silent expansion
     or contraction fires).
  2. Per-slot N-axis projection cardinalities (18/18/18/18/17/17/14).
  3. Bit-identical routing replay: consolidated lookup == legacy 7-chain
     oracle on full roster + hostile-shape sample.
  4. The 6 upstream-aliased exclusions stay absent (parametrised).
  5. Predicate-audit boundary cases — the rejected compact predicates
     ``A: N%64!=0 ∧ N≤384 ∧ K≥4096`` and ``B: N%128!=0 ∧ N≤384 ∧ K≥4096``
     would silently mis-route specific cells; those cells are pinned in
     ``test_predicate_audit_boundary_*`` so the audit's NO-PREDICATE
     conclusion becomes an enforced invariant rather than an offline note.
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120,
    _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18,
    _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18,
    _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18,
    _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18,
    _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17,
    _P37_SKINNY_N352_KCOMPL_VERIFIED_WIN_17,
    _P38_SKINNY_N384_KCOMPL_VERIFIED_WIN_14,
)


CONSOLIDATED = _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120

# Literal enumeration of the FULL production roster of cells promoted by
# P32–P38 at fix/K-1881 HEAD.  Sorted by (N, M, K, dtype).  This is the
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
    # ---- P37 N=352 (17 cells, K-1843 N=352 sub-cohort, full grid MINUS
    #      (2048, 352, 4096, bf16) upstream-aliased exclusion) ----
    (2048, 352,  4096, "torch.float16"),
    (2048, 352,  8192, "torch.bfloat16"),
    (2048, 352,  8192, "torch.float16"),
    (2048, 352, 16384, "torch.bfloat16"),
    (2048, 352, 16384, "torch.float16"),
    (4096, 352,  4096, "torch.bfloat16"),
    (4096, 352,  4096, "torch.float16"),
    (4096, 352,  8192, "torch.bfloat16"),
    (4096, 352,  8192, "torch.float16"),
    (4096, 352, 16384, "torch.bfloat16"),
    (4096, 352, 16384, "torch.float16"),
    (8192, 352,  4096, "torch.bfloat16"),
    (8192, 352,  4096, "torch.float16"),
    (8192, 352,  8192, "torch.bfloat16"),
    (8192, 352,  8192, "torch.float16"),
    (8192, 352, 16384, "torch.bfloat16"),
    (8192, 352, 16384, "torch.float16"),
    # ---- P38 N=384 (14 cells, K-1857 N=384 sub-cohort, full grid MINUS
    #      the 4 (M∈{2048,4096})×K=8192×{bf16,fp16} cells already routed
    #      by upstream P30) ----
    (2048, 384,  4096, "torch.bfloat16"),
    (2048, 384,  4096, "torch.float16"),
    (2048, 384, 16384, "torch.bfloat16"),
    (2048, 384, 16384, "torch.float16"),
    (4096, 384,  4096, "torch.bfloat16"),
    (4096, 384,  4096, "torch.float16"),
    (4096, 384, 16384, "torch.bfloat16"),
    (4096, 384, 16384, "torch.float16"),
    (8192, 384,  4096, "torch.bfloat16"),
    (8192, 384,  4096, "torch.float16"),
    (8192, 384,  8192, "torch.bfloat16"),
    (8192, 384,  8192, "torch.float16"),
    (8192, 384, 16384, "torch.bfloat16"),
    (8192, 384, 16384, "torch.float16"),
)


def _chained_7frozenset_lookup_PRE_K1881(cell):
    """Reference oracle: faithful replay of the would-be 7-chain dispatch
    path (7 chained ``cell in <frozenset>`` membership checks in
    P32→P33→P34→P35→P36→P37→P38 order, short-circuiting via ``or``).
    Used to assert that the K-1881 single-frozenset consolidation is
    byte-for-byte equivalent to the legacy chain on the full
    ``ALL_PROMOTED_CELLS`` roster AND on hostile-shape control cells."""
    return (
        cell in _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18
        or cell in _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18
        or cell in _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18
        or cell in _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18
        or cell in _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17
        or cell in _P37_SKINNY_N352_KCOMPL_VERIFIED_WIN_17
        or cell in _P38_SKINNY_N384_KCOMPL_VERIFIED_WIN_14
    )


def _consolidated_lookup(cell):
    """The K-1881 single-frozenset dispatch path."""
    return cell in CONSOLIDATED


# ---------------------------------------------------------------------------
# 1. Roster pinning — the canonical 120-cell set is exactly the literal
#    enumeration above.  Any silent expansion, contraction, or tuple-shape
#    drift fires here at module load.
# ---------------------------------------------------------------------------

def test_canonical_cardinality_is_120():
    assert len(CONSOLIDATED) == 120


def test_promoted_cell_roster_has_120_unique_cells():
    """``ALL_PROMOTED_CELLS`` must enumerate the full 120-cell roster (no
    duplicates, no missing cells, no extras)."""
    assert len(ALL_PROMOTED_CELLS) == 120
    assert len(set(ALL_PROMOTED_CELLS)) == 120, (
        "ALL_PROMOTED_CELLS must contain 120 UNIQUE cells (sorted by N,M,K,dtype)"
    )


def test_canonical_equals_promoted_cell_roster():
    """The canonical inlined frozenset MUST be exactly the literal roster
    enumerated in ``ALL_PROMOTED_CELLS`` — symmetric difference is empty."""
    canonical_minus_roster = CONSOLIDATED - frozenset(ALL_PROMOTED_CELLS)
    roster_minus_canonical = frozenset(ALL_PROMOTED_CELLS) - CONSOLIDATED
    assert canonical_minus_roster == frozenset(), (
        f"K-1881 canonical frozenset has cells NOT in ALL_PROMOTED_CELLS "
        f"(silent expansion?): {sorted(canonical_minus_roster)}"
    )
    assert roster_minus_canonical == frozenset(), (
        f"ALL_PROMOTED_CELLS has cells NOT in K-1881 canonical frozenset "
        f"(silent contraction?): {sorted(roster_minus_canonical)}"
    )


# ---------------------------------------------------------------------------
# 2. Per-slot derived view cardinalities — the dispatcher does not consult
#    these, but they back the existing per-slot tests
#    (tests/test_p3{2..8}_*_alias_stack.py).
# ---------------------------------------------------------------------------

def test_per_slot_view_cardinalities():
    assert len(_P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18) == 18
    assert len(_P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18) == 18
    assert len(_P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18) == 18
    assert len(_P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18) == 18
    assert len(_P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17) == 17
    assert len(_P37_SKINNY_N352_KCOMPL_VERIFIED_WIN_17) == 17
    assert len(_P38_SKINNY_N384_KCOMPL_VERIFIED_WIN_14) == 14


def test_per_slot_views_partition_canonical():
    """The 7 per-slot N-axis projections MUST exactly partition the
    canonical 120-cell roster (their union recovers it; pairwise empty)."""
    union = (
        _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18
        | _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18
        | _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18
        | _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18
        | _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17
        | _P37_SKINNY_N352_KCOMPL_VERIFIED_WIN_17
        | _P38_SKINNY_N384_KCOMPL_VERIFIED_WIN_14
    )
    assert union == CONSOLIDATED


# ---------------------------------------------------------------------------
# 3. Empirical disjointness — the actual N-axis projection of the canonical
#    roster is exactly the 7 distinct integers {96, 160, 224, 288, 320,
#    352, 384}, so per-slot N-axis projections cannot overlap.  This is
#    checked against the actual literal roster, not against a relabelled
#    set.
# ---------------------------------------------------------------------------

def test_canonical_n_axis_projection_is_exactly_seven_distinct_rungs():
    n_axis = sorted({c[1] for c in CONSOLIDATED})
    assert n_axis == [96, 160, 224, 288, 320, 352, 384], (
        f"Canonical roster's N-axis projection must be exactly the 7 "
        f"distinct K-COMPLEMENT-promoted rungs "
        f"{{96,160,224,288,320,352,384}}; observed: {n_axis}.  Per-slot "
        f"views' pairwise disjointness depends on this — any new N rung "
        f"MUST land via a separate slot OR the consolidation invariant "
        f"must be re-derived."
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
        ("P37(N=352)", _P37_SKINNY_N352_KCOMPL_VERIFIED_WIN_17),
        ("P38(N=384)", _P38_SKINNY_N384_KCOMPL_VERIFIED_WIN_14),
    )
    for i, (name_i, s_i) in enumerate(constituents):
        for j, (name_j, s_j) in enumerate(constituents):
            if i >= j:
                continue
            overlap = s_i & s_j
            assert overlap == frozenset(), (
                f"K-1881 consolidation requires pairwise-disjoint per-slot "
                f"views; {name_i} ∩ {name_j} = {sorted(overlap)}"
            )


# ---------------------------------------------------------------------------
# 4. Bit-identical routing replay — THE load-bearing contract.  Every
#    promoted cell AND every hostile-shape control cell must yield the
#    SAME verdict from the K-1881 single-frozenset lookup as from the
#    legacy 7-chain reference oracle.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell", ALL_PROMOTED_CELLS)
def test_bit_identical_routing_for_every_promoted_cell(cell):
    """For every cell in the full 120-cell promoted-cell roster, the
    K-1881 single-frozenset dispatch MUST return the same verdict (True,
    in this case) as the legacy 7-chain reference oracle.  Parametrised
    so any drift fires at the cell granularity, not as a generic test
    failure."""
    chained = _chained_7frozenset_lookup_PRE_K1881(cell)
    consolidated = _consolidated_lookup(cell)
    assert chained is True, (
        f"K-1881 reference-oracle invariant: every cell in "
        f"ALL_PROMOTED_CELLS must route True under the legacy 7-chain "
        f"path; {cell} routed {chained}"
    )
    assert consolidated == chained, (
        f"K-1881 consolidation drift on promoted cell {cell}: "
        f"chained={chained}, consolidated={consolidated}"
    )


# Hostile-shape control sample — cells that should NOT route via P32–P38.
# Every hostile cell is at an N-axis value OUTSIDE the 7 productionised
# rungs {96, 160, 224, 288, 320, 352, 384}, plus the 6 upstream-aliased
# exclusion cells that were deliberately removed from the roster.  Any
# drift between chained and consolidated lookups on these cells indicates
# a silent routing-set expansion — both must return False.
def _hostile_control_cells():
    """Generate the hostile-shape control sample.  Uses N values that span
    the gaps BETWEEN the 7 productionised rungs and the rungs adjacent to
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
        # Between N=320 and N=352 (off-by-16; pin off-by-32 boundary
        # against silent N=336 inclusion)
        336,
        # Between N=352 and N=384 (off-by-16)
        368,
        # Above N=384 (adjacent rungs that are NOT in P32-P38)
        400, 416, 448, 480, 512, 768, 1024, 1280, 1536, 2048, 4096,
    )
    for M in (2048, 4096, 8192):
        for N in hostile_n:
            for K in (2048, 4096, 8192, 16384, 32768):
                for dt in ("torch.bfloat16", "torch.float16"):
                    cells.append((M, N, K, dt))
    # Plus the 6 upstream-aliased exclusion cells — must be False under
    # both paths (deliberately omitted from the canonical roster per
    # R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST):
    cells.append((2048, 320, 4096, "torch.bfloat16"))  # P36 parity-band
    cells.append((2048, 352, 4096, "torch.bfloat16"))  # P37 parity-band
    cells.append((2048, 384, 8192, "torch.bfloat16"))  # P38 ↔ P30 alias
    cells.append((2048, 384, 8192, "torch.float16"))   # P38 ↔ P30 alias
    cells.append((4096, 384, 8192, "torch.bfloat16"))  # P38 ↔ P30 alias
    cells.append((4096, 384, 8192, "torch.float16"))   # P38 ↔ P30 alias
    return tuple(cells)


HOSTILE_CONTROL_CELLS = _hostile_control_cells()


def test_hostile_control_sample_is_substantive():
    """Sanity check: the hostile sample must be at least an order of
    magnitude larger than the promoted-cell roster so the bit-identical
    test exercises the False branch densely."""
    assert len(HOSTILE_CONTROL_CELLS) >= 750, (
        f"hostile sample too small: {len(HOSTILE_CONTROL_CELLS)} cells "
        f"(want >= 750 — ~6x the 120-cell promoted roster — for "
        f"substantive negative-case coverage)"
    )


@pytest.mark.parametrize("cell", HOSTILE_CONTROL_CELLS)
def test_bit_identical_routing_for_every_hostile_cell(cell):
    """For every hostile-shape control cell, the K-1881 single-frozenset
    dispatch MUST return the same verdict (False, in every case) as the
    legacy 7-chain reference oracle.  Parametrised so any drift fires at
    the cell granularity — protects against silent routing-set expansion
    when future PMC slots land at adjacent N values."""
    chained = _chained_7frozenset_lookup_PRE_K1881(cell)
    consolidated = _consolidated_lookup(cell)
    assert chained is False, (
        f"K-1881 hostile-control invariant: hostile cell {cell} routed "
        f"True under legacy 7-chain path (test sample must exclude "
        f"productionised cells)"
    )
    assert consolidated == chained, (
        f"K-1881 consolidation drift on hostile cell {cell}: "
        f"chained={chained}, consolidated={consolidated}"
    )


# ---------------------------------------------------------------------------
# 5. Upstream-aliased exclusions — the 6 cells deliberately dropped from
#    P36/P37/P38 to avoid duplicate routing must remain absent from the
#    canonical roster.
# ---------------------------------------------------------------------------

UPSTREAM_ALIASED_EXCLUSIONS = (
    (2048, 320, 4096, "torch.bfloat16"),  # P36 parity-band
    (2048, 352, 4096, "torch.bfloat16"),  # P37 parity-band
    (2048, 384, 8192, "torch.bfloat16"),  # P38 ↔ P30 alias
    (2048, 384, 8192, "torch.float16"),   # P38 ↔ P30 alias
    (4096, 384, 8192, "torch.bfloat16"),  # P38 ↔ P30 alias
    (4096, 384, 8192, "torch.float16"),   # P38 ↔ P30 alias
)


@pytest.mark.parametrize("cell", UPSTREAM_ALIASED_EXCLUSIONS)
def test_upstream_aliased_exclusions_are_preserved(cell):
    """The 6 upstream-aliased cells excluded from P36/P37/P38 per
    R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST MUST remain absent from
    the consolidated roster, otherwise the consolidation would silently
    re-introduce duplicate routing.  These boundary cells are exactly the
    cells the predicate-audit (output/k1881_predicate_audit.txt) called
    out as the disqualifier of any compact-predicate substitution — the
    rejected predicates would silently admit them and route LOSER cells
    down the WRONG kernel path (R-K1825 violation).  Pinning them as
    parametrised invariants here promotes the offline audit's
    boundary-case conclusions to enforced regressions."""
    assert cell not in CONSOLIDATED, (
        f"upstream-aliased exclusion cell {cell} leaked back into the "
        f"K-1881 canonical roster — duplicate routing would result"
    )
    assert cell not in ALL_PROMOTED_CELLS


# ---------------------------------------------------------------------------
# 6. Predicate-audit boundary cases — these cells are why no compact
#    predicate substitutes for the literal frozenset.  Each entry pins
#    one of the failure modes documented in
#    output/k1881_predicate_audit.txt against silent regression.
# ---------------------------------------------------------------------------


def _predicate_a(M, N, K, dt): return (N % 64 != 0) and (N <= 384) and (K >= 4096)
def _predicate_b(M, N, K, dt): return (N % 128 != 0) and (N <= 384) and (K >= 4096)


# (cell, predicate-A admits?, predicate-B admits?, in canon?, failure-mode)
PREDICATE_AUDIT_BOUNDARIES = (
    # parity-band LOSERS — predicate-B silently admits, canon excludes:
    ((2048, 320, 4096, "torch.bfloat16"), False, True,  False, "B-admits-LOSER"),
    ((2048, 352, 4096, "torch.bfloat16"), True,  True,  False, "AB-admit-LOSER"),
    # P30-alias overlaps — both predicates AND canon agree (canon-excludes):
    ((2048, 384, 8192, "torch.bfloat16"), False, False, False, "all-exclude"),
    ((4096, 384, 8192, "torch.float16"),  False, False, False, "all-exclude"),
    # canon-INCLUDES that predicate-A would silently DROP (N=320, N=384 ≡ 0 mod 64):
    ((4096, 320, 4096, "torch.bfloat16"), False, True,  True,  "A-drops-WIN"),
    ((8192, 384, 4096, "torch.float16"),  False, False, True,  "AB-drop-WIN-N384"),
    # canon-EXCLUDES at N rungs predicate-B would over-admit (N=64, N=192):
    ((2048,  64, 4096, "torch.bfloat16"), False, True,  False, "B-over-admits-N64"),
    ((4096, 192, 8192, "torch.float16"),  False, True,  False, "B-over-admits-N192"),
)


@pytest.mark.parametrize("cell,pa,pb,in_canon,mode", PREDICATE_AUDIT_BOUNDARIES)
def test_predicate_audit_boundary_canon_membership(cell, pa, pb, in_canon, mode):
    """Pin the 8 boundary cells that justify rejecting the compact
    predicate (per output/k1881_predicate_audit.txt).  This makes the
    audit's exclusion conclusions an enforced regression: any future
    refactor that swaps the literal frozenset for one of the rejected
    predicates will fire here at the cell granularity, not as a generic
    routing-drift hit."""
    assert (cell in CONSOLIDATED) is in_canon, (
        f"predicate-audit boundary cell {cell} (mode={mode}): expected "
        f"in-canon={in_canon}, got {cell in CONSOLIDATED}"
    )


@pytest.mark.parametrize("cell,pa,pb,in_canon,mode", PREDICATE_AUDIT_BOUNDARIES)
def test_predicate_audit_boundary_predicate_disagreement(cell, pa, pb, in_canon, mode):
    """For each boundary cell, confirm that at least one of the rejected
    predicates disagrees with the canonical set.  This locks the
    "no-predicate-substitutes" conclusion: if a future engineer claims a
    compact predicate matches canon, this test fails on the very cells
    that disprove it."""
    M, N, K, dt = cell
    assert _predicate_a(M, N, K, dt) is pa, f"pred-A on {cell}"
    assert _predicate_b(M, N, K, dt) is pb, f"pred-B on {cell}"
    if mode != "all-exclude":
        # mode != "all-exclude" implies at least one predicate disagrees with canon
        assert (pa != in_canon) or (pb != in_canon), (
            f"boundary cell {cell} (mode={mode}) does NOT actually expose "
            f"a predicate-vs-canon disagreement — audit is wrong or boundary "
            f"is mislabelled"
        )
