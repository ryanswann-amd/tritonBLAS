"""Unit fixture — K-1720 P30 skinny_Nmid (N ∈ {384, 768, 1536}) K-COMPLEMENT
34-cell alias-stack 20th-position route-OUT.

Productionizes the K-1711 cross-N envelope stitching audit at the
intermediate-large in-between-N values N ∈ {384, 768, 1536} that fall
between the productionized covered N-bins (P13/P21 N=256, P15/P17/P23/P27
N=512, P16 N=1024, P26 N=2048, P24 N=4096) as the 20th stacked frozenset.

Source measurement (still-of-record): K-1711 paired n=30 HIP-graph
hot-cache benchmarks on MI300X / gfx942 against the LIVE post-K-1685
P28 routing oracle (fork branch fix/K-1685 tip 6785fcd) with 2000-
sample paired bootstrap CI95 on `ratio = TB_time / hbl_time`:
36/72 cells flagged TB_speedup_over_hbl < 0.85 (cohort geomean tb/hbl
= 1.224×); 34/36 clear the strict CI95-lo > 1/0.85 = 1.176× gate;
cohort geomean ratio when persistently routed = 1.456×, range
1.18×-1.83×, 0 regressions.

The N=1024 control row in the audit (already covered by P16) reports
0/18 flagged cells with geomean 0.994× — falsifies the alternate
hypothesis that the gap is caused by some N-axis-orthogonal mechanism
(kernel correctness, hipBLASLt regression, harness noise) and confirms
the K-1685 alias-stack mechanism is load-bearing.

ALIAS-STACK structure: 34 NEW route-OUT cells + 0 alias cells (every
cell falls through every productionized predicate P1-P28 by construction
— the audit was on the in-between-N no-man's-land left by the P1-P28
N-projection {128, 256, 512, 1024, 2048, 4096, 16384, 32768}).  Sibling-
N firewall disjoint with all P1-P28.

Per-N admit shapes preserved via `_PER_N_ADMITS`:
  - N=384  : 10 cells (M ∈ {2048,4096,8192} × K ∈ {8192, 32768} ×
             {bf16, fp16}); K=2048 row excluded per R-1532 minimalist-
             admit-set (0/6 K=2048 N=384 cleared the K-1711 gate).
  - N=768  : 14 cells (full M × K × dtype admit at K ∈ {8192, 32768};
             K=2048 admits only the 2 M=8192 cells that cleared the gate);
             worst-seam N-bin per K-1711 (78% flag, geomean 1.372×).
  - N=1536 : 10 cells (full cohort-clear at K=32768 + M=4096 K ∈ {2048,
             8192}; M ∈ {2048, 8192} K=2048 cells fell below the CI95
             gate per K-1711 admit set).

Strict assertions per the K-1532 minimalist-admit-set + K-1175 stacked-
predicate convention; no GPU work performed by these tests (the K-1711
GPU evidence is the source-of-record, this fixture pins the productionized
shape against silent drift).
"""
import pytest

from tritonblas._route_predicate import (
    R_K979_P5_route_to_hbl,
    K971_ROUTE_TABLE,
    _p8_mfma_issue_stall_routeout,
    R_K1142_E1_route_to_hbl,
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4,
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT,
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29,
    _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17,
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30,
    _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT,
    _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30,
    _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30,
    _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30,
    _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30,
    _K1633_P27_SKINNY_N512_KCOMPL_ALIASSTACK_30,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _K1720_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34,
    _k1720_p30_skinny_nmid_kcompl_aliasstack_routeout,
    k971_route_decision,
)


# Per-N admit projections derived from the single source-of-truth frozenset.
# Per R-1532 / Minimalist feedback the source file holds ONE data structure;
# the per-N shape pins live in the test fixture below.
_PER_N_ADMITS = {
    n: frozenset(c for c in _K1720_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34
                 if c[1] == n)
    for n in (384, 768, 1536)
}


# ---------------------------------------------------------------------------
# (a) CARDINALITY — pinned to 34 cells (the K-1711 CI95-gated admit set:
#     10 N=384 + 14 N=768 + 10 N=1536).  Any deviation indicates an
#     authoring typo against the K-1711 admit set.
# ---------------------------------------------------------------------------
def test_p30_admit_set_cardinality_is_thirty_four():
    assert len(_K1720_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34) == 34


def test_p30_per_n_admit_shapes():
    assert len(_PER_N_ADMITS[384]) == 10
    assert len(_PER_N_ADMITS[768]) == 14
    assert len(_PER_N_ADMITS[1536]) == 10
    # Combined per-N admits == the 34-cell envelope (no per-N drift).
    union = _PER_N_ADMITS[384] | _PER_N_ADMITS[768] | _PER_N_ADMITS[1536]
    assert union == _K1720_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34


def test_p30_admit_set_n_axis_projection_is_three_intermediate_bins():
    n_values = {cell[1] for cell in _K1720_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34}
    assert n_values == {384, 768, 1536}


def test_p30_admit_set_m_axis_projection_is_canonical_three_rows():
    m_values = {cell[0] for cell in _K1720_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34}
    assert m_values == {2048, 4096, 8192}


def test_p30_admit_set_dtype_projection_is_bf16_and_fp16():
    dtypes = {cell[3] for cell in _K1720_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34}
    assert dtypes == {"torch.bfloat16", "torch.float16"}


# ---------------------------------------------------------------------------
# (b) ALIAS-STACK STRUCTURE — pinned at 34 NEW route-OUT cells + 0 alias
#     cells (no upstream firing-predicate alias).  Future widening of any
#     upstream P1-P28 admit set into N ∈ {384, 768, 1536} must trip these
#     tests so the K-1720 NEW route-OUT contribution is re-audited.
# ---------------------------------------------------------------------------
def test_p30_zero_upstream_alias_overlap():
    """Every K-1720 P30 cell is NEW route-OUT — no upstream predicate
    fires for any cell.  This is the K-1711 §4 mechanism falsifier."""
    new_routeout = frozenset(
        cell for cell in _K1720_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34
        if not (
            R_K979_P5_route_to_hbl(cell[0], cell[1], cell[2], cell[3])
            or _p8_mfma_issue_stall_routeout(cell[0], cell[1], cell[2], cell[3])
            or R_K1142_E1_route_to_hbl(cell[0], cell[1], cell[2], cell[3])
            or cell in K971_ROUTE_TABLE
        )
    )
    assert new_routeout == _K1720_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34
    assert len(new_routeout) == 34


def test_p30_no_p5_clause3_alias_overlap():
    """R-K979 P5 Clause-3 (bf16 minMN ≤ 192 ∧ K ≥ 2048) does NOT fire on
    K-1720 P30 because min(M, N) ≥ 384 for every cell."""
    p5_aliases = {
        cell for cell in _K1720_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34
        if R_K979_P5_route_to_hbl(cell[0], cell[1], cell[2], cell[3])
    }
    assert p5_aliases == frozenset()


def test_p30_no_p8_alias_overlap():
    """P8 MFMA-issue-stall envelope does NOT fire on K-1720 P30."""
    p8_aliases = {
        cell for cell in _K1720_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34
        if _p8_mfma_issue_stall_routeout(cell[0], cell[1], cell[2], cell[3])
    }
    assert p8_aliases == frozenset()


def test_p30_no_e1_alias_overlap():
    """K-1142 E1 axis-aligned envelope does NOT fire on K-1720 P30."""
    e1_aliases = {
        cell for cell in _K1720_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34
        if R_K1142_E1_route_to_hbl(cell[0], cell[1], cell[2], cell[3])
    }
    assert e1_aliases == frozenset()


def test_p30_no_k971_strict_table_alias_overlap():
    """K-905/K-971 strict-equality table does NOT contain any K-1720 P30
    cell (those anchors live at N ∈ {1024, 2048}, not N ∈ {384, 768, 1536})."""
    table_aliases = {
        cell for cell in _K1720_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34
        if cell in K971_ROUTE_TABLE
    }
    assert table_aliases == frozenset()


# ---------------------------------------------------------------------------
# (c) SIBLING-N FIREWALL — every K-COMPLEMENT predecessor frozenset must
#     be FULLY disjoint with K-1720 P30.  Productionized P1-P28 use N ∈
#     {128, 256, 512, 1024, 2048, 4096, 16384, 32768}; K-1720 P30 uses
#     N ∈ {384, 768, 1536} — disjoint by construction.  This is the
#     drift-detection guard required by K-1720 task spec ("no regressions
#     on prior productionized envelopes (P26/P27/P28)").
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sibling_name,sibling_set", [
    ("P12 (K-1361 M=N=K∈{2048,4096})",    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4),
    ("P13 N=128 (K-1367)",                _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18),
    ("P13 N=256 (K-1397)",                _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12),
    ("P15 N=512 (K-1409)",                _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT),
    ("P16 N=1024 (K-1429)",               _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29),
    ("P17 N=512 BASE (K-1437)",           _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17),
    ("P19 N=16384 (K-1478)",              _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30),
    ("P21 N=256 K-mid (K-1503)",          _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT),
    ("P22 N=32768 (K-1513)",              _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30),
    ("P23 N=512 alias (K-1552)",          _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30),
    ("P24 N=4096 (K-1566)",               _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30),
    ("P26 N=2048 alias (K-1611)",         _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30),
    ("P27 N=512 alias (K-1633)",          _K1633_P27_SKINNY_N512_KCOMPL_ALIASSTACK_30),
    ("P28 N=128 alias (K-1673)",          _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30),
])
def test_p30_disjoint_with_kcompl_predecessor(sibling_name, sibling_set):
    overlap = _K1720_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 & sibling_set
    assert overlap == frozenset(), (
        f"K-1720 P30 must be sibling-N-firewall disjoint with {sibling_name}; "
        f"overlap = {overlap}")


# ---------------------------------------------------------------------------
# (d) PER-N SUB-COHORT ADMIT-SHAPE PINS — preserves the K-1711 per-N
#     admit shape (10 N=384 / 14 N=768 / 10 N=1536) so a future per-N
#     audit can ablate / re-extend any of the three N-bins independently.
# ---------------------------------------------------------------------------
def test_p30_n384_subcohort_excludes_k2048_row():
    """K-1711 §5: 0/6 N=384 K=2048 cells cleared the 0.85x flag — per
    R-1532 minimalist-admit-set, those cells are NOT productionized."""
    n384 = _PER_N_ADMITS[384]
    k_values = {cell[2] for cell in n384}
    assert k_values == {8192, 32768}


def test_p30_n768_subcohort_includes_full_k_grid_at_m8192():
    """K-1711 §5: N=768 is the worst-seam N-bin (78% flag); the M=8192
    row admits the full K-grid {2048, 8192, 32768} per the K-1711 admit
    set."""
    n768 = _PER_N_ADMITS[768]
    m8192_k = {cell[2] for cell in n768 if cell[0] == 8192}
    assert m8192_k == {2048, 8192, 32768}


def test_p30_n1536_subcohort_admits_only_m4096_at_k2048():
    """K-1711 §5: N=1536 K=2048 admits only the 2 M=4096 cells that
    cleared the CI95 gate; M ∈ {2048, 8192} K=2048 cells fell below."""
    n1536 = _PER_N_ADMITS[1536]
    k2048_m = {cell[0] for cell in n1536 if cell[2] == 2048}
    assert k2048_m == {4096}


def test_p30_each_subcohort_covers_both_dtypes():
    """K-1711 admit pattern: cells are paired bf16/fp16 (every per-N
    per-(M,K) admit is dtype-mirrored)."""
    for n_bin, admits in _PER_N_ADMITS.items():
        # Group by (M, K); each group should have both dtypes.
        grouped = {}
        for cell in admits:
            grouped.setdefault((cell[0], cell[2]), set()).add(cell[3])
        for (m, k), dts in grouped.items():
            assert dts == {"torch.bfloat16", "torch.float16"}, (
                f"K-1720 P30 N={n_bin} (M={m}, K={k}) admits "
                f"only dtypes {dts}, expected both bf16 and fp16")


# ---------------------------------------------------------------------------
# (e) PREDICATE WIRING — `_k1720_p30_skinny_nmid_kcompl_aliasstack_routeout`
#     returns True for every cell in the admit set and False for nearby
#     cells outside the admit set.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell", sorted(_K1720_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34))
def test_p30_predicate_admits_every_envelope_cell(cell):
    M, N, K, dtype = cell
    assert _k1720_p30_skinny_nmid_kcompl_aliasstack_routeout(M, N, K, dtype)


def test_p30_predicate_rejects_nearby_uncovered_cells():
    """Cells that are NEAR the admit envelope but OUTSIDE it must reject."""
    # N=320 (between covered N=256 and uncovered N=384) — never measured,
    # not productionized.
    assert not _k1720_p30_skinny_nmid_kcompl_aliasstack_routeout(
        2048, 320, 8192, "torch.bfloat16")
    # N=512 (covered by P15/P17/P23/P27, NOT by P30).
    assert not _k1720_p30_skinny_nmid_kcompl_aliasstack_routeout(
        2048, 512, 8192, "torch.bfloat16")
    # N=384 K=2048 (excluded per R-1532 minimalist-admit-set).
    assert not _k1720_p30_skinny_nmid_kcompl_aliasstack_routeout(
        2048, 384, 2048, "torch.bfloat16")
    # N=1536 K=2048 M=2048 (fell below CI95 gate per K-1711).
    assert not _k1720_p30_skinny_nmid_kcompl_aliasstack_routeout(
        2048, 1536, 2048, "torch.bfloat16")
    # N=1024 K=8192 M=4096 (covered by P16 N=1024, NOT by P30).
    assert not _k1720_p30_skinny_nmid_kcompl_aliasstack_routeout(
        4096, 1024, 8192, "torch.bfloat16")


# ---------------------------------------------------------------------------
# (f) DRIFT-DETECTION on PRIOR PRODUCTIONIZED ENVELOPES (P26 / P27 / P28)
#     — the K-1720 task spec requires explicit no-regression assertions
#     on the most recent productionized alias-stacks.  K-1720 P30 must
#     not change the routing decision for any P26 / P27 / P28 cell (it's
#     a sibling-N-firewall add-only diff).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell", sorted(_K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30))
def test_p30_does_not_disturb_p26_n2048_routing(cell):
    M, N, K, dtype = cell
    # The new P30 predicate must NOT fire on any P26 cell.
    assert not _k1720_p30_skinny_nmid_kcompl_aliasstack_routeout(M, N, K, dtype)
    # End-to-end: P26 cells continue to route OUT (via the dispatch chain).
    decided = k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False)
    assert decided is True, (
        f"K-1720 P30 must not regress P26 routing for {cell}; "
        f"k971_route_decision = {decided}, expected True")


@pytest.mark.parametrize("cell", sorted(_K1633_P27_SKINNY_N512_KCOMPL_ALIASSTACK_30))
def test_p30_does_not_disturb_p27_n512_routing(cell):
    M, N, K, dtype = cell
    assert not _k1720_p30_skinny_nmid_kcompl_aliasstack_routeout(M, N, K, dtype)
    decided = k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False)
    assert decided is True, (
        f"K-1720 P30 must not regress P27 routing for {cell}; "
        f"k971_route_decision = {decided}, expected True")


@pytest.mark.parametrize("cell", sorted(_K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30))
def test_p30_does_not_disturb_p28_n128_routing(cell):
    M, N, K, dtype = cell
    assert not _k1720_p30_skinny_nmid_kcompl_aliasstack_routeout(M, N, K, dtype)
    decided = k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False)
    assert decided is True, (
        f"K-1720 P30 must not regress P28 routing for {cell}; "
        f"k971_route_decision = {decided}, expected True")


# ---------------------------------------------------------------------------
# (g) END-TO-END ROUTING DECISION — every K-1720 P30 cell routes OUT
#     under the LIVE oracle dispatch (`k971_route_decision`).  This is
#     the productionization smoke test: the new 20th-position predicate
#     fires AND no earlier slot pre-empts it.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell", sorted(_K1720_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34))
def test_p30_cell_routes_out_under_live_oracle(cell):
    M, N, K, dtype = cell
    decided = k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False)
    assert decided is True, (
        f"K-1720 P30 cell {cell} must route OUT under LIVE oracle; "
        f"k971_route_decision = {decided}")


# ---------------------------------------------------------------------------
# (h) STREAMK / WORK-STEALING SHORT-CIRCUIT — k971_route_decision early-
#     returns False when enable_streamk=True or work_stealing=True; K-1720
#     P30 inherits this gate from the dispatch chain.
# ---------------------------------------------------------------------------
def test_p30_streamk_short_circuits_routeout():
    M, N, K, dtype = 4096, 768, 32768, "torch.bfloat16"
    assert k971_route_decision(M, N, K, dtype, dtype,
                               enable_streamk=True, work_stealing=False) is False


def test_p30_work_stealing_short_circuits_routeout():
    M, N, K, dtype = 4096, 768, 32768, "torch.bfloat16"
    assert k971_route_decision(M, N, K, dtype, dtype,
                               enable_streamk=False, work_stealing=True) is False
