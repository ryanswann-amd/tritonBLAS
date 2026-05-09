"""Unit fixture — K-1673 P28 skinny_N128 K-COMPLEMENT 30-cell alias-stack
19th-position route-OUT.

Productionizes the K-1673 verification of the N=128 K-COMPLEMENT envelope
(M ∈ {2048, 4096, 8192} × N=128 × K ∈ {2048, 4096, 8192, 16384, 32768} ×
{bf16, fp16}) as the 19th stacked frozenset, closing the dtype-symmetry
gap left by K-1367 P13 (7th-slot, 18 cells, K ∈ {4096, 8192, 16384} only).

Source measurement (still-of-record): K-1673 paired n=30 HIP-graph hot-
cache benchmark on MI300X / gfx942 (per INFRA-0048 c42 SSH refused
fallback) against the LIVE post-K-1647 P27 routing
oracle on TB fork branch fix/K-1647 HEAD `8d010c6`; engines: TB →
tritonblas.matmul → persistent_matmul; HBL → hipBLASLt direct.
30/30 lose to hipBLASLt by ≥10% (cohort geomean 1.694×, worst 2.890×
at (8192, 128, 2048, fp16)); 24 cells already routed-OUT by P5 ⨄
P13_N128; 6 fp16 mirror cells slip through every existing predicate at
the K-EXTREMES sub-band — these are the load-bearing P28 contribution.

ALIAS-STACK structure: 6 NEW route-OUT cells (the fp16 mirror at K ∈
{2048, 32768}) + 24 alias cells (15 bf16 alias-of-R-K979-P5 ∪ 18
K∈{4096,8192,16384} alias-of-K-1367-P13_N128, intersection 9, union 24).
Alias overlaps fire BEFORE P28 in the dispatch chain (P5 at 4th-slot,
P13 at 7th-slot) so the 24 alias cells are unreachable under normal
dispatch — load-bearing only if either upstream layer is ablated.
"""
import pytest

from tritonblas._route_predicate import (
    K971_ROUTE_TABLE,
    R_K979_P5_route_to_hbl,
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30,
    _K1673_P28_NEW_ROUTEOUT_6,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _K1673_P28_VS_K1367_P13_ALIAS_OVERLAP,
    _K1673_P28_VS_R_K979_P5_BF16_ALIAS,
    _k1673_p28_skinny_n128_kcompl_aliasstack_routeout,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# (a) CARDINALITY — pinned to 30 cells (the K-1673 envelope: M ∈ {2048,
#     4096, 8192} × N=128 × K ∈ {2048, 4096, 8192, 16384, 32768} × {bf16,
#     fp16}).  Any deviation indicates an authoring typo against the
#     K-1673 paired n=30 admit set.
# ---------------------------------------------------------------------------
def test_p28_admit_set_cardinality_is_thirty():
    assert len(_K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30) == 30


def test_p28_admit_set_covers_full_n128_kcompl_grid():
    expected = {
        (M, 128, K, dtype)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dtype in ("torch.bfloat16", "torch.float16")
    }
    assert _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == expected


# ---------------------------------------------------------------------------
# (b) ALIAS-OVERLAP STRUCTURE — pinned at exactly 24 alias cells across 2
#     upstream predicates (R-K979 P5 closed-form covers 15 bf16; K-1367
#     P13 strict-equality covers 18 K∈{4096,8192,16384}; intersection 9
#     bf16 K∈{4096,8192,16384}; union 24).  Future contractions of either
#     upstream predicate must trip these tests so the K-1673 NEW route-
#     OUT contribution is re-audited.
# ---------------------------------------------------------------------------
def test_p28_p13_n128_alias_overlap_is_exactly_eighteen_cells():
    assert _K1673_P28_VS_K1367_P13_ALIAS_OVERLAP == frozenset({
        (M, 128, K, dtype)
        for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dtype in ("torch.bfloat16", "torch.float16")
    })
    assert len(_K1673_P28_VS_K1367_P13_ALIAS_OVERLAP) == 18


def test_p28_p5_bf16_alias_is_exactly_fifteen_cells():
    assert _K1673_P28_VS_R_K979_P5_BF16_ALIAS == frozenset({
        (M, 128, K, "torch.bfloat16")
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
    })
    assert len(_K1673_P28_VS_R_K979_P5_BF16_ALIAS) == 15


def test_p28_p5_alias_actually_admitted_by_p5_closed_form():
    # The alias-stack invariant: every cell declared as a P5 alias MUST
    # actually be admitted by R_K979_P5_route_to_hbl.  Module-load asserts
    # check this once; this test re-verifies at pytest time so a P5
    # contraction is caught before merge.
    for cell in _K1673_P28_VS_R_K979_P5_BF16_ALIAS:
        M, N, K, dtype = cell
        assert R_K979_P5_route_to_hbl(M, N, K, dtype), (
            f"K-1673 P28 vs R-K979 P5 alias cell {cell} not admitted by "
            "R_K979_P5_route_to_hbl; either the P5 closed-form predicate "
            "contracted or the K-1673 P28 alias-decomposition rationale "
            "is stale.")


def test_p28_new_routeout_contribution_is_exactly_six_fp16_cells():
    assert _K1673_P28_NEW_ROUTEOUT_6 == frozenset({
        (M, 128, K, "torch.float16")
        for M in (2048, 4096, 8192)
        for K in (2048, 32768)
    })
    assert len(_K1673_P28_NEW_ROUTEOUT_6) == 6
    # All 6 NEW cells must be fp16 at the K-EXTREMES sub-band.
    for cell in _K1673_P28_NEW_ROUTEOUT_6:
        M, N, K, dtype = cell
        assert N == 128
        assert dtype == "torch.float16"
        assert K in (2048, 32768)
        assert M in (2048, 4096, 8192)


# ---------------------------------------------------------------------------
# (c) SIBLING-N FIREWALL — P28 carries N=128 only; must be disjoint from
#     every K-COMPLEMENT predecessor that targets N != 128.  The only
#     N=128 predecessor (K-1367 P13) is exempt by the alias-overlap
#     invariant above.
# ---------------------------------------------------------------------------
def test_p28_disjoint_from_p26_n2048():
    assert _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30)


def test_p28_n_axis_is_strictly_128():
    for cell in _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30:
        _, N, _, _ = cell
        assert N == 128


# ---------------------------------------------------------------------------
# (d) ROUTING CONTRACT — every K-1673 admit cell dispatches to hipBLASLt
#     via the public `k971_route_decision`.  For NEW cells (the 6 fp16
#     mirror at K ∈ {2048, 32768}) the load-bearing path is the 19th-
#     position P28 predicate; for alias cells (P5 bf16 closed-form ∪
#     K-1367 P13 strict-equality at K∈{4096,8192,16384}) the load-bearing
#     path is the upstream firing predicate but the runtime verdict is
#     identical.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30)
)
def test_p28_admit_cell_routes_to_hbl(cell):
    M, N, K, dtype = cell
    assert (
        k971_route_decision(
            M, N, K, dtype, dtype,
            enable_streamk=False, work_stealing=False,
            disable_env_set=False,
        )
        is True
    ), f"K-1673 P28 admit cell {cell} failed to dispatch to hipBLASLt."


# ---------------------------------------------------------------------------
# (e) PREDICATE FUNCTION SHAPE — the strict-equality membership predicate
#     is True iff (M, N, K, str(dtype)) is in the admit set.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30)
)
def test_p28_predicate_fires_on_admit_cell(cell):
    M, N, K, dtype = cell
    assert _k1673_p28_skinny_n128_kcompl_aliasstack_routeout(M, N, K, dtype)


@pytest.mark.parametrize(
    "cell",
    [
        # N != 128 — sibling-N firewall negative.
        (2048, 256, 4096, "torch.bfloat16"),
        (2048, 512, 8192, "torch.float16"),
        (4096, 2048, 2048, "torch.bfloat16"),
        # M not in {2048, 4096, 8192} — outside the K-1673 envelope.
        (1024, 128, 4096, "torch.bfloat16"),
        (3072, 128, 8192, "torch.float16"),
        (16384, 128, 8192, "torch.bfloat16"),
        # K not in {2048, 4096, 8192, 16384, 32768} — outside K-grid.
        (4096, 128, 1024, "torch.bfloat16"),
        (8192, 128, 65536, "torch.float16"),
        # dtype not bf16/fp16.
        (4096, 128, 4096, "torch.float32"),
    ],
    ids=lambda c: f"{c[0]}x{c[1]}x{c[2]}_{c[3]}",
)
def test_p28_predicate_does_not_fire_outside_envelope(cell):
    M, N, K, dtype = cell
    assert (
        _k1673_p28_skinny_n128_kcompl_aliasstack_routeout(M, N, K, dtype)
        is False
    )


# ---------------------------------------------------------------------------
# (f) CARVE-OUT NEGATIVES — streamk / work_stealing / dtype-mismatch must
#     short-circuit routing OFF even on K-1673 admit cells.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable_streamk": True,  "work_stealing": False},
        {"enable_streamk": False, "work_stealing": True},
    ],
    ids=["streamk_on", "work_stealing_on"],
)
def test_p28_admit_cell_carved_out_by_flags(kwargs):
    # canonical K-1673 NEW admit (the worst-cohort fp16 cell).
    cell = (8192, 128, 2048, "torch.float16")
    M, N, K, dtype = cell
    assert k971_route_decision(
        M, N, K, dtype, dtype, disable_env_set=False, **kwargs,
    ) is False


def test_p28_admit_cell_carved_out_by_dtype_mismatch():
    M, N, K = 8192, 128, 2048
    assert k971_route_decision(
        M, N, K, "torch.bfloat16", "torch.float16",
        enable_streamk=False, work_stealing=False, disable_env_set=False,
    ) is False


# ---------------------------------------------------------------------------
# (g) ALIAS-COVERAGE INVARIANT — the 24 alias cells must remain covered
#     by the upstream predicate union (R-K979 P5 ∪ K-1367 P13).  If an
#     upstream contraction silently breaks this, the K-1673 audit handle
#     is no longer faithful and the alias cells become NEW route-OUT
#     (which would change the cardinality of `_K1673_P28_NEW_ROUTEOUT_6`).
# ---------------------------------------------------------------------------
def test_p28_alias_cells_covered_by_upstream_union():
    alias_cells = (
        _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30
        - _K1673_P28_NEW_ROUTEOUT_6
    )
    assert len(alias_cells) == 24
    # Every alias cell must be admitted by P13_N128 strict-equality OR
    # by R-K979 P5 closed-form.
    for cell in alias_cells:
        M, N, K, dtype = cell
        in_p13 = cell in _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18
        in_p5 = R_K979_P5_route_to_hbl(M, N, K, dtype)
        assert in_p13 or in_p5, (
            f"K-1673 P28 alias cell {cell} not covered by either P13_N128 "
            f"strict-equality or R-K979 P5 closed-form; alias-stack "
            "invariant violated."
        )


# ---------------------------------------------------------------------------
# (h) STRUCTURAL NON-INTERFERENCE — for every cell in the prior frozensets
#     of the K-COMPLEMENT chain (with N != 128 — sibling-N firewall), the
#     K-1673 P28 strict-equality predicate must return False.  This is
#     strictly stronger than any sampled empirical sweep for pure
#     membership-check predicates (R-1628.STRUCTURAL-NON-INTERFERENCE-VIA-
#     FULL-ENUMERATION-STRICTLY-STRONGER-THAN-SAMPLED-SWEEP-FOR-PURE-
#     MEMBERSHIP-PREDICATES).
# ---------------------------------------------------------------------------
def test_p28_does_not_fire_on_p26_n2048_cells():
    for cell in _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30:
        M, N, K, dtype = cell
        assert not _k1673_p28_skinny_n128_kcompl_aliasstack_routeout(
            M, N, K, dtype
        ), f"K-1673 P28 fires on K-1611 P26 cell {cell} — sibling-N firewall violated."
