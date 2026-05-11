"""Unit fixture — K-1611 P26 skinny_N2048 K-COMPLEMENT 30-cell alias-stack
17th-position route-OUT.

Productionizes the K-1611 verification of the N=2048 K-COMPLEMENT envelope
(M ∈ {2048, 4096, 8192} × N=2048 × K ∈ {2048, 4096, 8192, 16384, 32768} ×
{bf16, fp16}) as the 17th stacked frozenset, closing the LAST untested
mid-N rung of the K-COMPLEMENT N-ladder.  Coverage now spans the full
verified N range {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}.

Source measurement (still-of-record): K-1611 paired n=30 HIP-graph
hot-cache on MI300X / gfx942 (c42 partition), TRITONBLAS_DISABLE_K971=1,
B=10000 vectorised paired bootstrap against the live post-K-1581 routing
oracle; 30/30 admit at strict ratio_median ≥ 1.05 ∧ p(<1.05) < 0.01 gate;
cohort geomean tb/hbl = 1.451×, range 1.108×-1.853×, 0 regressions.

ALIAS-STACK structure: 22 NEW route-OUT cells + 8 alias cells (2 P12 at
the (2048, 2048, 2048, {bf16, fp16}) diagonal + 6 K971_ROUTE_TABLE union:
4 K-905/K-971 LDS-BC anchors at K∈{16384, 32768} ∪ 2 K-1335
longK_smallSquare bf16 cells at K∈{4096, 8192}).  Alias overlaps fire
BEFORE P26 in the dispatch chain so the 8 alias cells are unreachable
under normal dispatch — load-bearing only if any of the upstream layers
(P12, K971, K1335) is ablated.
"""
import pytest

from tritonblas._route_predicate import (
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4,
    _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4,
    _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30,
    _K1611_P26_NEW_ROUTEOUT_22,
    _K1611_P26_VS_K1335_ALIAS_OVERLAP,
    _K1611_P26_VS_K971_ALIAS_OVERLAP,
    _K1611_P26_VS_P12_ALIAS_OVERLAP,
    _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30,
    K971_ROUTE_TABLE,
    _k1611_p26_skinny_n2048_kcompl_aliasstack_routeout,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# (a) CARDINALITY — pinned to 30 cells (the K-1611 envelope: M ∈ {2048,
#     4096, 8192} × N=2048 × K ∈ {2048, 4096, 8192, 16384, 32768} × {bf16,
#     fp16}).  Any deviation indicates an authoring typo against the
#     K-1611 paired n=30 admit set.
# ---------------------------------------------------------------------------
def test_p26_admit_set_cardinality_is_thirty():
    assert len(_K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30) == 30


def test_p26_admit_set_covers_full_n2048_kcompl_grid():
    expected = {
        (M, 2048, K, dtype)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dtype in ("torch.bfloat16", "torch.float16")
    }
    assert _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30 == expected


# ---------------------------------------------------------------------------
# (b) ALIAS-OVERLAP STRUCTURE — pinned at exactly 8 alias cells across 3
#     upstream predicates.  Future contractions of any upstream predicate
#     must trip these tests so the K-1611 NEW route-OUT contribution is
#     re-audited.
# ---------------------------------------------------------------------------
def test_p26_p12_alias_overlap_is_exactly_two_cells():
    assert _K1611_P26_VS_P12_ALIAS_OVERLAP == frozenset({
        (2048, 2048, 2048, "torch.bfloat16"),
        (2048, 2048, 2048, "torch.float16"),
    })


def test_p26_k1335_alias_overlap_is_exactly_two_cells():
    assert _K1611_P26_VS_K1335_ALIAS_OVERLAP == frozenset({
        (2048, 2048, 4096, "torch.bfloat16"),
        (2048, 2048, 8192, "torch.bfloat16"),
    })


def test_p26_k971_alias_overlap_is_exactly_six_cells():
    assert _K1611_P26_VS_K971_ALIAS_OVERLAP == frozenset({
        (2048, 2048, 16384, "torch.bfloat16"),
        (2048, 2048, 16384, "torch.float16"),
        (2048, 2048, 32768, "torch.bfloat16"),
        (2048, 2048, 32768, "torch.float16"),
        (2048, 2048,  4096, "torch.bfloat16"),
        (2048, 2048,  8192, "torch.bfloat16"),
    })


def test_p26_new_routeout_contribution_is_exactly_twenty_two():
    assert len(_K1611_P26_NEW_ROUTEOUT_22) == 22
    # All 22 NEW cells must have M >= 4096 OR be the fp16 K-1335 siblings.
    for cell in _K1611_P26_NEW_ROUTEOUT_22:
        M, N, K, dtype = cell
        assert N == 2048
        is_m_ge_4096 = M >= 4096
        is_k1335_fp16_sibling = (
            M == 2048
            and K in (4096, 8192)
            and dtype == "torch.float16"
        )
        assert is_m_ge_4096 or is_k1335_fp16_sibling, (
            f"K-1611 P26 NEW route-OUT cell {cell} fails alias-stack "
            "decomposition: NEW cells must have M ∈ {4096, 8192} OR be "
            "the (2048, 2048, K, fp16) pair for K ∈ {4096, 8192} that "
            "K-1335 (bf16-only) does not cover."
        )


# ---------------------------------------------------------------------------
# (c) SIBLING-N FIREWALL — P26 carries N=2048 only; must be disjoint from
#     every K-COMPLEMENT predecessor (none of which carry N=2048 cells).
# ---------------------------------------------------------------------------
def test_p26_disjoint_from_p24_n4096():
    assert _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30)


def test_p26_n_axis_is_strictly_2048():
    for cell in _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30:
        _, N, _, _ = cell
        assert N == 2048


# ---------------------------------------------------------------------------
# (d) ROUTING CONTRACT — every K-1611 admit cell dispatches to hipBLASLt
#     via the public `k971_route_decision`.  For NEW cells (M ∈ {4096,
#     8192} or the K-1335 fp16 siblings) the load-bearing path is the
#     17th-position P26 predicate; for alias cells (M=2048 ∩ upstream)
#     the load-bearing path is the upstream firing predicate (P12 / K1335
#     / K971 LDS-BC anchors) but the runtime verdict is identical.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30)
)
def test_p26_admit_cell_routes_to_hbl(cell):
    M, N, K, dtype = cell
    assert (
        k971_route_decision(
            M, N, K, dtype, dtype,
            enable_streamk=False, work_stealing=False,
            disable_env_set=False,
        )
        is True
    ), f"K-1611 P26 admit cell {cell} failed to dispatch to hipBLASLt."


# ---------------------------------------------------------------------------
# (e) PREDICATE FUNCTION SHAPE — the strict-equality membership predicate
#     is True iff (M, N, K, str(dtype)) is in the admit set.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30)
)
def test_p26_predicate_fires_on_admit_cell(cell):
    M, N, K, dtype = cell
    assert _k1611_p26_skinny_n2048_kcompl_aliasstack_routeout(M, N, K, dtype)


@pytest.mark.parametrize(
    "cell",
    [
        # N != 2048 — sibling-N firewall negative.
        (2048, 4096, 4096, "torch.bfloat16"),
        (2048, 1024, 8192, "torch.float16"),
        (4096, 8192, 2048, "torch.bfloat16"),
        # M not in {2048, 4096, 8192} — outside the K-1611 envelope.
        (1024, 2048, 4096, "torch.bfloat16"),
        (3072, 2048, 8192, "torch.float16"),
        (16384, 2048, 8192, "torch.bfloat16"),
        # K not in {2048, 4096, 8192, 16384, 32768} — outside K-grid.
        (4096, 2048, 1024, "torch.bfloat16"),
        (8192, 2048, 65536, "torch.float16"),
        # dtype not bf16/fp16.
        (4096, 2048, 4096, "torch.float32"),
    ],
    ids=lambda c: f"{c[0]}x{c[1]}x{c[2]}_{c[3]}",
)
def test_p26_predicate_does_not_fire_outside_envelope(cell):
    M, N, K, dtype = cell
    assert (
        _k1611_p26_skinny_n2048_kcompl_aliasstack_routeout(M, N, K, dtype)
        is False
    )


# ---------------------------------------------------------------------------
# (f) CARVE-OUT NEGATIVES — streamk / work_stealing / dtype-mismatch must
#     short-circuit routing OFF even on K-1611 admit cells.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable_streamk": True,  "work_stealing": False},
        {"enable_streamk": False, "work_stealing": True},
    ],
    ids=["streamk_on", "work_stealing_on"],
)
def test_p26_admit_cell_carved_out_by_flags(kwargs):
    cell = (4096, 2048, 8192, "torch.bfloat16")  # canonical K-1611 NEW admit
    M, N, K, dtype = cell
    assert k971_route_decision(
        M, N, K, dtype, dtype, disable_env_set=False, **kwargs,
    ) is False


def test_p26_admit_cell_carved_out_by_dtype_mismatch():
    M, N, K = 4096, 2048, 8192
    assert k971_route_decision(
        M, N, K, "torch.bfloat16", "torch.float16",
        enable_streamk=False, work_stealing=False, disable_env_set=False,
    ) is False


# ---------------------------------------------------------------------------
# (g) ALIAS-COVERAGE INVARIANT — the 8 alias cells must remain covered by
#     the upstream predicate union (P12 ∪ K971_ROUTE_TABLE).  If an
#     upstream contraction silently breaks this, the K-1611 audit handle
#     is no longer faithful and the alias cells become NEW route-OUT
#     (which would change the cardinality of `_K1611_P26_NEW_ROUTEOUT_22`).
# ---------------------------------------------------------------------------
def test_p26_alias_cells_covered_by_upstream_union():
    upstream_union = (
        _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4
        | K971_ROUTE_TABLE
    )
    alias_cells = (
        _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30
        - _K1611_P26_NEW_ROUTEOUT_22
    )
    assert len(alias_cells) == 8
    uncovered = alias_cells - upstream_union
    assert not uncovered, (
        f"K-1611 P26 has {len(uncovered)} alias cells not covered by "
        f"P12 ∪ K971_ROUTE_TABLE: {sorted(uncovered)}"
    )


def test_p26_k1335_subset_of_k971_route_table():
    # K-1335 is unioned into K971_ROUTE_TABLE at module load; this pin
    # guards against future re-decompositions that would break the
    # K-1611 P26 alias-overlap structure.
    assert _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4.issubset(K971_ROUTE_TABLE)
