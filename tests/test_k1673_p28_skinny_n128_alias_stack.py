"""Unit fixture — K-1673 P28 skinny_N128 K-COMPLEMENT 30-cell alias-stack
19th-position route-OUT.

Productionizes the K-1673 verification of the N=128 K-COMPLEMENT envelope
(M ∈ {2048, 4096, 8192} × N=128 × K ∈ {2048, 4096, 8192, 16384, 32768} ×
{bf16, fp16}) as the 19th stacked frozenset, closing the LAST untested
small-N rung (N=128) of the K-COMPLEMENT N-ladder at the dtype-mirror gap
(fp16 K ∈ {2048, 32768}).  Coverage now spans the FULL N-ladder
{128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768} on the M ∈ {2048,
4096, 8192} rows for both bf16 AND fp16.

Source measurement (still-of-record): K-1673 paired n=30 HIP-graph
hot-cache on MI300X / gfx942 (OCI fallback amd-rccl partition) against
the LIVE post-K-1647 P27 routing oracle
(fork branch fix/K-1647 tip 8d010c6); 30/30 admit at strict ratio_median
≥ 1.05 ∧ p(<1.05) < 0.01 gate; cohort geomean tb/hbl = 1.694×, range
1.162×-2.890×, 0 regressions.

ALIAS-STACK structure: 6 NEW route-OUT cells (all fp16 at K ∈ {2048,
32768}) + 24 alias cells (15 bf16 via R-K979 P5 Clause-3 minMN ≤ 192 ∧
K ≥ 2048 + 9 fp16 K-mid via K-1367 P13 N=128).  Alias overlaps fire
BEFORE P28 in the dispatch chain so the 24 alias cells are unreachable
under normal dispatch — load-bearing only if any of P5 / P13 is ablated.
"""
import pytest

from tritonblas._route_predicate import (
    R_K979_P5_route_to_hbl,
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _K1673_P28_NEW_ROUTEOUT_6,
    _K1673_P28_VS_P5_ALIAS_OVERLAP,
    _K1673_P28_VS_P13_N128_ALIAS_OVERLAP,
    _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30,
    _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30,
    _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30,
    _K1633_P27_SKINNY_N512_KCOMPL_ALIASSTACK_30,
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
#     upstream predicates (P5 bf16 + P13 N=128 fp16 K-mid).  Future
#     contractions of any upstream predicate must trip these tests so the
#     K-1673 NEW route-OUT contribution is re-audited.
# ---------------------------------------------------------------------------
def test_p28_p5_alias_overlap_is_exactly_fifteen_bf16_cells():
    expected = frozenset({
        (M, 128, K, "torch.bfloat16")
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
    })
    assert _K1673_P28_VS_P5_ALIAS_OVERLAP == expected


def test_p28_p13_n128_alias_overlap_is_exactly_eighteen_cells():
    expected = frozenset({
        (M, 128, K, dtype)
        for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dtype in ("torch.bfloat16", "torch.float16")
    })
    assert _K1673_P28_VS_P13_N128_ALIAS_OVERLAP == expected


def test_p28_new_routeout_contribution_is_exactly_six_fp16_cells():
    expected = frozenset({
        (M, 128, K, "torch.float16")
        for M in (2048, 4096, 8192)
        for K in (2048, 32768)
    })
    assert _K1673_P28_NEW_ROUTEOUT_6 == expected
    assert len(_K1673_P28_NEW_ROUTEOUT_6) == 6


def test_p28_alias_decomposition_is_partition():
    # NEW ⊕ alias-union = full envelope; NEW ∩ alias-union = ∅.
    alias_union = (
        _K1673_P28_VS_P5_ALIAS_OVERLAP
        | _K1673_P28_VS_P13_N128_ALIAS_OVERLAP
    )
    assert _K1673_P28_NEW_ROUTEOUT_6.isdisjoint(alias_union)
    assert (
        _K1673_P28_NEW_ROUTEOUT_6 | alias_union
        == _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30
    )
    # Alias-union cardinality is 24 (15 bf16 from P5 + 18 from P13 N=128
    # − 9 P5∩P13 overlap = 24).
    assert len(alias_union) == 24


# ---------------------------------------------------------------------------
# (c) SIBLING-N FIREWALL — P28 carries N=128 only; must be disjoint from
#     every K-COMPLEMENT predecessor that uses N != 128.  The K-1367 P13
#     N=128 18-cell envelope is the only intentional N=128 overlap and is
#     asserted in (b) above as the alias subset.
# ---------------------------------------------------------------------------
def test_p28_disjoint_from_p26_n2048():
    assert _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30)


def test_p28_disjoint_from_p24_n4096():
    assert _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30)


def test_p28_disjoint_from_p23_n512():
    assert _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30)


def test_p28_disjoint_from_p27_n512_alias():
    # K-1633 P27 is a module-level alias of K-1552 P23; assert the
    # disjointness on the alias handle to catch any future P27 rebind.
    assert _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1633_P27_SKINNY_N512_KCOMPL_ALIASSTACK_30)


def test_p28_n_axis_is_strictly_128():
    for cell in _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30:
        _, N, _, _ = cell
        assert N == 128


# ---------------------------------------------------------------------------
# (d) ROUTING CONTRACT — every K-1673 admit cell dispatches to hipBLASLt
#     via the public `k971_route_decision`.  For NEW cells (the 6 fp16 at
#     K ∈ {2048, 32768}) the load-bearing path is the 19th-position P28
#     predicate; for alias cells the load-bearing path is the upstream
#     firing predicate (R-K979 P5 / K-1367 P13) but the runtime verdict
#     is identical.
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
        (4096, 1024, 2048, "torch.bfloat16"),
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
    cell = (2048, 128, 32768, "torch.float16")  # canonical K-1673 NEW admit (max-ratio)
    M, N, K, dtype = cell
    assert k971_route_decision(
        M, N, K, dtype, dtype, disable_env_set=False, **kwargs,
    ) is False


def test_p28_admit_cell_carved_out_by_dtype_mismatch():
    M, N, K = 2048, 128, 32768
    assert k971_route_decision(
        M, N, K, "torch.bfloat16", "torch.float16",
        enable_streamk=False, work_stealing=False, disable_env_set=False,
    ) is False


# ---------------------------------------------------------------------------
# (g) ALIAS-COVERAGE INVARIANT — the 24 alias cells must remain covered
#     by the upstream predicate union (P5 ∪ P13 N=128).  If an upstream
#     contraction silently breaks this, the K-1673 audit handle is no
#     longer faithful and the affected cells become NEW route-OUT (which
#     would change the cardinality of `_K1673_P28_NEW_ROUTEOUT_6`).
# ---------------------------------------------------------------------------
def test_p28_alias_cells_covered_by_upstream_union():
    alias_cells = (
        _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30
        - _K1673_P28_NEW_ROUTEOUT_6
    )
    assert len(alias_cells) == 24
    # P5 catches all bf16 cells via the structural minMN ≤ 192 ∧ K ≥
    # 2048 envelope; P13 catches the K-mid fp16 cells (K ∈ {4096, 8192,
    # 16384}); union must cover every alias cell.
    uncovered = []
    for cell in alias_cells:
        M, N, K, dtype = cell
        in_p5 = R_K979_P5_route_to_hbl(M, N, K, dtype)
        in_p13 = cell in _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18
        if not (in_p5 or in_p13):
            uncovered.append(cell)
    assert not uncovered, (
        f"K-1673 P28 has {len(uncovered)} alias cells not covered by "
        f"R-K979 P5 ∪ K-1367 P13 N=128: {sorted(uncovered)}"
    )


def test_p28_p5_p13_overlap_is_exactly_nine_bf16_cells():
    # The 9 cells (bf16, K ∈ {4096, 8192, 16384}) sit in BOTH P5 and P13;
    # P5 fires first at the 5th slot.  This pin guards against a future
    # P5 K-floor / minMN-ceiling contraction that would shift those 9
    # cells into the P13-only alias bucket (still routed but via a
    # different upstream predicate).
    p5_p13_overlap = (
        _K1673_P28_VS_P5_ALIAS_OVERLAP
        & _K1673_P28_VS_P13_N128_ALIAS_OVERLAP
    )
    expected = frozenset({
        (M, 128, K, "torch.bfloat16")
        for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
    })
    assert p5_p13_overlap == expected
