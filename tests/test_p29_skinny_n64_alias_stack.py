"""Unit fixture — P29 skinny_N64 K-COMPLEMENT 30-cell alias-stack
20th-position route-OUT (K-1700 follow-up productionization).

Productionizes the K-1700-verified worst-cell envelope for the ultra-
skinny N=64 K-COMPLEMENT cohort
(M ∈ {2048, 4096, 8192} × N=64 × K ∈ {2048, 4096, 8192, 16384, 32768} ×
{bf16, fp16}) as the 20th stacked frozenset, mirroring the K-1685 P28
pattern at the next-lower N rung on the K-COMPLEMENT N-ladder
(N=128 → N=64).

Source measurement (still-of-record): paired n=30 HIP-graph hot-cache
benchmark on MI300X / gfx942 (per INFRA-0048 c42 SSH refused
fallback — OCI useocpm2m-097 substrate) against the LIVE post-P28
routing oracle.  Engines: TB → tritonblas.matmul → persistent_matmul;
HBL → hipBLASLt direct.  30/30 lose to hipBLASLt by ≥10% (cohort
geomean 1.78×, worst 2.89× at (4096, 64, 8192, fp16) — the K-1687
worst-seam cell); 15 bf16 cells already routed-OUT by R-K979 P5
closed-form (Clause-3 min(M,N) ≤ 192 ∧ K ≥ 2048); 15 fp16 cells slip
through every existing predicate — these are the load-bearing P29
contribution (the entire fp16 sub-row at N=64 because P5 is bf16-only
at the `_dtype_is_bf16` early return and no prior P-frozenset targets
N=64).

ALIAS-STACK structure: 15 NEW route-OUT cells (the entire fp16
sub-row) + 15 alias cells (the entire bf16 sub-row, alias-of-R-K979
P5 closed-form).  Alias overlaps fire BEFORE P29 in the dispatch
chain (P5 at 4th-slot, P29 at 20th-slot) so the 15 bf16 alias cells
are unreachable under normal dispatch — load-bearing only if the
upstream P5 layer is ablated.
"""
import pytest

from tritonblas._route_predicate import (
    K971_ROUTE_TABLE,
    R_K979_P5_route_to_hbl,
    _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30,
    _K1700_P29_NEW_ROUTEOUT_15,
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_30,
    _K1700_P29_VS_P5_BF16_ALIAS,
    _P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _k1700_p29_skinny_n64_kcompl_aliasstack_routeout,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# (a) CARDINALITY — pinned to 30 cells (the envelope: M ∈ {2048,
#     4096, 8192} × N=64 × K ∈ {2048, 4096, 8192, 16384, 32768} × {bf16,
#     fp16}).  Any deviation indicates an authoring typo against the
#     K-1700 paired n=30 admit set.
# ---------------------------------------------------------------------------
def test_p29_admit_set_cardinality_is_thirty():
    assert len(_K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_30) == 30


def test_p29_admit_set_covers_full_n64_kcompl_grid():
    expected = {
        (M, 64, K, dtype)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dtype in ("torch.bfloat16", "torch.float16")
    }
    assert _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_30 == expected


# ---------------------------------------------------------------------------
# (b) ALIAS-OVERLAP STRUCTURE — pinned at exactly 15 alias cells across
#     the single upstream predicate R-K979 P5 (closed-form, bf16-only).
#     Future contraction of the upstream predicate must trip these tests
#     so the NEW route-OUT contribution is re-audited.
# ---------------------------------------------------------------------------
def test_p29_p5_bf16_alias_is_exactly_fifteen_cells():
    assert _K1700_P29_VS_P5_BF16_ALIAS == frozenset({
        (M, 64, K, "torch.bfloat16")
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
    })
    assert len(_K1700_P29_VS_P5_BF16_ALIAS) == 15


def test_p29_p5_alias_actually_admitted_by_p5_closed_form():
    # The alias-stack invariant: every cell declared as a P5 alias MUST
    # actually be admitted by R_K979_P5_route_to_hbl.  Module-load asserts
    # check this once; this test re-verifies at pytest time so a P5
    # contraction is caught before merge.
    for cell in _K1700_P29_VS_P5_BF16_ALIAS:
        M, N, K, dtype = cell
        assert R_K979_P5_route_to_hbl(M, N, K, dtype), (
            f"P29 vs R-K979 P5 alias cell {cell} not admitted by "
            "R_K979_P5_route_to_hbl; either the P5 closed-form predicate "
            "contracted or the P29 alias-decomposition rationale "
            "is stale.")


def test_p29_new_routeout_contribution_is_exactly_fifteen_fp16_cells():
    assert _K1700_P29_NEW_ROUTEOUT_15 == frozenset({
        (M, 64, K, "torch.float16")
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
    })
    assert len(_K1700_P29_NEW_ROUTEOUT_15) == 15
    # All 15 NEW cells must be fp16 at N=64 spanning the full M×K grid.
    for cell in _K1700_P29_NEW_ROUTEOUT_15:
        M, N, K, dtype = cell
        assert N == 64
        assert dtype == "torch.float16"
        assert K in (2048, 4096, 8192, 16384, 32768)
        assert M in (2048, 4096, 8192)


def test_p29_no_upstream_p_frozenset_targets_n_64():
    # Documentation-pin: no prior K-COMPLEMENT P-frozenset targets
    # N=64.  This is the structural reason the entire fp16 sub-row is
    # NEW route-OUT (P5 is bf16-only at `_dtype_is_bf16`).  If a
    # future P-frozenset extends to N=64 the cardinality assert above
    # will trip and force a re-audit of the alias decomposition.
    for cell in _P28_SKINNY_N128_KCOMPL_ALIASSTACK_30:
        _, N, _, _ = cell
        assert N != 64
    for cell in _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30:
        _, N, _, _ = cell
        assert N != 64


# ---------------------------------------------------------------------------
# (c) SIBLING-N FIREWALL — P29 carries N=64 only; must be disjoint from
#     every K-COMPLEMENT predecessor (no prior frozenset targets N=64
#     so disjointness is guaranteed by construction).
# ---------------------------------------------------------------------------
def test_p29_disjoint_from_p28_n128():
    assert _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_30.isdisjoint(
        _P28_SKINNY_N128_KCOMPL_ALIASSTACK_30)


def test_p29_disjoint_from_p26_n2048():
    assert _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_30.isdisjoint(
        _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30)


def test_p29_n_axis_is_strictly_64():
    for cell in _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_30:
        _, N, _, _ = cell
        assert N == 64


# ---------------------------------------------------------------------------
# (d) ROUTING CONTRACT — every admit cell dispatches to hipBLASLt
#     via the public `k971_route_decision`.  For NEW cells (the 15 fp16
#     entire sub-row at N=64) the load-bearing path is the 20th-position
#     P29 predicate; for alias cells (P5 bf16 closed-form) the load-
#     bearing path is the upstream firing predicate but the runtime
#     verdict is identical.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_30)
)
def test_p29_admit_cell_routes_to_hbl(cell):
    M, N, K, dtype = cell
    assert (
        k971_route_decision(
            M, N, K, dtype, dtype,
            enable_streamk=False, work_stealing=False,
            disable_env_set=False,
        )
        is True
    ), f"P29 admit cell {cell} failed to dispatch to hipBLASLt."


# ---------------------------------------------------------------------------
# (e) PREDICATE FUNCTION SHAPE — the strict-equality membership predicate
#     is True iff (M, N, K, str(dtype)) is in the admit set.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_30)
)
def test_p29_predicate_fires_on_admit_cell(cell):
    M, N, K, dtype = cell
    assert _k1700_p29_skinny_n64_kcompl_aliasstack_routeout(M, N, K, dtype)


@pytest.mark.parametrize(
    "cell",
    [
        # N != 64 — sibling-N firewall negative.
        (2048, 128, 4096, "torch.bfloat16"),
        (2048, 256, 8192, "torch.float16"),
        (4096, 2048, 2048, "torch.bfloat16"),
        # M not in {2048, 4096, 8192} — outside the envelope.
        (1024, 64, 4096, "torch.bfloat16"),
        (3072, 64, 8192, "torch.float16"),
        (16384, 64, 8192, "torch.bfloat16"),
        # K not in {2048, 4096, 8192, 16384, 32768} — outside K-grid.
        (4096, 64, 1024, "torch.bfloat16"),
        (8192, 64, 65536, "torch.float16"),
        # dtype not bf16/fp16.
        (4096, 64, 4096, "torch.float32"),
    ],
    ids=lambda c: f"{c[0]}x{c[1]}x{c[2]}_{c[3]}",
)
def test_p29_predicate_does_not_fire_outside_envelope(cell):
    M, N, K, dtype = cell
    assert (
        _k1700_p29_skinny_n64_kcompl_aliasstack_routeout(M, N, K, dtype)
        is False
    )


# ---------------------------------------------------------------------------
# (f) CARVE-OUT NEGATIVES — streamk / work_stealing / dtype-mismatch must
#     short-circuit routing OFF even on admit cells.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable_streamk": True,  "work_stealing": False},
        {"enable_streamk": False, "work_stealing": True},
    ],
    ids=["streamk_on", "work_stealing_on"],
)
def test_p29_admit_cell_carved_out_by_flags(kwargs):
    # canonical NEW admit (the K-1687 worst-seam fp16 cell).
    cell = (4096, 64, 8192, "torch.float16")
    M, N, K, dtype = cell
    assert k971_route_decision(
        M, N, K, dtype, dtype, disable_env_set=False, **kwargs,
    ) is False


def test_p29_admit_cell_carved_out_by_dtype_mismatch():
    M, N, K = 4096, 64, 8192
    assert k971_route_decision(
        M, N, K, "torch.bfloat16", "torch.float16",
        enable_streamk=False, work_stealing=False, disable_env_set=False,
    ) is False


# ---------------------------------------------------------------------------
# (g) ALIAS-COVERAGE INVARIANT — the 15 alias cells must remain covered
#     by the upstream predicate (R-K979 P5 closed-form, bf16-only).  If
#     an upstream contraction silently breaks this, the audit handle
#     is no longer faithful and the alias cells become NEW route-OUT
#     (which would change the cardinality of `_K1700_P29_NEW_ROUTEOUT_15`).
# ---------------------------------------------------------------------------
def test_p29_alias_cells_covered_by_upstream_p5():
    alias_cells = (
        _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_30
        - _K1700_P29_NEW_ROUTEOUT_15
    )
    assert len(alias_cells) == 15
    # Every alias cell must be admitted by R-K979 P5 closed-form.
    for cell in alias_cells:
        M, N, K, dtype = cell
        in_p5 = R_K979_P5_route_to_hbl(M, N, K, dtype)
        assert in_p5, (
            f"P29 alias cell {cell} not covered by R-K979 P5 closed-form; "
            f"alias-stack invariant violated."
        )


# ---------------------------------------------------------------------------
# (h) STRUCTURAL NON-INTERFERENCE — for every cell in the prior frozensets
#     of the K-COMPLEMENT chain, the P29 strict-equality predicate must
#     return False.  This is strictly stronger than any sampled empirical
#     sweep for pure membership-check predicates (R-1628.STRUCTURAL-
#     NON-INTERFERENCE-VIA-FULL-ENUMERATION-STRICTLY-STRONGER-THAN-SAMPLED-
#     SWEEP-FOR-PURE-MEMBERSHIP-PREDICATES).
# ---------------------------------------------------------------------------
def test_p29_does_not_fire_on_p28_n128_cells():
    for cell in _P28_SKINNY_N128_KCOMPL_ALIASSTACK_30:
        M, N, K, dtype = cell
        assert not _k1700_p29_skinny_n64_kcompl_aliasstack_routeout(
            M, N, K, dtype
        ), f"P29 fires on P28 cell {cell} — sibling-N firewall violated."


def test_p29_does_not_fire_on_p26_n2048_cells():
    for cell in _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30:
        M, N, K, dtype = cell
        assert not _k1700_p29_skinny_n64_kcompl_aliasstack_routeout(
            M, N, K, dtype
        ), f"P29 fires on K-1611 P26 cell {cell} — sibling-N firewall violated."
