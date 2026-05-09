"""Unit fixture — K-1717 P29 NIB (N-In-Between) envelope 65-cell alias-stack
20th-position route-OUT.

Productionises the K-1704 paired n=30 hot-cache HIP-graph audit (108 cells)
on MI300X / gfx942 against the LIVE post-K-1685 P28 routing oracle (fork
branch ``fix/K-1685-p28-skinny-n128-kcompl-aliasstack`` tip ``6785fcd``):
65 of 108 cells cleared the strict 5% admit gate (oracle/hbl
``speedup_ci95_hi`` < ``1/1.05`` = 0.95238 — i.e. forced-hipBLASLt is
reliably ≥5% faster than the live oracle with non-overlapping bootstrap
CIs).

Cohort layout (six per-N frozensets, partitioned by N for per-N audit
handles per the K-1175 stacked-predicate convention):

  ``N= 80``  8 fp16-only cells; oracle/hbl geomean 0.670 → 1.49× lift
  ``N=112``  8 fp16-only cells; oracle/hbl geomean 0.623 → 1.61× lift
  ``N=144``  9 fp16-only cells; oracle/hbl geomean 0.588 → 1.70× lift
  ``N=176``  8 fp16-only cells; oracle/hbl geomean 0.707 → 1.41× lift
  ``N=208``  15 cells (8 fp16 + 7 bf16); oracle/hbl geomean 0.707 → 1.41×
  ``N=240``  17 cells (8 fp16 + 9 bf16); oracle/hbl geomean 0.723 → 1.38×
  Total: 65 cells; cohort oracle/hbl geomean 0.678 → 1.475× hbl-route lift

bf16 cells reappear at N ∈ {208, 240} only because R-K979 P5 Clause-3
(``min(M, N) ≤ 192 ∧ K ≥ 2048``, bf16-only) excludes them at the
``minMN > 192`` ceiling.  This is the same dtype-mirror gap shape that
K-1685 P28 closed at N=128 along the K-axis, here applied along the
N-axis.

Sibling-N firewall: P29 N-projection {80, 112, 144, 176, 208, 240} is
disjoint by construction from every prior K-COMPLEMENT N-ladder value
{128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}; the fixture pins
the projection so an N typo trips at module load.

Drift-detection: the K-1502-family 18-shape baseline is exercised in
``tests/test_k1502_drift_detection_baseline.py`` (canonical productionised
N-ladder); P29 leaves those shapes untouched because P29's N-axis is
strictly disjoint from the productionised ladder, and the fixture below
includes a dedicated ``test_p29_does_not_fire_on_k1502_baseline_shapes``
sanity check.
"""
import pytest

from tritonblas._route_predicate import (
    R_K979_P5_route_to_hbl,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _K1704_P29_NIB_ENVELOPE_ALIASSTACK_80,
    _K1704_P29_NIB_ENVELOPE_ALIASSTACK_112,
    _K1704_P29_NIB_ENVELOPE_ALIASSTACK_144,
    _K1704_P29_NIB_ENVELOPE_ALIASSTACK_176,
    _K1704_P29_NIB_ENVELOPE_ALIASSTACK_208,
    _K1704_P29_NIB_ENVELOPE_ALIASSTACK_240,
    _K1704_P29_NIB_ENVELOPE_BY_N,
    _K1704_P29_NIB_ENVELOPE_UNION_65,
    _K1704_P29_VS_P5_DISJOINT_BF16,
    _k1704_p29_nib_envelope_aliasstack_routeout,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# (a) CARDINALITY — pinned to the K-1704 admit-set per-N counts and union.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "frozen, expected_count",
    [
        (_K1704_P29_NIB_ENVELOPE_ALIASSTACK_80,   8),
        (_K1704_P29_NIB_ENVELOPE_ALIASSTACK_112,  8),
        (_K1704_P29_NIB_ENVELOPE_ALIASSTACK_144,  9),
        (_K1704_P29_NIB_ENVELOPE_ALIASSTACK_176,  8),
        (_K1704_P29_NIB_ENVELOPE_ALIASSTACK_208, 15),
        (_K1704_P29_NIB_ENVELOPE_ALIASSTACK_240, 17),
    ],
    ids=["N80", "N112", "N144", "N176", "N208", "N240"],
)
def test_p29_per_n_admit_set_cardinality(frozen, expected_count):
    assert len(frozen) == expected_count


def test_p29_union_cardinality_is_sixty_five():
    assert len(_K1704_P29_NIB_ENVELOPE_UNION_65) == 65
    assert _K1704_P29_NIB_ENVELOPE_UNION_65 == frozenset().union(
        *_K1704_P29_NIB_ENVELOPE_BY_N.values()
    )


def test_p29_per_n_dispatch_table_keys_match_envelope_n_axis():
    assert set(_K1704_P29_NIB_ENVELOPE_BY_N.keys()) == {
        80, 112, 144, 176, 208, 240
    }


# ---------------------------------------------------------------------------
# (b) ENVELOPE STRUCTURE — N-axis pinned to the in-between values, M and K
#     pinned to the K-1704 audit grid, dtype restricted to bf16/fp16.
# ---------------------------------------------------------------------------
def test_p29_n_axis_is_in_between_set():
    actual_n = {n for (_M, n, _K, _dt) in _K1704_P29_NIB_ENVELOPE_UNION_65}
    assert actual_n == {80, 112, 144, 176, 208, 240}


def test_p29_m_axis_is_subset_of_audit_grid():
    actual_m = {m for (m, _N, _K, _dt) in _K1704_P29_NIB_ENVELOPE_UNION_65}
    assert actual_m <= {2048, 4096, 8192}


def test_p29_k_axis_is_subset_of_audit_grid():
    actual_k = {k for (_M, _N, k, _dt) in _K1704_P29_NIB_ENVELOPE_UNION_65}
    assert actual_k <= {2048, 8192, 32768}


def test_p29_dtype_is_bf16_or_fp16_only():
    actual_dtypes = {dt for (_M, _N, _K, dt) in _K1704_P29_NIB_ENVELOPE_UNION_65}
    assert actual_dtypes <= {"torch.bfloat16", "torch.float16"}


def test_p29_bf16_cells_only_at_n_208_and_240():
    bf16_cells = {
        c for c in _K1704_P29_NIB_ENVELOPE_UNION_65
        if c[3] == "torch.bfloat16"
    }
    bf16_n = {n for (_M, n, _K, _dt) in bf16_cells}
    assert bf16_n == {208, 240}, (
        "K-1704 bf16 admits sit only at N ∈ {208, 240} (where R-K979 P5 "
        "Clause-3's minMN ≤ 192 ceiling excludes bf16 from upstream "
        "route-OUT); deviation indicates either a P5 ceiling change or "
        "a P29 authoring typo.")


# ---------------------------------------------------------------------------
# (c) SIBLING-N FIREWALL — disjoint by N-projection from every prior
#     K-COMPLEMENT N-ladder value.  Pinned at module load via the
#     `_K1704_P29_NIB_ENVELOPE_UNION_65` N-projection assert; the fixture
#     re-pins it here so a regression trips the test suite as well.
# ---------------------------------------------------------------------------
def test_p29_disjoint_from_p28_n128_envelope():
    assert _K1704_P29_NIB_ENVELOPE_UNION_65.isdisjoint(
        _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30
    )


def test_p29_n_axis_disjoint_from_productionised_n_ladder():
    productionised_ladder = {
        128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    }
    p29_n = {n for (_M, n, _K, _dt) in _K1704_P29_NIB_ENVELOPE_UNION_65}
    assert p29_n.isdisjoint(productionised_ladder), (
        "K-1704 P29 N-axis must be disjoint from the productionised "
        "K-COMPLEMENT N-ladder; deviation breaks the sibling-N firewall."
    )


def test_p29_bf16_cells_disjoint_from_p5_clause3():
    # P29 bf16 cells live at N ∈ {208, 240} where minMN > 192, so P5
    # Clause-3 (minMN ≤ 192 ∧ K ≥ 2048, bf16-only) cannot fire.  This
    # pin guards a future P5 ceiling-widening from silently making P29's
    # bf16 cells unreachable via the upstream slot.
    assert _K1704_P29_VS_P5_DISJOINT_BF16 == frozenset()
    for cell in _K1704_P29_NIB_ENVELOPE_UNION_65:
        if cell[3] != "torch.bfloat16":
            continue
        M, N, K, dtype = cell
        assert not R_K979_P5_route_to_hbl(M, N, K, dtype), (
            f"K-1704 P29 bf16 cell {cell} unexpectedly caught by R-K979 "
            "P5 Clause-3 — re-audit the P29 bf16 cohort.")


# ---------------------------------------------------------------------------
# (d) PREDICATE FUNCTION SHAPE — strict-equality membership,
#     constant-time N switch then per-N frozenset O(1) membership.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1704_P29_NIB_ENVELOPE_UNION_65)
)
def test_p29_predicate_fires_on_admit_cell(cell):
    M, N, K, dtype = cell
    assert _k1704_p29_nib_envelope_aliasstack_routeout(M, N, K, dtype)


@pytest.mark.parametrize(
    "cell",
    [
        # N on the productionised ladder — sibling-N firewall negative.
        (2048,  64, 32768, "torch.float16"),
        (4096, 128, 32768, "torch.float16"),
        (4096, 256, 32768, "torch.float16"),
        # N in the in-between set but K outside the audit grid.
        (4096, 144,  4096, "torch.float16"),
        (4096, 144, 16384, "torch.float16"),
        # N in the in-between set but M outside the audit grid.
        (1024, 144,  8192, "torch.float16"),
        (3072, 144,  8192, "torch.float16"),
        (16384, 144, 8192, "torch.float16"),
        # bf16 at N ∈ {80, 112, 144, 176} — outside the K-1704 admit set.
        (2048,  80,  8192, "torch.bfloat16"),
        (4096, 144, 32768, "torch.bfloat16"),
        (8192, 176, 32768, "torch.bfloat16"),
        # dtype not bf16/fp16.
        (4096, 144,  8192, "torch.float32"),
    ],
    ids=lambda c: f"{c[0]}x{c[1]}x{c[2]}_{c[3]}",
)
def test_p29_predicate_does_not_fire_outside_envelope(cell):
    M, N, K, dtype = cell
    assert _k1704_p29_nib_envelope_aliasstack_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# (e) ROUTING CONTRACT — every K-1704 admit cell dispatches to hipBLASLt
#     via the public `k971_route_decision` (the 20th-position predicate
#     is reachable because no upstream slot fires for in-between N).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1704_P29_NIB_ENVELOPE_UNION_65)
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
    ), f"K-1704 P29 admit cell {cell} failed to dispatch to hipBLASLt."


# ---------------------------------------------------------------------------
# (f) CARVE-OUT NEGATIVES — streamk / work_stealing / dtype-mismatch must
#     short-circuit routing OFF even on K-1704 admit cells.
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
    # Canonical K-1704 worst-cell admit (M=2048, N=144, K=32768, fp16;
    # speedup=0.274× → 3.65× hbl-route lift).
    cell = (2048, 144, 32768, "torch.float16")
    M, N, K, dtype = cell
    assert k971_route_decision(
        M, N, K, dtype, dtype, disable_env_set=False, **kwargs,
    ) is False


def test_p29_admit_cell_carved_out_by_dtype_mismatch():
    M, N, K = 2048, 144, 32768
    assert k971_route_decision(
        M, N, K, "torch.bfloat16", "torch.float16",
        enable_streamk=False, work_stealing=False, disable_env_set=False,
    ) is False


# ---------------------------------------------------------------------------
# (g) DRIFT-DETECTION SANITY — the K-1502 family 18-shape baseline (the
#     canonical productionised N-ladder cohort used to confirm zero
#     regressions post-merge) must NOT have any cell touched by P29.
#     The K-1502 baseline shapes live at N ∈ {128, 256, 512, 1024,
#     2048, 4096, 8192, 16384, 32768} which is the productionised
#     N-ladder — fully disjoint from P29's in-between N-axis.  This
#     test pins the disjointness directly so any future K-1502
#     baseline-shape rotation that drifts into the in-between values
#     surfaces immediately rather than via a silent regression on a
#     downstream drift-detection sweep.
# ---------------------------------------------------------------------------
def test_p29_does_not_fire_on_k1502_baseline_shapes():
    # K-1502 18-shape baseline (the canonical drift-detection cohort
    # used in K-1701 / K-1502-family iter11 lock-in): M ∈ {2048, 4096,
    # 8192} × N ∈ {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}
    # × K ∈ {4096} × dtype=fp16.  Exhaustively sampled (a single K
    # value is enough — drift detection runs the full 18-shape sweep
    # at a fixed K).
    k1502_baseline_shapes = [
        (M, N, K, dtype)
        for M in (2048, 4096, 8192)
        for N in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
        for K in (4096,)
        for dtype in ("torch.float16",)
    ]
    for cell in k1502_baseline_shapes:
        M, N, K, dtype = cell
        assert _k1704_p29_nib_envelope_aliasstack_routeout(M, N, K, dtype) is False, (
            f"K-1502 baseline shape {cell} unexpectedly admitted by "
            "K-1704 P29 — drift-detection cohort would regress."
        )


# ---------------------------------------------------------------------------
# (h) PER-N SUB-COHORT INVARIANT — each per-N frozenset is internally
#     consistent: every cell shares the right N, and the union of the six
#     per-N frozensets equals the full envelope (no orphan cells, no
#     accidental cross-N collisions).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "expected_n, frozen",
    [
        ( 80, _K1704_P29_NIB_ENVELOPE_ALIASSTACK_80),
        (112, _K1704_P29_NIB_ENVELOPE_ALIASSTACK_112),
        (144, _K1704_P29_NIB_ENVELOPE_ALIASSTACK_144),
        (176, _K1704_P29_NIB_ENVELOPE_ALIASSTACK_176),
        (208, _K1704_P29_NIB_ENVELOPE_ALIASSTACK_208),
        (240, _K1704_P29_NIB_ENVELOPE_ALIASSTACK_240),
    ],
    ids=["N80", "N112", "N144", "N176", "N208", "N240"],
)
def test_p29_per_n_frozenset_n_axis_pinned(expected_n, frozen):
    for cell in frozen:
        _, N, _, _ = cell
        assert N == expected_n, (
            f"P29 per-N frozenset for N={expected_n} contains foreign "
            f"cell {cell}; cross-N collision in the K-1704 admit set."
        )


def test_p29_per_n_frozensets_are_pairwise_disjoint():
    # Per-N frozensets are pairwise disjoint by construction (N-axis
    # partition); pin it so a future per-N expansion that accidentally
    # duplicates a cell across two N buckets surfaces.
    items = list(_K1704_P29_NIB_ENVELOPE_BY_N.items())
    for i, (n_i, fs_i) in enumerate(items):
        for n_j, fs_j in items[i + 1 :]:
            assert fs_i.isdisjoint(fs_j), (
                f"K-1704 P29 per-N frozensets for N={n_i} and N={n_j} "
                "overlap — cross-N collision in the audit set."
            )
