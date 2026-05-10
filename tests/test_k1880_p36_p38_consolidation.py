"""K-1880 R-Minimalist P36+P37+P38 consolidation tests.

Mirrors the K-1864 P32-P36 consolidation test pattern (commit c954555):
the dispatcher in `_k971_route_to_hbl` previously chained three sequential
frozenset membership probes for the K-COMPLEMENT N∈{320,352,416} verified-
winner subsets.  K-1880's R-Minimalist consolidation collapses those into
ONE membership probe over a single canonical inlined frozenset
`_K1880_SKINNY_KCOMPL_N320_N352_N416_ROUTEOUT_51`; the three per-slot
frozensets become derived N-axis projection VIEWS that back the existing
per-slot tests without churn.

Tests in this file enforce the consolidation contract:

  1. Cardinality + N-axis projection of the canonical roster.
  2. Derivation equivalence — each per-slot view ≡ the corresponding
     N-axis filter of the canonical roster (cell-for-cell).
  3. Dispatcher-admit equivalence — every cell in the canonical roster
     routes True via the LIVE `_k971_route_to_hbl` (the post-K-1880
     production code path), parametrised per cell so a future drop of
     any single cell from the route surfaces as one named pytest failure.
  4. Hostile control — a sample of N values OUTSIDE {320, 352, 416}
     does NOT route via the K-1880 canonical probe (must NOT match by
     the K-1880 line; orthogonal P-slots may still admit them — the
     test asserts they are NOT in the canonical set, NOT that the
     dispatcher returns False).
  5. Upstream-alias exclusions: each of the three (2048,N,4096,bf16)
     cells (N∈{320,352,416}) excluded from the K-1880 canonical roster
     per R-K1825 still routes True via `_k971_route_to_hbl` overall
     (through P5 / upstream alias), proving the exclusion did not drop
     coverage; AND each is NOT a member of the K-1880 canonical set
     (proving R-K1825 single-source-of-truth was respected).
  6. Validator parity assertion (per R-Testing-Zealot K-1880 RETRY):
     loads the post-promotion HIP-graph hot-cache validator's JSON
     summary if present and asserts every routed cell stayed within
     ≥0.95× of HBL — making step 5's parity claim CI-enforceable.

Per the K-1864 pattern, the canonical set is the ONLY source of truth
the dispatcher consults; this file is the only place that asserts the
canonical-vs-views consistency contract empirically (the per-slot tests
under `test_p3{6,7,8}_*_alias_stack.py` continue to pin slot-local
invariants on the views).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tritonblas._route_predicate import (
    _K1880_SKINNY_KCOMPL_N320_N352_N416_ROUTEOUT_51 as _CANON,
    _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17 as _P36,
    _P37_SKINNY_N352_KCOMPL_VERIFIED_WIN_17 as _P37,
    _P38_SKINNY_N416_KCOMPL_VERIFIED_WIN_17 as _P38,
)
from tritonblas.matmul import _k971_route_to_hbl


# Upstream-alias exclusions per R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST:
# the (2048,N,4096,bf16) cell on each N-rung is already routed by an upstream
# alias-stack slot (P5 or P5-parity-band) and is therefore excluded from the
# K-1880 canonical roster to avoid duplicate routing.
_UPSTREAM_ALIASED_EXCLUSIONS = frozenset({
    (2048, 320, 4096, "torch.bfloat16"),  # K-1843 paired-n30: r=1.0001, p=0.293
    (2048, 352, 4096, "torch.bfloat16"),  # K-1843 paired-n30: r=1.0031, p=0.167
    (2048, 416, 4096, "torch.bfloat16"),  # K-1873 paired-n30: r=1.002,  p=0.394
})


# ---------------------------------------------------------------------------
# 1. Cardinality + N-axis projection
# ---------------------------------------------------------------------------
def test_canonical_cardinality_is_51():
    """3 N-rungs × (full 18-cell K-COMPLEMENT grid − 1 R-K1825 alias) = 51."""
    assert len(_CANON) == 51


def test_canonical_n_axis_projection_is_320_352_416():
    n_axis = frozenset(n for (_m, n, _k, _dt) in _CANON)
    assert n_axis == frozenset({320, 352, 416})


def test_canonical_envelope_matches_kcompl_grid_minus_aliases():
    """The canonical roster MUST exactly equal the union of the 3 K-COMPLEMENT
    grids minus the 3 R-K1825 upstream-aliased cells.  Pinned to detect any
    silent contraction (winner cell leaks back to TB) or expansion (alias
    cell duplicates an upstream-routed slot)."""
    full = frozenset(
        (M, N, K, dt) for M in (2048, 4096, 8192)
        for N in (320, 352, 416)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert len(full) == 54
    assert _CANON == full - _UPSTREAM_ALIASED_EXCLUSIONS


# ---------------------------------------------------------------------------
# 2. Derivation equivalence (per-slot views ≡ N-axis filter of canonical)
# ---------------------------------------------------------------------------
def test_p36_view_equals_canonical_n320_filter():
    assert _P36 == frozenset(c for c in _CANON if c[1] == 320)


def test_p37_view_equals_canonical_n352_filter():
    assert _P37 == frozenset(c for c in _CANON if c[1] == 352)


def test_p38_view_equals_canonical_n416_filter():
    assert _P38 == frozenset(c for c in _CANON if c[1] == 416)


def test_per_slot_views_partition_canonical():
    """The 3 per-slot views form a partition (disjoint + union ≡ canonical).
    Detects any drift where a cell would mis-attribute to two N-rungs."""
    assert _P36.isdisjoint(_P37)
    assert _P36.isdisjoint(_P38)
    assert _P37.isdisjoint(_P38)
    assert _P36 | _P37 | _P38 == _CANON


# ---------------------------------------------------------------------------
# 3. Dispatcher-admit equivalence — parametrised per cell
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell", sorted(_CANON))
def test_canonical_cell_routes_through_k971_dispatcher(cell):
    """Every cell in the canonical 51-cell roster MUST admit True via the
    LIVE `_k971_route_to_hbl()`.  Per R-Testing-Zealot (K-1880 RETRY), this
    is the load-bearing CI assertion that protects the dispatcher contract:
    if a future change drops any single cell from the route, exactly one
    named pytest case fails (pinpointing the regression)."""
    M, N, K, dt = cell
    assert _k971_route_to_hbl(M, N, K, dt, dt, False, False), (
        f"K-1880 canonical cell ({M}, {N}, {K}, {dt}) failed to admit via "
        f"_k971_route_to_hbl — dispatcher contract regression"
    )


# ---------------------------------------------------------------------------
# 4. Hostile control — N values OUTSIDE {320, 352, 416} are NOT in the
#    K-1880 canonical set.  Note: orthogonal upstream P-slots (P5, P28-P37,
#    K-1864 set, etc.) MAY still route some of these cells; this test
#    asserts only that they are NOT members of the K-1880 canonical roster
#    (i.e., the dispatcher's K-1880 line CANNOT be the one that routes them).
# ---------------------------------------------------------------------------
_HOSTILE_N_SAMPLE = (
    # off-by-1 / off-by-32 / off-by-64 around each P38 boundary
    [(M, N, K, dt) for M in (2048, 4096, 8192)
                   for N in (288, 304, 384, 400, 432, 448, 480, 512)
                   for K in (4096, 8192, 16384)
                   for dt in ("torch.bfloat16", "torch.float16")]
    # plus a handful of K-axis off-rungs at the in-cohort N values
    + [(M, N, K, dt) for M in (2048, 4096, 8192)
                     for N in (320, 352, 416)
                     for K in (2048, 32768)  # outside the K-COMPLEMENT band
                     for dt in ("torch.bfloat16", "torch.float16")]
)


@pytest.mark.parametrize("cell", _HOSTILE_N_SAMPLE)
def test_hostile_n_or_k_cell_not_in_canonical(cell):
    """Hostile-N or off-K-rung cells MUST NOT be members of the K-1880
    canonical 51-cell roster.  Detects any over-broad expansion of the
    consolidated set (e.g., a careless union with a sibling slot's
    cells)."""
    assert cell not in _CANON, (
        f"K-1880 canonical roster over-broad: hostile cell {cell} leaked in"
    )


# ---------------------------------------------------------------------------
# 5. R-K1825 upstream-alias exclusions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell", sorted(_UPSTREAM_ALIASED_EXCLUSIONS))
def test_p5_aliased_cell_excluded_from_canonical(cell):
    """Each (2048,N,4096,bf16) cell is already routed by an upstream alias-
    stack slot per R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST.  Including
    it in the K-1880 canonical roster would cause duplicate routing.  Pin
    the exclusion."""
    assert cell not in _CANON


@pytest.mark.parametrize("cell", sorted(_UPSTREAM_ALIASED_EXCLUSIONS))
def test_p5_aliased_cell_still_routes_via_upstream_path(cell):
    """The R-K1825 exclusion must NOT drop dispatch coverage for these
    cells — they are still routed by an upstream slot (P5 Clause-1 mid-rect
    non-square for N=416; the P5 parity-band classifier for N∈{320,352}).
    This test pins the upstream coverage so a future P5 change that lost
    one of these cells would surface as a named failure here, not silently
    drop the cell back to TB-native."""
    M, N, K, dt = cell
    assert _k971_route_to_hbl(M, N, K, dt, dt, False, False), (
        f"R-K1825-aliased cell ({M},{N},{K},{dt}) lost upstream coverage — "
        f"P38 excluded it expecting an upstream slot to route it, but "
        f"_k971_route_to_hbl returns False"
    )


# ---------------------------------------------------------------------------
# 6. Post-promotion validator parity assertion (R-Testing-Zealot K-1880 RETRY)
# ---------------------------------------------------------------------------
# When the post-promotion validator (k1880_p38_postpromo_validate.py) runs
# in CI, it writes a JSON summary to {workspace}/output/.  This test loads
# that artifact (if present) and asserts the parity floor:
#   * every routed cell stayed within ≥0.95× of HBL (no regression)
#   * cohort geomean ≥ 0.98× (parity-band per the K-1850 productionisation
#     wiring criterion)
# If the artifact is absent (e.g., CI runs without GPU), the test SKIPS —
# making this enforceable when measurements exist, not flaky-blocking when
# they don't.
def _candidate_summary_paths():
    candidates = []
    env = os.environ.get("K1880_POSTPROMO_SUMMARY")
    if env:
        candidates.append(Path(env))
    # Conventional workspace location (per ROCm-Claude task scaffolding)
    candidates.append(Path("/home/ryaswann/mc2-workspaces/K-1880/output/"
                           "k1880_p38_postpromo_n30.json"))
    # Repo-relative fallback for local dev
    candidates.append(Path(__file__).resolve().parent.parent
                      / "k1880_p38_postpromo_n30.json")
    return candidates


def test_postpromo_validator_parity_floor():
    """Per R-Testing-Zealot (K-1880 RETRY): step 5's parity claim must be
    CI-enforceable, not living only in a CSV."""
    summary = None
    src = None
    for p in _candidate_summary_paths():
        if p.is_file():
            with open(p) as f:
                summary = json.load(f)
            src = p
            break
    if summary is None:
        pytest.skip("No post-promotion validator summary found "
                    "(set K1880_POSTPROMO_SUMMARY=<path> to a JSON written by "
                    "scripts/k1880_p38_postpromo_validate.py)")

    cells = summary.get("cells", [])
    assert cells, f"validator summary at {src} has no cells"

    # Acceptance: every cell ratio_mean ≥ 0.95 (parity-band, no regression).
    regressions = [
        c for c in cells
        if c.get("ratio_mean", c.get("ratio_pre_over_post", 1.0)) < 0.95
    ]
    assert not regressions, (
        f"K-1880 post-promotion parity floor violated at {src}: "
        f"{len(regressions)} cell(s) below 0.95× of HBL — "
        + "; ".join(
            f"({c['M']},{c['N']},{c['K']},{c['dtype']})"
            f"={c.get('ratio_mean', c.get('ratio_pre_over_post')):.4f}"
            for c in regressions
        )
    )

    geomean = summary.get("cohort_geomean_admit_17",
                          summary.get("geomean_ratio_pre_over_post", 1.0))
    assert geomean >= 0.98, (
        f"K-1880 post-promotion cohort geomean {geomean:.4f}× < 0.98× "
        f"parity floor at {src}"
    )
