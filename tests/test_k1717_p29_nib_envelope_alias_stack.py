"""Unit fixture — K-1717 P29 NIB (N-In-Between) envelope 65-cell route-OUT.

Productionises the K-1704 paired n=30 hot-cache HIP-graph audit (108 cells)
on MI300X / gfx942 against the LIVE post-K-1685 P28 routing oracle: 65 of
108 cells cleared the strict 5% admit gate (oracle/hbl
``speedup_ci95_hi`` < ``1/1.05`` = 0.95238, i.e. forced-hipBLASLt is
reliably ≥5% faster than the live oracle with non-overlapping bootstrap
CIs).  Cohort oracle/hbl geomean 0.678 → 1.475× hbl-route lift.

Per the K-1717 minimalist refactor: one 65-cell ``frozenset`` and one
membership check.  This fixture pins the four invariants that are NOT
already covered by the existing alias-stack tests or by the K-1502-family
drift cadence:

1. **cardinality** — the frozenset is exactly the K-1704 admit set;
2. **routing contract** — every admit cell routes-OUT through the live
   ``k971_route_decision``; no admit cell falls through the chain;
3. **N-axis disjointness** — P29 N-axis is disjoint from the productionised
   N-ladder used by every prior K-COMPLEMENT alias-stack;
4. **K-1502 drift sanity** — the canonical 18-shape M=N square cohort
   (per K-1701 / K-1502-family) is structurally untouched by P29 (no
   N=128/256/512/1024/2048/4096/8192/16384 cell can possibly be in a
   frozenset whose N-axis is {80,112,144,176,208,240}).

The actual K-1502 perf harness run is captured at
``output/k1502_drift_harness_post_K1717.log`` (see PR description).
"""
import torch
import pytest

from tritonblas._route_predicate import (
    R_K979_P5_route_to_hbl,
    k971_route_decision,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _K1704_P29_NIB_ENVELOPE,
    _k1704_p29_nib_envelope_aliasstack_routeout,
)


# ---------------------------------------------------------------------------
# 1. Cardinality + structural invariants
# ---------------------------------------------------------------------------
def test_envelope_cardinality_is_65():
    """Single audit handle — the 65-cell K-1704 admit set."""
    assert len(_K1704_P29_NIB_ENVELOPE) == 65


def test_envelope_n_axis_is_strictly_in_between():
    """N-axis is strictly disjoint from the productionised N-ladder
    {128,256,512,1024,2048,4096,8192,16384,32768} used by every prior
    K-COMPLEMENT alias-stack frozenset.  This is the sibling-N firewall."""
    n_axis = {n for (_M, n, _K, _dt) in _K1704_P29_NIB_ENVELOPE}
    productionised_ladder = {128, 256, 512, 1024, 2048, 4096, 8192,
                             16384, 32768}
    assert n_axis == {80, 112, 144, 176, 208, 240}
    assert n_axis.isdisjoint(productionised_ladder)


def test_envelope_disjoint_from_p28():
    """Disjoint from K-1685 P28 (N=128 only) by N-axis projection."""
    assert _K1704_P29_NIB_ENVELOPE.isdisjoint(
        _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30)


def test_bf16_only_appears_when_p5_excludes_it():
    """bf16 cells in P29 are reachable here only because R-K979 P5
    Clause-3 (min(M,N) ≤ 192 ∧ K ≥ 2048, bf16-only) does NOT catch them
    — i.e. minMN > 192 (true at N ∈ {208,240} given M ≥ 2048)."""
    bf16_cells = {c for c in _K1704_P29_NIB_ENVELOPE
                  if c[3] == "torch.bfloat16"}
    assert bf16_cells, "bf16 must appear at N ∈ {208,240}"
    overlap = {c for c in bf16_cells
               if R_K979_P5_route_to_hbl(c[0], c[1], c[2], c[3])}
    assert overlap == set(), (
        f"bf16 P29 cells must be DISJOINT from P5 Clause-3; overlap = "
        f"{overlap!r} indicates P5's minMN-ceiling has been widened > 192")


# ---------------------------------------------------------------------------
# 2. Routing contract — every admit cell ends up routed-OUT through the
#    live k971_route_decision (P29 dispatch slot wired correctly).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "M,N,K,dtype_str", sorted(_K1704_P29_NIB_ENVELOPE),
    ids=lambda c: f"{c}",
)
def test_admit_cell_routes_out(M, N, K, dtype_str):
    """Every K-1704 admit cell must end up routed-OUT via the live
    ``k971_route_decision``.  Either P29 catches it OR an upstream
    P1–P28 catches it (alias overlap is allowed by design); what is
    NOT allowed is an admit cell falling through to ``return False``."""
    dtype = torch.float16 if dtype_str == "torch.float16" else torch.bfloat16
    assert k971_route_decision(M, N, K, dtype, dtype,
                               enable_streamk=False,
                               work_stealing=False) is True, (
        f"K-1704 admit cell {(M, N, K, dtype_str)} fell through every "
        f"P1–P29 predicate — P29 dispatch slot regression?")


# ---------------------------------------------------------------------------
# 3. K-1502 18-shape canonical drift cohort (CANON18) — structurally
#    untouched by P29 because P29's N-axis is strictly off-ladder.
# ---------------------------------------------------------------------------
CANON18_MNK = [(mn, mn, k)
               for mn in (512, 1024, 2048, 4096, 8192, 16384)
               for k in (2048, 8192, 32768)]


@pytest.mark.parametrize("M,N,K", CANON18_MNK,
                         ids=[f"M=N={mn},K={k}" for (mn, _, k) in CANON18_MNK])
def test_p29_does_not_fire_on_k1502_canonical_18(M, N, K):
    """P29 must NEVER fire on the K-1502-family CANON18 cohort (M=N
    square, M=N ∈ {512..16384}, K ∈ {2048,8192,32768}, bf16) — structural
    by N-axis projection, but pinned here so a future predicate rewrite
    can't accidentally widen P29's N-axis without breaking this test.
    The actual K-1502 perf harness run is in
    ``output/k1502_drift_harness_post_K1717.log``."""
    for dtype_str in ("torch.bfloat16", "torch.float16"):
        assert _k1704_p29_nib_envelope_aliasstack_routeout(
            M, N, K, dtype_str) is False, (
            f"P29 misfired on CANON18 cell {(M, N, K, dtype_str)}; "
            f"would silently regress the K-1502 drift cadence baseline.")
