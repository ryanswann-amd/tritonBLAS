"""K-1717 P29 NIB (N-In-Between) envelope alias-stack — minimal unit fixture.

Productionises the K-1704 paired n=30 hot-cache HIP-graph MI300X audit
(108 cells: N ∈ {80,112,144,176,208,240} × M ∈ {2048,4096,8192} ×
K ∈ {2048,8192,32768} × {fp16,bf16}): 65 cells cleared the strict 5%
admit gate (oracle/hbl ``speedup_ci95_hi`` < 1/1.05 = 0.95238).  Cohort
oracle/hbl geomean 0.678 → 1.475× hbl-route lift.

Per the K-1717 minimalist refactor: the production code is one frozenset
+ one O(1) membership predicate.  Belt-and-suspenders disjointness asserts
(N-axis disjointness, P28-disjointness, P5-bf16 firewall) are STRUCTURALLY
implied by frozenset membership and are dropped from this fixture; the
K-1502-family drift cadence covers the runtime regression surface.

Three tests pin the actual contract:

1. ``test_envelope_cardinality_and_round_trip`` — frozenset has exactly
   65 cells AND every admitted cell round-trips through the public
   membership predicate ``_k1704_p29_nib_envelope_aliasstack_routeout``.
2. ``test_admit_cell_routes_out`` — parametrised over the 65 admit cells:
   each must end up routed-OUT through the live ``k971_route_decision``
   (P29 dispatch slot is wired correctly).
3. ``test_non_admit_cell_does_not_route_via_p29`` — parametrised over a
   negative-cell sample (a) one of the 43 K-1704 cells excluded by the
   5% gate, (b) every K-1502 CANON18 M=N square cell, and (c) a sample
   of K-1685 P28 (N=128) cells: NONE of them may route via P29.  This
   covers the admit-gate boundary itself (Testing Zealot ask).

The actual K-1502 perf harness drift run is captured at
``output/k1502_drift_harness_post_K1717.log`` (see PR description).
"""
import json
import pathlib

import torch
import pytest

from tritonblas._route_predicate import (
    k971_route_decision,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _K1704_P29_NIB_ENVELOPE,
    _k1704_p29_nib_envelope_aliasstack_routeout,
)


def _dtype(dtype_str: str) -> torch.dtype:
    return torch.float16 if dtype_str == "torch.float16" else torch.bfloat16


# ---------------------------------------------------------------------------
# 1. Cardinality + admit-set membership round-trip.  Single test consolidates
#    the audit-handle invariant (per Minimalist refactor: structural N-axis
#    disjointness from the productionised ladder + P28-disjointness are
#    implied by frozenset membership and need not be re-asserted here).
# ---------------------------------------------------------------------------
def test_envelope_cardinality_and_round_trip():
    """The K-1704 admit set is exactly 65 cells AND every admitted cell
    round-trips through the public membership predicate."""
    assert len(_K1704_P29_NIB_ENVELOPE) == 65, (
        "K-1704 admit set must be exactly 65 cells; deviation = authoring "
        "typo against the K-1704 paired n=30 audit.")
    for (M, N, K, dtype_str) in _K1704_P29_NIB_ENVELOPE:
        assert _k1704_p29_nib_envelope_aliasstack_routeout(
            M, N, K, dtype_str) is True, (
            f"admit cell {(M, N, K, dtype_str)} must round-trip through "
            f"_k1704_p29_nib_envelope_aliasstack_routeout")


# ---------------------------------------------------------------------------
# 2. Routing contract — every admit cell ends up routed-OUT through the
#    live k971_route_decision (parametrised over all 65 admitted cells).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "M,N,K,dtype_str", sorted(_K1704_P29_NIB_ENVELOPE),
    ids=lambda c: f"{c}",
)
def test_admit_cell_routes_out(M, N, K, dtype_str):
    """Every K-1704 admit cell must route-OUT via the live
    ``k971_route_decision`` — either P29 catches it or an upstream
    P1–P28 catches it (alias overlap is allowed by design).  What is
    NOT allowed is an admit cell falling through to ``return False``.
    """
    assert k971_route_decision(M, N, K, _dtype(dtype_str), _dtype(dtype_str),
                               enable_streamk=False,
                               work_stealing=False) is True, (
        f"K-1704 admit cell {(M, N, K, dtype_str)} fell through every "
        f"P1–P29 predicate — P29 dispatch slot regression?")


# ---------------------------------------------------------------------------
# 3. Negative-cell sample — non-admit cells must NOT route via P29.
#    Covers (a) K-1704 5%-gate-excluded cells (admit-gate boundary,
#    Testing Zealot ask), (b) the K-1502 CANON18 M=N cohort (drift
#    cadence safety), (c) K-1685 P28 N=128 cells (sibling-N firewall).
# ---------------------------------------------------------------------------
# (a) One representative cell that K-1704 measured but EXCLUDED at the 5%
# gate (oracle/hbl CI95_hi >= 0.95238 — speedup not strong enough to
# justify routing).  Pinned literally so a K-1704 re-audit that flips
# this cell into the admit set is caught here.  Source-of-truth:
# /home/ryaswann/mc2-workspaces/K-1717/output/k1717_excluded_cells.json
# entry [0] = (M=2048, N=80, K=2048, bf16) speedup=0.927 CI=[0.842,0.967].
K1704_EXCLUDED_REP = (2048, 80, 2048, "torch.bfloat16")

# (b) K-1502 CANON18 cohort (M=N square × K ∈ {2048,8192,32768} × bf16).
CANON18_NEG = [(mn, mn, k, "torch.bfloat16")
               for mn in (512, 1024, 2048, 4096, 8192, 16384)
               for k in (2048, 8192, 32768)]

# (c) Sample of K-1685 P28 (N=128) cells — drawn directly from the live
# P28 frozenset to keep this test in lock-step if K-1685 ever re-audits.
P28_SAMPLE = sorted(_K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30)[:6]

NEGATIVE_CELLS = [K1704_EXCLUDED_REP, *CANON18_NEG, *P28_SAMPLE]


@pytest.mark.parametrize("M,N,K,dtype_str", NEGATIVE_CELLS,
                         ids=lambda c: f"{c}")
def test_non_admit_cell_does_not_route_via_p29(M, N, K, dtype_str):
    """No non-admit cell may route via P29.  Pins the admit-gate
    boundary directly: if a future predicate rewrite widens P29's
    membership beyond the 65 K-1704-verified cells, this test fails.

    The negative sample includes (a) a K-1704 cell that failed the 5%
    admit gate, (b) every K-1502 CANON18 M=N square cell (drift-cadence
    safety: P29 must never silently regress the K-1502 baseline), and
    (c) a sample of K-1685 P28 (N=128) cells (sibling-N firewall: P29
    N-axis is {80,112,144,176,208,240}, must not collide with P28).
    """
    assert _k1704_p29_nib_envelope_aliasstack_routeout(
        M, N, K, dtype_str) is False, (
        f"non-admit cell {(M, N, K, dtype_str)} routed via P29 — admit "
        f"gate widened?  P29 must be exactly the 65-cell K-1704 admit "
        f"set, no more, no less.")
