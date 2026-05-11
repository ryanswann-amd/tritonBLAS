"""K-1472 P19 skinny_N4096 K-COMPLEMENT pin tests — torch-free.

Minimal pin coverage for the K-1472 productionisation of the 28-cell
`_K1472_P19_SKINNY_N4096_KCOMPL_ROUTEOUT` frozenset stacked as the
12th-position envelope on top of the K-1437 P17 11-predicate stack
(envelope grows 127 → 155 cells; +28 admits at the N=4096 column-narrow
tier of the K-COMPLEMENT methodology).

Cardinality and cross-frozenset disjointness are already asserted at module
load in `_route_predicate.py`; we do not duplicate those here (per
R-1406.MODULE-LOAD-ASSERTS-MAKE-PYTEST-CARDINALITY-DISJOINTNESS-DUPS-DEAD-WEIGHT).
The pytest suite focuses on the orthogonal behavioural pins:

  * exact membership of the K-1472 P19 28 cells
    (M ∈ {2048,4096,8192} × N=4096 × K ∈ {2048,4096,8192,16384,32768} ×
    {bf16, fp16} MINUS the 2 (4096,4096,4096,*) P12-claimed cells)
  * full-chain dispatch routes every admit cell to hipBLASLt (covers the
    12th-position predicate's interaction with the 11-predicate precedence
    chain — and verifies the K-1437 P17 invariant is preserved)
  * carve-out short-circuits (streamk / work-stealing / mismatched-dtype)
    correctly suppress the route-OUT for an admit cell
  * P17 (K-1437 EXTENDED, N=1024) cells STILL route to hipBLASLt under the
    new chain (zero regression on the 11th-position envelope at the
    sibling N column — the load-bearing stacking invariant)
  * five nearest-neighbour negative pin tests for shapes one axis off the
    K-1472 admit grid that must NOT be admitted by the P19 strict-equality
    predicate function (firewall against silent over-routing from a future
    range/pattern predicate refactor — R-1417 #2 / Skeptic guidance carried
    forward through K-1433 / K-1437 / K-1472)
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K1437_P17_SKINNY_N1024_KCOMPL_EXT_ROUTEOUT_12,
    _K1472_P19_SKINNY_N4096_KCOMPL_ROUTEOUT,
    _k1472_p19_skinny_n4096_kcompl_routeout,
    k971_route_decision,
)


# Bucket rule per K-1472 success criteria: M ∈ {2048,4096,8192} ×
# N=4096 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16} MINUS the 2
# (4096,4096,4096,*) cells already claimed by K-1361 P12 square_mid.
_P12_CLAIMED_4096CUBED = frozenset({
    (4096, 4096, 4096, "torch.bfloat16"),
    (4096, 4096, 4096, "torch.float16"),
})
_EXPECTED_K1472_28_CELLS = frozenset({
    (M, 4096, K, dt)
    for M in (2048, 4096, 8192)
    for K in (2048, 4096, 8192, 16384, 32768)
    for dt in ("torch.bfloat16", "torch.float16")
}) - _P12_CLAIMED_4096CUBED


def test_k1472_membership_exactly_28_kcomplement_cells():
    """The frozenset must equal the bucket rule exactly — no missing or
    unknown cells; 30-cell N=4096 grid minus the 2 P12-claimed 4096³ cells."""
    diff_missing = sorted(_EXPECTED_K1472_28_CELLS
                          - _K1472_P19_SKINNY_N4096_KCOMPL_ROUTEOUT)
    diff_unknown = sorted(_K1472_P19_SKINNY_N4096_KCOMPL_ROUTEOUT
                          - _EXPECTED_K1472_28_CELLS)
    assert (
        _K1472_P19_SKINNY_N4096_KCOMPL_ROUTEOUT == _EXPECTED_K1472_28_CELLS
    ), (
        f"K-1472 P19 frozenset diverges from the K-COMPLEMENT bucket rule "
        f"(M ∈ {{2048,4096,8192}} × N=4096 × K ∈ {{2048,4096,8192,16384,"
        f"32768}} × {{bf16,fp16}} MINUS the 2 P12-claimed 4096³ cells).\n"
        f"  missing : {diff_missing}\n"
        f"  unknown : {diff_unknown}")


def test_k1472_p12_4096cubed_cells_explicitly_excluded():
    """Explicit firewall: the 2 (4096,4096,4096,{bf16,fp16}) cells claimed
    by K-1361 P12 square_mid MUST NOT appear in the K-1472 P19 frozenset
    (A4 no-double-admit firewall against P12 — without this exclusion the
    bucket cardinality would be 30, not 28)."""
    for cell in _P12_CLAIMED_4096CUBED:
        assert cell not in _K1472_P19_SKINNY_N4096_KCOMPL_ROUTEOUT, (
            f"K-1472 P19 strict-equality firewall LEAK: {cell} is claimed "
            f"by K-1361 P12 square_mid 4096³ and must NOT appear in the "
            f"P19 frozenset; the P12 strict-equality match must be the "
            f"single owner of these 2 cells (the 4096³ K-1472 measurement "
            f"shows ratio_median ≈ 1.0 because tritonblas reaches the same "
            f"hipBLASLt kernel through P12 anyway).")


@pytest.mark.parametrize("M,N,K,dtype", sorted(_EXPECTED_K1472_28_CELLS))
def test_full_dispatch_routes_every_k1472_cell_to_hbl(M, N, K, dtype):
    """Every K-1472 P19 admit cell must route to hipBLASLt under the full
    12-position precedence chain with default carve-out settings."""
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


@pytest.mark.parametrize("enable_streamk,work_stealing,b_dtype", [
    (True,  False, "torch.bfloat16"),  # streamk requested
    (False, True,  "torch.bfloat16"),  # work-stealing requested
    (False, False, "torch.float16"),   # mismatched dtype (a bf16, b fp16)
])
def test_k1472_admit_cell_carveout_short_circuits(
    enable_streamk, work_stealing, b_dtype,
):
    """Admit cells must NOT route to hipBLASLt when streamk / work-stealing
    / mismatched-dtype is requested.  Pick (2048, 4096, 2048, bf16) — the
    small-K starvation corner where K-1472's measured speedup is largest
    (3.79×) — as the representative carve-out probe."""
    a_dtype = "torch.bfloat16"
    M, N, K = 2048, 4096, 2048
    assert (M, N, K, a_dtype) in _K1472_P19_SKINNY_N4096_KCOMPL_ROUTEOUT
    routed = k971_route_decision(
        M, N, K, a_dtype, b_dtype, enable_streamk, work_stealing,
    )
    assert routed is False, (
        f"K-1472 P19 admit cell ({M},{N},{K},{a_dtype}) must NOT route to "
        f"hipBLASLt when carve-out applies (streamk={enable_streamk}, "
        f"work_stealing={work_stealing}, b_dtype={b_dtype}); short-circuit "
        f"in _k971_route_to_hbl is load-bearing.")


@pytest.mark.parametrize("M,N,K,dtype",
                         sorted(_K1437_P17_SKINNY_N1024_KCOMPL_EXT_ROUTEOUT_12))
def test_k1437_p17_extended_envelope_unchanged_under_p19_stack(M, N, K, dtype):
    """Regression: every K-1437 P17 (11th-position, N=1024 EXTENDED) cell
    must STILL route to hipBLASLt under the new 12-position stack.
    Asserts bit-identical routing behaviour on the prior productionisation
    envelope at the sibling N column — the load-bearing stacking invariant
    for the K-COMPLEMENT N-axis lineage."""
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


# ---------------------------------------------------------------------------
# Negative pin tests — strict-equality firewall against silent over-routing
# from the P19 predicate.
# ---------------------------------------------------------------------------
# Per Skeptic guidance carried forward from K-1417 / K-1433 / K-1437: a
# strict-equality frozenset must be explicitly fenced off against adjacent-
# but-not-admitted shapes that a future broadening (e.g. range / pattern
# predicate refactor) could silently absorb.  The five nearest non-member
# neighbours each vary exactly ONE axis off the K-1472 admit grid:
#
#   * (2048, 2048, 4096, bf16)  — M/K admit, N=2048 OFF (one column-step
#                                 back into the K-1438 N=2048 regime; would
#                                 be claimed by K-1438 P17 sister branch
#                                 once productionised — but P19 itself
#                                 MUST NOT claim it)
#   * (2048, 8192, 4096, bf16)  — M/K admit, N=8192 OFF (one column-step
#                                 forward into the K-1474 N=8192 sister
#                                 envelope — but P19 itself MUST NOT claim it)
#   * (1024, 4096, 4096, bf16)  — N/K admit, M=1024 OFF (M=1024 not in
#                                 K-1472 anchor row; firewall test asserts
#                                 P19 does NOT fire)
#   * (4096, 4096, 4096, bf16)  — M/N/K all "admit" but THIS is the P12
#                                 4096³ cell explicitly excluded from
#                                 the K-1472 bucket; P19 firewall MUST
#                                 NOT claim it (the load-bearing P12/P19
#                                 partition firewall on the 4096³ cell)
#   * (4096, 4096, 1024, bf16)  — M/N admit, K=1024 OFF the K-COMPLEMENT
#                                 mesh (K=1024 not in the K-1389→K-1472
#                                 K-mesh); P19 firewall test
#
# These tests target the P19 predicate function directly (NOT the full
# `k971_route_decision`), because some of these neighbours may be
# legitimately routed by an *earlier* predicate (e.g. P12 for the 4096³
# cell) in the precedence chain.  The P19 firewall guarantee is: P19
# itself does NOT contribute a route decision for any shape outside its
# 28-cell admit set.
_K1472_NEAREST_NEIGHBOUR_NEGATIVES = [
    # (M,    N,    K,     a_dtype,           reason)
    (2048,  2048,  4096, "torch.bfloat16",
     "N=2048 OFF column-narrow tier (P19 fixes N=4096); K-1438 sister scope"),
    (2048,  8192,  4096, "torch.bfloat16",
     "N=8192 OFF column-narrow tier; K-1474 sister scope, NOT P19"),
    (1024,  4096,  4096, "torch.bfloat16",
     "M=1024 OFF anchor row (P19 anchors M ∈ {2048,4096,8192})"),
    (4096,  4096,  4096, "torch.bfloat16",
     "P12 4096³ EXCLUSION (claimed by K-1361 P12 square_mid, NOT P19)"),
    (4096,  4096,  1024, "torch.bfloat16",
     "K=1024 OFF K-COMPLEMENT mesh (K ∈ {2048,4096,8192,16384,32768})"),
]


@pytest.mark.parametrize("M,N,K,dtype,reason",
                         _K1472_NEAREST_NEIGHBOUR_NEGATIVES)
def test_k1472_nearest_neighbour_shapes_not_admitted_by_p19_predicate(
    M, N, K, dtype, reason,
):
    """Negative pin: nearest non-member neighbours one axis off the K-1472
    admit grid must NOT be admitted by the P19 strict-equality predicate
    function `_k1472_p19_skinny_n4096_kcompl_routeout`.  Targets the P19
    firewall directly (independent of earlier-precedence predicates that
    may legitimately claim these shapes for their own reasons — P12 for
    the 4096³ cell)."""
    # Sanity: the neighbour shape is genuinely outside the P19 admit set.
    assert (M, N, K, dtype) not in _K1472_P19_SKINNY_N4096_KCOMPL_ROUTEOUT, (
        f"Test scaffolding bug: neighbour ({M},{N},{K},{dtype}) is actually "
        f"in the P19 admit set — the negative test would pass trivially.")
    p19_admit = _k1472_p19_skinny_n4096_kcompl_routeout(M, N, K, dtype)
    assert p19_admit is False, (
        f"P19 strict-equality firewall LEAK: ({M},{N},{K},{dtype}) — {reason} "
        f"— P19 predicate returned True, but the K-1472 admit set is exactly "
        f"the 28-cell K-COMPLEMENT grid (M ∈ {{2048,4096,8192}} × N=4096 × "
        f"K ∈ {{2048,4096,8192,16384,32768}} × {{bf16,fp16}} MINUS the 2 "
        f"P12-claimed 4096³ cells); any True outside this set signals an "
        f"over-broad predicate.")
