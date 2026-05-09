"""K-1437 P17 skinny_N1024 K-COMPLEMENT EXTENDED pin tests — torch-free.

Minimal pin coverage for the K-1437 productionisation of the 12-cell
`_K1437_P17_SKINNY_N1024_KCOMPL_EXT_ROUTEOUT_12` frozenset stacked as the
11th-position envelope on top of the K-1433 P16 10-predicate stack
(envelope grows 115 → 127 cells; +12 admits at the EXTREMES K-band of the
N=1024 column-narrow regime).

Cardinality and cross-frozenset disjointness are already asserted at module
load in `_route_predicate.py`; we do not duplicate those here (per
R-1406.MODULE-LOAD-ASSERTS-MAKE-PYTEST-CARDINALITY-DISJOINTNESS-DUPS-DEAD-WEIGHT).
The pytest suite focuses on the orthogonal behavioural pins:

  * exact membership of the K-1437 P17 12 cells
    (M ∈ {2048, 4096, 8192} × N=1024 × K ∈ {2048, 32768} × {bf16, fp16})
  * full-chain dispatch routes every admit cell to hipBLASLt (covers the
    11th-position predicate's interaction with the 10-predicate precedence
    chain — and verifies the K-1433 P16 BASE invariant is preserved)
  * carve-out short-circuits (streamk / work-stealing / mismatched-dtype)
    correctly suppress the route-OUT for an admit cell
  * P16 (K-1433 BASE, N=1024) cells STILL route to hipBLASLt under the new
    chain (zero regression on the 10th-position envelope at the same
    N column — this is the load-bearing stacking invariant)
  * four nearest-neighbour negative pin tests for shapes one axis off the
    K-1437 EXTENDED admit grid (N-axis off via N=512 K=2048; N-axis off via
    N=2048 K=2048; M-axis off via M=1024; K-axis off via K=4096 BASE) that
    must NOT be admitted by the P17 strict-equality predicate function
    (firewall against silent over-routing from a future range/pattern
    predicate refactor — R-1417 #2 and Skeptic guidance carried forward
    from K-1433)
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K1433_P16_SKINNY_N1024_KCOMPL_ROUTEOUT,
    _K1437_P17_SKINNY_N1024_KCOMPL_EXT_ROUTEOUT_12,
    _k1437_p17_skinny_n1024_kcompl_ext_routeout,
    k971_route_decision,
)


# Bucket rule per K-1437 success criteria: M ∈ {2048,4096,8192} ×
# N=1024 × K ∈ {2048,32768} × {bf16, fp16}.
_EXPECTED_K1437_12_CELLS = frozenset({
    (M, 1024, K, dt)
    for M in (2048, 4096, 8192)
    for K in (2048, 32768)
    for dt in ("torch.bfloat16", "torch.float16")
})


def test_k1437_membership_exactly_12_kcomplement_extended_cells():
    """The frozenset must equal the bucket rule exactly — no missing or
    unknown cells."""
    diff_missing = sorted(_EXPECTED_K1437_12_CELLS
                          - _K1437_P17_SKINNY_N1024_KCOMPL_EXT_ROUTEOUT_12)
    diff_unknown = sorted(_K1437_P17_SKINNY_N1024_KCOMPL_EXT_ROUTEOUT_12
                          - _EXPECTED_K1437_12_CELLS)
    assert (
        _K1437_P17_SKINNY_N1024_KCOMPL_EXT_ROUTEOUT_12
        == _EXPECTED_K1437_12_CELLS
    ), (
        f"K-1437 P17 frozenset diverges from the K-COMPLEMENT EXTENDED bucket "
        f"rule (M ∈ {{2048,4096,8192}} × N=1024 × K ∈ {{2048,32768}} × "
        f"{{bf16,fp16}}).\n"
        f"  missing : {diff_missing}\n"
        f"  unknown : {diff_unknown}")


@pytest.mark.parametrize("M,N,K,dtype", sorted(_EXPECTED_K1437_12_CELLS))
def test_full_dispatch_routes_every_k1437_cell_to_hbl(M, N, K, dtype):
    """Every K-1437 admit cell must route to hipBLASLt under the full
    11-position precedence chain with default carve-out settings."""
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


@pytest.mark.parametrize("enable_streamk,work_stealing,b_dtype", [
    (True,  False, "torch.bfloat16"),  # streamk requested
    (False, True,  "torch.bfloat16"),  # work-stealing requested
    (False, False, "torch.float16"),   # mismatched dtype (a bf16, b fp16)
])
def test_k1437_admit_cell_carveout_short_circuits(
    enable_streamk, work_stealing, b_dtype,
):
    """Admit cells must NOT route to hipBLASLt when streamk / work-stealing
    / mismatched-dtype is requested.  Pick (8192, 1024, 32768, bf16) — the
    long-K large-M corner where K-1437's measured speedup is largest — as
    the representative carve-out probe."""
    a_dtype = "torch.bfloat16"
    M, N, K = 8192, 1024, 32768
    assert (M, N, K, a_dtype) in _K1437_P17_SKINNY_N1024_KCOMPL_EXT_ROUTEOUT_12
    routed = k971_route_decision(
        M, N, K, a_dtype, b_dtype, enable_streamk, work_stealing,
    )
    assert routed is False, (
        f"K-1437 P17 admit cell ({M},{N},{K},{a_dtype}) must NOT route to "
        f"hipBLASLt when carve-out applies (streamk={enable_streamk}, "
        f"work_stealing={work_stealing}, b_dtype={b_dtype}); short-circuit "
        f"in _k971_route_to_hbl is load-bearing.")


@pytest.mark.parametrize("M,N,K,dtype",
                         sorted(_K1433_P16_SKINNY_N1024_KCOMPL_ROUTEOUT))
def test_k1433_p16_base_envelope_unchanged_under_p17_stack(M, N, K, dtype):
    """Regression: every K-1433 P16 (10th-position, N=1024 BASE) cell
    must STILL route to hipBLASLt under the new 11-position stack.
    Asserts bit-identical routing behaviour on the prior productionisation
    envelope at the SAME N column — the load-bearing stacking invariant
    for the BASE/EXTENDED partition at N=1024."""
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


# ---------------------------------------------------------------------------
# Negative pin tests — strict-equality firewall against silent over-routing
# from the P17 predicate.
# ---------------------------------------------------------------------------
# Per Skeptic guidance carried forward from K-1417 / K-1433: a strict-equality
# frozenset must be explicitly fenced off against adjacent-but-not-admitted
# shapes that a future broadening (e.g. range / pattern predicate refactor)
# could silently absorb.  The four nearest non-member neighbours each vary
# exactly ONE axis off the K-1437 EXTENDED admit grid:
#
#   * (2048,  512,  2048, bf16)  — M/K admit, N=512 OFF (one column-step
#                                  back into the K-1409/K-1417 N=512
#                                  EXTENSION regime; legitimately routed
#                                  by P15 in the precedence chain — but
#                                  P17 itself MUST NOT claim it)
#   * (2048, 2048,  2048, bf16)  — M/K admit, N=2048 OFF (one column-step
#                                  forward past the N=1024 column-narrow
#                                  regime; collides with K971 anchor table
#                                  on the M=N=2048 square cells but P17
#                                  itself MUST NOT claim it)
#   * (1024, 1024,  2048, bf16)  — N/K admit, M=1024 OFF (M=1024 not in
#                                  K-1437 anchor row; collides with
#                                  K971_ROUTE_TABLE square cells but P17
#                                  firewall test asserts P17 does NOT fire)
#   * (4096, 1024,  4096, bf16)  — M/N admit, K=4096 OFF (K=4096 is in
#                                  K-1433 P16 BASE — legitimately routed by
#                                  P16 in the precedence chain — but P17
#                                  itself MUST NOT claim it; this is the
#                                  load-bearing BASE/EXTENDED partition
#                                  firewall on the K-axis)
#
# These tests target the P17 predicate function directly (NOT the full
# `k971_route_decision`), because some of these neighbours may be
# legitimately routed by an *earlier* predicate (P15 / K971_ROUTE_TABLE /
# P16) in the precedence chain.  The P17 firewall guarantee is: P17 itself
# does NOT contribute a route decision for any shape outside its 12-cell
# admit set.
_K1437_NEAREST_NEIGHBOUR_NEGATIVES = [
    # (M,    N,    K,     a_dtype,           reason)
    (2048,   512,  2048, "torch.bfloat16",
     "N=512 OFF column-narrow tier (P17 fixes N=1024); claimed by K-1417 P15"),
    (2048,  2048,  2048, "torch.bfloat16",
     "N=2048 OFF column-narrow tier; collides with K971 square anchors not P17"),
    (1024,  1024,  2048, "torch.bfloat16",
     "M=1024 OFF anchor row (P17 anchors M ∈ {2048,4096,8192})"),
    (4096,  1024,  4096, "torch.bfloat16",
     "K=4096 BASE (claimed by K-1433 P16), NOT in K-1437 EXTENDED set"),
]


@pytest.mark.parametrize("M,N,K,dtype,reason",
                         _K1437_NEAREST_NEIGHBOUR_NEGATIVES)
def test_k1437_nearest_neighbour_shapes_not_admitted_by_p17_predicate(
    M, N, K, dtype, reason,
):
    """Negative pin: nearest non-member neighbours one axis off the
    K-1437 EXTENDED admit grid must NOT be admitted by the P17 strict-
    equality predicate function `_k1437_p17_skinny_n1024_kcompl_ext_routeout`.
    Targets the P17 firewall directly (independent of earlier-precedence
    predicates that may legitimately claim these shapes for their own
    reasons)."""
    # Sanity: the neighbour shape is genuinely outside the P17 admit set.
    assert (M, N, K, dtype) not in _K1437_P17_SKINNY_N1024_KCOMPL_EXT_ROUTEOUT_12, (
        f"Test scaffolding bug: neighbour ({M},{N},{K},{dtype}) is actually "
        f"in the P17 admit set — the negative test would pass trivially.")
    p17_admit = _k1437_p17_skinny_n1024_kcompl_ext_routeout(M, N, K, dtype)
    assert p17_admit is False, (
        f"P17 strict-equality firewall LEAK: ({M},{N},{K},{dtype}) — {reason} "
        f"— P17 predicate returned True, but the K-1437 admit set is exactly "
        f"the 12-cell EXTENDED grid (M ∈ {{2048,4096,8192}} × N=1024 × "
        f"K ∈ {{2048,32768}} × {{bf16,fp16}}); any True outside this set "
        f"signals an over-broad predicate.")
