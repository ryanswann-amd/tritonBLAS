"""K-1465 P19 skinny_N8192 K-COMPLEMENT pin tests — torch-free.

Minimal pin coverage for the K-1474 productionisation of the 29-cell
`_K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT` frozenset stacked as the
12th-position envelope on top of the K-1437 P17 11-predicate stack
(envelope grows 127 -> 156 cells; +29 admits at the unified
K {2048,4096,8192,16384,32768} band of the N=8192 column-narrow regime).

Cardinality and cross-frozenset disjointness are already asserted at module
load in `_route_predicate.py`; we do not duplicate those here (per
R-1406.MODULE-LOAD-ASSERTS-MAKE-PYTEST-CARDINALITY-DISJOINTNESS-DUPS-DEAD-WEIGHT).
The pytest suite focuses on the orthogonal behavioural pins:

  * exact membership of the K-1465 P19 29 cells
    (M in {2048,4096,8192} x N=8192 x K in {2048,4096,8192,16384,32768} x
    {bf16, fp16}, less the lone reject (2048,8192,32768,fp16) at r=1.087)
  * full-chain dispatch routes every admit cell to hipBLASLt (covers the
    12th-position predicate's interaction with the 11-predicate precedence
    chain — and verifies the K-1437 P17 EXTENDED + K-1433 P16 BASE
    invariants are preserved at N=1024)
  * the lone reject cell (2048,8192,32768,fp16) does NOT route to hipBLASLt
    via P19 (firewall against accidental over-broadening; it must fall
    through to native triton dispatch)
  * carve-out short-circuits (streamk / work-stealing / mismatched-dtype)
    correctly suppress the route-OUT for an admit cell
  * P17 (K-1437 EXTENDED, N=1024) cells STILL route to hipBLASLt under the
    new chain (zero regression on the 11th-position envelope at the prior
    N column — this is the load-bearing stacking invariant)
  * P16 (K-1433 BASE, N=1024) cells STILL route to hipBLASLt under the new
    chain (regression on 10th-position envelope)
  * four nearest-neighbour negative pin tests for shapes one axis off the
    K-1465 P19 admit grid (N-axis off via N=4096 and N=16384; M-axis off
    via M=1024; dtype-axis off via fp32) that must NOT be admitted by the
    P19 strict-equality predicate function (firewall against silent
    over-routing from a future range/pattern predicate refactor)
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K1433_P16_SKINNY_N1024_KCOMPL_ROUTEOUT,
    _K1437_P17_SKINNY_N1024_KCOMPL_EXT_ROUTEOUT_12,
    _K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT,
    _k1465_p19_skinny_n8192_routeout,
    k971_route_decision,
)


# Bucket rule per K-1465 success criteria, less the 1 reject:
#   M in {2048,4096,8192} x N=8192 x K in {2048,4096,8192,16384,32768}
#   x dtype in {bf16, fp16}, MINUS (2048, 8192, 32768, fp16) which fell
#   below the 1.10 strict admit gate at r=1.087.
_K1465_P19_REJECT_CELL = (2048, 8192, 32768, "torch.float16")
_EXPECTED_K1465_29_CELLS = frozenset({
    (M, 8192, K, dt)
    for M in (2048, 4096, 8192)
    for K in (2048, 4096, 8192, 16384, 32768)
    for dt in ("torch.bfloat16", "torch.float16")
}) - {_K1465_P19_REJECT_CELL}
assert len(_EXPECTED_K1465_29_CELLS) == 29


def test_k1465_membership_exactly_29_kcomplement_n8192_cells():
    """The frozenset must equal the bucket rule (minus reject) exactly —
    no missing or unknown cells."""
    diff_missing = sorted(_EXPECTED_K1465_29_CELLS
                          - _K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT)
    diff_unknown = sorted(_K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT
                          - _EXPECTED_K1465_29_CELLS)
    assert (
        _K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT
        == _EXPECTED_K1465_29_CELLS
    ), (
        f"K-1465 P19 frozenset diverges from the K-COMPLEMENT N=8192 bucket "
        f"rule (M in {{2048,4096,8192}} x N=8192 x K in "
        f"{{2048,4096,8192,16384,32768}} x {{bf16,fp16}}, less the 1 reject "
        f"(2048,8192,32768,fp16) at r=1.087).\n"
        f"  missing : {diff_missing}\n"
        f"  unknown : {diff_unknown}")


@pytest.mark.parametrize("M,N,K,dtype", sorted(_EXPECTED_K1465_29_CELLS))
def test_full_dispatch_routes_every_k1465_cell_to_hbl(M, N, K, dtype):
    """Every K-1465 P19 admit cell must route to hipBLASLt under the full
    12-position precedence chain with default carve-out settings."""
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


def test_k1465_reject_cell_not_admitted_by_p19_predicate():
    """The 1 reject cell (2048, 8192, 32768, fp16) at r=1.087 (below the
    strict 1.10 admit floor) MUST NOT be admitted by the P19 predicate
    itself.  Falls through to native triton dispatch."""
    M, N, K, dt = _K1465_P19_REJECT_CELL
    assert (M, N, K, dt) not in _K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT, (
        f"K-1465 P19 admit-set scaffolding bug: the reject cell {(M,N,K,dt)} "
        f"is in the productionised frozenset; it must not be (r=1.087 < 1.10).")
    p19_admit = _k1465_p19_skinny_n8192_routeout(M, N, K, dt)
    assert p19_admit is False, (
        f"K-1465 P19 admit gate firewall LEAK: reject cell {(M,N,K,dt)} at "
        f"measured r=1.087 (below the strict 1.10 admit floor) was admitted "
        f"by the P19 predicate; the K-1442 admit gate convention requires "
        f"sub-1.10 cells to fall through to native triton dispatch.")


@pytest.mark.parametrize("enable_streamk,work_stealing,b_dtype", [
    (True,  False, "torch.bfloat16"),  # streamk requested
    (False, True,  "torch.bfloat16"),  # work-stealing requested
    (False, False, "torch.float16"),   # mismatched dtype (a bf16, b fp16)
])
def test_k1465_admit_cell_carveout_short_circuits(
    enable_streamk, work_stealing, b_dtype,
):
    """Admit cells must NOT route to hipBLASLt when streamk / work-stealing
    / mismatched-dtype is requested.  Pick (8192, 8192, 2048, bf16) — the
    cell where K-1465's measured speedup is largest (r=1.413x) — as the
    representative carve-out probe."""
    a_dtype = "torch.bfloat16"
    M, N, K = 8192, 8192, 2048
    assert (M, N, K, a_dtype) in _K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT
    routed = k971_route_decision(
        M, N, K, a_dtype, b_dtype, enable_streamk, work_stealing,
    )
    assert routed is False, (
        f"K-1465 P19 admit cell ({M},{N},{K},{a_dtype}) must NOT route to "
        f"hipBLASLt when carve-out applies (streamk={enable_streamk}, "
        f"work_stealing={work_stealing}, b_dtype={b_dtype}); short-circuit "
        f"in _k971_route_to_hbl is load-bearing.")


@pytest.mark.parametrize("M,N,K,dtype",
                         sorted(_K1437_P17_SKINNY_N1024_KCOMPL_EXT_ROUTEOUT_12))
def test_k1437_p17_extended_envelope_unchanged_under_p19_stack(M, N, K, dtype):
    """Regression: every K-1437 P17 EXTENDED (11th-position, N=1024) cell
    must STILL route to hipBLASLt under the new 12-position stack.
    Asserts bit-identical routing behaviour on the immediately prior
    productionisation envelope — the load-bearing stacking invariant."""
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


@pytest.mark.parametrize("M,N,K,dtype",
                         sorted(_K1433_P16_SKINNY_N1024_KCOMPL_ROUTEOUT))
def test_k1433_p16_base_envelope_unchanged_under_p19_stack(M, N, K, dtype):
    """Regression: every K-1433 P16 BASE (10th-position, N=1024) cell must
    STILL route to hipBLASLt under the new 12-position stack."""
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


# ---------------------------------------------------------------------------
# Negative pin tests — strict-equality firewall against silent over-routing
# from the P19 predicate.
# ---------------------------------------------------------------------------
# Per Skeptic guidance carried forward from K-1417 / K-1433 / K-1437: a
# strict-equality frozenset must be explicitly fenced off against
# adjacent-but-not-admitted shapes that a future broadening (e.g. range /
# pattern predicate refactor) could silently absorb.  Each negative shape
# varies exactly ONE axis off the K-1465 P19 admit grid.
_K1465_NEAREST_NEIGHBOUR_NEGATIVES = [
    # (M,    N,    K,     a_dtype,           reason)
    (4096,  4096,  4096, "torch.bfloat16",
     "N=4096 OFF column tier (P19 fixes N=8192); reserved for K-1458 P18"),
    (4096, 16384,  4096, "torch.bfloat16",
     "N=16384 OFF column tier (one column-step past N=8192 K-COMPLEMENT)"),
    (1024,  8192,  8192, "torch.bfloat16",
     "M=1024 OFF anchor row (P19 anchors M in {2048,4096,8192})"),
    (4096,  8192,  4096, "torch.float32",
     "dtype OFF (P19 admits {bf16,fp16}); fp32 not in K-COMPLEMENT cohort"),
]


@pytest.mark.parametrize("M,N,K,dtype,reason",
                         _K1465_NEAREST_NEIGHBOUR_NEGATIVES)
def test_k1465_nearest_neighbour_shapes_not_admitted_by_p19_predicate(
    M, N, K, dtype, reason,
):
    """Negative pin: nearest non-member neighbours one axis off the
    K-1465 P19 admit grid must NOT be admitted by the P19 strict-
    equality predicate function `_k1465_p19_skinny_n8192_routeout`.
    Targets the P19 firewall directly (independent of earlier-precedence
    predicates that may legitimately claim these shapes for their own
    reasons)."""
    # Sanity: the neighbour shape is genuinely outside the P19 admit set.
    assert (M, N, K, dtype) not in _K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT, (
        f"Test scaffolding bug: neighbour ({M},{N},{K},{dtype}) is actually "
        f"in the P19 admit set — the negative test would pass trivially.")
    p19_admit = _k1465_p19_skinny_n8192_routeout(M, N, K, dtype)
    assert p19_admit is False, (
        f"P19 strict-equality firewall LEAK: ({M},{N},{K},{dtype}) — {reason} "
        f"— P19 predicate returned True, but the K-1465 admit set is exactly "
        f"the 29-cell N=8192 grid (M in {{2048,4096,8192}} x N=8192 x "
        f"K in {{2048,4096,8192,16384,32768}} x {{bf16,fp16}}, less the 1 "
        f"reject (2048,8192,32768,fp16)); any True outside this set "
        f"signals an over-broad predicate.")
