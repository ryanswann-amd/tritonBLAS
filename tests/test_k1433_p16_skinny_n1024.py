"""K-1433 P16 skinny_N1024 K-COMPLEMENT BASE pin tests — torch-free.

Minimal pin coverage for the K-1433 productionisation of the 18-cell
`_K1433_P16_SKINNY_N1024_KCOMPL_ROUTEOUT` frozenset stacked as the
10th-position envelope on top of the K-1417 P15 9-predicate stack
(envelope grows 97 → 115 cells; +18 admits at the BASE region of the
N=1024 column-narrow regime).

Cardinality and cross-frozenset disjointness are already asserted at module
load in `_route_predicate.py`; we do not duplicate those here (per
R-1406.MODULE-LOAD-ASSERTS-MAKE-PYTEST-CARDINALITY-DISJOINTNESS-DUPS-DEAD-WEIGHT).
The pytest suite focuses on the orthogonal behavioural pins:

  * exact membership of the K-1433 P16 18 cells
    (M ∈ {2048, 4096, 8192} × N=1024 × K ∈ {4096, 8192, 16384} × {bf16, fp16})
  * full-chain dispatch routes every admit cell to hipBLASLt (covers the
    10th-position predicate's interaction with the 9-predicate precedence
    chain)
  * carve-out short-circuits (streamk / work-stealing / mismatched-dtype)
    correctly suppress the route-OUT for an admit cell
  * P15 (K-1417 N=512) cells STILL route to hipBLASLt under the new chain
    (zero regression on the prior 9th-position envelope)
  * five negative pin tests for near-miss shapes one axis off the K-1433
    admit grid (M-axis off, K-axis off [both EXTREMES sides], N-axis off,
    dtype off) that must NOT be admitted by the P16 strict-equality
    predicate function (firewall against silent over-routing from a future
    range/pattern predicate refactor — R-1417 #2 and Skeptic guidance on
    K-1417 RETRY)
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT,
    _K1433_P16_SKINNY_N1024_KCOMPL_ROUTEOUT,
    _k1433_p16_skinny_n1024_routeout,
    k971_route_decision,
)


# Bucket rule per K-1433 success criteria: M ∈ {2048,4096,8192} ×
# N=1024 × K ∈ {4096,8192,16384} × {bf16, fp16}.
_EXPECTED_K1433_18_CELLS = frozenset({
    (M, 1024, K, dt)
    for M in (2048, 4096, 8192)
    for K in (4096, 8192, 16384)
    for dt in ("torch.bfloat16", "torch.float16")
})


def test_k1433_membership_exactly_18_kcomplement_base_cells():
    """The frozenset must equal the bucket rule exactly — no missing or
    unknown cells."""
    diff_missing = sorted(_EXPECTED_K1433_18_CELLS
                          - _K1433_P16_SKINNY_N1024_KCOMPL_ROUTEOUT)
    diff_unknown = sorted(_K1433_P16_SKINNY_N1024_KCOMPL_ROUTEOUT
                          - _EXPECTED_K1433_18_CELLS)
    assert _K1433_P16_SKINNY_N1024_KCOMPL_ROUTEOUT == _EXPECTED_K1433_18_CELLS, (
        f"K-1433 P16 frozenset diverges from the K-COMPLEMENT BASE bucket "
        f"rule (M ∈ {{2048,4096,8192}} × N=1024 × K ∈ {{4096,8192,16384}} × "
        f"{{bf16,fp16}}).\n"
        f"  missing : {diff_missing}\n"
        f"  unknown : {diff_unknown}")


@pytest.mark.parametrize("M,N,K,dtype", sorted(_EXPECTED_K1433_18_CELLS))
def test_full_dispatch_routes_every_k1433_cell_to_hbl(M, N, K, dtype):
    """Every K-1433 admit cell must route to hipBLASLt under the full
    10-position precedence chain with default carve-out settings."""
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


@pytest.mark.parametrize("enable_streamk,work_stealing,b_dtype", [
    (True,  False, "torch.bfloat16"),  # streamk requested
    (False, True,  "torch.bfloat16"),  # work-stealing requested
    (False, False, "torch.float16"),   # mismatched dtype (a bf16, b fp16)
])
def test_k1433_admit_cell_carveout_short_circuits(
    enable_streamk, work_stealing, b_dtype,
):
    """Admit cells must NOT route to hipBLASLt when streamk / work-stealing
    / mismatched-dtype is requested.  Pick (4096, 1024, 16384, bf16) as a
    representative K-1433 admit cell."""
    a_dtype = "torch.bfloat16"
    M, N, K = 4096, 1024, 16384
    assert (M, N, K, a_dtype) in _K1433_P16_SKINNY_N1024_KCOMPL_ROUTEOUT
    routed = k971_route_decision(
        M, N, K, a_dtype, b_dtype, enable_streamk, work_stealing,
    )
    assert routed is False, (
        f"K-1433 P16 admit cell ({M},{N},{K},{a_dtype}) must NOT route to "
        f"hipBLASLt when carve-out applies (streamk={enable_streamk}, "
        f"work_stealing={work_stealing}, b_dtype={b_dtype}); short-circuit "
        f"in _k971_route_to_hbl is load-bearing.")


@pytest.mark.parametrize("M,N,K,dtype", sorted(_K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT))
def test_k1417_p15_envelope_unchanged_under_p16_stack(M, N, K, dtype):
    """Regression: every K-1417 (P15, 9th-position, N=512 EXTREMES) cell
    must STILL route to hipBLASLt under the new 10-position stack.
    Asserts bit-identical routing behaviour on the prior productionisation
    envelope."""
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


# ---------------------------------------------------------------------------
# Negative pin tests — strict-equality firewall against silent over-routing
# from the P16 predicate.
# ---------------------------------------------------------------------------
# Per Skeptic guidance on K-1417 RETRY (carried forward to K-1433): a
# strict-equality frozenset must be explicitly fenced off against
# adjacent-but-not-admitted shapes that a future broadening (e.g. range /
# pattern predicate refactor) could silently absorb.  These near-miss
# shapes each vary exactly ONE axis off the K-1433 admit grid:
#
#   * (2048, 1024,  2048, bf16)  — N/M admit, K=2048 OFF (would-be EXTREMES,
#                                  reserved for K-1433-FOLLOW-A)
#   * (1024, 1024,  4096, bf16)  — N/K admit, M=1024 OFF (M-axis OFF;
#                                  M=1024 collides with K971_ROUTE_TABLE
#                                  square cells but P16 firewall test
#                                  asserts P16 itself does NOT fire)
#   * (8192, 1024, 32768, fp16)  — N/M admit, K=32768 OFF (would-be EXTREMES,
#                                  reserved for K-1433-FOLLOW-A)
#   * (4096, 1024,  4096, fp32)  — M/N/K admit, dtype=fp32 OFF (P16 admits
#                                  bf16/fp16 only)
#   * (4096,  512,  4096, bf16)  — M/K admit, N=512 OFF (one column-step
#                                  back into the K-1409/K-1417 N=512 regime)
#
# These tests target the P16 predicate function directly (NOT the full
# `k971_route_decision`), because some of these near-miss shapes may be
# legitimately routed by an *earlier* predicate (P5 / K971_ROUTE_TABLE / P12
# / P15) in the precedence chain.  The P16 firewall guarantee is: P16 itself
# does NOT contribute a route decision for any shape outside its 18-cell
# admit set.
_K1433_NEAR_MISS_NEGATIVES = [
    # (M,    N,    K,     a_dtype,           reason)
    (2048,  1024,  2048, "torch.bfloat16", "K=2048 EXTREMES, NOT in K-1433 BASE set"),
    (1024,  1024,  4096, "torch.bfloat16", "M=1024 NOT in K-1433 anchor row"),
    (8192,  1024, 32768, "torch.float16",  "K=32768 EXTREMES, NOT in K-1433 BASE set"),
    (4096,  1024,  4096, "torch.float32",  "dtype=fp32 NOT admitted by P16"),
    (4096,   512,  4096, "torch.bfloat16", "N=512 OFF column-narrow tier (P16 fixes N=1024)"),
]


@pytest.mark.parametrize("M,N,K,dtype,reason", _K1433_NEAR_MISS_NEGATIVES)
def test_k1433_near_miss_shapes_not_admitted_by_p16_predicate(
    M, N, K, dtype, reason,
):
    """Negative pin: near-miss shapes one axis off the K-1433 admit grid
    must NOT be admitted by the P16 strict-equality predicate function
    `_k1433_p16_skinny_n1024_routeout`.  Targets the P16 firewall directly
    (independent of earlier-precedence predicates that may legitimately
    claim these shapes for their own reasons)."""
    # Sanity: the near-miss shape is genuinely outside the P16 admit set.
    assert (M, N, K, dtype) not in _K1433_P16_SKINNY_N1024_KCOMPL_ROUTEOUT, (
        f"Test scaffolding bug: near-miss ({M},{N},{K},{dtype}) is actually "
        f"in the P16 admit set — the negative test would pass trivially.")
    p16_admit = _k1433_p16_skinny_n1024_routeout(M, N, K, dtype)
    assert p16_admit is False, (
        f"P16 strict-equality firewall LEAK: ({M},{N},{K},{dtype}) — {reason} "
        f"— P16 predicate returned True, but the K-1433 admit set is exactly "
        f"the 18-cell BASE grid (M ∈ {{2048,4096,8192}} × N=1024 × "
        f"K ∈ {{4096,8192,16384}} × {{bf16,fp16}}); any True outside this "
        f"set signals an over-broad predicate.")
