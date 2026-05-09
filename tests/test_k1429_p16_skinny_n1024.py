"""K-1429 P16 skinny_N1024 K-COMPLEMENT 30-cell pin tests — torch-free.

Minimal pin coverage for the K-1429 productionisation of the 30-cell
`_K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_30` frozenset stacked as the
10th-position envelope on top of the K-1417/K-1425 P15 9-predicate stack
(envelope grows 97 → 127 cells; +30 admits across the FULL K range
(BASE ∪ EXTREMES) of the N=1024 column-narrow regime).  Subsumes the
K-1433 BASE-only 18-cell precursor by adding the 12-cell EXTREMES band
(K ∈ {2048, 32768}) at the same 10th-position slot.

Cardinality and cross-frozenset disjointness vs the 9-predicate precedence
stack are pin-tested HERE (not at module load); per R-1406 refinement,
import-time defensive checks for properties already covered by pytest are
dead weight on every production load.  Behavioural pins:

  * exact membership of the K-1429 P16 30 cells
    (M ∈ {2048,4096,8192} × N=1024 × K ∈ {2048,4096,8192,16384,32768} ×
    {bf16, fp16})
  * full-chain dispatch routes every admit cell to hipBLASLt (covers the
    10th-position predicate's interaction with the 9-predicate precedence
    chain)
  * carve-out short-circuits (streamk / work-stealing / mismatched-dtype)
    correctly suppress the route-OUT for an admit cell
  * P15 (K-1417/K-1425 N=512) cells STILL route to hipBLASLt under the new
    chain (zero regression on the prior 9th-position envelope)
  * negative pin tests for adjacent non-cohort shapes one axis off the
    K-1429 admit grid (M-axis off, N-axis off both directions, K-axis off,
    dtype off) that must NOT be admitted by the P16 strict-equality
    predicate function (firewall against silent over-routing from a future
    range/pattern predicate refactor — R-1417 #2 and Skeptic guidance
    carried forward to K-1429)
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    K971_ROUTE_TABLE,
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4,
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT,
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_30,
    _P8_MFMA_ISSUE_STALL_ROUTEOUT,
    _k1429_p16_skinny_n1024_routeout,
    k971_route_decision,
)


# Bucket rule per K-1429 success criteria:
# M ∈ {2048,4096,8192} × N=1024 × K ∈ {2048,4096,8192,16384,32768} ×
# {bf16, fp16} = 30 cells.
_EXPECTED_K1429_30_CELLS = frozenset({
    (M, 1024, K, dt)
    for M in (2048, 4096, 8192)
    for K in (2048, 4096, 8192, 16384, 32768)
    for dt in ("torch.bfloat16", "torch.float16")
})


def test_k1429_membership_exactly_30_kcomplement_cells():
    """The frozenset must equal the bucket rule exactly — no missing or
    unknown cells (BASE ∪ EXTREMES at N=1024)."""
    diff_missing = sorted(_EXPECTED_K1429_30_CELLS
                          - _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_30)
    diff_unknown = sorted(_K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_30
                          - _EXPECTED_K1429_30_CELLS)
    assert _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_30 == _EXPECTED_K1429_30_CELLS, (
        f"K-1429 P16 30-cell frozenset diverges from the K-COMPLEMENT bucket "
        f"rule (M ∈ {{2048,4096,8192}} × N=1024 × "
        f"K ∈ {{2048,4096,8192,16384,32768}} × {{bf16,fp16}}).\n"
        f"  missing : {diff_missing}\n"
        f"  unknown : {diff_unknown}")


def test_k1429_cardinality_is_30():
    """Defense-in-depth: explicit cardinality check that the BASE+EXTREMES
    envelope is exactly 30 cells (= 3 M × 1 N × 5 K × 2 dtype)."""
    assert len(_K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_30) == 30


# ---------------------------------------------------------------------------
# Cross-frozenset disjointness invariants (formerly module-load asserts;
# moved here per R-1406 refinement).  Each invariant pins that the K-1429
# 30-cell P16 envelope is disjoint from one of the 9 prior precedence
# frozensets — required to prove "no double-admit" against the K-1175
# stacked-predicate precedence chain.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("other_name,other_set", [
    ("_P8_MFMA_ISSUE_STALL_ROUTEOUT",          _P8_MFMA_ISSUE_STALL_ROUTEOUT),
    ("K971_ROUTE_TABLE",                       K971_ROUTE_TABLE),
    ("_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4",   _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4),
    ("_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18", _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18),
    ("_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12", _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12),
    ("_K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT",    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT),
])
def test_k1429_p16_disjoint_from_prior_stack(other_name, other_set):
    """K-1429 P16 30-cell envelope must be disjoint from every prior
    frozenset in the 9-predicate precedence chain (A4 no-double-admit).
    By construction: P8/K-1367/K-1397/K-1409 use N ∈ {128, 256, 512},
    K971_ROUTE_TABLE uses M=N square shapes, P12 uses M=N=K square shapes —
    K-1429 uses N=1024 with non-square M ∈ {2048, 4096, 8192}, so the
    intersection is empty."""
    overlap = sorted(_K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_30 & other_set)
    assert not overlap, (
        f"K-1429 P16 envelope overlaps {other_name}: {overlap}; the four "
        f"K-COMPLEMENT predicates partition the skinny-N column-narrow "
        f"regime by N-axis at {{128, 256, 512, 1024}} — any overlap is a "
        f"double-admit bug.")


@pytest.mark.parametrize("M,N,K,dtype", sorted(_EXPECTED_K1429_30_CELLS))
def test_full_dispatch_routes_every_k1429_cell_to_hbl(M, N, K, dtype):
    """Every K-1429 admit cell must route to hipBLASLt under the full
    10-position precedence chain with default carve-out settings.  Covers
    all 30 cells (BASE band K ∈ {4096,8192,16384} AND EXTREMES band
    K ∈ {2048, 32768})."""
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


@pytest.mark.parametrize("M,N,K,dtype", sorted(_EXPECTED_K1429_30_CELLS))
def test_k1429_predicate_admits_every_cell_directly(M, N, K, dtype):
    """The K-1429 P16 predicate function itself must admit every one of the
    30 cells (independent of upstream predicate ordering)."""
    assert _k1429_p16_skinny_n1024_routeout(M, N, K, dtype) is True


@pytest.mark.parametrize("enable_streamk,work_stealing,b_dtype", [
    (True,  False, "torch.bfloat16"),  # streamk requested
    (False, True,  "torch.bfloat16"),  # work-stealing requested
    (False, False, "torch.float16"),   # mismatched dtype (a bf16, b fp16)
])
def test_k1429_admit_cell_carveout_short_circuits(
    enable_streamk, work_stealing, b_dtype,
):
    """Admit cells must NOT route to hipBLASLt when streamk / work-stealing
    / mismatched-dtype is requested.  Pick (4096, 1024, 16384, bf16) as a
    representative K-1429 admit cell."""
    a_dtype = "torch.bfloat16"
    M, N, K = 4096, 1024, 16384
    assert (M, N, K, a_dtype) in _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_30
    routed = k971_route_decision(
        M, N, K, a_dtype, b_dtype, enable_streamk, work_stealing,
    )
    assert routed is False, (
        f"K-1429 P16 admit cell ({M},{N},{K},{a_dtype}) must NOT route to "
        f"hipBLASLt when carve-out applies (streamk={enable_streamk}, "
        f"work_stealing={work_stealing}, b_dtype={b_dtype}); short-circuit "
        f"in _k971_route_to_hbl is load-bearing.")


@pytest.mark.parametrize("M,N,K,dtype", sorted(_K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT))
def test_k1417_p15_envelope_unchanged_under_p16_stack(M, N, K, dtype):
    """Regression: every K-1417/K-1425 (P15, 9th-position, N=512 EXTREMES)
    cell must STILL route to hipBLASLt under the new 10-position stack.
    Asserts bit-identical routing behaviour on the prior productionisation
    envelope (no regression on the K-1295-style cohort sibling)."""
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


# ---------------------------------------------------------------------------
# Negative pin tests — strict-equality firewall against silent over-routing
# from the P16 predicate.  Targets adjacent non-cohort shapes one axis off
# the K-1429 admit grid.
# ---------------------------------------------------------------------------
# Per K-1441 success criteria: assert that adjacent non-cohort cells
# (e.g. N=2048, K=1024) do NOT route-OUT via the P16 predicate.  These tests
# also assert the M-axis firewall (M=1024 NOT in K-1429 anchor row) and the
# dtype firewall (only bf16/fp16 admitted), and the N-axis sibling firewalls
# at N=512 (K-1417/K-1425 territory) and N=2048 (no productionised
# K-COMPLEMENT above N=1024 yet).
#
# These tests target the P16 predicate function directly (NOT the full
# `k971_route_decision`), because some near-miss shapes may be legitimately
# routed by an *earlier* predicate (P5 / K971_ROUTE_TABLE / P12 / P15) in
# the precedence chain.  The P16 firewall guarantee is: P16 itself does NOT
# contribute a route decision for any shape outside its 30-cell admit set.
_K1429_NEAR_MISS_NEGATIVES = [
    # (M,    N,     K,     a_dtype,           reason)
    # N-axis OFF — adjacent non-cohort cells per K-1441 success criteria
    (4096,  2048,  4096, "torch.bfloat16", "N=2048 NOT in K-1429 (P16 fixes N=1024)"),
    (8192,  2048, 32768, "torch.float16",  "N=2048 NOT in K-1429 (one column-step beyond)"),
    (4096,   512,  4096, "torch.bfloat16", "N=512 OFF — sibling K-1417/K-1425 P15 territory"),
    # K-axis OFF — non-admitted K values
    (4096,  1024,  1024, "torch.bfloat16", "K=1024 NOT in K-1429 K-set (would-be K-NULL)"),
    (4096,  1024,  1024, "torch.float16",  "K=1024 NOT in K-1429 K-set (would-be K-NULL)"),
    (2048,  1024, 65536, "torch.bfloat16", "K=65536 above K-1429 EXTREMES top (32768)"),
    # M-axis OFF — non-anchor M rows
    (1024,  1024,  4096, "torch.bfloat16", "M=1024 NOT in K-1429 anchor row"),
    (1024,  1024, 32768, "torch.float16",  "M=1024 NOT in K-1429 anchor row (EXTREMES K)"),
    (16384, 1024,  4096, "torch.bfloat16", "M=16384 above K-1429 anchor row top (8192)"),
    # dtype OFF — only bf16 / fp16 admitted
    (4096,  1024,  4096, "torch.float32",  "dtype=fp32 NOT admitted by P16 (bf16/fp16 only)"),
    (4096,  1024,  2048, "torch.float8_e4m3fnuz", "dtype=fp8 NOT admitted by P16"),
]


@pytest.mark.parametrize("M,N,K,dtype,reason", _K1429_NEAR_MISS_NEGATIVES)
def test_k1429_adjacent_non_cohort_not_admitted_by_p16_predicate(
    M, N, K, dtype, reason,
):
    """Negative pin: adjacent non-cohort shapes one axis off the K-1429
    admit grid must NOT be admitted by the P16 strict-equality predicate
    function `_k1429_p16_skinny_n1024_routeout`.  Targets the P16 firewall
    directly (independent of earlier-precedence predicates that may
    legitimately claim these shapes for their own reasons).

    Per K-1441 success criteria, asserts non-routing on the canonical
    adjacent points (e.g. N=2048, K=1024) plus the M-axis / dtype firewalls.
    """
    # Sanity: the near-miss shape is genuinely outside the P16 admit set.
    assert (M, N, K, dtype) not in _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_30, (
        f"Test scaffolding bug: near-miss ({M},{N},{K},{dtype}) is actually "
        f"in the P16 admit set — the negative test would pass trivially.")
    p16_admit = _k1429_p16_skinny_n1024_routeout(M, N, K, dtype)
    assert p16_admit is False, (
        f"P16 strict-equality firewall LEAK: ({M},{N},{K},{dtype}) — {reason} "
        f"— P16 predicate returned True, but the K-1429 admit set is exactly "
        f"the 30-cell BASE∪EXTREMES grid (M ∈ {{2048,4096,8192}} × N=1024 × "
        f"K ∈ {{2048,4096,8192,16384,32768}} × {{bf16,fp16}}); any True "
        f"outside this set signals an over-broad predicate.")
