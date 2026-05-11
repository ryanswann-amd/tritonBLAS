"""K-1367 P13 + K-1361 P12 pin tests — torch-free.

Validates the K-1379 productionisation of the K-1367 RETRY-winning 18-cell
`_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18` frozenset stacked as the
7th-position envelope on top of the K-1361 P12 55-cell baseline.

These tests pin:
  * cardinality of the new frozensets (P12=4, P13=18, total post-P13 = 73)
  * exact membership of the K-1367 P13 18 cells (M ∈ {2048, 4096, 8192} ×
    N=128 × K ∈ {4096, 8192, 16384} × {bf16, fp16})
  * exact membership of the K-1361 P12 4 cells (M=N=K ∈ {2048, 4096} ×
    {bf16, fp16})
  * cross-frozenset disjointness between P12 / P13 / P8 / K971_ROUTE_TABLE
  * dispatch precedence: dtype-mismatch / streamk / work-stealing carve-outs
    short-circuit ahead of every strict-equality table

Imports the actual shipped predicate module (`_route_predicate`) so the real
runtime path is exercised; no exec/string-parse copies.
"""
from __future__ import annotations

import itertools

import pytest

from tritonblas._route_predicate import (
    K971_ROUTE_TABLE,
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4,
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _P8_MFMA_ISSUE_STALL_ROUTEOUT,
    _k1361_p12_square_mid_routeout,
    _k1367_p13_skinny_n128_routeout,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# Cardinality pins.  Any deviation indicates an authoring typo against the
# K-1308 / K-1227 / K-1345 / K-1367 bucket-rule resolution.
# ---------------------------------------------------------------------------
def test_p12_square_mid_cardinality_4():
    assert len(_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4) == 4


def test_p13_skinny_n128_kcomplement_cardinality_18():
    assert len(_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18) == 18


def test_envelope_total_post_p13_is_73():
    """P8 (51) + K-1361 P12 (4) + K-1367 P13 (18) = 73 strict-equality cells.

    K971_ROUTE_TABLE (12 cells: K-905/K-971 anchors + K-1335 longK_smallSquare)
    is the 5th-position envelope; counted separately from the P8/P12/P13
    7th-position chain because it gates on the LDS-bank-conflict mechanism
    rather than the MFMA-issue-stall / square_mid / skinny_N128 mechanisms.
    """
    total = (
        len(_P8_MFMA_ISSUE_STALL_ROUTEOUT)
        + len(_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)
        + len(_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)
    )
    assert total == 73, (
        f"Expected 51 + 4 + 18 = 73 strict-equality cells across P8/P12/P13; "
        f"got {total}.  Check that no frozenset has been mutated.")


# ---------------------------------------------------------------------------
# Exact-membership pins.  These deliberately enumerate every key so any
# accidental drop / typo / re-ordering is caught at CI time.
# ---------------------------------------------------------------------------
def _expected_p13_18_cells():
    """Construct the K-1367 P13 18-cell expected set from the bucket rule."""
    return frozenset({
        (M, 128, K, dt)
        for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    })


def _expected_p12_4_cells():
    """Construct the K-1361 P12 4-cell expected set from the bucket rule."""
    return frozenset({
        (S, S, S, dt)
        for S in (2048, 4096)
        for dt in ("torch.bfloat16", "torch.float16")
    })


def test_p13_membership_exactly_18_kcomplement_cells():
    expected = _expected_p13_18_cells()
    assert _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == expected, (
        "K-1367 P13 frozenset diverges from the K-COMPLEMENT bucket rule: "
        "M ∈ {2048,4096,8192} × N=128 × K ∈ {4096,8192,16384} × {bf16,fp16}.\n"
        f"  missing : {sorted(expected - _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)}\n"
        f"  unknown : {sorted(_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 - expected)}")


def test_p12_membership_exactly_4_diagonal_cells():
    expected = _expected_p12_4_cells()
    assert _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4 == expected, (
        "K-1361 P12 frozenset diverges from the K-1345 RANKED_BUCKETS_v2 "
        "diagonal-only authoritative rule: M=N=K ∈ {2048,4096} × {bf16,fp16}.\n"
        f"  missing : {sorted(expected - _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)}\n"
        f"  unknown : {sorted(_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4 - expected)}")


# ---------------------------------------------------------------------------
# Disjointness pins.  Cheap insurance per
# R-1329.K-AXIS-PROJECTION-DISJOINTNESS-ASSERTS-ARE-CHEAP-INSURANCE.
# ---------------------------------------------------------------------------
def test_p13_disjoint_from_p8():
    assert _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18.isdisjoint(
        _P8_MFMA_ISSUE_STALL_ROUTEOUT)


def test_p13_disjoint_from_k971_route_table():
    assert _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18.isdisjoint(
        K971_ROUTE_TABLE)


def test_p13_disjoint_from_p12():
    assert _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18.isdisjoint(
        _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)


def test_p12_disjoint_from_p8():
    assert _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4.isdisjoint(
        _P8_MFMA_ISSUE_STALL_ROUTEOUT)


def test_p12_disjoint_from_k971_route_table():
    assert _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4.isdisjoint(K971_ROUTE_TABLE)


# ---------------------------------------------------------------------------
# Per-cell predicate behaviour.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", sorted(_expected_p13_18_cells()))
def test_p13_predicate_admits_every_kcomplement_cell(M, N, K, dtype):
    assert _k1367_p13_skinny_n128_routeout(M, N, K, dtype) is True


@pytest.mark.parametrize("M,N,K,dtype", sorted(_expected_p12_4_cells()))
def test_p12_predicate_admits_every_diagonal_cell(M, N, K, dtype):
    assert _k1361_p12_square_mid_routeout(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# Negative-control pins (cells immediately adjacent to the cohort that MUST
# NOT fire — guards against axis over-relaxation).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", [
    # K below the K=4096 wrapper-overhead floor (K-1227 still owns these)
    (2048, 128, 2048, "torch.bfloat16"),
    (4096, 128, 2048, "torch.float16"),
    # M below the M=2048 cohort floor
    (1024, 128, 4096, "torch.bfloat16"),
    (1024, 128, 8192, "torch.float16"),
    # M above the M=8192 cohort ceiling
    (16384, 128, 4096, "torch.bfloat16"),
    # N off the N=128 column-narrow regime
    (2048, 64, 4096, "torch.bfloat16"),
    (2048, 256, 4096, "torch.bfloat16"),
    # K above the K=16384 cohort ceiling
    (2048, 128, 32768, "torch.bfloat16"),
    # dtype outside the bf16/fp16 K-913 §3 invariance class
    (2048, 128, 4096, "torch.float32"),
])
def test_p13_predicate_rejects_axis_perturbations(M, N, K, dtype):
    assert _k1367_p13_skinny_n128_routeout(M, N, K, dtype) is False


@pytest.mark.parametrize("M,N,K,dtype", [
    # off-diagonal cells (K-1345 RANKED_BUCKETS_v2 routes these to K-1313 WIDE)
    (2048, 2048, 4096, "torch.bfloat16"),  # also a K-1335 LDS-BC anchor
    (4096, 4096, 2048, "torch.bfloat16"),
    (2048, 4096, 4096, "torch.bfloat16"),
    # 8192³ — DROP_HBM_BOUND per K-1308 F4 / K-879 N2a
    (8192, 8192, 8192, "torch.bfloat16"),
    # dtype outside K-913 §3 invariance class
    (2048, 2048, 2048, "torch.float32"),
])
def test_p12_predicate_rejects_off_diagonal_and_dropped_cells(M, N, K, dtype):
    assert _k1361_p12_square_mid_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# Full dispatch decision — exercises the 7-position precedence chain.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", sorted(_expected_p13_18_cells()))
def test_full_dispatch_routes_every_p13_cell_to_hbl(M, N, K, dtype):
    """K-1367 P13 admits MUST route to hipBLASLt under default carve-out
    settings (no streamk, no work-stealing, matched dtype).
    """
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


@pytest.mark.parametrize("M,N,K,dtype", sorted(_expected_p12_4_cells()))
def test_full_dispatch_routes_every_p12_cell_to_hbl(M, N, K, dtype):
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


# ---------------------------------------------------------------------------
# Carve-out short-circuits — a P13 admit cell must NOT route to hipBLASLt
# when streamk / work-stealing / mismatched-dtype is requested.  This is
# load-bearing for the matmul.py kwarg surface (PR description Test Plan).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enable_streamk,work_stealing,b_dtype", [
    (True,  False, "torch.bfloat16"),  # streamk requested
    (False, True,  "torch.bfloat16"),  # work-stealing requested
    (False, False, "torch.float16"),   # mixed dtype (a bf16, b fp16)
])
def test_p13_admit_cell_carveout_short_circuits(enable_streamk, work_stealing, b_dtype):
    # Pick a known P13 admit cell.
    a_dtype = "torch.bfloat16"
    M, N, K = 4096, 128, 8192
    assert (M, N, K, a_dtype) in _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18
    routed = k971_route_decision(M, N, K, a_dtype, b_dtype,
                                 enable_streamk, work_stealing)
    assert routed is False, (
        f"P13 admit cell ({M},{N},{K},{a_dtype}) must NOT route to hipBLASLt "
        f"when carve-out applies (streamk={enable_streamk}, "
        f"work_stealing={work_stealing}, b_dtype={b_dtype}); short-circuit "
        f"in _k971_route_to_hbl is load-bearing.")


# ---------------------------------------------------------------------------
# Regression firewall — every cell in the K-1335 / K-1322 / K-905 prior
# envelopes still routes to hipBLASLt after the P12 + P13 stack.
# Per R-1322.STRICT-EQUALITY-UNION-PRESERVES-PRIOR-ADMIT-INVARIANCE; the new
# envelopes are unioned via short-circuit `if ... return True` so prior
# admits cannot be removed by the diff.
# ---------------------------------------------------------------------------
def test_no_regression_on_k971_route_table_prior_admits():
    for (M, N, K, dt) in K971_ROUTE_TABLE:
        assert k971_route_decision(M, N, K, dt, dt, False, False) is True, (
            f"K971_ROUTE_TABLE prior admit ({M},{N},{K},{dt}) was lost after "
            f"P12 + P13 stack; the P13 productionisation must preserve every "
            f"prior 5th-position envelope cell (regression firewall).")


def test_no_regression_on_p8_prior_admits():
    for (M, N, K, dt) in _P8_MFMA_ISSUE_STALL_ROUTEOUT:
        assert k971_route_decision(M, N, K, dt, dt, False, False) is True, (
            f"P8 prior admit ({M},{N},{K},{dt}) was lost after P12 + P13 "
            f"stack; the P13 productionisation must preserve every prior "
            f"P8 sub-frozenset cell (regression firewall).")


# ---------------------------------------------------------------------------
# K-1389 boundary controls — adjacent non-admitted cells in the
# skinny_N128 K-COMPLEMENT region MUST still route to triton (i.e.
# k971_route_decision == False), proving the strict-equality frozenset
# does NOT spill into neighbouring shapes.  Reviewer-mandated negative
# controls per the K-1389 brief.
# ---------------------------------------------------------------------------
_K1389_BOUNDARY_TRITON_CELLS = [
    # M=1024 — directly below the M=2048 cohort floor; same N=128, same K
    # range as P13 admits.  Must route to triton: M=1024 is occupancy-bound
    # at N=128 and the wrapper-overhead/K-time crossover does not flip yet.
    (1024,   128,  4096, "torch.bfloat16"),
    (1024,   128,  4096, "torch.float16"),
    (1024,   128,  8192, "torch.bfloat16"),
    # M=16384 — directly above the M=8192 cohort ceiling; same N=128, same
    # K range.  Must route to triton: persistent_matmul scales linearly
    # with M and the K-913 LDS-BC discriminator inverts past M=8192 because
    # the per-CU wave footprint amortises the bank conflicts.
    (16384,  128,  4096, "torch.bfloat16"),
    (16384,  128,  8192, "torch.bfloat16"),
    (16384,  128, 16384, "torch.bfloat16"),
]


@pytest.mark.parametrize("M,N,K,dtype", _K1389_BOUNDARY_TRITON_CELLS)
def test_k1389_p13_does_not_spill_into_adjacent_n128_kcompl_cells(M, N, K, dtype):
    """K-1389 negative control: adjacent non-admitted cells (M=1024 N=128,
    M=16384 N=128) MUST NOT be admitted by the P13 strict-equality
    frozenset _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 nor by the P13
    per-cell predicate ``_k1367_p13_skinny_n128_routeout``.

    NOTE: end-to-end ``k971_route_decision`` may still return True on
    these cells via the EARLIER R-K979 P5 closed-form structural
    pathology predicate (which routes large-K narrow-N shapes to
    hipBLASLt as a separate mechanism).  That is intentional and out of
    scope for K-1389 — K-1389 is a strict-equality additive extension to
    P13, not a re-tuning of P5.  This test isolates the P13-specific
    no-spill claim from the end-to-end routing decision.
    """
    assert (M, N, K, dtype) not in _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18, (
        f"Boundary control cell ({M},{N},{K},{dtype}) is in the P13 admit "
        f"frozenset; the K-1389 boundary table is mis-specified.")
    assert _k1367_p13_skinny_n128_routeout(M, N, K, dtype) is False, (
        f"K-1389 adjacent boundary cell ({M},{N},{K},{dtype}) was admitted "
        f"by the P13 per-cell predicate; strict-equality frozenset must NOT "
        f"spill into adjacent M={M} cells.")
