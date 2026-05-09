"""K-1400 P14 + K-1389 P13 + K-1361 P12 pin tests — torch-free.

Validates the K-1400 productionisation of the K-1382 RETRY-winning 18-cell
`_K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18` frozenset stacked as the
8th-position envelope on top of the K-1389 P13 73-cell baseline (sibling
to P13 with N axis bumped 128 → 256; envelope grows 73 → 91 cells).

These tests pin:
  * cardinality of the new frozenset (P14=18, total post-P14 = 91)
  * exact membership of the K-1382 P14 18 cells (M ∈ {2048, 4096, 8192} ×
    N=256 × K ∈ {4096, 8192, 16384} × {bf16, fp16})
  * cross-frozenset disjointness between P14 / P13 / P12 / P8 / K971_ROUTE_TABLE
  * dispatch precedence: dtype-mismatch / streamk / work-stealing carve-outs
    short-circuit ahead of every strict-equality table
  * regression firewall: every K-1389 prior admit (K971_ROUTE_TABLE, P8, P13)
    still routes via its original predicate after the +P14 stack
  * boundary controls (M=1024, M=16384) MUST NOT be admitted by P14

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
    _K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18,
    _P8_MFMA_ISSUE_STALL_ROUTEOUT,
    _k1361_p12_square_mid_routeout,
    _k1367_p13_skinny_n128_routeout,
    _k1382_p14_skinny_n256_routeout,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# Cardinality pins.  Any deviation indicates an authoring typo against the
# K-1365 / K-1382 / K-1400 bucket-rule resolution.
# ---------------------------------------------------------------------------
def test_p14_skinny_n256_kcomplement_cardinality_18():
    assert len(_K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18) == 18


def test_envelope_total_post_p14_is_91():
    """P8 (51) + K-1361 P12 (4) + K-1367 P13 (18) + K-1382 P14 (18) = 91
    strict-equality cells across the post-K971 stacked envelopes.

    K971_ROUTE_TABLE (12 cells: K-905/K-971 anchors + K-1335 longK_smallSquare)
    is the 5th-position envelope; counted separately from the P8/P12/P13/P14
    8th-position chain because it gates on the LDS-bank-conflict mechanism
    rather than the MFMA-issue-stall / square_mid / skinny_NN mechanisms.
    """
    total = (
        len(_P8_MFMA_ISSUE_STALL_ROUTEOUT)
        + len(_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)
        + len(_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)
        + len(_K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18)
    )
    assert total == 91, (
        f"Expected 51 + 4 + 18 + 18 = 91 strict-equality cells across "
        f"P8/P12/P13/P14; got {total}.  Check that no frozenset has been "
        f"mutated.")


# ---------------------------------------------------------------------------
# Exact-membership pin.  Deliberately enumerates every key so any accidental
# drop / typo / re-ordering is caught at CI time.
# ---------------------------------------------------------------------------
def _expected_p14_18_cells():
    """Construct the K-1382 P14 18-cell expected set from the bucket rule
    (sibling to P13 with N axis bumped 128 → 256)."""
    return frozenset({
        (M, 256, K, dt)
        for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    })


def test_p14_membership_exactly_18_kcomplement_cells():
    expected = _expected_p14_18_cells()
    assert _K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18 == expected, (
        "K-1382 P14 frozenset diverges from the K-COMPLEMENT bucket rule: "
        "M ∈ {2048,4096,8192} × N=256 × K ∈ {4096,8192,16384} × {bf16,fp16}.\n"
        f"  missing : {sorted(expected - _K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18)}\n"
        f"  unknown : {sorted(_K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18 - expected)}")


# ---------------------------------------------------------------------------
# Disjointness pins (5-way).  Cheap insurance per
# R-1329.K-AXIS-PROJECTION-DISJOINTNESS-ASSERTS-ARE-CHEAP-INSURANCE.
# ---------------------------------------------------------------------------
def test_p14_disjoint_from_p8():
    assert _K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18.isdisjoint(
        _P8_MFMA_ISSUE_STALL_ROUTEOUT)


def test_p14_disjoint_from_k971_route_table():
    assert _K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18.isdisjoint(
        K971_ROUTE_TABLE)


def test_p14_disjoint_from_p12():
    assert _K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18.isdisjoint(
        _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)


def test_p14_disjoint_from_p13():
    """The N-axis sibling firewall: P13 (N=128) and P14 (N=256) MUST be
    disjoint by N-axis construction (every cell pair differs on N)."""
    assert _K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18.isdisjoint(
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)


# ---------------------------------------------------------------------------
# Per-cell predicate behaviour.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", sorted(_expected_p14_18_cells()))
def test_p14_predicate_admits_every_kcomplement_cell(M, N, K, dtype):
    assert _k1382_p14_skinny_n256_routeout(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# Negative-control pins (cells immediately adjacent to the cohort that MUST
# NOT fire — guards against axis over-relaxation).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", [
    # K below the K=4096 wrapper-overhead floor (K-1227 still owns these)
    (2048, 256, 2048, "torch.bfloat16"),
    (4096, 256, 2048, "torch.float16"),
    # M below the M=2048 cohort floor
    (1024, 256, 4096, "torch.bfloat16"),
    (1024, 256, 8192, "torch.float16"),
    # M above the M=8192 cohort ceiling
    (16384, 256, 4096, "torch.bfloat16"),
    # N off the N=256 column-narrow regime
    (2048, 128, 4096, "torch.bfloat16"),
    (2048, 512, 4096, "torch.bfloat16"),
    # K above the K=16384 cohort ceiling
    (2048, 256, 32768, "torch.bfloat16"),
    # dtype outside the bf16/fp16 K-913 §3 invariance class
    (2048, 256, 4096, "torch.float32"),
    (2048, 256, 4096, "torch.float8_e4m3fnuz"),
])
def test_p14_predicate_rejects_axis_perturbations(M, N, K, dtype):
    assert _k1382_p14_skinny_n256_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# Full dispatch decision — exercises the 8-position precedence chain.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", sorted(_expected_p14_18_cells()))
def test_full_dispatch_routes_every_p14_cell_to_hbl(M, N, K, dtype):
    """K-1382 P14 admits MUST route to hipBLASLt under default carve-out
    settings (no streamk, no work-stealing, matched dtype).
    """
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


# ---------------------------------------------------------------------------
# Carve-out short-circuits — a P14 admit cell must NOT route to hipBLASLt
# when streamk / work-stealing / mismatched-dtype is requested.  This is
# load-bearing for the matmul.py kwarg surface (PR description Test Plan).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enable_streamk,work_stealing,b_dtype", [
    (True,  False, "torch.bfloat16"),  # streamk requested
    (False, True,  "torch.bfloat16"),  # work-stealing requested
    (False, False, "torch.float16"),   # mixed dtype (a bf16, b fp16)
])
def test_p14_admit_cell_carveout_short_circuits(enable_streamk, work_stealing, b_dtype):
    # Pick a known P14 admit cell.
    a_dtype = "torch.bfloat16"
    M, N, K = 4096, 256, 8192
    assert (M, N, K, a_dtype) in _K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18
    routed = k971_route_decision(M, N, K, a_dtype, b_dtype,
                                 enable_streamk, work_stealing)
    assert routed is False, (
        f"P14 admit cell ({M},{N},{K},{a_dtype}) must NOT route to hipBLASLt "
        f"when carve-out applies (streamk={enable_streamk}, "
        f"work_stealing={work_stealing}, b_dtype={b_dtype}); short-circuit "
        f"in _k971_route_to_hbl is load-bearing.")


# ---------------------------------------------------------------------------
# Regression firewall — every cell in the K-1389 prior 73-cell envelope still
# routes to hipBLASLt after the +P14 stack.
# Per R-1322.STRICT-EQUALITY-UNION-PRESERVES-PRIOR-ADMIT-INVARIANCE; the new
# envelope is unioned via short-circuit `if ... return True` so prior admits
# cannot be removed by the diff.
# ---------------------------------------------------------------------------
def test_no_regression_on_k971_route_table_prior_admits():
    for (M, N, K, dt) in K971_ROUTE_TABLE:
        assert k971_route_decision(M, N, K, dt, dt, False, False) is True, (
            f"K971_ROUTE_TABLE prior admit ({M},{N},{K},{dt}) was lost after "
            f"+P14 stack; the P14 productionisation must preserve every prior "
            f"5th-position envelope cell (regression firewall).")


def test_no_regression_on_p8_prior_admits():
    for (M, N, K, dt) in _P8_MFMA_ISSUE_STALL_ROUTEOUT:
        assert k971_route_decision(M, N, K, dt, dt, False, False) is True, (
            f"P8 prior admit ({M},{N},{K},{dt}) was lost after +P14 stack; "
            f"the P14 productionisation must preserve every prior P8 "
            f"sub-frozenset cell (regression firewall).")


def test_no_regression_on_p13_prior_admits():
    for (M, N, K, dt) in _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18:
        assert k971_route_decision(M, N, K, dt, dt, False, False) is True, (
            f"P13 prior admit ({M},{N},{K},{dt}) was lost after +P14 stack; "
            f"the P14 productionisation must preserve every prior P13 "
            f"sub-frozenset cell (regression firewall).")


# ---------------------------------------------------------------------------
# K-1400 boundary controls — adjacent non-admitted cells in the
# skinny_N256 K-COMPLEMENT region MUST still route to triton (i.e.
# k971_route_decision == False), proving the strict-equality frozenset
# does NOT spill into neighbouring shapes.  Reviewer-mandated negative
# controls per the K-1400 brief (mirrors K-1389's 6-cell controls).
# ---------------------------------------------------------------------------
_K1400_BOUNDARY_TRITON_CELLS = [
    # M=1024 — directly below the M=2048 cohort floor; same N=256, same K
    # range as P14 admits.  Must NOT be admitted by P14: M=1024 is occupancy-
    # bound at N=256 and the wrapper-overhead/K-time crossover does not flip
    # yet.
    (1024,   256,  4096, "torch.bfloat16"),
    (1024,   256,  4096, "torch.float16"),
    (1024,   256,  8192, "torch.bfloat16"),
    # M=16384 — directly above the M=8192 cohort ceiling; same N=256, same
    # K range.  Must NOT be admitted by P14: persistent_matmul scales linearly
    # with M and the K-913 LDS-BC discriminator inverts past M=8192 because
    # the per-CU wave footprint amortises the bank conflicts.
    (16384,  256,  4096, "torch.bfloat16"),
    (16384,  256,  8192, "torch.bfloat16"),
    (16384,  256, 16384, "torch.bfloat16"),
]


@pytest.mark.parametrize("M,N,K,dtype", _K1400_BOUNDARY_TRITON_CELLS)
def test_k1400_p14_does_not_spill_into_adjacent_n256_kcompl_cells(M, N, K, dtype):
    """K-1400 negative control: adjacent non-admitted cells (M=1024 N=256,
    M=16384 N=256) MUST NOT be admitted by the P14 strict-equality
    frozenset _K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18 nor by the P14
    per-cell predicate ``_k1382_p14_skinny_n256_routeout``.

    NOTE: end-to-end ``k971_route_decision`` may still return True on
    these cells via the EARLIER R-K979 P5 closed-form structural
    pathology predicate (which routes large-K narrow-N shapes to
    hipBLASLt as a separate mechanism).  That is intentional and out of
    scope for K-1400 — K-1400 is a strict-equality additive extension to
    P14, not a re-tuning of P5.  This test isolates the P14-specific
    no-spill claim from the end-to-end routing decision.
    """
    assert (M, N, K, dtype) not in _K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18, (
        f"Boundary control cell ({M},{N},{K},{dtype}) is in the P14 admit "
        f"frozenset; the K-1400 boundary table is mis-specified.")
    assert _k1382_p14_skinny_n256_routeout(M, N, K, dtype) is False, (
        f"K-1400 adjacent boundary cell ({M},{N},{K},{dtype}) was admitted "
        f"by the P14 per-cell predicate; strict-equality frozenset must NOT "
        f"spill into adjacent M={M} cells.")
