"""K-1409 P15 + K-1400 P14 + K-1389 P13 + K-1361 P12 pin tests — torch-free.

Validates the K-1409 productionisation of the 18-cell
`_K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT_18` frozenset stacked as the
9th-position envelope on top of the K-1400 P14 91-cell baseline (sibling
to P14 with N axis bumped 256 → 512; envelope grows 91 → 109 cells).

These tests pin:
  * cardinality of the new frozenset (P15=18, total post-P15 = 109)
  * exact membership of the K-1409 P15 18 cells (M ∈ {2048, 4096, 8192} ×
    N=512 × K ∈ {4096, 8192, 16384} × {bf16, fp16})
  * cross-frozenset disjointness between P15 / P14 / P13 / P12 / P8 /
    K971_ROUTE_TABLE
  * dispatch precedence: dtype-mismatch / streamk / work-stealing carve-outs
    short-circuit ahead of every strict-equality table
  * regression firewall: every K-1400 prior admit (K971_ROUTE_TABLE, P8, P13,
    P14) still routes via its original predicate after the +P15 stack
  * boundary controls (M=1024, M=16384, N=128/256/1024) MUST NOT be admitted
    by P15

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
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT_18,
    _P8_MFMA_ISSUE_STALL_ROUTEOUT,
    _k1361_p12_square_mid_routeout,
    _k1367_p13_skinny_n128_routeout,
    _k1382_p14_skinny_n256_routeout,
    _k1409_p15_skinny_n512_routeout,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# Cardinality pins.  Any deviation indicates an authoring typo against the
# K-1365 / K-1382 / K-1400 / K-1409 bucket-rule resolution.
# ---------------------------------------------------------------------------
def test_p15_skinny_n512_kcomplement_cardinality_18():
    assert len(_K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT_18) == 18


def test_envelope_total_post_p15_is_109():
    """P8 (51) + K-1361 P12 (4) + K-1367 P13 (18) + K-1382 P14 (18) +
    K-1409 P15 (18) = 109 strict-equality cells across the post-K971
    stacked envelopes.

    K971_ROUTE_TABLE (12 cells: K-905/K-971 anchors + K-1335 longK_smallSquare)
    is the 5th-position envelope; counted separately from the
    P8/P12/P13/P14/P15 9th-position chain because it gates on the LDS-bank-
    conflict mechanism rather than the MFMA-issue-stall / square_mid /
    skinny_NN mechanisms.
    """
    total = (
        len(_P8_MFMA_ISSUE_STALL_ROUTEOUT)
        + len(_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)
        + len(_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)
        + len(_K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18)
        + len(_K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT_18)
    )
    assert total == 109, (
        f"Expected 51 + 4 + 18 + 18 + 18 = 109 strict-equality cells across "
        f"P8/P12/P13/P14/P15; got {total}.  Check that no frozenset has been "
        f"mutated.")


# ---------------------------------------------------------------------------
# Exact-membership pin.  Deliberately enumerates every key so any accidental
# drop / typo / re-ordering is caught at CI time.
# ---------------------------------------------------------------------------
def _expected_p15_18_cells():
    """Construct the K-1409 P15 18-cell expected set from the bucket rule
    (sibling to P14 with N axis bumped 256 → 512)."""
    return frozenset({
        (M, 512, K, dt)
        for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    })


def test_p15_membership_exactly_18_kcomplement_cells():
    expected = _expected_p15_18_cells()
    assert _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT_18 == expected, (
        "K-1409 P15 frozenset diverges from the K-COMPLEMENT bucket rule: "
        "M ∈ {2048,4096,8192} × N=512 × K ∈ {4096,8192,16384} × {bf16,fp16}.\n"
        f"  missing : {sorted(expected - _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT_18)}\n"
        f"  unknown : {sorted(_K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT_18 - expected)}")


# ---------------------------------------------------------------------------
# Disjointness pins (5-way).  Cheap insurance per
# R-1329.K-AXIS-PROJECTION-DISJOINTNESS-ASSERTS-ARE-CHEAP-INSURANCE.
# ---------------------------------------------------------------------------
def test_p15_disjoint_from_p8():
    assert _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT_18.isdisjoint(
        _P8_MFMA_ISSUE_STALL_ROUTEOUT)


def test_p15_disjoint_from_k971_route_table():
    assert _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT_18.isdisjoint(
        K971_ROUTE_TABLE)


def test_p15_disjoint_from_p12():
    assert _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT_18.isdisjoint(
        _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)


def test_p15_disjoint_from_p13():
    """The N-axis sibling firewall: P13 (N=128) and P15 (N=512) MUST be
    disjoint by N-axis construction (every cell pair differs on N)."""
    assert _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT_18.isdisjoint(
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)


def test_p15_disjoint_from_p14():
    """The N-axis sibling firewall: P14 (N=256) and P15 (N=512) MUST be
    disjoint by N-axis construction (every cell pair differs on N)."""
    assert _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT_18.isdisjoint(
        _K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18)


# ---------------------------------------------------------------------------
# Per-cell predicate behaviour.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", sorted(_expected_p15_18_cells()))
def test_p15_predicate_admits_every_kcomplement_cell(M, N, K, dtype):
    assert _k1409_p15_skinny_n512_routeout(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# Negative-control pins (cells immediately adjacent to the cohort that MUST
# NOT fire — guards against axis over-relaxation).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", [
    # K below the K=4096 wrapper-overhead floor
    (2048, 512, 2048, "torch.bfloat16"),
    (4096, 512, 2048, "torch.float16"),
    # M below the M=2048 cohort floor
    (1024, 512, 4096, "torch.bfloat16"),
    (1024, 512, 8192, "torch.float16"),
    # M above the M=8192 cohort ceiling
    (16384, 512, 4096, "torch.bfloat16"),
    # N off the N=512 column-narrow regime (P13 / P14 / non-skinny)
    (2048, 128, 4096, "torch.bfloat16"),
    (2048, 256, 4096, "torch.bfloat16"),
    (2048, 1024, 4096, "torch.bfloat16"),
    # K above the K=16384 cohort ceiling
    (2048, 512, 32768, "torch.bfloat16"),
    # dtype outside the bf16/fp16 K-913 §3 invariance class
    (2048, 512, 4096, "torch.float32"),
    (2048, 512, 4096, "torch.float8_e4m3fnuz"),
])
def test_p15_predicate_rejects_axis_perturbations(M, N, K, dtype):
    assert _k1409_p15_skinny_n512_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# Full dispatch decision — exercises the 9-position precedence chain.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", sorted(_expected_p15_18_cells()))
def test_full_dispatch_routes_every_p15_cell_to_hbl(M, N, K, dtype):
    """K-1409 P15 admits MUST route to hipBLASLt under default carve-out
    settings (no streamk, no work-stealing, matched dtype).
    """
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


# ---------------------------------------------------------------------------
# Carve-out short-circuits — a P15 admit cell must NOT route to hipBLASLt
# when streamk / work-stealing / mismatched-dtype is requested.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enable_streamk,work_stealing,b_dtype", [
    (True,  False, "torch.bfloat16"),  # streamk requested
    (False, True,  "torch.bfloat16"),  # work-stealing requested
    (False, False, "torch.float16"),   # mixed dtype (a bf16, b fp16)
])
def test_p15_admit_cell_carveout_short_circuits(enable_streamk, work_stealing, b_dtype):
    a_dtype = "torch.bfloat16"
    M, N, K = 4096, 512, 8192
    assert (M, N, K, a_dtype) in _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT_18
    routed = k971_route_decision(M, N, K, a_dtype, b_dtype,
                                 enable_streamk, work_stealing)
    assert routed is False, (
        f"P15 admit cell ({M},{N},{K},{a_dtype}) must NOT route to hipBLASLt "
        f"when carve-out applies (streamk={enable_streamk}, "
        f"work_stealing={work_stealing}, b_dtype={b_dtype}); short-circuit "
        f"in _k971_route_to_hbl is load-bearing.")


# ---------------------------------------------------------------------------
# Regression firewall — every cell in the K-1400 prior 91-cell envelope still
# routes to hipBLASLt after the +P15 stack.
# Per R-1322.STRICT-EQUALITY-UNION-PRESERVES-PRIOR-ADMIT-INVARIANCE.
# ---------------------------------------------------------------------------
def test_no_regression_on_k971_route_table_prior_admits():
    for (M, N, K, dt) in K971_ROUTE_TABLE:
        assert k971_route_decision(M, N, K, dt, dt, False, False) is True, (
            f"K971_ROUTE_TABLE prior admit ({M},{N},{K},{dt}) was lost after "
            f"+P15 stack; the P15 productionisation must preserve every prior "
            f"5th-position envelope cell (regression firewall).")


def test_no_regression_on_p8_prior_admits():
    for (M, N, K, dt) in _P8_MFMA_ISSUE_STALL_ROUTEOUT:
        assert k971_route_decision(M, N, K, dt, dt, False, False) is True, (
            f"P8 prior admit ({M},{N},{K},{dt}) was lost after +P15 stack; "
            f"the P15 productionisation must preserve every prior P8 "
            f"sub-frozenset cell (regression firewall).")


def test_no_regression_on_p13_prior_admits():
    for (M, N, K, dt) in _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18:
        assert k971_route_decision(M, N, K, dt, dt, False, False) is True, (
            f"P13 prior admit ({M},{N},{K},{dt}) was lost after +P15 stack; "
            f"the P15 productionisation must preserve every prior P13 "
            f"sub-frozenset cell (regression firewall).")


def test_no_regression_on_p14_prior_admits():
    for (M, N, K, dt) in _K1382_P14_SKINNY_N256_KCOMPL_ROUTEOUT_18:
        assert k971_route_decision(M, N, K, dt, dt, False, False) is True, (
            f"P14 prior admit ({M},{N},{K},{dt}) was lost after +P15 stack; "
            f"the P15 productionisation must preserve every prior P14 "
            f"sub-frozenset cell (regression firewall).")


# ---------------------------------------------------------------------------
# K-1409 boundary controls — adjacent non-admitted cells in the
# skinny_N512 K-COMPLEMENT region MUST still NOT be admitted by the P15
# strict-equality predicate.  Mirrors K-1400's 6-cell controls with N=512.
# ---------------------------------------------------------------------------
_K1409_BOUNDARY_TRITON_CELLS = [
    # M=1024 — directly below the M=2048 cohort floor; same N=512, same K
    # range as P15 admits.  Must NOT be admitted by P15: M=1024 is occupancy-
    # bound at N=512 and the wrapper-overhead/K-time crossover does not flip
    # yet.
    (1024,   512,  4096, "torch.bfloat16"),
    (1024,   512,  4096, "torch.float16"),
    (1024,   512,  8192, "torch.bfloat16"),
    # M=16384 — directly above the M=8192 cohort ceiling; same N=512, same
    # K range.  Must NOT be admitted by P15.
    (16384,  512,  4096, "torch.bfloat16"),
    (16384,  512,  8192, "torch.bfloat16"),
    (16384,  512, 16384, "torch.bfloat16"),
]


@pytest.mark.parametrize("M,N,K,dtype", _K1409_BOUNDARY_TRITON_CELLS)
def test_k1409_p15_does_not_spill_into_adjacent_n512_kcompl_cells(M, N, K, dtype):
    """K-1409 negative control: adjacent non-admitted cells (M=1024 N=512,
    M=16384 N=512) MUST NOT be admitted by the P15 strict-equality
    frozenset _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT_18 nor by the P15
    per-cell predicate ``_k1409_p15_skinny_n512_routeout``.

    NOTE: end-to-end ``k971_route_decision`` may still return True on
    these cells via the EARLIER R-K979 P5 closed-form structural
    pathology predicate (which routes large-K narrow-N shapes to
    hipBLASLt as a separate mechanism).  That is intentional and out of
    scope for K-1409 — K-1409 is a strict-equality additive extension to
    P15, not a re-tuning of P5.  This test isolates the P15-specific
    no-spill claim from the end-to-end routing decision.
    """
    assert (M, N, K, dtype) not in _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT_18, (
        f"Boundary control cell ({M},{N},{K},{dtype}) is in the P15 admit "
        f"frozenset; the K-1409 boundary table is mis-specified.")
    assert _k1409_p15_skinny_n512_routeout(M, N, K, dtype) is False, (
        f"K-1409 adjacent boundary cell ({M},{N},{K},{dtype}) was admitted "
        f"by the P15 per-cell predicate; strict-equality frozenset must NOT "
        f"spill into adjacent M={M} cells.")
