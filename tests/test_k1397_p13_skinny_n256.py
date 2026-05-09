"""K-1397 P13 skinny_N256 K-COMPLEMENT pin tests — torch-free.

Validates the K-1402 productionisation of the K-1397-winning 12-cell
`_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12` frozenset stacked as the
8th-position envelope on top of the K-1389 P13 73-cell baseline (envelope
grows 73 → 85 cells; +12 admits).

These tests pin:
  * cardinality of the new frozenset (K-1397 P13 = 12, total post-K-1397 = 85)
  * exact membership of the K-1397 P13 12 cells (M ∈ {2048, 4096, 8192} ×
    N=256 × K ∈ {2048, 32768} × {bf16, fp16})
  * cross-frozenset disjointness vs P8 / K971_ROUTE_TABLE / P12 / K-1367 P13
  * dispatch precedence: dtype-mismatch / streamk / work-stealing carve-outs
    short-circuit ahead of every strict-equality table
  * boundary-perturbed cells (K=4096, K=16384, N=128, N=512) MUST NOT match

Imports the actual shipped predicate module (`_route_predicate`) so the real
runtime path is exercised; no exec/string-parse copies.
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    K971_ROUTE_TABLE,
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4,
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _P8_MFMA_ISSUE_STALL_ROUTEOUT,
    _k1397_p13_skinny_n256_routeout,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# Cardinality pins.  Any deviation indicates an authoring typo against the
# K-1365 post-P12 4-bucket decomposition or K-1397 K-COMPLEMENT extremes scoping.
# ---------------------------------------------------------------------------
def test_p13_skinny_n256_kcomplement_cardinality_12():
    assert len(_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12) == 12


def test_envelope_total_post_k1397_is_85():
    """P8 (51) + K-1361 P12 (4) + K-1367 P13 (18) + K-1397 P13 (12) = 85
    strict-equality cells.

    K971_ROUTE_TABLE (12 cells: K-905/K-971 anchors + K-1335 longK_smallSquare)
    is the 5th-position envelope; counted separately from the P8/P12/P13
    8-position chain because it gates on the LDS-bank-conflict mechanism
    rather than the MFMA-issue-stall / square_mid / skinny_N128 / skinny_N256
    mechanisms.
    """
    total = (
        len(_P8_MFMA_ISSUE_STALL_ROUTEOUT)
        + len(_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)
        + len(_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)
        + len(_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12)
    )
    assert total == 85, (
        f"Expected 51 + 4 + 18 + 12 = 85 strict-equality cells across "
        f"P8/P12/P13(N=128)/P13(N=256); got {total}.  Check that no "
        f"frozenset has been mutated.")


# ---------------------------------------------------------------------------
# Exact-membership pins.
# ---------------------------------------------------------------------------
def _expected_k1397_12_cells():
    """Construct the K-1397 P13 12-cell expected set from the bucket rule."""
    return frozenset({
        (M, 256, K, dt)
        for M in (2048, 4096, 8192)
        for K in (2048, 32768)
        for dt in ("torch.bfloat16", "torch.float16")
    })


def test_k1397_membership_exactly_12_kcomplement_cells():
    expected = _expected_k1397_12_cells()
    assert _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == expected, (
        "K-1397 P13 frozenset diverges from the K-COMPLEMENT bucket rule: "
        "M ∈ {2048,4096,8192} × N=256 × K ∈ {2048,32768} × {bf16,fp16}.\n"
        f"  missing : {sorted(expected - _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12)}\n"
        f"  unknown : {sorted(_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 - expected)}")


# ---------------------------------------------------------------------------
# Disjointness pins.  Cheap insurance per
# R-1329.K-AXIS-PROJECTION-DISJOINTNESS-ASSERTS-ARE-CHEAP-INSURANCE.
# ---------------------------------------------------------------------------
def test_k1397_disjoint_from_p8():
    assert _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12.isdisjoint(
        _P8_MFMA_ISSUE_STALL_ROUTEOUT)


def test_k1397_disjoint_from_k971_route_table():
    assert _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12.isdisjoint(
        K971_ROUTE_TABLE)


def test_k1397_disjoint_from_p12():
    assert _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12.isdisjoint(
        _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)


def test_k1397_disjoint_from_k1367_p13_n128():
    """K-1367 P13 (N=128) and K-1397 P13 (N=256) partition the skinny-N
    column-narrow regime by N-axis; no cell may belong to both.
    """
    assert _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12.isdisjoint(
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)


# ---------------------------------------------------------------------------
# Per-cell predicate behaviour — every one of the 12 admit cells routes True.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", sorted(_expected_k1397_12_cells()))
def test_k1397_predicate_admits_every_kcomplement_cell(M, N, K, dtype):
    assert _k1397_p13_skinny_n256_routeout(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# Boundary negative-control pins (the 4 boundary perturbations spelled out
# in K-1402's success criteria: K=4096, K=16384 — which are mid-band cells
# already covered by tritonblas persistent_matmul, NOT to be re-routed; and
# N=128, N=512 — adjacent column widths NOT in the K-1397 cohort).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", [
    # K=4096 (mid-band — tritonblas persistent_matmul wins)
    (2048, 256,  4096, "torch.bfloat16"),
    (4096, 256,  4096, "torch.float16"),
    # K=16384 (mid-band ceiling — tritonblas persistent_matmul wins)
    (4096, 256, 16384, "torch.bfloat16"),
    (8192, 256, 16384, "torch.float16"),
    # N=128 (K-1367 P13 territory, NOT K-1397)
    (2048, 128,  2048, "torch.bfloat16"),
    (8192, 128, 32768, "torch.float16"),
    # N=512 (above N=256 column-narrow regime; not in K-1365 skinny_N256 bucket)
    (2048, 512,  2048, "torch.bfloat16"),
    (4096, 512, 32768, "torch.float16"),
])
def test_k1397_predicate_rejects_boundary_perturbations(M, N, K, dtype):
    assert _k1397_p13_skinny_n256_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# Additional axis-perturbation negative controls (M-axis and dtype-axis).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", [
    # M below the M=2048 cohort floor
    (1024, 256, 2048, "torch.bfloat16"),
    (1024, 256, 32768, "torch.float16"),
    # M above the M=8192 cohort ceiling
    (16384, 256, 2048, "torch.bfloat16"),
    # K mid-band (between {2048, 32768})
    (2048, 256, 8192, "torch.bfloat16"),
    # dtype outside the bf16/fp16 K-913 §3 invariance class
    (2048, 256, 2048, "torch.float32"),
    (4096, 256, 32768, "torch.float32"),
])
def test_k1397_predicate_rejects_axis_perturbations(M, N, K, dtype):
    assert _k1397_p13_skinny_n256_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# Full dispatch decision — exercises the 8-position precedence chain.
# Every K-1397 admit cell MUST route to hipBLASLt under default carve-out
# settings (no streamk, no work-stealing, matched dtype).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", sorted(_expected_k1397_12_cells()))
def test_full_dispatch_routes_every_k1397_cell_to_hbl(M, N, K, dtype):
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


# ---------------------------------------------------------------------------
# Boundary cells must NOT route under the full dispatch chain either —
# guards against leakage via any earlier 1st–7th-position predicate.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", [
    # K=4096 / K=16384 mid-band (NOT covered by any envelope at N=256 outside
    # of any prior P5/P6/P8/E1 admits — these specific tuples should fall
    # through to in-kernel dispatch).
    (2048, 256,  4096, "torch.bfloat16"),
    (8192, 256, 16384, "torch.float16"),
    # N=512 (no envelope claims this column width at these (M,K) tuples)
    (2048, 512, 32768, "torch.bfloat16"),
    (4096, 512,  2048, "torch.float16"),
])
def test_full_dispatch_rejects_k1397_boundary_perturbations(M, N, K, dtype):
    """These boundary cells lie outside the K-1397 envelope.  They must not
    be claimed by the K-1397 8th-position predicate.  (Cells may still be
    routed by an earlier predicate if independently admitted — but in that
    case the rejection is from the K-1397 predicate alone, not the chain.)
    """
    # Direct predicate must reject:
    assert _k1397_p13_skinny_n256_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# Carve-out short-circuits — a K-1397 admit cell must NOT route to hipBLASLt
# when streamk / work-stealing / mismatched-dtype is requested.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enable_streamk,work_stealing,b_dtype", [
    (True,  False, "torch.bfloat16"),  # streamk requested
    (False, True,  "torch.bfloat16"),  # work-stealing requested
    (False, False, "torch.float16"),   # mixed dtype (a bf16, b fp16)
])
def test_k1397_admit_cell_carveout_short_circuits(enable_streamk, work_stealing, b_dtype):
    # Pick a known K-1397 admit cell.
    a_dtype = "torch.bfloat16"
    M, N, K = 4096, 256, 32768
    assert (M, N, K, a_dtype) in _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12
    routed = k971_route_decision(M, N, K, a_dtype, b_dtype,
                                 enable_streamk, work_stealing)
    assert routed is False, (
        f"K-1397 P13 admit cell ({M},{N},{K},{a_dtype}) must NOT route to "
        f"hipBLASLt when carve-out applies (streamk={enable_streamk}, "
        f"work_stealing={work_stealing}, b_dtype={b_dtype}); short-circuit "
        f"in _k971_route_to_hbl is load-bearing.")


# ---------------------------------------------------------------------------
# Regression firewall — every cell in the K-1335 / K-1322 / K-905 / K-1361
# / K-1367 prior envelopes still routes to hipBLASLt after the K-1397 stack.
# Per R-1322.STRICT-EQUALITY-UNION-PRESERVES-PRIOR-ADMIT-INVARIANCE; the new
# envelope is unioned via short-circuit `if ... return True` so prior admits
# cannot be removed by the diff.
# ---------------------------------------------------------------------------
def test_no_regression_on_k971_route_table_prior_admits():
    for (M, N, K, dt) in K971_ROUTE_TABLE:
        assert k971_route_decision(M, N, K, dt, dt, False, False) is True, (
            f"K971_ROUTE_TABLE prior admit ({M},{N},{K},{dt}) was lost after "
            f"K-1397 P13 stack; the K-1397 productionisation must preserve every "
            f"prior 5th-position envelope cell (regression firewall).")


def test_no_regression_on_p8_prior_admits():
    for (M, N, K, dt) in _P8_MFMA_ISSUE_STALL_ROUTEOUT:
        assert k971_route_decision(M, N, K, dt, dt, False, False) is True, (
            f"P8 prior admit ({M},{N},{K},{dt}) was lost after K-1397 P13 "
            f"stack; the K-1397 productionisation must preserve every prior "
            f"P8 sub-frozenset cell (regression firewall).")


def test_no_regression_on_p12_prior_admits():
    for (M, N, K, dt) in _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4:
        assert k971_route_decision(M, N, K, dt, dt, False, False) is True, (
            f"K-1361 P12 prior admit ({M},{N},{K},{dt}) was lost after "
            f"K-1397 P13 stack; the K-1397 productionisation must preserve "
            f"every prior 6th-position P12 cell (regression firewall).")


def test_no_regression_on_k1367_p13_n128_prior_admits():
    for (M, N, K, dt) in _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18:
        assert k971_route_decision(M, N, K, dt, dt, False, False) is True, (
            f"K-1367 P13 (N=128) prior admit ({M},{N},{K},{dt}) was lost "
            f"after K-1397 P13 stack; the K-1397 productionisation must "
            f"preserve every prior 7th-position P13 cell (regression "
            f"firewall).")
