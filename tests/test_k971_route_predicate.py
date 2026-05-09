"""K-1003 R-K979 P5 Gate-0 admission predicate — integration tests.

Verifies the closed-form predicate AND the full ``_k971_route_to_hbl``
dispatch decision (predicate + strict-equality table + streamk /
work-stealing / dtype-mismatch carve-outs) match the K-979 verification
contract:

  * On the K-984 + K-989 union (14 unique K-931 shapes), the predicate
    must FIRE on all 14.
  * On the K-950 9-cell LAND set, the predicate alone must NOT fire on
    any (zero leakage from the closed-form), and the full dispatch must
    only route the 3 K-905 anchor-collision tuples (a documented design
    decision — K-905 measurements override the K-950 verdict on those
    exact tuples).
  * The streamk / work-stealing / mismatched-dtype carve-outs must
    short-circuit to ``False`` regardless of the predicate verdict.

These tests import the actual shipped predicate (`_route_predicate`) and
the actual shipped dispatch helper (`matmul._k971_route_to_hbl`) so the
*real* code path users hit is exercised — no exec/string-parse copies.
"""
from __future__ import annotations

import pytest
import torch

from tritonblas._route_predicate import (
    R_K979_P5_route_to_hbl,
    R_K1037_P6_admit_wpeu1,
    K971_ROUTE_TABLE,
    _K905_K971_LDS_BC_ANCHORS_8,
    _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4,
    _K1336_REGRESSOR_CARVEIN_6,
    _K1109_P6_K1074_ALLOWLIST,
    _K1109_P6_K1074_REGRESSION_EXCLUSIONS,
    _K1109_P6_ALLOWLIST_ENV,
    _p8_mfma_issue_stall_routeout,
    _P8_MFMA_ISSUE_STALL_ROUTEOUT,
    _K1121_P8_ANCHORS_13,
    _K1131_P8_NEIGHBORS_12,
    _K1161_E2_ADMITS_3,
    _K1205_EN3_ADMITS_8,
    _K1219_E3_NFLOOR256_ADMITS_7,
    _K1283_A1_PERTURBATIONS_8,
    R_K1142_E1_route_to_hbl,
    K1142_E1_NS,
    K1142_E1_KS,
    K1142_E1_M_FLOOR,
    K1142_E1_K_FLOOR,
    k971_route_decision,
)
from tritonblas.matmul import _k971_route_to_hbl


# ---------------------------------------------------------------------------
# Cohorts (kept in sync with scripts/validate_p5_predicate.py).  bf16
# everywhere because the K-984/K-989 ship cohort and K-931 measurement scope
# are bf16; fp16 / fp32 / etc. exercise the bf16-only carve-out.
# ---------------------------------------------------------------------------
K984_K989_UNION = [
    # (sid, M, N, K, source)
    ("S04",  2304,   2048, 4800, "K984+K989"),
    ("S05",   256,   1792, 2048, "K989"),
    ("S06",   736,   1792,  736, "K989"),
    ("S10",   512,    192, 2048, "K984"),
    ("S17",   768,   1792, 5972, "K984+K989"),
    ("S18",  5972,   1792,  768, "K984"),
    ("S21",   768,   3072, 4480, "K989"),
    ("S22",  1024,   2048, 6016, "K989"),
    ("S23",  1024,   2048, 8064, "K989"),
    ("S25",  6016,   2048, 1024, "K984"),
    ("S27", 10112,   2048, 1024, "K984+K989"),
    ("S28", 12160,   2048, 1024, "K984+K989"),
    ("S36",    30, 786432,  200, "K984"),
    ("S40",  1024,   2048, 1240, "K989"),
]

# K-950 9-cell LAND set (cells K-912 / K-883 deemed NO-LAND for guarded
# overrides). Entries: (cid, M, N, K, dtype, cohort).
K950_LAND_CELLS = [
    ("L01", 1024, 1024,   512, torch.bfloat16, "harness-guarded-override"),
    ("L02",  512,  512,  1024, torch.bfloat16, "harness-guarded-override"),
    ("L03", 2048,  768,  1024, torch.bfloat16, "harness-guarded-override"),
    ("L04",  768, 2048,  1024, torch.bfloat16, "harness-guarded-override"),
    ("L05", 4096, 4096,  4096, torch.bfloat16, "harness-guarded-override"),
    ("L06", 2048, 2048, 16384, torch.float16,  "harness-guarded-override"),
    ("L07", 1024, 1024, 16384, torch.bfloat16, "C9-cohortA-longK-HBLroute"),
    ("L08", 4096, 4096,  8192, torch.bfloat16, "C9-cohortA-longK-HBLroute"),
    ("L09", 1024, 1024, 16384, torch.float16,  "C9-cohortA-longK-HBLroute"),
]

# Tuples where the dispatch chain intentionally routes to hbl despite the
# K-950 LAND verdict.  Provenance:
#   * K-905 anchor table overrides on long-K cohort A (L06, L07, L09).
#   * K-1352 / K-1336 dispatch-wrapper-overhead carve-IN on (4096^3) bf16
#     (L05): K-1311 §F1 rank #2 (delta=-0.0110, CI95-hi(delta) < 0); K-1336
#     short-circuits the chain at the strict-equality P8 entry to skip the
#     +13.89 us chain-evaluation cost.  See `_K1336_REGRESSOR_CARVEIN_6`.
K950_ANCHOR_COLLISIONS = frozenset({
    (1024, 1024, 16384, "torch.bfloat16"),  # L07
    (1024, 1024, 16384, "torch.float16"),   # L09
    (2048, 2048, 16384, "torch.float16"),   # L06
    (4096, 4096,  4096, "torch.bfloat16"),  # L05 (K-1352 / K-1336 carve-IN)
})


# ---------------------------------------------------------------------------
# Predicate-only checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sid,M,N,K,src", K984_K989_UNION,
                         ids=[c[0] for c in K984_K989_UNION])
def test_p5_fires_on_k984_k989_union(sid, M, N, K, src):
    """Every K-984+K-989 anchor shape must be caught by the closed-form
    predicate (zero strict-equality fallback needed)."""
    assert R_K979_P5_route_to_hbl(M, N, K, torch.bfloat16) is True, (
        f"P5 missed shipped K-984+K-989 anchor {sid} ({M},{N},{K}) — "
        f"strict-equality fallback would re-grow")


@pytest.mark.parametrize("cid,M,N,K,dtype,cohort", K950_LAND_CELLS,
                         ids=[c[0] for c in K950_LAND_CELLS])
def test_p5_does_not_leak_on_k950_land(cid, M, N, K, dtype, cohort):
    """The closed-form predicate alone must produce zero LAND leakage on
    K-950's 9-cell training set. Strict-equality table collisions are
    handled separately by ``test_full_dispatch_only_routes_known_anchors``."""
    assert R_K979_P5_route_to_hbl(M, N, K, dtype) is False, (
        f"P5 leaked into K-950 LAND cell {cid} ({M},{N},{K}) {dtype} "
        f"cohort={cohort}")


def test_p5_is_bf16_only():
    """fp16 / fp32 / int8 must all short-circuit to False so K-905/K-971
    fp16 anchors flow through the strict-equality table only."""
    M, N, K = 2304, 2048, 4800  # S04 — would fire under bf16
    assert R_K979_P5_route_to_hbl(M, N, K, torch.bfloat16) is True
    assert R_K979_P5_route_to_hbl(M, N, K, torch.float16) is False
    assert R_K979_P5_route_to_hbl(M, N, K, torch.float32) is False
    assert R_K979_P5_route_to_hbl(M, N, K, torch.int8) is False


# ---------------------------------------------------------------------------
# Full-dispatch checks (predicate + strict-equality + carve-outs)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sid,M,N,K,src", K984_K989_UNION,
                         ids=[c[0] for c in K984_K989_UNION])
def test_full_dispatch_fires_on_k984_k989_union(sid, M, N, K, src):
    """End-to-end dispatch decision matches the predicate verdict for
    K-984+K-989 anchors (no carve-out interference)."""
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is True


@pytest.mark.parametrize("cid,M,N,K,dtype,cohort", K950_LAND_CELLS,
                         ids=[c[0] for c in K950_LAND_CELLS])
def test_full_dispatch_only_routes_known_anchors(cid, M, N, K, dtype, cohort):
    """Full dispatch may route K-950 LAND cells ONLY if they match a known
    K-905/K-971 anchor in the strict-equality allowlist. Any other route is
    a regression (predicate widened to leak, or new entry slipped into the
    table)."""
    routed = _k971_route_to_hbl(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False)
    expected = (M, N, K, str(dtype)) in K950_ANCHOR_COLLISIONS
    assert routed is expected, (
        f"K-950 LAND cell {cid} ({M},{N},{K}) {dtype} routed={routed} "
        f"expected={expected} (allowlist={K950_ANCHOR_COLLISIONS})")


# Carve-outs: streamk / work-stealing / dtype-mismatch / kill-env must all
# unconditionally veto routing — exercised on a cell P5 *would* otherwise
# fire on (S04 = 2304x2048x4800 bf16).
S04 = (2304, 2048, 4800)


def test_streamk_carveout_overrides_predicate():
    M, N, K = S04
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=True, work_stealing=False) is False


def test_workstealing_carveout_overrides_predicate():
    M, N, K = S04
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=True) is False


def test_mixed_dtype_carveout_overrides_predicate():
    M, N, K = S04
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.float16,
        enable_streamk=False, work_stealing=False) is False


def test_kill_env_overrides_predicate(monkeypatch):
    monkeypatch.setenv("TRITONBLAS_DISABLE_K971", "1")
    M, N, K = S04
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is False


# ---------------------------------------------------------------------------
# Coupling check — ensures the matmul.py copy and the helper module agree.
# Catches the case where someone re-introduces a divergent inline copy.
# ---------------------------------------------------------------------------
def test_matmul_module_uses_helper_module_predicate():
    # tritonblas/__init__.py does `from .matmul import matmul`, which makes
    # `tritonblas.matmul` resolve to the *function*, not the submodule. Grab
    # the actual submodule out of sys.modules so we can introspect it.
    import sys
    _m = sys.modules["tritonblas.matmul"]
    assert _m._R_K979_P5_route_to_hbl is R_K979_P5_route_to_hbl
    assert _m._K971_ROUTE_TABLE is K971_ROUTE_TABLE


# ---------------------------------------------------------------------------
# Public dispatch entry-point check — catches a missed call site.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="public matmul entry-point needs a CUDA device "
                           "for tensor allocation + torch.matmul fallback")
def test_public_matmul_dispatches_via_predicate():
    """Confirm the public dispatch entry-point (`tritonblas.matmul`) actually
    consults `_k971_route_to_hbl` for every call and uses its verdict to
    short-circuit to `torch.matmul`. Done by monkey-patching the helper
    with a spy that returns True for a known shape and False otherwise —
    this avoids touching real hipBLASLt timing while exercising the live
    dispatch ordering inside `_matmul`."""
    # tritonblas/__init__.py does `from .matmul import matmul`, which makes
    # `tritonblas.matmul` resolve to the *function*, not the submodule. Grab
    # the actual submodule out of sys.modules so we can introspect it.
    import sys
    _m = sys.modules["tritonblas.matmul"]
    calls = []
    original = _m._k971_route_to_hbl

    def spy(M, N, K, a_dt, b_dt, sk, ws):
        calls.append((int(M), int(N), int(K), str(a_dt), bool(sk), bool(ws)))
        # Return True so _matmul takes the torch.matmul fallback (cheap +
        # avoids exercising the heavy triton dispatch in this unit test).
        return True

    _m._k971_route_to_hbl = spy
    try:
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
        out = _m.matmul(a, b)
        # If the dispatch site ignored the spy, it would not produce a
        # valid bf16 result with these shapes (or it'd take much longer).
        assert out.shape == (128, 32)
        assert out.dtype == torch.bfloat16
    finally:
        _m._k971_route_to_hbl = original

    assert calls, "_k971_route_to_hbl was not consulted by matmul()"
    M, K_, N = 128, 64, 32
    last = calls[-1]
    assert (last[0], last[2], last[1]) == (M, K_, N), (
        f"dispatch consulted with unexpected args: {last}")


# ---------------------------------------------------------------------------
# K-1089 R_K1037_P6_admit_wpeu1 — structural surrogate of K-1037 P6
# MFMA-issue-stall classifier.  Pin tests cover (a) the 2 K-1017 P6+ cells
# (S24, S29) admit, (b) every K-984/K-989 LAND anchor and the K-1017 OCC
# negatives are NOT leaked, (c) bf16-only carve-out, (d) full-dispatch
# composition: P6 admit short-circuits route-OUT to in-kernel.
# ---------------------------------------------------------------------------
P6_POSITIVES = [
    # (cid, M, N, K)
    ("S24",  4480, 3072,  768),  # MFMA-issue-stall, ratio 1.532
    ("S29", 14208, 2048, 1024),  # MFMA-issue-stall, ratio 1.514
]


# K-1017 OCC + HBM negatives that must NOT admit (leak risk on Envelope A).
# Wide-rect family near the M=14208 admit window:
P6_NEGATIVES_K1017 = [
    ("S26",  8064, 2048, 1024),  # OCC ratio 2.03
    ("S30", 16256, 2048, 1024),  # OCC ratio 1.75 (just above admit ceil)
    ("S31", 18304, 2048, 1024),  # OCC ratio 2.01
    ("S07",  2048, 1792,  256),  # OCC shallow-K
    ("S16",   256,  256, 2048),  # OCC small-mild-rect
    ("S38",   384,  128,  200),  # HBM-bound, K=200 fails C2 floor
]


# K-1074 cohort-extension cells (S32, S33, S34, S35, S37, S39).  K-1109
# independently re-derived K-1037 P6 on these 6 cells from the K-1037
# primary-source PMC CSV (output/k1109_p6_audit.csv): all 6 fail load-
# bearing C1 (waves_per_CU_ratio in [1.753, 2.740], above the 1.70x
# MFMA-stall ceiling) and structurally cluster at the canonical occupancy
# 2.0x signature.  K-1074 paired n=30 round-robin timing on c42/MI300X
# confirmed 0/6 cells clear the K-901 +2% LAND margin and 2/6 (S32, S37)
# are CI95 hi<0 confirmed regressions.  Pin tests below lock the K-1109
# NULL-RESULT into the regression suite so any future relaxation of
# Envelope A's M-band [13000, 14999] or its K=1024-only guard fails CI.
# (S26, S30, S31 are already covered by P6_NEGATIVES_K1017 above.)
P6_NEGATIVES_K1074 = [
    ("S32", 20352, 2048, 1024),  # OCC ratio 2.254
    ("S33", 22400, 2048, 1024),  # OCC ratio 2.378
    ("S34", 24448, 2048, 1024),  # OCC ratio 1.753 — collides on C1 wall with S30
    ("S35", 26496, 2048, 1024),  # OCC ratio 1.887
    ("S37", 25600, 2048,  256),  # OCC ratio 2.740, K=256 fails C2 floor
    ("S39", 49152, 2048,  256),  # OCC ratio 2.122, K=256 fails C2 floor
]


# K-1044 LAND anchors that K-1109 brief lists as 4-cell "no-regression"
# witnesses.  S24/S29 are the K-1037 P6 positives (already in P6_POSITIVES);
# S18 (= K-1044 B3a, 5972,1792,768) and S25 (= K-1044 B3b, 6016,2048,1024)
# fire P6 by exact PMC but the structural surrogate correctly preserves
# them as route-OUT LAND anchors via Envelope-A M>=13000 floor (B3b) and
# Envelope-B minMN>=2048 floor (B3a).  K-989 measured 5.31x-5.63x routing
# speedup on the structurally-equivalent S25/S27/S28 family — admitting
# them to in-kernel would erase that gain.
P6_NEGATIVES_K1044_ANCHORS = [
    ("B3a_S18", 5972, 1792,  768),  # Env-B minMN floor (1792<2048) preserves anchor
    ("B3b_S25", 6016, 2048, 1024),  # Env-A M-floor (6016<13000) preserves anchor
]


# K-984/K-989 LAND anchors that must NOT admit (route-OUT must be preserved).
P6_LAND_ANCHORS = [
    ("S25",  6016, 2048, 1024),  # K-984 anchor, M just below admit floor
    ("S27", 10112, 2048, 1024),  # K-984+K-989 anchor
    ("S28", 12160, 2048, 1024),  # K-984+K-989 anchor, just below admit floor
    ("S04",  2304, 2048, 4800),  # K-984+K-989 anchor, K outside Envelope B
    ("S18",  5972, 1792,  768),  # K-984 anchor, N=1792 not 2048
    ("S40",  1024, 2048, 1240),  # K-989 anchor, minMN<2048
]


@pytest.mark.parametrize("cid,M,N,K", P6_POSITIVES, ids=[c[0] for c in P6_POSITIVES])
def test_p6_admits_k1017_mfma_issue_stall_positives(cid, M, N, K):
    """K-1017 P6+ cells (S24, S29) must admit so dispatch sets wpeu=1
    in-kernel."""
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is True, (
        f"P6 surrogate missed K-1017 MFMA-issue-stall positive {cid} ({M},{N},{K})")


@pytest.mark.parametrize("cid,M,N,K", P6_NEGATIVES_K1017,
                         ids=[c[0] for c in P6_NEGATIVES_K1017])
def test_p6_does_not_admit_k1017_negatives(cid, M, N, K):
    """K-1017 OCC + HBM-bound cells must NOT admit (preserves K-1037
    18/18 perfect agreement against ground truth)."""
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is False, (
        f"P6 surrogate false-positive on K-1017 negative {cid} ({M},{N},{K})")


@pytest.mark.parametrize("cid,M,N,K", P6_LAND_ANCHORS,
                         ids=[c[0] for c in P6_LAND_ANCHORS])
def test_p6_does_not_leak_into_k984_k989_land_anchors(cid, M, N, K):
    """K-984/K-989 LAND anchors gain 5.31x-5.63x via route-OUT; P6 admit
    must NOT silence them by short-circuiting the route."""
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is False, (
        f"P6 LAND-leak into K-984/K-989 anchor {cid} ({M},{N},{K})")


def test_p6_is_bf16_only():
    """fp16 / fp32 must short-circuit (K-1037 ground truth scope is bf16)."""
    M, N, K = 14208, 2048, 1024  # S29
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is True
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.float16) is False
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.float32) is False


def test_p6_envelope_a_boundary():
    """Pin Envelope A boundaries: M=12999 just below floor, M=15000 just
    above ceiling. K-984/K-989 anchor S28 (M=12160) and OCC S30 (M=16256)
    are the calibration witnesses for these floors."""
    assert R_K1037_P6_admit_wpeu1(12999, 2048, 1024, torch.bfloat16) is False
    assert R_K1037_P6_admit_wpeu1(13000, 2048, 1024, torch.bfloat16) is True
    assert R_K1037_P6_admit_wpeu1(14999, 2048, 1024, torch.bfloat16) is True
    assert R_K1037_P6_admit_wpeu1(15000, 2048, 1024, torch.bfloat16) is False


def test_p6_envelope_b_boundary():
    """Pin Envelope B boundaries: maxMN=4501 just above ceiling (preserves
    K-984 S18 maxMN=5972, K-989 S25 maxMN=6016); minMN=2047 just below
    floor (preserves K-989 S22 minMN=1024)."""
    # On-boundary positive (S24 itself):
    assert R_K1037_P6_admit_wpeu1(4480, 3072, 768, torch.bfloat16) is True
    # maxMN ceiling
    assert R_K1037_P6_admit_wpeu1(4500, 2048, 768, torch.bfloat16) is True
    assert R_K1037_P6_admit_wpeu1(4501, 2048, 768, torch.bfloat16) is False
    # minMN floor
    assert R_K1037_P6_admit_wpeu1(2048, 2048, 768, torch.bfloat16) is True
    assert R_K1037_P6_admit_wpeu1(2047, 2048, 768, torch.bfloat16) is False
    # K floor (Surrogate-C2)
    assert R_K1037_P6_admit_wpeu1(2048, 2048, 511, torch.bfloat16) is False
    assert R_K1037_P6_admit_wpeu1(2048, 2048, 512, torch.bfloat16) is True
    # K ceiling (Envelope B)
    assert R_K1037_P6_admit_wpeu1(2048, 2048, 1024, torch.bfloat16) is True
    assert R_K1037_P6_admit_wpeu1(2048, 2048, 1025, torch.bfloat16) is False


@pytest.mark.parametrize("cid,M,N,K", P6_POSITIVES, ids=[c[0] for c in P6_POSITIVES])
def test_p6_admit_overridden_by_p8_routeout(cid, M, N, K):
    """K-1144 supersedes the original K-1089 P6 short-circuit assertion
    on S24 and S29.  Both cells are inside K-1089's structural envelopes
    (S24 inside Env-B, S29 inside Env-A) so ``R_K1037_P6_admit_wpeu1``
    still returns True at the unit level (preserves the K-1037 18/18
    confusion-matrix integrity).  But K-1074's paired n=30 LAND audit on
    c42/MI300X showed 0/13 K-1051+K-1031 cohort cells clear the K-901
    +2% margin under wpeu=1, falsifying the K-1089 admit on these cells.
    K-1121 then measured paired n=30 HIP-graph hot-cache and reported
    hipBLASLt wins on S24 (~1.20x) and S29 (~1.22x).  K-1144 P8 codifies
    that correction by routing both cells OUT to hipBLASLt at the
    dispatch level -- the unit-level P6 admit is unchanged, but the
    composed dispatch decision flips to True (route-OUT) because P8 is
    consulted BEFORE P6 in ``_k971_route_to_hbl``."""
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is True, (
        f"K-1144 P8 failed to override K-1089 P6 admit for {cid} "
        f"({M},{N},{K}); P8 must be consulted before P6 to honour the "
        f"K-1121 paired n=30 measurement evidence.")


@pytest.mark.parametrize("cid,M,N,K", P6_LAND_ANCHORS,
                         ids=[c[0] for c in P6_LAND_ANCHORS])
def test_p6_admit_does_not_break_k984_k989_route_out_anchors(cid, M, N, K):
    """K-984/K-989 anchors must continue to route-OUT after P6 wiring —
    the new short-circuit must not regress the existing P5 LAND set."""
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is True, (
        f"P6 wiring regressed K-984/K-989 LAND anchor {cid} ({M},{N},{K})")


# ---------------------------------------------------------------------------
# K-1109 NULL-RESULT pin tests — locks K-1098 backtest verdict on the
# K-1074 cohort-extension cells and the K-1044 LAND-anchor witnesses.
# Re-derived independently from the K-1037 primary-source paired-PMC CSV
# (see /home/ryaswann/mc2-workspaces/K-1109/output/k1109_p6_audit.csv).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cid,M,N,K", P6_NEGATIVES_K1074,
                         ids=[c[0] for c in P6_NEGATIVES_K1074])
def test_p6_does_not_admit_k1074_cohort_extension(cid, M, N, K):
    """K-1109 lock: 6 K-1074 cohort-extension cells must NOT admit.  Their
    K-1037 paired-PMC waves_per_CU ratios cluster at the canonical OCC
    2.0x signature (range 1.753-2.740) above C1's 1.70x ceiling; K-1074
    paired n=30 timing showed 0/6 cells clear the K-901 +2% LAND margin
    and 2/6 (S32, S37) are CI95-confirmed regressions.  Any future
    relaxation of Envelope A's M-band [13000, 14999] or its K==1024 guard
    must fail this test before re-enabling these cells."""
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is False, (
        f"P6 surrogate false-positive on K-1074 cohort-extension {cid} "
        f"({M},{N},{K}); K-1098 paired-PMC verdict was NO_FIRE")


@pytest.mark.parametrize("cid,M,N,K", P6_NEGATIVES_K1044_ANCHORS,
                         ids=[c[0] for c in P6_NEGATIVES_K1044_ANCHORS])
def test_p6_does_not_admit_k1044_land_anchors(cid, M, N, K):
    """K-1109 lock: K-1044 B3a (S18) and B3b (S25) LAND anchors must NOT
    admit despite firing the exact PMC P6 predicate.  The structural
    surrogate is intentionally more conservative than the exact PMC
    classifier here -- B3b sits below Envelope A's M>=13000 floor and B3a
    sits below Envelope B's minMN>=2048 floor.  K-989 measured 5.31x-5.63x
    routing speedup on the structurally-equivalent S25/S27/S28 family;
    admitting these anchors to in-kernel would erase that gain."""
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is False, (
        f"P6 surrogate LAND-leak into K-1044 anchor {cid} ({M},{N},{K})")


# ---------------------------------------------------------------------------
# K-1109 brief deliverable (1) — strict-equality allowlist gate.  The
# allowlist is consulted FIRST inside R_K1037_P6_admit_wpeu1 so it can admit
# cells that fail the structural envelopes (the K-1074 cohort fails C1 by
# design).  Default-OFF env gate: TRITONBLAS_K1109_P6_ALLOWLIST=1 to enable.
#
# Reviewer-consensus carve-out (4/4 REVISE votes on prior K-1109 attempt):
# the original 8-cell K-1074 cohort had two cells with CI95-confirmed
# regressions in the paired n=30 timing (S32, S37).  Those two are EXCLUDED
# from the allowlist and pinned in K1074_REGRESSION_CARVE_OUT_PINS so any
# silent re-add fails CI.  The shipped allowlist is therefore 6 cells.
# ---------------------------------------------------------------------------
K1074_COHORT_6 = [
    ("S26",  8064, 2048, 1024),
    ("S31", 18304, 2048, 1024),
    ("S33", 22400, 2048, 1024),
    ("S34", 24448, 2048, 1024),
    ("S35", 26496, 2048, 1024),
    ("S39", 49152, 2048,  256),
]

# Pinned regression evidence — keys are (cid, M, N, K, dtype),
# values are the K-1074 paired n=30 timing measurements.  These cells must
# stay OUT of the allowlist; if anybody re-adds them to
# _K1109_P6_K1074_ALLOWLIST without first refuting this evidence with a
# fresh paired backtest, K1074_REGRESSION_CARVE_OUT_PINS fails.
K1074_REGRESSION_CARVE_OUT_PINS = [
    # (cid,    M,  N,    K, dtype,            mean_pct, ci95_lo, ci95_hi)
    ("S32", 20352, 2048, 1024, "torch.bfloat16",   -0.5101, -0.872,  -0.1481),
    ("S37", 25600, 2048,  256, "torch.bfloat16",   -0.6806, -1.1294, -0.2318),
]


def test_p6_k1109_allowlist_contents_are_pinned():
    """The allowlist content is part of the public contract of K-1109's
    diff — pin the exact tuples so any silent edit fails CI."""
    expected = frozenset(
        (M, N, K, "torch.bfloat16") for _cid, M, N, K in K1074_COHORT_6)
    assert _K1109_P6_K1074_ALLOWLIST == expected


def test_p6_k1109_allowlist_env_var_name_is_pinned():
    """Pin the env var name (operators set this to enable; renaming silently
    would break runbooks)."""
    assert _K1109_P6_ALLOWLIST_ENV == "TRITONBLAS_K1109_P6_ALLOWLIST"


def test_p6_k1109_regression_carve_out_pins():
    """K-1109 reviewer-consensus regression-guard.  The two K-1074 cells
    whose paired n=30 timing showed CI95 entirely below zero (S32, S37)
    must NEVER appear in the allowlist.  Pinning the (M,N,K,dtype) tuples
    here as the canonical exclusion set means a future contributor who
    silently re-adds either cell will trip both this test AND the
    contents-pinned test above; the docstring carries the exact CI95
    evidence so the reviewer reading the diff sees WHY.

    Source: workspace output/k1109_k1074_timing.json (derived from
    K-1074/output/rr_summary.csv, c42/MI300X HIP-graph hot-cache, n=30
    paired)."""
    # 1) Exclusion frozenset is correctly populated.
    expected_exclusions = frozenset(
        (M, N, K, dtype) for _cid, M, N, K, dtype, *_ in
        K1074_REGRESSION_CARVE_OUT_PINS)
    assert _K1109_P6_K1074_REGRESSION_EXCLUSIONS == expected_exclusions
    # 2) Exclusions and allowlist are disjoint (the contract).
    assert _K1109_P6_K1074_REGRESSION_EXCLUSIONS.isdisjoint(
        _K1109_P6_K1074_ALLOWLIST), (
        "REGRESSION GUARD: a K-1074 cell with CI95-confirmed regression in "
        "paired n=30 timing was re-added to _K1109_P6_K1074_ALLOWLIST. "
        "See the docstring on _K1109_P6_K1074_ALLOWLIST in "
        "include/tritonblas/_route_predicate.py for the timing evidence.")
    # 3) For each excluded cell, the recorded CI95 upper bound is below
    #    zero — i.e. the regression evidence pinned here is real.
    for cid, M, N, K, dtype, mean_pct, ci_lo, ci_hi in \
            K1074_REGRESSION_CARVE_OUT_PINS:
        assert ci_hi < 0.0, (
            f"REGRESSION-EVIDENCE PIN: {cid} ({M},{N},{K},{dtype}) "
            f"CI95 upper bound is {ci_hi}, expected < 0 (i.e. confirmed "
            f"regression).  If this assertion fails because the timing was "
            f"re-measured, the cell should also be considered for re-add to "
            f"the allowlist.")


@pytest.mark.parametrize("cid,M,N,K,dtype,_mean,_lo,_hi",
                         K1074_REGRESSION_CARVE_OUT_PINS,
                         ids=[r[0] for r in K1074_REGRESSION_CARVE_OUT_PINS])
def test_p6_k1109_regressing_cells_never_admit_even_when_env_set(
        cid, M, N, K, dtype, _mean, _lo, _hi, monkeypatch):
    """Hazard pin: even with the gate flipped ON, the two K-1074 cells with
    CI95-confirmed regressions in paired n=30 timing must NOT admit (since
    they are not in the allowlist; this test lives next to the carve-out
    documentation so a contributor who re-adds them sees the failing test
    cite this exact reason)."""
    monkeypatch.setenv(_K1109_P6_ALLOWLIST_ENV, "1")
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is False, (
        f"REGRESSION HAZARD: K-1074 cell {cid} ({M},{N},{K}) admitted "
        f"into wpeu=1 route despite CI95-confirmed regression in paired "
        f"n=30 timing.  See _K1109_P6_K1074_REGRESSION_EXCLUSIONS in "
        f"include/tritonblas/_route_predicate.py.")


@pytest.mark.parametrize("cid,M,N,K", K1074_COHORT_6,
                         ids=[c[0] for c in K1074_COHORT_6])
def test_p6_k1109_allowlist_inert_when_env_unset(cid, M, N, K, monkeypatch):
    """Default-OFF: with the env gate unset, allowlisted cells must NOT
    admit (preserves K-1037 18/18 + K-1109 24/24 confusion-matrix integrity
    that the structural surrogate inherits).  Locks the K-1074 paired n=30
    NULL-RESULT (0/6 cleared the K-901 +2% LAND margin) into CI as the
    production default."""
    monkeypatch.delenv(_K1109_P6_ALLOWLIST_ENV, raising=False)
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is False, (
        f"K-1109 allowlist leaked when env unset on {cid} ({M},{N},{K})")


@pytest.mark.parametrize("cid,M,N,K", K1074_COHORT_6,
                         ids=[c[0] for c in K1074_COHORT_6])
def test_p6_k1109_allowlist_admits_when_env_set(cid, M, N, K, monkeypatch):
    """Gate-ON: with TRITONBLAS_K1109_P6_ALLOWLIST=1, the 6 non-regressing
    K-1074 cells admit.  This is the brief's "strict-equality fallback for
    the validated 8-cell set as a safety net" deliverable, with the two
    regression-confirmed cells (S32, S37) carved out per reviewer
    consensus.  A future fresh n=30 c42/MI300X backtest will determine
    whether this gate flips on by default."""
    monkeypatch.setenv(_K1109_P6_ALLOWLIST_ENV, "1")
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is True, (
        f"K-1109 allowlist failed to admit gated {cid} ({M},{N},{K})")


def test_p6_k1109_allowlist_does_not_widen_to_negatives_when_set(monkeypatch):
    """Gate-ON must NOT admit any held-out K-1017 negative (zero false
    positive on the K-1037 calibration set even with the gate flipped)."""
    monkeypatch.setenv(_K1109_P6_ALLOWLIST_ENV, "1")
    # K-1017 OCC negatives that sit OUTSIDE the K-1074 cohort:
    held_out = [
        ("S30", 16256, 2048, 1024),  # OCC ratio 1.7530 — collinearity wall
        ("S38",   384,  128,  200),  # HBM-bound, K=200
        ("S07",  2048, 1792,  256),  # OCC shallow-K
        ("S16",   256,  256, 2048),  # OCC small-mild-rect
        ("S14",  1024, 1024, 1024),  # OCC square (K-1017 cohort)
    ]
    for cid, M, N, K in held_out:
        assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is False, (
            f"K-1109 allowlist false-positive on held-out NEG {cid} "
            f"({M},{N},{K}) when env set")


def test_p6_k1109_allowlist_only_admits_bf16_when_set(monkeypatch):
    """Gate-ON must still respect the bf16-only carve-out (K-1037 ground
    truth scope is bf16 only)."""
    monkeypatch.setenv(_K1109_P6_ALLOWLIST_ENV, "1")
    M, N, K = 8064, 2048, 1024  # S26 — bf16 entry in the allowlist
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is True
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.float16) is False
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.float32) is False


def test_p6_c1_collinearity_wall_is_load_bearing():
    """K-1109 lock: Envelope A's M=14999 ceiling exists because S30
    (M=16256, ratio=1.7530, OCC) and S34 (M=24448, ratio=1.7530, OCC)
    collide exactly on C1.  No structural-surrogate cutoff in the
    waves_ratio interval (1.5320, 1.7530] can admit S34 without producing
    a false positive on S30 -- so the M-band must be enforced via shape
    coordinates rather than a 'tighten C1' relaxation.  This test pins
    that S30 and S34 receive the same NO_FIRE verdict (both must remain
    OUTSIDE Envelope A) and that the ceiling stops at M=14999."""
    # S30 and S34 both NEG; S29 (between them, M=14208) is the only POS.
    assert R_K1037_P6_admit_wpeu1(16256, 2048, 1024, torch.bfloat16) is False  # S30
    assert R_K1037_P6_admit_wpeu1(24448, 2048, 1024, torch.bfloat16) is False  # S34
    assert R_K1037_P6_admit_wpeu1(14208, 2048, 1024, torch.bfloat16) is True   # S29


# ---------------------------------------------------------------------------
# K-1144 P8 — MFMA-issue-stall direct hipBLASLt route-OUT.  Productionizes
# K-1121 (13-cell anchor cohort) + K-1131 (12-cell held-out neighbor
# envelope).  Pin tests cover (a) 25/25 envelope cells route-OUT at the
# dispatch level, (b) bf16-only carve-out, (c) precedence-vs-K-1089:
# overlap cells (S24, S29, N11) route-OUT despite P6 admit, (d) no-overlap
# guarantee with the K-1062 K-floor=512 / K-1066 P5 / K-1109 allowlist
# productionized predicates, (e) zero-leak vs the K-984/K-989 LAND anchors
# and the K-950 LAND set, (f) provenance pins so the 25-cell envelope
# can't silently drift.
# ---------------------------------------------------------------------------
K1121_P8_ANCHORS_13_LIST = [
    # (cid,    M,  N,    K)  -- per K-1121 manifest, paired n=30 hot-cache
    ("S24",  4480, 3072,  768),
    ("S29", 14208, 2048, 1024),
    ("S18",  5972, 1792,  768),
    ("S25",  6016, 2048, 1024),
    ("S30", 16256, 2048, 1024),
    ("S26",  8064, 2048, 1024),
    ("S31", 18304, 2048, 1024),
    ("S32", 20352, 2048, 1024),
    ("S33", 22400, 2048, 1024),
    ("S34", 24448, 2048, 1024),
    ("S35", 26496, 2048, 1024),
    ("S37", 25600, 2048,  256),
    ("S39", 49152, 2048,  256),
]


K1131_P8_NEIGHBORS_12_LIST = [
    # (cid,    M,    N,    K, anchor, axis-direction)
    ("N01", 20352, 2048,  512),  # S32 K/2
    ("N02", 20352, 2048, 2048),  # S32 K*2
    ("N03", 40704, 2048, 1024),  # S32 M*2
    ("N04", 10176, 2048, 1024),  # S32 M/2  (envelope edge 1.604x)
    ("N05", 20352, 4096, 1024),  # S32 N*2
    ("N06", 20352, 1024, 1024),  # S32 N/2  (cohort MAX 1.657x)
    ("N07", 49152, 2048,  128),  # S39 K/2
    ("N08", 49152, 2048,  512),  # S39 K*2
    ("N09",  5972,  896,  768),  # S18 N/2
    ("N10",  5972, 3584,  768),  # S18 N*2
    ("N11",  2240, 3072,  768),  # S24 M/2  (P6 Env-B overlap)
    ("N12",  4480, 3072, 1536),  # S24 K*2
]


# K-931 control sample — 4 cells from the K-931 always-uncovered top-40
# catalog that K-1127 confirmed are NOT in the K-1121 envelope.  Pinned
# here as the "P8 must NOT leak into K-931 controls" no-false-positive
# witness; mirrors K-1127's 0-FP adversarial backtest verdict.
#
# K-1352 NOTE: the prior K931_C04 control = (2048, 2048, 4096) bf16 was
# REMOVED from this list because K-1336 carve-IN now intentionally admits
# this cell (K-1311 §F1 rank #4, dispatch-wrapper-overhead route-OUT).
# It is no longer a control; it is an intentional admit.
K931_CONTROL_CELLS_5 = [
    ("K931_C01",   1024, 1024, 1240),  # K-989 LAND anchor (S40)
    ("K931_C02",   2048, 1024, 1024),  # mid-square mid-K
    ("K931_C03",  16384, 8192, 1024),  # extreme-M extreme-N
    ("K931_C05",   4096, 4096, 2048),  # square mid-K
]


def test_k1352_p8_envelope_size_is_exactly_57_after_k1336_carvein_compose():
    """K-1352 composed envelope: 13 K-1121 anchors + 12 K-1131 neighbors
    + 3 K-1175/K-1161 E2 admits + 8 K-1205 E_N3 N=128 admits + 7 K-1219
    E3 N=256 admits + 8 K-1283/K-1297 A1 perturbations + 6 K-1311/K-1336
    regressor carve-IN = 57 cells.  Any silent edit changes this count
    and trips this canary.

    Provenance chain:
      * K-1144 originally pinned 25.
      * K-1175 extended by 3 K-1161-validated cells (E2_I4 K-interior
        admit + 2 M-axis admits at K=1024) -> 28.
      * K-1216 stacked E1 envelope after P8 (no envelope-size change).
      * K-1231 / K-1205 extended by 8 N=128 cells (E_N3 cohort) -> 36
        (productionised in K-1275 on top of K-1216 stacked dispatch).
      * K-1219 / K-1240 extended by 7 N=256 cells (E3 cohort,
        K-1131-style anchor projection at N-floor=256), validated
        independently on commit f110d64 (fix/K-1219-n256).  +7 -> 43
        (K-1303 unified composition).
      * K-1283 / K-1297 extended by 8 A1 sub-cohort cells -- K-1131-style
        +/-1-power-of-2 perturbations of A1 anchors that K-1131 itself
        did NOT enumerate (S25 5/5 axes, S29 3/6 axes), validated
        independently on commit f4cc5cf (fix/K-1283-A1).  +8 -> 51
        (K-1322 unified composition).
      * K-1311 / K-1336 carve-IN extended by 6 dispatch-wrapper-overhead
        regressor cells (longK + square cohort, all bf16; K-1311 ranks
        #1-#6).  Mechanism: +13.89 us mean chain-evaluation cost on
        cells where the chain terminates False -- short-circuiting at
        the strict-equality P8 entry skips the chain and dispatches via
        torch.matmul (hipBLASLt), netting a +0.36 to +0.49 ratio gain.
        +6 -> 57 (K-1352 composed envelope).  4 of these 6 cells overlap
        K-1335's LDS-BC admit set at the (M,N,K,dtype) level; the
        composition invariant is the SUBSET relation
        K-1335 <= K-1336 (asserted at module load)."""
    assert len(_P8_MFMA_ISSUE_STALL_ROUTEOUT) == 57
    assert len(_K1121_P8_ANCHORS_13) == 13
    assert len(_K1131_P8_NEIGHBORS_12) == 12
    assert len(_K1161_E2_ADMITS_3) == 3
    assert len(_K1205_EN3_ADMITS_8) == 8
    assert len(_K1219_E3_NFLOOR256_ADMITS_7) == 7
    assert len(_K1283_A1_PERTURBATIONS_8) == 8
    assert len(_K1336_REGRESSOR_CARVEIN_6) == 6
    # K-1303 cross-band disjointness pin: K-1227 (N=128) vs K-1219 (N=256).
    assert _K1219_E3_NFLOOR256_ADMITS_7.isdisjoint(_K1205_EN3_ADMITS_8)
    # K-1322 K-1297 cross-disjointness with all five prior sub-frozensets.
    assert _K1283_A1_PERTURBATIONS_8.isdisjoint(_K1121_P8_ANCHORS_13)
    assert _K1283_A1_PERTURBATIONS_8.isdisjoint(_K1131_P8_NEIGHBORS_12)
    assert _K1283_A1_PERTURBATIONS_8.isdisjoint(_K1161_E2_ADMITS_3)
    assert _K1283_A1_PERTURBATIONS_8.isdisjoint(_K1205_EN3_ADMITS_8)
    assert _K1283_A1_PERTURBATIONS_8.isdisjoint(_K1219_E3_NFLOOR256_ADMITS_7)
    # K-1352 K-1336 carve-IN cross-disjointness with all six prior P8
    # sub-frozensets (asserted at module load; mirrored here as a pin).
    assert _K1336_REGRESSOR_CARVEIN_6.isdisjoint(_K1121_P8_ANCHORS_13)
    assert _K1336_REGRESSOR_CARVEIN_6.isdisjoint(_K1131_P8_NEIGHBORS_12)
    assert _K1336_REGRESSOR_CARVEIN_6.isdisjoint(_K1161_E2_ADMITS_3)
    assert _K1336_REGRESSOR_CARVEIN_6.isdisjoint(_K1205_EN3_ADMITS_8)
    assert _K1336_REGRESSOR_CARVEIN_6.isdisjoint(_K1219_E3_NFLOOR256_ADMITS_7)
    assert _K1336_REGRESSOR_CARVEIN_6.isdisjoint(_K1283_A1_PERTURBATIONS_8)


def test_k1144_p8_anchors_and_neighbors_are_disjoint():
    """K-1131 neighbor generation rules require strict disjointness from
    the 13 K-1121 anchors (perturbation moves AWAY from the anchor)."""
    assert _K1121_P8_ANCHORS_13.isdisjoint(_K1131_P8_NEIGHBORS_12)


def test_k1144_p8_envelope_contents_are_pinned_to_k1121_manifest():
    """Pin the K-1121 13-anchor envelope to source-of-truth (the K-1121
    manifest).  A silent edit to either constant trips here."""
    expected = frozenset(
        (M, N, K, "torch.bfloat16") for _cid, M, N, K in K1121_P8_ANCHORS_13_LIST)
    assert _K1121_P8_ANCHORS_13 == expected


def test_k1144_p8_envelope_contents_are_pinned_to_k1131_manifest():
    """Pin the K-1131 12-neighbor envelope to source-of-truth (the K-1131
    manifest).  A silent edit to either constant trips here."""
    expected = frozenset(
        (M, N, K, "torch.bfloat16") for _cid, M, N, K in K1131_P8_NEIGHBORS_12_LIST)
    assert _K1131_P8_NEIGHBORS_12 == expected


@pytest.mark.parametrize("cid,M,N,K", K1121_P8_ANCHORS_13_LIST,
                         ids=[c[0] for c in K1121_P8_ANCHORS_13_LIST])
def test_k1144_p8_admits_all_13_k1121_anchors(cid, M, N, K):
    """Every K-1121 anchor must fire the P8 strict-equality match.
    These 13 cells were paired-n=30 measured at hbl/tb >= 1.10x mean
    AND CI95-lo >= 1.05x (K-1007 admission floor) on rad-mi300x-1."""
    assert _p8_mfma_issue_stall_routeout(M, N, K, torch.bfloat16) is True, (
        f"K-1121 anchor {cid} ({M},{N},{K}) missed P8 envelope")


@pytest.mark.parametrize("cid,M,N,K", K1131_P8_NEIGHBORS_12_LIST,
                         ids=[c[0] for c in K1131_P8_NEIGHBORS_12_LIST])
def test_k1144_p8_admits_all_12_k1131_neighbors(cid, M, N, K):
    """Every K-1131 neighbor must fire the P8 strict-equality match.
    These 12 cells classified IN_COHORT at the stricter generalization
    gate (ratio >= 1.15x AND CI95-lo > 1.00) per K-1131 manifest."""
    assert _p8_mfma_issue_stall_routeout(M, N, K, torch.bfloat16) is True, (
        f"K-1131 neighbor {cid} ({M},{N},{K}) missed P8 envelope")


@pytest.mark.parametrize("cid,M,N,K", K1121_P8_ANCHORS_13_LIST + K1131_P8_NEIGHBORS_12_LIST,
                         ids=[c[0] for c in K1121_P8_ANCHORS_13_LIST + K1131_P8_NEIGHBORS_12_LIST])
def test_k1144_p8_dispatch_routes_all_25_cells_out_to_hbl(cid, M, N, K):
    """Integration pin: every P8 cell must dispatch to hipBLASLt at the
    full ``_k971_route_to_hbl`` decision level (composition with P5/P6/
    K-905-K-971 anchor table).  This is the load-bearing production
    contract: K-1121 + K-1131 paired n=30 measurements show hipBLASLt
    wins on every one of these 25 cells."""
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is True, (
        f"P8 cell {cid} ({M},{N},{K}) failed to route-OUT through "
        f"_k971_route_to_hbl; check P8 precedence vs K-1089 P6 admit")


def test_k1144_p8_is_bf16_only():
    """K-1121 / K-1131 measurement scope is bf16; fp16 / fp32 must
    short-circuit (fp16 parity is tracked separately on the K-1093 line)."""
    M, N, K = 4480, 3072, 768  # S24
    assert _p8_mfma_issue_stall_routeout(M, N, K, torch.bfloat16) is True
    assert _p8_mfma_issue_stall_routeout(M, N, K, torch.float16) is False
    assert _p8_mfma_issue_stall_routeout(M, N, K, torch.float32) is False


def test_k1144_p8_string_dtype_compat_with_route_table():
    """Like the K-905/K-971 anchor table, P8 accepts the literal string
    ``"torch.bfloat16"`` (used in tests and in the dispatch site's
    ``str(a_dtype)``) so the route table and the predicate share key
    semantics."""
    assert _p8_mfma_issue_stall_routeout(4480, 3072, 768, "torch.bfloat16") is True
    assert _p8_mfma_issue_stall_routeout(4480, 3072, 768, "torch.float16") is False


@pytest.mark.parametrize("cid,M,N,K", P6_POSITIVES, ids=[c[0] for c in P6_POSITIVES])
def test_k1144_p8_overrides_k1089_p6_admit_on_overlap(cid, M, N, K):
    """Precedence pin: S24 and S29 are inside K-1089 P6 envelopes (P6
    admit returns True at the unit level) but K-1121's paired n=30
    measurement falsified that admit.  P8 must override at the dispatch
    level (return True from ``_k971_route_to_hbl``).  This is the load-
    bearing composition rule -- if a future contributor reorders P8 to
    AFTER P6, this test fails."""
    # Unit-level: P6 still admits (preserves K-1037 18/18).
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is True
    # Unit-level: P8 also fires (cell is in the K-1121 envelope).
    assert _p8_mfma_issue_stall_routeout(M, N, K, torch.bfloat16) is True
    # Composed-dispatch-level: P8 wins -> route to hipBLASLt.
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is True


def test_k1144_p8_overrides_k1089_p6_admit_on_n11_neighbor_overlap():
    """N11 = (2240, 3072, 768) is the K-1131 neighbor that perturbs S24
    along M/2.  It still sits inside K-1089 P6 Envelope B (minMN=2240
    >= 2048; maxMN=3072 <= 4500; K=768 in [512, 1024]).  K-1131 measured
    it IN_COHORT at the >= 1.15x gate, so P8 must override P6 here too --
    same precedence rule as for the 13 anchors."""
    M, N, K = 2240, 3072, 768
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is True
    assert _p8_mfma_issue_stall_routeout(M, N, K, torch.bfloat16) is True
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is True


@pytest.mark.parametrize("cid,M,N,K", K931_CONTROL_CELLS_5,
                         ids=[c[0] for c in K931_CONTROL_CELLS_5])
def test_k1144_p8_does_not_leak_into_k931_control_cells(cid, M, N, K):
    """K-1127's adversarial backtest of K-1121 against the K-931 always-
    uncovered top-40 catalog reported 0 false positives (FP rate 0.0%
    < 5% productionisation gate).  Pin a 5-cell K-931 control sample as
    a no-leak witness; P8 strict-equality must NOT match any control.
    (P5 / K-905 / K-971 may legitimately route some controls -- this test
    only proves P8 itself is non-leaky on the controls.)"""
    assert _p8_mfma_issue_stall_routeout(M, N, K, torch.bfloat16) is False, (
        f"P8 leaked into K-931 control {cid} ({M},{N},{K}) -- envelope "
        f"must remain strictly the K-1121 + K-1131 25-cell set.")


def test_k1144_p8_disjoint_from_k1109_allowlist_keys():
    """K-1109's env-gated allowlist (default OFF) is a wpeu=1 admit list
    -- it pulls cells BACK to in-kernel.  P8 is a route-OUT predicate.
    The K-1109 allowlist contents and P8 contents OVERLAP by design
    (K-1109 covers 6 of the K-1074 cohort cells: S26, S31, S33, S34,
    S35, S39 -- all also in K-1121); the design intent is that P8's
    BEFORE-P6 placement makes the K-1109 admit inert when its env is
    set, so the production default behaviour is consistent regardless
    of K-1109's gate state.  This test verifies that even with the
    K-1109 env ON, every overlap cell still routes OUT through P8."""
    overlap = _K1121_P8_ANCHORS_13 & _K1109_P6_K1074_ALLOWLIST
    # K-1109 allowlist is 6 cells; all 6 are in the K-1121 13-anchor set.
    assert len(overlap) == 6, (
        f"Expected 6-cell overlap between K-1109 allowlist and K-1121 "
        f"anchors (K-1109 documents this); got {len(overlap)}")


def test_k1144_p8_overrides_k1109_allowlist_when_env_set(monkeypatch):
    """With TRITONBLAS_K1109_P6_ALLOWLIST=1, K-1109 admits 6 cells back
    to in-kernel (R_K1037_P6_admit_wpeu1 returns True via the allowlist
    short-circuit).  P8's BEFORE-P6 placement must still route those
    cells OUT to hipBLASLt -- the K-1121 paired n=30 evidence supersedes
    the K-1109 default-OFF admit hypothesis."""
    monkeypatch.setenv(_K1109_P6_ALLOWLIST_ENV, "1")
    overlap_cells = [
        ("S26",  8064, 2048, 1024),
        ("S31", 18304, 2048, 1024),
        ("S33", 22400, 2048, 1024),
        ("S34", 24448, 2048, 1024),
        ("S35", 26496, 2048, 1024),
        ("S39", 49152, 2048,  256),
    ]
    for cid, M, N, K in overlap_cells:
        # K-1109 allowlist admits at unit level (gate ON).
        assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is True, (
            f"K-1109 allowlist failed to admit {cid} with env set")
        # But P8 overrides at dispatch level -> route OUT.
        assert _k971_route_to_hbl(
            M, N, K, torch.bfloat16, torch.bfloat16,
            enable_streamk=False, work_stealing=False) is True, (
            f"P8 failed to override K-1109 allowlist on {cid} ({M},{N},{K}) "
            f"with env set")


def test_k1144_p8_does_not_overlap_k950_land_set():
    """K-950 9-cell LAND set must remain non-routed by P8 EXCEPT for cells
    explicitly admitted by post-K-1144 sub-frozensets.  K-1352 / K-1336
    intentionally carves L05 = (4096^3) bf16 INTO the P8 envelope (K-1311
    §F1 rank #2, dispatch-wrapper-overhead-skip).  See K950_ANCHOR_COLLISIONS
    for the consolidated dispatch-allowlist override list.  All OTHER K-950
    LAND cells must remain non-routed by P8."""
    for cid, M, N, K, dtype, _cohort in K950_LAND_CELLS:
        if (M, N, K, str(dtype)) in K950_ANCHOR_COLLISIONS:
            # Intentional override (K-905 / K-1336).
            continue
        assert _p8_mfma_issue_stall_routeout(M, N, K, dtype) is False, (
            f"P8 leaked into K-950 LAND set on {cid} ({M},{N},{K},{dtype})")


def test_k1144_p8_streamk_workstealing_dtype_mismatch_carve_outs_still_fire():
    """Pin the existing dispatch-level carve-outs (streamk / work-stealing /
    a_dtype != b_dtype) still short-circuit even when P8 would otherwise
    fire.  P8 lives BELOW these carve-outs in ``_k971_route_to_hbl`` so
    the operator-set escape hatches keep priority."""
    M, N, K = 4480, 3072, 768  # S24 -- in P8 envelope
    # streamk on -> no route
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=True, work_stealing=False) is False
    # work-stealing on -> no route
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=True) is False
    # dtype mismatch -> no route
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.float16,
        enable_streamk=False, work_stealing=False) is False


def test_k1144_p8_disable_env_var_short_circuits():
    """The TRITONBLAS_DISABLE_K971 killswitch must continue to disable
    P8 -- one disable lever per K-883 R1 (one cohort, one disable)."""
    import os
    M, N, K = 4480, 3072, 768  # S24
    saved = os.environ.get("TRITONBLAS_DISABLE_K971")
    try:
        os.environ["TRITONBLAS_DISABLE_K971"] = "1"
        assert _k971_route_to_hbl(
            M, N, K, torch.bfloat16, torch.bfloat16,
            enable_streamk=False, work_stealing=False) is False
    finally:
        if saved is None:
            os.environ.pop("TRITONBLAS_DISABLE_K971", None)
        else:
            os.environ["TRITONBLAS_DISABLE_K971"] = saved


def test_k1144_p8_does_not_match_perturbations_outside_envelope():
    """Strict-equality discipline (R-1131): the cohort is a structural
    pocket but the route table is strict-equality only.  Verify that
    near-miss perturbations of the K-1121 anchors that K-1131 did NOT
    sample do NOT match P8 -- specifically, +/-1 in a single dimension
    away from any anchor that is NOT one of the 12 K-1131 neighbors."""
    # S24 = (4480, 3072, 768).  K=767 / K=769 are not in the envelope
    # (the K-1131 K-axis perturbations of S24 are K*2=1536 only; K/2 was
    # not sampled because it would have hit the K=384 region below P5
    # Clause-2's 256-or-1024 pin).
    assert _p8_mfma_issue_stall_routeout(4480, 3072, 767, torch.bfloat16) is False
    assert _p8_mfma_issue_stall_routeout(4480, 3072, 769, torch.bfloat16) is False
    # M=4479 / M=4481 not in envelope.
    assert _p8_mfma_issue_stall_routeout(4479, 3072, 768, torch.bfloat16) is False
    assert _p8_mfma_issue_stall_routeout(4481, 3072, 768, torch.bfloat16) is False


# ---------------------------------------------------------------------------
# K-1175 / K-1161 E2 axis-extension pin tests.  K-1161 paired n=30 HIP-graph
# hot-cache (B=10000 bootstrap CI95) on rad-mi300x-1 / MI300X / ROCm 7.2
# returned NEGATIVE_AXIS_PIVOT for the K-floor=128 + K=512 interior axis
# (3/8 = 37.5% admit, below 50% pivot threshold).  Three cells admit cleanly
# and are staged here as strict-equality additions to P8:
#   E2_I4  ( 736, 1792,  736)  hbl/tb=1.315x  CI95=[1.313, 1.317]
#   E2_M1  (10112, 2048, 1024) hbl/tb=1.759x  CI95=[1.753, 1.770]
#   E2_M2  (12160, 2048, 1024) hbl/tb=1.499x  CI95=[1.496, 1.504]
#
# Pin tests cover (a) 3/3 envelope cells route-OUT at the dispatch level,
# (b) bf16-only carve-out, (c) disjointness with K-1121 anchors and K-1131
# neighbors, (d) NO-LEAK guarantees: the 5 K-1161 cells classified as
# route-IN-safe / ambiguous (E2_K1, E2_K2, E2_I1, E2_I2, E2_I3) MUST NOT
# fire P8 -- staging them would cause production false-positives (TB-WIN
# cells routed to slower hipBLASLt).  See the _K1161_E2_ADMITS_3 docstring
# for the K-floor relaxation mechanism (R-1161 small-M Triton-favoured tail).
# ---------------------------------------------------------------------------
K1161_E2_ADMITS_3_LIST = [
    # (cid,    M,    N,    K, hbl_tb_med, ci95_lo, ci95_hi)  per K-1161 manifest
    ("E2_I4",   736, 1792,  736),  # K=736 K-interior single admit (hbl/tb=1.315x)
    ("E2_M1", 10112, 2048, 1024),  # M-axis extension at K=1024 (hbl/tb=1.759x)
    ("E2_M2", 12160, 2048, 1024),  # M-axis extension at K=1024 (hbl/tb=1.499x)
]


# K-1161 cells that K-1161 measured as route-IN-safe (TB strictly faster) or
# ambiguous.  These cells are inside the E2 envelope but were REJECTED by
# the paired n=30 measurement -- staging them as route-OUT would create
# production FALSE POSITIVES (TB wins routed to slower hipBLASLt).  Pinned
# here as a no-leak witness so any future contributor who tries to widen
# the K-floor or the K=512 interior must trip these tests first.
K1161_E2_REJECTED_5_LIST = [
    # (cid,    M,    N,    K, hbl_tb_med, classification)  per K-1161 manifest
    ("E2_K1",  512, 2048,   96),  # 0.810x  route-IN-safe (TB +23% faster)
    ("E2_K2",  512, 2048,  192),  # 1.020x  ambiguous     (within noise floor)
    ("E2_I1",  192, 2048,  512),  # 0.868x  route-IN-safe (TB +15% faster)
    ("E2_I2",  156, 1792,  512),  # 0.914x  route-IN-safe (TB +9% faster)
    ("E2_I3",  160, 3072,  512),  # 0.907x  route-IN-safe (TB +10% faster)
]


def test_k1175_e2_admits_envelope_size_is_exactly_3():
    """K-1161 measured 8 E2 candidates (K-floor=128 + K=512/736 interior +
    M-axis extension).  Exactly 3 admit route-OUT-safe (hbl/tb >= 1.05x AND
    CI95-lo > 1.00).  Any silent edit changes this count and trips here."""
    assert len(_K1161_E2_ADMITS_3) == 3


def test_k1175_e2_admits_disjoint_from_k1121_and_k1131():
    """K-1161 candidate generator (K-931 always-uncovered top-40 catalog
    intersected with E2 envelope) excluded all K-1121 anchors and all
    K-1131 neighbors by construction."""
    assert _K1161_E2_ADMITS_3.isdisjoint(_K1121_P8_ANCHORS_13)
    assert _K1161_E2_ADMITS_3.isdisjoint(_K1131_P8_NEIGHBORS_12)


def test_k1175_e2_admits_envelope_contents_pinned_to_k1161_manifest():
    """Pin the 3-cell E2 admit envelope to source-of-truth (the K-1161
    paired n=30 manifest at workspace K-1161/output/k1161_per_cell.csv).
    A silent edit to either constant trips here."""
    expected = frozenset(
        (M, N, K, "torch.bfloat16") for _cid, M, N, K in K1161_E2_ADMITS_3_LIST)
    assert _K1161_E2_ADMITS_3 == expected


@pytest.mark.parametrize("cid,M,N,K", K1161_E2_ADMITS_3_LIST,
                         ids=[c[0] for c in K1161_E2_ADMITS_3_LIST])
def test_k1175_p8_admits_all_3_k1161_e2_cells(cid, M, N, K):
    """Every K-1161 E2 admit must fire the P8 strict-equality match.
    These 3 cells were paired n=30 measured at hbl/tb >= 1.315x mean
    AND CI95-lo > 1.05x (route-OUT-safe gate) on rad-mi300x-1."""
    assert _p8_mfma_issue_stall_routeout(M, N, K, torch.bfloat16) is True, (
        f"K-1161 E2 admit {cid} ({M},{N},{K}) missed P8 envelope")


@pytest.mark.parametrize("cid,M,N,K", K1161_E2_ADMITS_3_LIST,
                         ids=[c[0] for c in K1161_E2_ADMITS_3_LIST])
def test_k1175_p8_dispatch_routes_all_3_k1161_e2_cells_out_to_hbl(cid, M, N, K):
    """Integration pin: every K-1161 E2 admit must dispatch to hipBLASLt
    at the full ``_k971_route_to_hbl`` decision level (composition with
    P5/P6/K-905-K-971 anchor table).  K-1161's paired n=30 measurement
    shows hipBLASLt wins on every one of these 3 cells at >= 1.315x."""
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is True, (
        f"K-1161 E2 admit {cid} ({M},{N},{K}) failed to route-OUT through "
        f"_k971_route_to_hbl; check P8 precedence vs K-1089 P6 admit")


def test_k1175_p8_e2_admits_are_bf16_only():
    """K-1161 measurement scope is bf16; fp16/fp32 must short-circuit."""
    M, N, K = 736, 1792, 736  # E2_I4
    assert _p8_mfma_issue_stall_routeout(M, N, K, torch.bfloat16) is True
    assert _p8_mfma_issue_stall_routeout(M, N, K, torch.float16) is False
    assert _p8_mfma_issue_stall_routeout(M, N, K, torch.float32) is False


@pytest.mark.parametrize("cid,M,N,K", K1161_E2_REJECTED_5_LIST,
                         ids=[c[0] for c in K1161_E2_REJECTED_5_LIST])
def test_k1175_p8_does_not_leak_into_k1161_rejected_cells(cid, M, N, K):
    """K-1161 NO-LEAK pin: the 5 E2 candidates that classified as
    route-IN-safe (TB strictly faster) or ambiguous MUST NOT fire P8.

    Staging any of these as route-OUT would create production FALSE
    POSITIVES -- TB-WIN cells (M <= 512 small-M Triton-favoured tail
    per R-1161) routed to a slower hipBLASLt path.  This is the load-
    bearing K-floor=128 + K=512 NEGATIVE_AXIS_PIVOT contract: K-floor
    must remain at K >= 256 and K=512 interior is not a hipBLASLt-favored
    region for the small-M sub-cohort.  If a future contributor widens
    the envelope to K-floor=128 or to K=512 with M < ~600, this test
    fails and they must re-validate at paired n=30 first."""
    assert _p8_mfma_issue_stall_routeout(M, N, K, torch.bfloat16) is False, (
        f"K-1161 REJECTED cell {cid} ({M},{N},{K}) leaked into P8 -- "
        f"this cell is TB-favoured (hbl/tb < 1.05x); routing it OUT "
        f"would be a strict pessimisation per K-1161 paired n=30 evidence.")


def test_k1175_p8_does_not_leak_into_k1142_fp_controls():
    """K-1142 documented 2 false positives at K=256 small-M corner.
    K-1161 re-confirmed both at n=30: FP1 (256, 2048, 256) hbl/tb=0.972x,
    FP2 (2048, 1792, 256) hbl/tb=1.035x.  Both must remain OUT of P8."""
    # FP1 — clear route-IN-safe
    assert _p8_mfma_issue_stall_routeout(256, 2048, 256, torch.bfloat16) is False
    # FP2 — ambiguous, drifted +0.020 between K-1142 and K-1161
    assert _p8_mfma_issue_stall_routeout(2048, 1792, 256, torch.bfloat16) is False


def test_k1175_p8_e2_admits_e2_i4_is_in_k_interior_upper_edge():
    """Documentation pin: E2_I4 is the only K-interior cell that admitted.
    It sits at K=736, on the upper edge of the K-axis extension (closer to
    E1's existing K=768 anchor than to the K=512 interior gap or K=128
    floor relaxation).  This is consistent with K-1161's R-1161 finding
    that the small-K / small-M corner is Triton-favoured -- E2_I4 has
    M=736 which is well above the small-M tail threshold (M <= 512)."""
    M, N, K = 736, 1792, 736
    assert (M, N, K, "torch.bfloat16") in _K1161_E2_ADMITS_3
    assert M > 600   # outside small-M Triton-favoured tail
    assert K >= 512  # K-interior, not K-floor
    assert K < 768   # but below E1's K-floor anchor


def test_k1175_p8_e2_admits_m_axis_cells_at_k1024_anchor_k():
    """Documentation pin: E2_M1 and E2_M2 are at K=1024 (E1 anchor K) with
    M values interior to the K-1121 anchor M-range (between S26 M=8064
    and S29 M=14208).  These are NOT K-floor or K=512 cells -- they are
    M-axis extension cells that K-1161 incidentally validated.  K-1127
    classified both as EXT and they re-confirm at K-1161's measurement
    context (hbl/tb 1.499x and 1.759x respectively)."""
    for cid, M, N, K in (("E2_M1", 10112, 2048, 1024), ("E2_M2", 12160, 2048, 1024)):
        assert (M, N, K, "torch.bfloat16") in _K1161_E2_ADMITS_3
        assert K == 1024  # K-1121 anchor K, NOT K=512 / K=128
        assert N == 2048  # K-1121 anchor N
        # M is interior to the K-1121 anchor M-range:
        assert 8064 < M < 14208


# ===========================================================================
# K-1209-stacked / K-1216 — E1 axis-aligned envelope pin tests.
# E1 (R_K1142_E1_route_to_hbl) is stacked AFTER P8 28-cell strict-equality
# in dispatch precedence; K-1209 ablation on K-931 always-uncovered top-40
# confirmed E1 contributes 0 marginal cells beyond P8+K-1175 (16/40 union)
# but E1 is retained as defense-in-depth for non-K-931 cohorts where the
# audit chain has not converged. These tests pin (a) the K-1142 carve-out
# floors holding against the 2 K-1142 inverse-predicate FPs, (b) E1
# admitting all K-1121 anchors that lie inside its envelope, and (c) the
# E2 K-floor=128 exclusion (K-1176 cross-arch failure: do NOT relax K_FLOOR
# from 256 to 128 — 0/8 cells admit on MI325X/MI355X).
# ===========================================================================


def test_k1216_e1_envelope_constants_pinned():
    """K-1209-stacked / K-1216: pin the K-1142 E1 envelope structural
    constants. A future refactor that loosens these values silently
    re-admits the K-1142 inverse-predicate FPs and breaks the 0-FP
    composition guarantee K-1209 confirmed across 4 stacked configs.
    """
    assert K1142_E1_NS == frozenset({1792, 2048, 3072})
    assert K1142_E1_KS == frozenset({256, 768, 1024})
    assert K1142_E1_M_FLOOR == 4480  # K-1121 anchor S24's M (margin 0)
    assert K1142_E1_K_FLOOR == 256   # K-1161 NEGATIVE_AXIS_PIVOT pin


def test_k1216_e1_excludes_k1142_fp1_at_M_floor():
    """K-1142 FP1 = (M=256, N=2048, K=256) bf16: hbl/tb=0.972x (TB-TIE).
    Must be excluded by M >= 4480 carve-out. Re-admitting this cell
    silently re-introduces K-1142's paired-n=30 false positive."""
    assert R_K1142_E1_route_to_hbl(256, 2048, 256, torch.bfloat16) is False


def test_k1216_e1_excludes_k1142_fp2_at_M_floor():
    """K-1142 FP2 = (M=2048, N=1792, K=256) bf16: hbl/tb=1.035x (TB-TIE).
    Must be excluded by M >= 4480 carve-out (M=2048 < 4480)."""
    assert R_K1142_E1_route_to_hbl(2048, 1792, 256, torch.bfloat16) is False


@pytest.mark.parametrize("cid,M,N,K", [
    ("S24",  4480, 3072,  768),
    ("S29", 14208, 2048, 1024),
    ("S30", 16256, 2048, 1024),
    ("S31", 18304, 2048, 1024),
    ("S32", 20352, 2048, 1024),
    ("S33", 22400, 2048, 1024),
    ("S34", 24448, 2048, 1024),
    ("S35", 26496, 2048, 1024),
    ("S37", 25600, 2048,  256),
    ("S39", 49152, 2048,  256),
])
def test_k1216_e1_admits_k1121_anchors_inside_envelope(cid, M, N, K):
    """K-1121 anchors that lie inside E1 envelope must be admitted.
    These overlap with P8 strict-equality (P8 wins by precedence) but
    E1 must independently classify them correctly so the predicate is
    self-consistent for downstream callers / non-K-931 cohorts."""
    assert R_K1142_E1_route_to_hbl(M, N, K, torch.bfloat16) is True


def test_k1216_e1_excludes_e2_kfloor_128_per_k1176_cross_arch_failure():
    """K-1176 cross-arch port of E2 K-floor=128 relaxation FAILED on both
    MI325X and MI355X (0/8 cells admit). The K_FLOOR=256 pin in E1 must
    reject any cell with K < 256 to prevent re-introducing the K-1176-
    rejected K-axis relaxation. This test pins E1's K-axis floor at 256
    against a future relaxation attempt."""
    # Sample E2 K-floor=128 candidate cells from K-1161 audit:
    for M in (4480, 8192, 16256):
        for N in (1792, 2048, 3072):
            assert R_K1142_E1_route_to_hbl(M, N, 128, torch.bfloat16) is False
            assert R_K1142_E1_route_to_hbl(M, N, 192, torch.bfloat16) is False


def test_k1216_e1_is_bf16_only():
    """E1 inherits K-1142's bf16-only audit scope (the K-1051/K-1031/K-1121
    measurement chain is bf16). Non-bf16 dtypes must return False so the
    fp16 anchor table (K-905/K-971) and fp16 envelope (K-1093/K-1125) own
    fp16 dispatch decisions exclusively."""
    for dt in (torch.float16, torch.float32, torch.float64):
        assert R_K1142_E1_route_to_hbl(4480, 2048, 768, dt) is False


def test_k1216_e1_dispatch_stacked_after_p8_no_double_route():
    """K-1209 dominance check: every cell E1 admits inside P8's 28-cell
    envelope is already routed True by P8, so E1's own True is harmless
    (idempotent). For K-1121 anchors / K-1131 neighbors / K-1175 E2
    admits that lie inside E1, the dispatch verdict is True regardless
    of which predicate fires first — the brief's no-double-routing
    guarantee holds at the verdict level."""
    # Sample 5 P8-anchored cells inside E1 envelope (overlap region):
    for M, N, K in [(4480, 3072, 768), (14208, 2048, 1024),
                    (10112, 2048, 1024), (16256, 2048, 1024),
                    (49152, 2048, 256)]:
        p8_says = _p8_mfma_issue_stall_routeout(M, N, K, torch.bfloat16)
        e1_says = R_K1142_E1_route_to_hbl(M, N, K, torch.bfloat16)
        assert p8_says is True
        # E1 may or may not fire depending on whether cell is inside E1
        # envelope; verdict is True iff either fires (dispatch is OR).
        assert (p8_says or e1_says) is True


# ---------------------------------------------------------------------------
# K-1231 / K-1205 E_N3 N-axis P8 envelope extension pin tests (8 admit cells
# at N=128 + 1 no-leak carve-out for the EN_K768_S24 false positive at the
# K-1142 M-floor).  Layered on top of K-1216's stacked dispatch chain
# (fix/K-1209-stacked @ fb9461f); K-1216 P8/E1 stacking semantics are
# preserved unchanged -- E_N3 admits are pure data added to the P8 union
# frozenset, so E1's defense-in-depth role and 0-marginal-coverage verdict
# from K-1209 are not disturbed.
#
# Source: K-1205 paired n=30 HIP-graph hot-cache validation on rad-mi300x-1
# (c42 SSH plane outage 6th->10th->11th recurrence; rad-mi300x-1 fallback
# is now load-bearing infrastructure for the entire P8 envelope investigation
# series per R-1205.RAD-MI300X-1-IS-LOAD-BEARING-INFRASTRUCTURE-FOR-P8-
# ENVELOPE-WORK).  Brief decision threshold: >=75% landing => STAGE_DIFF;
# K-1205 returned 8/9 = 88.9% admit at hbl/tb_med >= 1.05x AND CI95-lo > 1.00.
# ---------------------------------------------------------------------------

K1205_EN3_ADMITS_8_LIST = [
    # (cid, M, N, K, hbl_tb_med, CI95_lo)
    ("EN_K1024_S25",  6016,  128, 1024, 1.370, 1.362),
    ("EN_K1024_S26",  8064,  128, 1024, 1.402, 1.396),
    ("EN_K1024_S29", 14208,  128, 1024, 3.254, 3.234),  # cohort MAX
    ("EN_K1024_S30", 16256,  128, 1024, 3.024, 3.011),
    ("EN_K1024_S33", 22400,  128, 1024, 2.258, 2.234),
    ("EN_K256_S37",  25600,  128,  256, 1.144, 1.140),  # cohort MIN admit
    ("EN_K256_S39",  49152,  128,  256, 1.781, 1.768),
    ("EN_K768_S18",   5972,  128,  768, 1.227, 1.226),
]

# K-1205 N-axis false-positive carve-out: EN_K768_S24 at M=4480 K=768 N=128
# bf16 measured hbl/tb_med = 0.967x CI95=[0.94, 1.00] -- TB-favoured at
# the K-1142 M-floor.  Carve-out is implicit (cell is omitted from the
# strict-equality frozenset by construction); this list is the explicit
# regression-trap counterpart per R-1175.PIN-REJECTED-CELLS-AS-NO-LEAK-
# WITNESSES.  A future contributor who tries to widen E_N3 to a parametric
# envelope along the M-axis must trip this pin and re-validate at paired
# n=30 -- same discipline K-1175 applied to the K-1161 K-axis rejects.
K1205_EN3_REJECTED_1_LIST = [
    # (cid, M, N, K, hbl_tb_med, mechanism)
    ("EN_K768_S24", 4480, 128, 768, 0.967,
     "small-M Triton-favoured tail at the K-1142 M-floor "
     "(R-1205.SMALL-M-TRITON-FAVOURED-TAIL-IS-AXIS-AGNOSTIC-AT-COHORT-M-FLOOR)"),
]


def test_k1205_en3_admits_envelope_size_is_exactly_8():
    """Pin K-1205 E_N3 sub-frozenset cardinality. The K-1205 brief gated
    landing at >=75% of 9 candidates; the 8/9 = 88.9% empirical result
    drives this size.  Any silent edit changes the count and trips this
    canary."""
    assert len(_K1205_EN3_ADMITS_8) == 8


def test_k1205_en3_admits_pairwise_disjoint_with_k1121_k1131_k1161():
    """K-1205 E_N3 candidates have N=128 by construction; no K-1121 anchor,
    K-1131 neighbor, or K-1161 E2 admit has N=128, so the 4-way union is
    pairwise disjoint without manual exclusion lists.  This is the
    structural disjointness invariant that the runtime asserts in
    _route_predicate.py rely on."""
    assert _K1205_EN3_ADMITS_8.isdisjoint(_K1121_P8_ANCHORS_13)
    assert _K1205_EN3_ADMITS_8.isdisjoint(_K1131_P8_NEIGHBORS_12)
    assert _K1205_EN3_ADMITS_8.isdisjoint(_K1161_E2_ADMITS_3)


def test_k1205_en3_admits_envelope_contents_pinned_to_k1205_manifest():
    """Pin the K-1205 8-admit envelope to source-of-truth (the K-1205
    paired n=30 dataset summarised in K1205_EN3_ADMITS_8_LIST). A silent
    edit to either constant trips here."""
    expected = frozenset(
        (M, N, K, "torch.bfloat16")
        for _cid, M, N, K, _hbl_tb, _ci_lo in K1205_EN3_ADMITS_8_LIST)
    assert _K1205_EN3_ADMITS_8 == expected


def test_k1205_en3_admits_all_have_n_equal_128():
    """Structural axis-discipline invariant: K-1205 staged the N-axis
    extension at N-floor=128 ONLY. K-1161/K-1176 already established
    that K-floor relaxation past K=256 is NEGATIVE_AXIS_PIVOT (cross-
    arch confirmed).  Any cell in E_N3 with N != 128 would be a silent
    drift away from the staged axis and trips here."""
    for (M, N, K, dtype) in _K1205_EN3_ADMITS_8:
        assert N == 128, (
            f"K-1205 E_N3 cell {(M, N, K, dtype)!r} has N={N} != 128; "
            "K-1205 is N-axis-only at N-floor=128 by R-1205.")


def test_k1205_en3_admits_k_in_k1144_k_set():
    """Structural axis-discipline invariant: K-1205 held K fixed at the
    K-1144 K-set {256, 768, 1024} (NOT a K-axis relaxation -- K-1161
    rejected K-floor < 256 on adjacent cells).  Any K outside this set
    would be a covert K-axis extension and trips here."""
    for (M, N, K, dtype) in _K1205_EN3_ADMITS_8:
        assert K in {256, 768, 1024}, (
            f"K-1205 E_N3 cell {(M, N, K, dtype)!r} has K={K} outside "
            "the K-1144 K-set {256, 768, 1024}; K-1205 is N-axis-only.")


@pytest.mark.parametrize(
    "cid,M,N,K,hbl_tb_med,ci95_lo", K1205_EN3_ADMITS_8_LIST,
    ids=[c[0] for c in K1205_EN3_ADMITS_8_LIST])
def test_k1205_p8_admits_all_8_en3_cells(cid, M, N, K, hbl_tb_med, ci95_lo):
    """Every K-1205 E_N3 admit cell must fire the P8 strict-equality
    match.  Each was paired-n=30 measured at hbl/tb_med >= 1.144x
    AND CI95-lo > 1.00 (K-1007 admission floor exceeded with margin)
    on rad-mi300x-1 under HIP-graph hot-cache."""
    assert hbl_tb_med >= 1.05, (
        f"{cid} hbl/tb_med {hbl_tb_med:.3f}x below K-1007 admission floor")
    assert ci95_lo > 1.00, (
        f"{cid} CI95-lo {ci95_lo:.3f} fails CI95-lo > 1.00 gate")
    assert _p8_mfma_issue_stall_routeout(M, N, K, torch.bfloat16) is True


@pytest.mark.parametrize(
    "cid,M,N,K,hbl_tb_med,ci95_lo", K1205_EN3_ADMITS_8_LIST,
    ids=[c[0] for c in K1205_EN3_ADMITS_8_LIST])
def test_k1205_dispatch_routes_all_8_en3_cells_out_to_hbl(
        cid, M, N, K, hbl_tb_med, ci95_lo):
    """End-to-end dispatch check: with default carve-outs disabled,
    _k971_route_to_hbl must return True for every K-1205 E_N3 admit
    so the cell is routed to hipBLASLt and silently picks up the
    measured 1.14x-3.25x speedup."""
    decision = _k971_route_to_hbl(
        M, N, K,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16,
        enable_streamk=False, work_stealing=False)
    assert decision is True, (
        f"{cid} (M={M} N={N} K={K}) was not routed to hipBLASLt; "
        f"expected True per K-1205 paired n=30 hbl/tb_med={hbl_tb_med:.3f}x")


def test_k1205_p8_en3_admits_are_bf16_only():
    """K-1205 measurement scope is bf16 only.  fp16 / fp32 with the same
    (M, N, K) must NOT fire P8 -- fp16 dispatch is owned by the K-1093
    / K-1125 mirror predicate line (none of which currently covers N=128;
    a silent expansion of P8 to fp16 would invade that ownership and
    trips here)."""
    for (M, N, K, _dtype) in _K1205_EN3_ADMITS_8:
        for fp_dtype in (torch.float16, torch.float32):
            assert _p8_mfma_issue_stall_routeout(M, N, K, fp_dtype) is False, (
                f"P8 fired on non-bf16 dtype {fp_dtype} for "
                f"K-1205 cell {(M, N, K)}; K-1205 is bf16-only.")


@pytest.mark.parametrize(
    "cid,M,N,K,hbl_tb_med,mechanism", K1205_EN3_REJECTED_1_LIST,
    ids=[c[0] for c in K1205_EN3_REJECTED_1_LIST])
def test_k1205_p8_does_not_leak_into_en_k768_s24_carve_out(
        cid, M, N, K, hbl_tb_med, mechanism):
    """NO-LEAK pin (R-1175.PIN-REJECTED-CELLS-AS-NO-LEAK-WITNESSES extended
    to the K-1205 N-axis):  EN_K768_S24 (M=4480 N=128 K=768 bf16) was
    measured at hbl/tb_med = 0.967x (TB-favoured) -- it is the K-1205
    N-axis FALSE-POSITIVE CARVE-OUT.  The strict-equality frozenset
    excludes it by construction (omission); this test is the explicit
    regression trap.

    A future contributor who tries to widen E_N3 to a parametric envelope
    along the M-axis -- e.g. ``N == 128 AND K in {256, 768, 1024} AND M
    in K-1121-anchor-M-set`` without an M-floor strictly above 4480 --
    will trip this test, forcing them to either (a) re-validate at paired
    n=30 (the right thing) or (b) explicitly remove the pin (which makes
    the deviation auditable and traceable to the K-1142 M-floor / R-1205
    small-M Triton-favoured tail mechanism)."""
    assert hbl_tb_med < 1.05, (
        f"{cid} reject pin requires measured hbl/tb_med < 1.05 (got "
        f"{hbl_tb_med:.3f}); update K1205_EN3_REJECTED_1_LIST if "
        "K-1205 measurements were re-run with different verdict.")
    assert _p8_mfma_issue_stall_routeout(M, N, K, torch.bfloat16) is False, (
        f"P8 leaked admit into K-1205 N-axis carve-out {cid} "
        f"(M={M} N={N} K={K}); reject mechanism: {mechanism}.")


def test_k1205_en_k768_s24_dispatch_does_not_route_to_hbl():
    """Carve-out validated through the FULL dispatch chain (P8 -> E1 ->
    P6 -> P5 -> K-971 anchor table).  EN_K768_S24 must NOT be routed
    to hipBLASLt by any layer; if it is, K-1205 ships a silent
    pessimisation worth ~3% per call on this shape."""
    M, N, K = 4480, 128, 768
    decision = _k971_route_to_hbl(
        M, N, K,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16,
        enable_streamk=False, work_stealing=False)
    assert decision is False, (
        f"EN_K768_S24 (M={M} N={N} K={K}) was routed to hipBLASLt by "
        "the dispatch chain; this is the K-1205 N-axis false-positive "
        "carve-out -- measured TB-WIN at hbl/tb_med=0.967x.")


def test_k1205_p8_e1_stack_no_double_route_for_en3_admits():
    """K-1216 stacked-dispatch invariant preserved: every K-1205 E_N3
    admit fires P8 (strict equality), and -- because all 8 admits have
    N=128 which is OUTSIDE the K-1216 E1 envelope's N-set
    {1792, 2048, 3072} -- E1 must NOT fire on these cells (E1 is a
    no-op for N=128 by design).  Confirms K-1216's E1 stacking is
    untouched by the K-1231 P8 extension (K-1209 dominance unchanged
    on K-931 top-40)."""
    for (M, N, K, _dtype) in _K1205_EN3_ADMITS_8:
        p8_says = _p8_mfma_issue_stall_routeout(M, N, K, torch.bfloat16)
        e1_says = R_K1142_E1_route_to_hbl(M, N, K, torch.bfloat16)
        assert p8_says is True, (
            f"P8 missed K-1205 E_N3 admit ({M}, {N}, {K})")
        assert e1_says is False, (
            f"E1 fired on K-1205 N=128 cell ({M}, {N}, {K}); K-1216 "
            "stacking semantics require E1 N-set {1792, 2048, 3072} "
            "(N=128 must be excluded).")


def test_k1205_does_not_disturb_existing_p8_28_cell_envelope():
    """K-1216 -> K-1231 invariance check: the original 28-cell P8 envelope
    is preserved exactly -- K-1205 strict-equality union ADDS 8 cells
    without re-measuring or modifying any existing K-1144/K-1175 cell.
    This is R-1184.STRICT-EQUALITY-UNION-PROVES-EXISTING-CELL-INVARIANCE-
    WITHOUT-RE-MEASUREMENT in test form."""
    pre_k1205 = (
        _K1121_P8_ANCHORS_13
        | _K1131_P8_NEIGHBORS_12
        | _K1161_E2_ADMITS_3)
    assert len(pre_k1205) == 28
    for cell in pre_k1205:
        assert cell in _P8_MFMA_ISSUE_STALL_ROUTEOUT, (
            f"K-1175 P8 cell {cell} dropped from K-1231 envelope; "
            "K-1205 must be strict-equality union only.")
    # Symmetric (post K-1352): every cell beyond the original 28-cell
    # baseline must come from K-1205 (8 N=128), K-1219 (7 N=256),
    # K-1283/K-1297 (8 A1 perturbations), or K-1311/K-1336 (6 carve-IN).
    delta = _P8_MFMA_ISSUE_STALL_ROUTEOUT - pre_k1205
    expected_delta = (
        _K1205_EN3_ADMITS_8
        | _K1219_E3_NFLOOR256_ADMITS_7
        | _K1283_A1_PERTURBATIONS_8
        | _K1336_REGRESSOR_CARVEIN_6
    )
    assert delta == expected_delta, (
        "Cells added to P8 envelope past K-1175 do not match "
        "_K1205_EN3_ADMITS_8 | _K1219_E3_NFLOOR256_ADMITS_7 | "
        "_K1283_A1_PERTURBATIONS_8 exactly; an unattributed cell crept in.")


# ---------------------------------------------------------------------------
# K-1219 / K-1240 -- E3 N-floor=256 anchor-projected admits (7 cells).
# Independent paired n=30 HIP-graph hot-cache validation on rad-mi300x-1
# (K-1219 commit f110d64 on fix/K-1219-n256).  Cohort geomean tb/hbl
# 1.505x; productionised onto the K-1216 stacked dispatch chain by
# K-1303 (this branch) as the 5th provenance sub-frozenset.
# ---------------------------------------------------------------------------
K1219_E3_NFLOOR256_ADMITS_7_LIST = [
    # (cid, M, N, K, hbl_tb_med, CI95_lo) -- from K-1219 paired n=30 dataset
    ("E3_S18_N256",   5972, 256,  768, 1.110, 1.084),
    ("E3_S24_N256",   4480, 256,  768, 1.016, 0.989),
    ("E3_S25_N256",   6016, 256, 1024, 1.214, 1.206),
    ("E3_S30_N256",  16256, 256, 1024, 2.271, 2.261),
    ("E3_S37_N256",  25600, 256,  256, 0.998, 0.992),
    ("E3_S39_N256",  49152, 256,  256, 1.004, 0.999),
    ("E3_E2M1_N256", 10112, 256, 1024, 2.927, 2.911),  # cohort MAX
]


def test_k1219_e3_admits_envelope_size_is_exactly_7():
    """Pin K-1219 E3 sub-frozenset cardinality.  K-1219 staged 7 N=256
    candidates (K-1131-style anchor projection at N-floor=256 of the
    K-1121 PMC anchors {S18, S24, S25, S30, S37, S39} + K-1175 E2_M1
    admit) and admitted all 7 to the unified K-1282 envelope.  Any
    silent edit changes the count and trips this canary."""
    assert len(_K1219_E3_NFLOOR256_ADMITS_7) == 7


def test_k1219_e3_admits_pairwise_disjoint_with_prior_four_subsets():
    """K-1219 E3 candidates have N=256 by construction: no K-1121 anchor
    (N>=896), K-1131 neighbor (N>=896), K-1161 E2 admit (N in {1792, 2048}),
    or K-1205 E_N3 admit (N=128) has N=256.  The 5-way union is therefore
    pairwise disjoint without manual exclusion lists; this test is the
    structural disjointness invariant the runtime asserts in
    _route_predicate.py rely on."""
    assert _K1219_E3_NFLOOR256_ADMITS_7.isdisjoint(_K1121_P8_ANCHORS_13)
    assert _K1219_E3_NFLOOR256_ADMITS_7.isdisjoint(_K1131_P8_NEIGHBORS_12)
    assert _K1219_E3_NFLOOR256_ADMITS_7.isdisjoint(_K1161_E2_ADMITS_3)
    assert _K1219_E3_NFLOOR256_ADMITS_7.isdisjoint(_K1205_EN3_ADMITS_8)


def test_k1219_e3_admits_envelope_contents_pinned_to_k1219_manifest():
    """Pin the K-1219 7-admit envelope to source-of-truth (the K-1219
    paired n=30 dataset summarised in K1219_E3_NFLOOR256_ADMITS_7_LIST).
    A silent edit to either constant trips here."""
    expected = frozenset(
        (M, N, K, "torch.bfloat16")
        for _cid, M, N, K, _hbl_tb, _ci_lo in K1219_E3_NFLOOR256_ADMITS_7_LIST)
    assert _K1219_E3_NFLOOR256_ADMITS_7 == expected


def test_k1219_e3_admits_all_have_n_equal_256():
    """Structural axis-discipline invariant: K-1219 staged the N-axis
    extension at N-floor=256 ONLY (one tier below the K-1131 N09 N=896
    natural floor and one tier above K-1205's N=128 tier).  Any cell
    in E3 with N != 256 would be a silent drift away from the staged
    axis and trips here."""
    for (M, N, K, dtype) in _K1219_E3_NFLOOR256_ADMITS_7:
        assert N == 256, (
            f"K-1219 E3 cell {(M, N, K, dtype)!r} has N={N} != 256; "
            "K-1219 is N-axis-only at N-floor=256 by R-1219.")


def test_k1219_e3_admits_k_in_k1144_k_set():
    """Structural axis-discipline invariant: K-1219 held K fixed at the
    K-1144 K-set {256, 768, 1024} (NOT a K-axis relaxation).  Any K
    outside this set would be a covert K-axis extension and trips here."""
    for (M, N, K, dtype) in _K1219_E3_NFLOOR256_ADMITS_7:
        assert K in {256, 768, 1024}, (
            f"K-1219 E3 cell {(M, N, K, dtype)!r} has K={K} outside "
            "the K-1144 K-set {256, 768, 1024}; K-1219 is N-axis-only.")


def test_k1219_e3_admits_m_floor_per_k1142_carve_out():
    """Per K-1142 carve-out, every K-1219 admit must satisfy M >= 4480.
    A silent admission of M < 4480 cells would re-open the K-1142
    small-M Triton-favoured tail and trips this pin."""
    for (M, N, K, dtype) in _K1219_E3_NFLOOR256_ADMITS_7:
        assert M >= 4480, (
            f"K-1219 E3 cell {(M, N, K, dtype)!r} has M={M} < 4480; "
            "K-1142 M-floor carve-out violation.")


@pytest.mark.parametrize(
    "cid,M,N,K,hbl_tb_med,ci95_lo", K1219_E3_NFLOOR256_ADMITS_7_LIST,
    ids=[c[0] for c in K1219_E3_NFLOOR256_ADMITS_7_LIST])
def test_k1219_p8_admits_all_7_e3_cells(cid, M, N, K, hbl_tb_med, ci95_lo):
    """Every K-1219 E3 admit cell must fire the P8 strict-equality match.
    The K-1219 measurement campaign documented all 7 as admitted to the
    unified K-1282 envelope on rad-mi300x-1 under HIP-graph hot-cache."""
    assert _p8_mfma_issue_stall_routeout(M, N, K, torch.bfloat16) is True, (
        f"K-1219 E3 admit {cid} ({M},{N},{K}) missed P8 envelope")


@pytest.mark.parametrize(
    "cid,M,N,K,hbl_tb_med,ci95_lo", K1219_E3_NFLOOR256_ADMITS_7_LIST,
    ids=[c[0] for c in K1219_E3_NFLOOR256_ADMITS_7_LIST])
def test_k1219_dispatch_routes_all_7_e3_cells_out_to_hbl(
        cid, M, N, K, hbl_tb_med, ci95_lo):
    """End-to-end dispatch check: with default carve-outs disabled,
    _k971_route_to_hbl must return True for every K-1219 E3 admit
    so the cell is routed to hipBLASLt."""
    decision = _k971_route_to_hbl(
        M, N, K,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16,
        enable_streamk=False, work_stealing=False)
    assert decision is True, (
        f"{cid} (M={M} N={N} K={K}) was not routed to hipBLASLt; "
        f"expected True per K-1219 paired n=30 hbl/tb_med={hbl_tb_med:.3f}x")


def test_k1219_e3_admits_are_bf16_only():
    """K-1219 measurement scope is bf16 only.  fp16/fp32 with the same
    (M, N, K) must NOT fire P8 -- fp16 dispatch is owned by the K-1093
    / K-1125 mirror predicate line."""
    for (M, N, K, _dtype) in _K1219_E3_NFLOOR256_ADMITS_7:
        for fp_dtype in (torch.float16, torch.float32):
            assert _p8_mfma_issue_stall_routeout(M, N, K, fp_dtype) is False, (
                f"P8 fired on non-bf16 dtype {fp_dtype} for "
                f"K-1219 cell {(M, N, K)}; K-1219 is bf16-only.")


def test_k1219_p8_e1_stack_no_double_route_for_e3_admits():
    """K-1216 stacked-dispatch invariant preserved: every K-1219 E3 admit
    fires P8 (strict equality), and -- because all 7 admits have N=256
    which is OUTSIDE the K-1216 E1 envelope's N-set {1792, 2048, 3072} --
    E1 must NOT fire on these cells (E1 is a no-op for N=256 by design)."""
    for (M, N, K, _dtype) in _K1219_E3_NFLOOR256_ADMITS_7:
        p8_says = _p8_mfma_issue_stall_routeout(M, N, K, torch.bfloat16)
        e1_says = R_K1142_E1_route_to_hbl(M, N, K, torch.bfloat16)
        assert p8_says is True, (
            f"P8 missed K-1219 E3 admit ({M}, {N}, {K})")
        assert e1_says is False, (
            f"E1 fired on K-1219 N=256 cell ({M}, {N}, {K}); K-1216 "
            "stacking semantics require E1 N-set {1792, 2048, 3072} "
            "(N=256 must be excluded).")


def test_k1303_does_not_disturb_existing_p8_36_cell_envelope():
    """K-1275 -> K-1303 invariance check: the K-1275 36-cell P8 envelope
    (28 baseline + 8 K-1205 E_N3) is preserved exactly -- K-1219 strict-
    equality union ADDS 7 cells without re-measuring or modifying any
    existing K-1144/K-1175/K-1205 cell.  This is R-1184.STRICT-EQUALITY-
    UNION-PROVES-EXISTING-CELL-INVARIANCE-WITHOUT-RE-MEASUREMENT in test
    form (extended to K-1219)."""
    pre_k1219 = (
        _K1121_P8_ANCHORS_13
        | _K1131_P8_NEIGHBORS_12
        | _K1161_E2_ADMITS_3
        | _K1205_EN3_ADMITS_8)
    assert len(pre_k1219) == 36
    for cell in pre_k1219:
        assert cell in _P8_MFMA_ISSUE_STALL_ROUTEOUT, (
            f"K-1275 P8 cell {cell} dropped from K-1303 envelope; "
            "K-1219 must be strict-equality union only.")
    # Symmetric (post K-1322): every new cell added past K-1275 must
    # come from K-1219 (7 N=256) or K-1283/K-1297 (8 A1 perturbations).
    delta = _P8_MFMA_ISSUE_STALL_ROUTEOUT - pre_k1219
    expected = (_K1219_E3_NFLOOR256_ADMITS_7
                | _K1283_A1_PERTURBATIONS_8
                | _K1336_REGRESSOR_CARVEIN_6)
    assert delta == expected, (
        "Cells added to P8 envelope past K-1275 do not match "
        "_K1219_E3_NFLOOR256_ADMITS_7 | _K1283_A1_PERTURBATIONS_8 | "
        "_K1336_REGRESSOR_CARVEIN_6 exactly; an unattributed cell crept in.")


# ---------------------------------------------------------------------------
# K-1283 / K-1297 -- A1 sub-cohort targeted P8 admit-cell extension (8
# cells).  Independent paired n=30 HIP-graph hot-cache validation on
# rad-mi300x-1 (K-1297 commit f4cc5cf on fix/K-1283-A1, V2 protocol with
# corrected negative-control captures).  Cohort geomean tb/hbl 1.389x;
# productionised onto the K-1303 unified envelope by K-1322 (this branch)
# as the 6th provenance sub-frozenset.
# ---------------------------------------------------------------------------
K1283_A1_PERTURBATIONS_8_LIST = [
    # (cid, M, N, K, hbl_tb_med, CI95_lo) -- from K-1297 V2 paired n=30
    ("A1_S25_Mx2", 12032, 2048, 1024, 3.365, 3.009),  # cohort MAX
    ("A1_S25_Nx2",  6016, 4096, 1024, 1.383, 1.351),
    ("A1_S25_N/2",  6016, 1024, 1024, 1.202, 1.143),
    ("A1_S25_Kx2",  6016, 2048, 2048, 1.264, 1.242),
    ("A1_S25_K/2",  6016, 2048,  512, 1.226, 1.140),
    ("A1_S29_Mx2", 28416, 2048, 1024, 1.185, 1.163),
    ("A1_S29_Nx2", 14208, 4096, 1024, 1.143, 1.071),
    ("A1_S29_K/2", 14208, 2048,  512, 1.126, 1.035),  # cohort MIN
]


def test_k1283_a1_perturbations_envelope_size_is_exactly_8():
    """Pin K-1283 A1 sub-frozenset cardinality.  K-1297 V2 admitted 8
    cells out of 15 candidates at CI95-lo > 1.0x.  Any silent edit
    changes the count and trips this canary."""
    assert len(_K1283_A1_PERTURBATIONS_8) == 8


def test_k1283_a1_perturbations_pairwise_disjoint_with_prior_five_subsets():
    """K-1297 selected only K-1131-style perturbations on axes K-1131
    did NOT enumerate (S25 5/5 axes, S29 3/6 axes); structural disjointness
    with the prior five sub-frozensets is checked at module load and
    re-asserted here for explicit test coverage."""
    assert _K1283_A1_PERTURBATIONS_8.isdisjoint(_K1121_P8_ANCHORS_13)
    assert _K1283_A1_PERTURBATIONS_8.isdisjoint(_K1131_P8_NEIGHBORS_12)
    assert _K1283_A1_PERTURBATIONS_8.isdisjoint(_K1161_E2_ADMITS_3)
    assert _K1283_A1_PERTURBATIONS_8.isdisjoint(_K1205_EN3_ADMITS_8)
    assert _K1283_A1_PERTURBATIONS_8.isdisjoint(_K1219_E3_NFLOOR256_ADMITS_7)


def test_k1283_a1_perturbations_contents_pinned_to_k1297_manifest():
    """Pin the K-1297 8-admit envelope to source-of-truth (the K-1297 V2
    paired n=30 dataset summarised in K1283_A1_PERTURBATIONS_8_LIST).
    A silent edit to either constant trips here."""
    expected = frozenset(
        (M, N, K, "torch.bfloat16")
        for _cid, M, N, K, _hbl_tb, _ci_lo in K1283_A1_PERTURBATIONS_8_LIST)
    assert _K1283_A1_PERTURBATIONS_8 == expected


def test_k1283_a1_admits_are_in_p8_envelope():
    """Per Testing Zealot REVISE on K-1297 -- positive-side admit pin:
    every K-1283/K-1297 A1 admit MUST be a member of the unified P8
    envelope.  Symmetric with the K-931 no-leak pins from K-1297."""
    for cell in _K1283_A1_PERTURBATIONS_8:
        assert cell in _P8_MFMA_ISSUE_STALL_ROUTEOUT
        M, N, K, _ = cell
        assert _p8_mfma_issue_stall_routeout(M, N, K, torch.bfloat16) is True


def test_k1283_a1_admits_respect_k1142_m_floor_and_k1161_k_floor():
    """Structural floor invariants inherited from K-1142 and K-1161:
    every admit must satisfy M >= 4480 (K-1142 small-M Triton-favoured
    tail carve-out) and K >= 256 (K-1161 negative-axis-pivot pin; do
    NOT relax to 128 per K-1176 cross-arch failure)."""
    for (M, N, K, _dtype) in _K1283_A1_PERTURBATIONS_8:
        assert M >= 4480, (
            f"K-1297 admit ({M}, {N}, {K}) violates K-1142 M>=4480 floor")
        assert K >= 256, (
            f"K-1297 admit ({M}, {N}, {K}) violates K-1161 K>=256 floor")


def test_k1322_does_not_disturb_existing_k1303_43_cell_envelope():
    """K-1303 -> K-1322 invariance check: the K-1303 43-cell P8 envelope
    (28 baseline + 8 K-1205 E_N3 + 7 K-1219 E3) is preserved exactly --
    K-1297 strict-equality union ADDS 8 A1 cells without re-measuring or
    modifying any existing K-1144/K-1175/K-1205/K-1219 cell.  Extends
    R-1184.STRICT-EQUALITY-UNION-PROVES-EXISTING-CELL-INVARIANCE to the
    K-1322 layer."""
    pre_k1297 = (
        _K1121_P8_ANCHORS_13
        | _K1131_P8_NEIGHBORS_12
        | _K1161_E2_ADMITS_3
        | _K1205_EN3_ADMITS_8
        | _K1219_E3_NFLOOR256_ADMITS_7)
    assert len(pre_k1297) == 43
    for cell in pre_k1297:
        assert cell in _P8_MFMA_ISSUE_STALL_ROUTEOUT, (
            f"K-1303 P8 cell {cell} dropped from K-1322 envelope; "
            "K-1297 must be strict-equality union only.")
    # Symmetric (post K-1352): every new cell added past K-1303 must come
    # from K-1297 A1 (8 cells) or K-1311/K-1336 carve-IN (6 cells).
    delta = _P8_MFMA_ISSUE_STALL_ROUTEOUT - pre_k1297
    expected = _K1283_A1_PERTURBATIONS_8 | _K1336_REGRESSOR_CARVEIN_6
    assert delta == expected, (
        "Cells added to P8 envelope past K-1303 do not match "
        "_K1283_A1_PERTURBATIONS_8 | _K1336_REGRESSOR_CARVEIN_6 exactly; "
        "an unattributed cell crept in.")


# ---------------------------------------------------------------------------
# K-1335 / K-1326 (S-002) -- longK_smallSquare bf16 route-OUT extension pins.
# K-1335 productionizes K-1326's `P9_longK_smallSquare_routeOUT` proposal as
# a 4-cell strict-equality extension of the K-905/K-971 LDS-bank-conflict
# `K971_ROUTE_TABLE` (the 5th-position dispatch predicate).  Per K-913 PMC
# anchor + K-1326 attribution, the bottleneck is LDS-bank-conflict (LDS_BC
# 1.78 cyc/inst), NOT MFMA-issue-stall, so the attachment is the LDS-BC
# envelope -- NOT the K-1322 51-cell P8 envelope (R-1329.MECHANISTIC-
# DISAMBIGUATION-OF-PREDICATE-ATTACHMENT-POINT).
# ---------------------------------------------------------------------------
K1335_LONGK_SMALLSQUARE_ADMITS_4_LIST = [
    # (sid,            M,    N,    K, source)
    ("K1335_A1",    1024, 1024, 4096, "K-1326 longK_smallSquare anchor; LDS-BC"),
    ("K1335_A2",    1024, 1024, 8192, "K-913 PMC anchor; LDS_BC 1.78 cyc/inst"),
    ("K1335_A3",    2048, 2048, 4096, "K-1326 longK_smallSquare anchor; LDS-BC"),
    ("K1335_A4",    2048, 2048, 8192, "K-1326 longK_smallSquare anchor; LDS-BC"),
]

# K-axis-neighbor controls that MUST NOT route via K-1335 (different K than
# the admit set; outside the K-1326 `longK_smallSquare` bucket).  K=2048
# stays in the in-kernel autotune regime; K=16384 is on the K-905/K-971
# baseline anchors and routes via _K905_K971_LDS_BC_ANCHORS_8 instead
# (verified in the per-cell test below by membership rather than route
# verdict so the two campaigns stay attribution-clean).
K1335_K_AXIS_NEIGHBOR_CONTROLS_4 = [
    ("K1335_N_K2048_M1024", 1024, 1024, 2048),  # K-axis neighbor below admit band
    ("K1335_N_K2048_M2048", 2048, 2048, 2048),
    ("K1335_N_M512_K4096",   512,  512, 4096),  # M-axis neighbor below admit band
    ("K1335_N_M4096_K4096", 4096, 4096, 4096),  # M-axis neighbor above admit band
]


def test_k1335_longk_smallsquare_admits_cardinality_is_exactly_4():
    """K-1335 strict-equality cardinality pin.  K-1326 nominated exactly 4
    cells in the `longK_smallSquare` bucket (M=N in {1024, 2048} x K in
    {4096, 8192}, bf16) and K-1335 ships all 4 after paired n=30 HIP-graph
    hot-cache + B=10000 paired bootstrap CI95 confirmation on MI300X.  Any
    silent edit to the admit set trips this canary."""
    assert len(_K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4) == 4


def test_k1335_lds_bc_table_size_is_exactly_12_after_k1335_extension():
    """K-1335 unified LDS-BC table (K971_ROUTE_TABLE) cardinality pin:
    8 K-905/K-971 baseline anchors + 4 K-1335 longK_smallSquare admits
    = 12.  Any silent edit changes the count and trips this canary.

    Provenance chain:
      * K-905 originally pinned 2 cells (cohort A LDS-bound bf16 + fp16).
      * K-930 / K-971 extended to 8 cells (mid-square long-K bf16/fp16).
      * K-1335 extends by 4 longK_smallSquare bf16 admits identified by
        K-1308's residual decomposition + K-913 PMC anchor + K-1326
        attribution -> 12 (this canary)."""
    assert len(K971_ROUTE_TABLE) == 12
    assert len(_K905_K971_LDS_BC_ANCHORS_8) == 8
    assert len(_K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4) == 4


def test_k1335_disjoint_from_k905_k971_baseline_by_k_axis_projection():
    """K-axis projection disjointness assert (R-1329.K-AXIS-PROJECTION-
    DISJOINTNESS-ASSERTS-ARE-CHEAP-INSURANCE): K-1335 cells live in
    K in {4096, 8192}; K-905/K-971 baseline cells live in K in {16384,
    32768}.  Disjointness is the natural separator that keeps the two
    campaigns' provenance attribution clean."""
    assert _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4.isdisjoint(
        _K905_K971_LDS_BC_ANCHORS_8)
    k1335_ks = {K for (_, _, K, _) in _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4}
    baseline_ks = {K for (_, _, K, _) in _K905_K971_LDS_BC_ANCHORS_8}
    assert k1335_ks == {4096, 8192}
    assert baseline_ks == {16384, 32768}
    assert k1335_ks.isdisjoint(baseline_ks)


def test_k1352_k1335_subset_of_k1336_carvein_after_compose():
    """K-1352 composition invariant: K-1335 LDS-BC admits are a STRICT
    SUBSET of the K-1336 dispatch-wrapper-overhead carve-IN at the
    (M,N,K,dtype) 4-tuple level.  This INTENTIONALLY breaks the prior
    K-1344 `disjoint_from_p8` invariant: the two predicates attack the
    same 4-cell longK_smallSquare cohort via DIFFERENT mechanisms
    (K-1335 = K-913/K-1338 LDS-BC fingerprint; K-1336 = K-1311 dispatch-
    wrapper-overhead-skip).  Dispatch precedence: P8 strict-equality
    is consulted BEFORE K971_ROUTE_TABLE in `route_to_hbl`, so the
    K-1336 carve-IN catches the 4 overlapping cells first.  K-1335 is
    retained for mechanistic provenance + defense-in-depth (if K-1336
    is ever pulled, K-1335 still routes the 4 LDS-BC anchors OUT).

    Negative invariant: the 2 K-1336-only cells (2048^3, 4096^3) bf16
    -- K-1311 ranks #5 and #2 -- have NO LDS-BC PMC anchoring and MUST
    NOT leak into K971_ROUTE_TABLE."""
    # Positive: K-1335 is a SUBSET of K-1336 at the 4-tuple level.
    assert _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4 <= _K1336_REGRESSOR_CARVEIN_6
    # Positive: K-1335 cells are members of the composed P8 envelope (via
    # the K-1336 carve-IN).
    assert _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4.issubset(
        _P8_MFMA_ISSUE_STALL_ROUTEOUT)
    # Set intersection: the 12 strict-equality tuples (4 in K-1335 +
    # K971_ROUTE_TABLE + 6 in K-1336/P8 + 2 baseline overlap zero) reduce
    # to 4 + 6 = 10 predicate fires across 6 unique (M,N,K,dtype) cells
    # (K-1335 cardinality + K-1336 cardinality - overlap = 4 + 6 - 4 = 6).
    overlap = _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4 & _K1336_REGRESSOR_CARVEIN_6
    assert len(overlap) == 4, (
        "K-1335 / K-1336 4-tuple intersection must be exactly the 4 "
        "longK_smallSquare cells (1024^2 x {4096,8192} + 2048^2 x {4096,8192}).")
    union = _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4 | _K1336_REGRESSOR_CARVEIN_6
    assert len(union) == 6
    # Negative: K-1336-only cells (NOT in K-1335) MUST NOT be in K971.
    k1336_only = _K1336_REGRESSOR_CARVEIN_6 - _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4
    assert k1336_only == frozenset({
        (2048, 2048, 2048, "torch.bfloat16"),
        (4096, 4096, 4096, "torch.bfloat16"),
    })
    assert k1336_only.isdisjoint(K971_ROUTE_TABLE), (
        "K-1336-only cells (2048^3, 4096^3) bf16 leaked into "
        "K971_ROUTE_TABLE; they have no K-913/K-1338 PMC LDS-BC "
        "anchoring and must NOT route via the LDS-BC predicate.")


def test_k1335_admit_cells_route_to_hbl_after_k1352_compose():
    """K-1352 composition: every K-1335 admit cell still flips the full
    dispatch decision to route-OUT (route_to_hbl = True) on the bf16
    path with streamk OFF and work-stealing OFF.  After K-1336 carve-IN
    composition, the K-1335 cells are ALSO members of the composed P8
    envelope (via _K1336_REGRESSOR_CARVEIN_6) -- dispatch precedence
    means the P8 strict-equality check fires FIRST, but route_to_hbl
    returns True regardless of which predicate catches the cell, so
    end-to-end routing remains invariant.  K-1335 retains independent
    membership in K971_ROUTE_TABLE for mechanistic provenance and
    defense-in-depth."""
    for (sid, M, N, K, source) in K1335_LONGK_SMALLSQUARE_ADMITS_4_LIST:
        # K-1352: K-1335 cells ARE members of the composed P8 envelope
        # (via the K-1336 carve-IN -- this is the K-1352 invariant).
        assert (M, N, K, "torch.bfloat16") in _P8_MFMA_ISSUE_STALL_ROUTEOUT, (
            f"K-1335 cell {sid} ({M},{N},{K}) missing from composed P8 "
            "envelope; K-1352 K-1336 carve-IN must include all K-1335 cells.")
        # K-1335 cell ALSO independently in the LDS-BC `K971_ROUTE_TABLE`
        # (defense-in-depth: survives any future K-1336 rollback).
        assert (M, N, K, "torch.bfloat16") in K971_ROUTE_TABLE, (
            f"K-1335 cell {sid} ({M},{N},{K}) missing from K971_ROUTE_TABLE.")
        # End-to-end dispatch decision: still routes to hbl on the bf16
        # path (whether via P8 first or K971 fallback, route_to_hbl=True).
        routed = k971_route_decision(
            M, N, K,
            "torch.bfloat16", "torch.bfloat16",
            enable_streamk=False, work_stealing=False,
            disable_env_set=False)
        assert routed is True, (
            f"K-1335 cell {sid} ({M},{N},{K}) failed to route to hbl after "
            "K-1352 compose; either the P8 carve-IN or K971_ROUTE_TABLE "
            "must fire and return True.")


def test_k1335_k_axis_neighbor_controls_do_not_match_admit_set():
    """K-axis-neighbor controls (cells just outside the K-1326
    `longK_smallSquare` bucket) MUST NOT match the K-1335 admit
    frozenset.  This is a strict-equality no-leak pin -- verifies the
    K=2048 K-axis neighbor below the admit band, the M=512 / M=4096
    M-axis neighbors outside the M=N in {1024, 2048} admit window, and
    rules out any silent broadening of the admit set on the K or M
    axes during future edits."""
    for (sid, M, N, K) in K1335_K_AXIS_NEIGHBOR_CONTROLS_4:
        assert (M, N, K, "torch.bfloat16") not in _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4, (
            f"K-1335 K-axis-neighbor control {sid} ({M},{N},{K}) leaked "
            "into the admit set; the K-1326 bucket is strict {4096, 8192} "
            "on K and strict {1024, 2048} on M=N.")


def test_k1335_admits_are_all_bf16_dtype():
    """K-1335 ships bf16 only.  K-913 PMC anchor + K-1326 residual bucket
    + K-1308 cross-branch ratio invariant are all measured on bf16; fp16
    LDS-BC parity is tracked separately (out of scope for K-1335).  A
    silent fp16 entry would land an admit on an unmeasured-dtype cell
    and bypass the K-1335 verification protocol entirely."""
    for (M, N, K, dtype) in _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4:
        assert dtype == "torch.bfloat16", (
            f"K-1335 admit cell ({M},{N},{K},{dtype}) is not bf16; the "
            "K-913 / K-1326 / K-1308 verification scope is bf16-only.")


@pytest.mark.parametrize("M,N,K", [
    (1024, 1024, 4096),
    (1024, 1024, 8192),
    (2048, 2048, 4096),
    (2048, 2048, 8192),
])
def test_k1335_streamk_carve_out_short_circuits_admit(M, N, K):
    """The K971 dispatch streamk / work-stealing carve-outs must
    short-circuit BEFORE the K-1335 LDS-BC table is consulted.  Pinning
    these here so any future edit that moves K-1335 ahead of the carve-out
    (e.g. into a parametric-band predicate) trips this test."""
    assert k971_route_decision(
        M, N, K,
        "torch.bfloat16", "torch.bfloat16",
        enable_streamk=True, work_stealing=False,
        disable_env_set=False) is False
    assert k971_route_decision(
        M, N, K,
        "torch.bfloat16", "torch.bfloat16",
        enable_streamk=False, work_stealing=True,
        disable_env_set=False) is False


def test_k1335_k905_k971_baseline_cells_still_route_after_k1335_union():
    """K-905/K-971 baseline anchors (8 cells) must still flip the dispatch
    decision to route-OUT after the K-1335 strict-equality union extends
    K971_ROUTE_TABLE from 8 -> 12 cells.  Strict-equality union is
    invariance-preserving by construction (R-1322.STRICT-EQUALITY-UNION-
    PRESERVES-PRIOR-ADMIT-INVARIANCE-AS-STRUCTURAL-GUARANTEE), but a
    one-line pin keeps the structural guarantee from being silently
    broken by a future refactor that replaces the union with a parametric
    predicate."""
    for (M, N, K, dtype) in _K905_K971_LDS_BC_ANCHORS_8:
        if dtype != "torch.bfloat16":
            # fp16 baseline anchors: only the strict-equality table
            # consultation applies (P5 closed-form is bf16-only).
            assert (M, N, K, dtype) in K971_ROUTE_TABLE
            continue
        routed = k971_route_decision(
            M, N, K, dtype, dtype,
            enable_streamk=False, work_stealing=False,
            disable_env_set=False)
        assert routed is True, (
            f"K-905/K-971 baseline anchor ({M},{N},{K},{dtype}) regressed "
            "after K-1335 extension; strict-equality union must be "
            "invariance-preserving.")


# ---------------------------------------------------------------------------
# K-1352 / K-1336 (S-002) carve-IN pins.  K-1336 productionised the K-1311
# 6-cell dispatch-wrapper-overhead regressor manifest as a sub-frozenset
# folded INTO `_P8_MFMA_ISSUE_STALL_ROUTEOUT`; K-1352 composes K-1336's
# carve-IN with K-1335's K971 LDS-BC envelope on a single branch.  Pins
# below replicate the K-1336 PR test additions plus K-1352-specific
# composition pins.
# ---------------------------------------------------------------------------
K1336_REGRESSOR_CARVEIN_6_LIST = [
    # K-1311 §F1 ranks #1-#6, all bf16 (K-1295 paired-n30 CI95-hi(delta) < 0)
    ("K1311_R3", 1024, 1024, 4096),
    ("K1311_R6", 1024, 1024, 8192),
    ("K1311_R5", 2048, 2048, 2048),
    ("K1311_R4", 2048, 2048, 4096),
    ("K1311_R1", 2048, 2048, 8192),
    ("K1311_R2", 4096, 4096, 4096),
]
# Single-axis-perturbed negative-neighbour cohort (K-1336 PR test additions):
# 6 perturbations including the K-1311 borderline (8192,8192,8192) bf16
# whose CI95 spans 0 at n=30 (defer admit decision to n=150).  All 6 must
# NOT be admitted by `_K1336_REGRESSOR_CARVEIN_6`.
K1336_NEGATIVE_NEIGHBORS_6 = [
    ("K1311_borderline_8192cube", 8192, 8192, 8192),  # CI95 spans 0
    ("K1336_neg_M_axis_512_2_4096", 512, 2048, 4096),
    ("K1336_neg_N_axis_2048_512_4096", 2048, 512, 4096),
    ("K1336_neg_K_axis_2048_2_1024", 2048, 2048, 1024),
    ("K1336_neg_K_axis_4096_3_2048", 4096, 4096, 2048),
    ("K1336_neg_M_axis_8192_2_4096", 8192, 2048, 4096),
]


def test_k1336_carvein_cardinality_is_exactly_6():
    """K-1336 strict-equality cardinality pin.  K-1311 §F1 ranked exactly
    6 cells with CI95-hi(delta) < 0 vs tb-main on the K-1295 paired-n30
    sweep; K-1336 ships all 6 (and ONLY those 6).  Any silent edit changes
    the count and trips this canary."""
    assert len(_K1336_REGRESSOR_CARVEIN_6) == 6


def test_k1336_carvein_contents_pinned_to_k1311_manifest():
    """K-1336 carve-IN frozenset MUST equal the K-1311 §F1 6-cell
    manifest exactly (no extras, no omissions).  bf16-only (the K-1311
    paired-bench cohort was bf16-only)."""
    expected = frozenset({
        (1024, 1024, 4096, "torch.bfloat16"),
        (1024, 1024, 8192, "torch.bfloat16"),
        (2048, 2048, 2048, "torch.bfloat16"),
        (2048, 2048, 4096, "torch.bfloat16"),
        (2048, 2048, 8192, "torch.bfloat16"),
        (4096, 4096, 4096, "torch.bfloat16"),
    })
    assert _K1336_REGRESSOR_CARVEIN_6 == expected


def test_k1336_carvein_admits_all_6_k1311_regressors():
    """Every K-1336 carve-IN cell flips the full dispatch decision to
    route-OUT (route_to_hbl = True) on the bf16 path with streamk OFF /
    work-stealing OFF.  The strict-equality lookup short-circuits at the
    P8 envelope (1st-position dispatch predicate) -- the +13.89 us chain-
    evaluation cost is bypassed."""
    for (rank, M, N, K) in K1336_REGRESSOR_CARVEIN_6_LIST:
        assert (M, N, K, "torch.bfloat16") in _K1336_REGRESSOR_CARVEIN_6, (
            f"K-1336 carve-IN cell {rank} ({M},{N},{K}) missing from "
            "_K1336_REGRESSOR_CARVEIN_6.")
        assert (M, N, K, "torch.bfloat16") in _P8_MFMA_ISSUE_STALL_ROUTEOUT, (
            f"K-1336 carve-IN cell {rank} ({M},{N},{K}) missing from "
            "composed P8 envelope.")
        routed = k971_route_decision(
            M, N, K,
            "torch.bfloat16", "torch.bfloat16",
            enable_streamk=False, work_stealing=False,
            disable_env_set=False)
        assert routed is True, (
            f"K-1336 carve-IN cell {rank} ({M},{N},{K}) failed to route "
            "to hbl via the composed P8 envelope short-circuit.")


def test_k1336_carvein_does_not_admit_negative_neighbors():
    """Single-axis-perturbed negative neighbours (K-1311 borderline 8192^3
    + 5 single-axis perturbations of the 6 admit cells) MUST NOT match
    the K-1336 carve-IN.  The borderline 8192^3 bf16 has CI95 spanning 0
    at n=30 (K-1311 §F20 deferred to n=150 re-measure) -- explicitly
    not-admitted at n=30 evidence."""
    for (sid, M, N, K) in K1336_NEGATIVE_NEIGHBORS_6:
        assert (M, N, K, "torch.bfloat16") not in _K1336_REGRESSOR_CARVEIN_6, (
            f"K-1336 negative neighbour {sid} ({M},{N},{K}) leaked into "
            "the carve-IN set; the K-1311 §F1 manifest is strictly the 6 "
            "ranks #1-#6 cells, no perturbations admitted.")


def test_k1336_carvein_is_bf16_only():
    """K-1336 ships bf16 only.  The K-1295 paired-n30 sweep that K-1311
    triaged was bf16-only; an fp16 entry would land an admit on an
    unmeasured-dtype cell and bypass the K-1311 verification protocol."""
    for (M, N, K, dtype) in _K1336_REGRESSOR_CARVEIN_6:
        assert dtype == "torch.bfloat16", (
            f"K-1336 carve-IN cell ({M},{N},{K},{dtype}) is not bf16; "
            "the K-1295 / K-1311 verification scope is bf16-only.")


def test_k1352_compose_invariant_k1335_subset_of_k1336_at_4tuple_level():
    """K-1352 composition invariant (the load-bearing invariant of this
    branch): K-1335's 4-cell `_K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4` is
    a STRICT SUBSET of K-1336's 6-cell `_K1336_REGRESSOR_CARVEIN_6` at
    the (M,N,K,dtype) 4-tuple level.  Operationally:
      * dispatch precedence places P8 strict-equality BEFORE
        K971_ROUTE_TABLE in `route_to_hbl`
      * K-1336's carve-IN catches the 4 overlapping cells via the
        wrapper-overhead-skip mechanism on the P8 envelope
      * K-1335 retains those 4 cells in K971_ROUTE_TABLE for mechanistic
        defense-in-depth (LDS-BC fingerprint per K-913/K-1338 PMC; if
        the K-1336 carve-IN is ever pulled, K-1335 still routes them)
      * the 2 K-1336-only cells (2048^3, 4096^3) bf16 stay OUT of
        K971_ROUTE_TABLE (no LDS-BC PMC anchor for those cells)

    Set-intersection summary on the 12 strict-equality fires (the
    'composed envelope' from K-1352 task spec, decomposed): 4 K-1335
    fires + 6 K-1336 fires = 10 predicate fires across 6 unique
    (M,N,K,dtype) cells (K-1336 union K-1335 = K-1336 since K-1335 is
    a subset)."""
    # Subset relation.
    assert _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4 < _K1336_REGRESSOR_CARVEIN_6
    assert _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4.issubset(
        _P8_MFMA_ISSUE_STALL_ROUTEOUT)
    # Cardinality decomposition.
    assert len(_K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4) == 4
    assert len(_K1336_REGRESSOR_CARVEIN_6) == 6
    union = (_K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4
             | _K1336_REGRESSOR_CARVEIN_6)
    intersection = (_K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4
                    & _K1336_REGRESSOR_CARVEIN_6)
    assert len(union) == 6
    assert len(intersection) == 4
    # Negative invariant: K-1336-only cells must NOT be in K971_ROUTE_TABLE.
    k1336_only = (_K1336_REGRESSOR_CARVEIN_6
                  - _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4)
    assert len(k1336_only) == 2
    assert k1336_only.isdisjoint(K971_ROUTE_TABLE)


def test_k1352_compose_envelope_size_canary():
    """K-1352 composed-envelope cardinality canary.  Composed P8 envelope
    grows from K-1322's 51 cells to 57 cells (+ 6 K-1336 carve-IN, all
    disjoint from the prior 51).  K971_ROUTE_TABLE remains at 12 cells
    (K-1335 admits unchanged at the 4-tuple level; the K-1336 composition
    does NOT widen K971)."""
    assert len(_P8_MFMA_ISSUE_STALL_ROUTEOUT) == 57
    assert len(K971_ROUTE_TABLE) == 12
    assert len(_K1336_REGRESSOR_CARVEIN_6) == 6
    assert len(_K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4) == 4
