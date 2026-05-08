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
    R_K1142_E1_route_to_hbl,
    is_k1121_e1_admit_safe,
    K1142_E1_NS,
    K1142_E1_KS,
    K1142_E1_M_FLOOR,
    K1142_E1_K_FLOOR,
    K971_ROUTE_TABLE,
    _K1109_P6_K1074_ALLOWLIST,
    _K1109_P6_K1074_REGRESSION_EXCLUSIONS,
    _K1109_P6_ALLOWLIST_ENV,
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

# Tuples where K-905 anchor table intentionally routes to hbl despite the
# K-950 LAND verdict (K-905 measurement overrides — documented in the PR
# description).
K950_ANCHOR_COLLISIONS = frozenset({
    (1024, 1024, 16384, "torch.bfloat16"),  # L07
    (1024, 1024, 16384, "torch.float16"),   # L09
    (2048, 2048, 16384, "torch.float16"),   # L06
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
def test_p6_admit_supersedes_only_when_e1_does_not_fire(cid, M, N, K):
    """K-1151 (S-002) — K-1121 pivot inversion of the K-1109 P6 short-circuit.

    Pre-K-1151: P6 admit short-circuited route-OUT for S24 / S29 so they
    dispatched in-kernel with waves_per_eu=1 — at K-1121 paired n=30 this
    was confirmed slower than hipBLASLt (S24 hbl/tb=1.304x, S29 1.352x)
    and was the empirical root cause of the K-1121 pivot.

    Post-K-1151: the productionised K-1142 E1 envelope route-OUT precedes
    the P6 admit gate, so S24 and S29 (which sit inside E1 with M=4480 and
    M=14208 respectively) route OUT to hipBLASLt regardless of P6's admit
    verdict.  The P6 admit predicate STILL fires structurally on these
    cells (R_K1037_P6_admit_wpeu1 returns True so the K-1109 NULL-RESULT
    pins continue to hold) but it is no longer load-bearing for the route
    decision on the K-1121 cohort.

    This test pins the K-1151 inversion: P6-positive cells inside the
    productionised E1 envelope route OUT (route_to_hbl == True), while
    the underlying P6 admit predicate continues to classify them as
    structural P6 positives (separate test below)."""
    # Both S24 and S29 are inside E1 by construction.
    assert R_K1142_E1_route_to_hbl(M, N, K, torch.bfloat16) is True, (
        f"K-1142 E1 must fire on K-1037 P6 positive {cid} ({M},{N},{K}) "
        f"so K-1151 routes the K-1121-pivoted cell OUT")
    # K-1151 full-dispatch: routes OUT (E1 supersedes P6 admit).
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is True, (
        f"K-1151 productionised E1 must route P6-positive K-1121 anchor "
        f"{cid} ({M},{N},{K}) OUT (supersedes K-1037 P6 admit)")
    # P6 admit predicate STILL fires structurally — the K-1109 NULL-RESULT
    # pin tests below depend on this remaining True.
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is True, (
        f"R_K1037_P6_admit_wpeu1 must still classify {cid} as a P6 "
        f"positive — only the dispatch consequence changes under K-1151")


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
# K-1151 R_K1142_E1_route_to_hbl — productionised K-1142 E1 envelope
# (axis-aligned structural extrapolation of the K-1121 13-cell strict-equality
# cohort) with the K-1142 minimum two-axis carve-out.
#
# Pins:
#   (a) all 13 K-1121 anchors fire True (route OUT to hipBLASLt)
#   (b) the 2 K-1142 inverse-predicate FPs fire False (excluded by carve-out)
#   (c) the 2 K-1131 HBL-leaning E1 admit candidates fire True
#   (d) bf16-only carve-out (fp16/fp32 short-circuit to False)
#   (e) full-dispatch composition: E1 route-OUT supersedes K-1037 P6 admit
#       so S24 / S29 (which P6 would otherwise admit back to in-kernel)
#       route OUT to hipBLASLt — preserves the K-1121 pivot at runtime.
#   (f) is_k1121_e1_admit_safe helper rejects the 2 FPs and accepts every
#       anchor — the natural unit-test guard for any future structural
#       relaxation of the K-1142 carve-out.
# ---------------------------------------------------------------------------

# K-1121 anchor cells (paired n=30 K-1142 reconfirmation, K-1143 cross-arch
# reconfirmation on gfx950 — see knowledge/k1142_*.md and k1143_*.md).
K1121_ANCHORS_13 = [
    # (sid, M, N, K, source-cohort)
    ("S18",  5972, 1792,  768, "K-1031"),
    ("S24",  4480, 3072,  768, "K-1051"),  # also fires K-1037 P6 envelope B
    ("S25",  6016, 2048, 1024, "K-1051"),
    ("S26",  8064, 2048, 1024, "K-1031"),
    ("S29", 14208, 2048, 1024, "K-1051"),  # also fires K-1037 P6 envelope A
    ("S30", 16256, 2048, 1024, "K-1051"),
    ("S31", 18304, 2048, 1024, "K-1031"),
    ("S32", 20352, 2048, 1024, "K-1031"),
    ("S33", 22400, 2048, 1024, "K-1031"),
    ("S34", 24448, 2048, 1024, "K-1031"),
    ("S35", 26496, 2048, 1024, "K-1031"),
    ("S37", 25600, 2048,  256, "K-1031"),
    ("S39", 49152, 2048,  256, "K-1031"),
]

# K-1142 inverse-predicate false positives — cells inside the natural
# axis-aligned envelope E1 that hipBLASLt does NOT actually win at paired
# n=30 (FP1 is a 2.7% slowdown TB-TIE; FP2 is +1.5% TB-TIE).  Both must
# remain route-IN under the K-1142 carve-out (M >= 4480) so productionising
# E1 does not silently regress them.
K1142_INVERSE_PRED_FPS = [
    ("FP1",  256, 2048, 256, 0.973, "TB-TIE — −2.7% per K-1142 paired n=30"),
    ("FP2", 2048, 1792, 256, 1.015, "TB-TIE — +1.5% per K-1142 paired n=30"),
]

# K-1131 HBL-leaning E1 admit candidates — cells inside E1 that K-1131
# held-out neighbour validation confirmed as productionisation-safe
# (paired-n speedup >= K-1007 1.05x admission floor).  Productionised E1
# must continue to route them OUT — otherwise the +3.23% K-1142 geomean
# uplift on the K-931 ∩ E1 cohort is lost.  The exact (M, N, K) values
# fall inside the K-1121 anchor convex hull on the M axis (between S18
# and S39); the same predicate that fires on the anchors fires on these.
K1131_E1_ADMIT_CANDIDATES = [
    # (cid, M, N, K) -- both cells share the K=1024 / N=2048 family and
    # sit clear of the M=4480 carve-out.
    ("K1131A", 10112, 2048, 1024),
    ("K1131B", 12160, 2048, 1024),
]

# K-931 always-uncovered "neighbour" cells OUTSIDE the K-1142 envelope on
# at least one axis — used to pin that the productionised E1 predicate
# does NOT reach beyond the K-1142 audit (no envelope drift on
# unaudited cells).  Mix of (a) E1 N-axis miss, (b) E1 K-axis miss,
# (c) carve-out M-floor miss, (d) carve-out fires but cell is wider than
# E1 — they all must fire False under R_K1142_E1_route_to_hbl.
K931_NEIGHBOURS_OUT_OF_E1 = [
    ("N01",  1024, 2048, 1240),  # K=1240 outside E1 K-set
    ("N02",  2304, 2048, 4800),  # K=4800 outside E1 K-set (S04)
    ("N03",   768, 1792, 5972),  # K=5972 outside (S17)
    ("N04",  1024, 2048, 6016),  # K=6016 outside (S22)
    ("N05",   768, 3072, 4480),  # K=4480 outside (S21)
    ("N06",   512,  192, 2048),  # N=192 outside E1 N-set (S10)
    ("N07",    30, 786432, 200), # N=786432 outside (S36)
    ("N08",   256, 1792, 2048),  # K=2048 outside (S05)
    ("N09",   736, 1792,  736),  # K=736 outside (S06)
    ("N10",  1024, 2048, 8064),  # K=8064 outside (S23)
]


@pytest.mark.parametrize("sid,M,N,K,src", K1121_ANCHORS_13,
                         ids=[c[0] for c in K1121_ANCHORS_13])
def test_k1142_e1_fires_on_k1121_anchors(sid, M, N, K, src):
    """Every K-1121 strict-equality anchor must fire under the
    productionised K-1142 E1 envelope predicate so K-1151 ships the same
    route-OUT verdicts that K-1121's strict-equality table produced at
    paired n=30."""
    assert R_K1142_E1_route_to_hbl(M, N, K, torch.bfloat16) is True, (
        f"K-1142 E1 missed K-1121 anchor {sid} ({M},{N},{K}) — "
        f"productionisation regression: K-1121 paired-n=30 verdict not preserved")


@pytest.mark.parametrize("fid,M,N,K,hbl_over_tb,note", K1142_INVERSE_PRED_FPS,
                         ids=[c[0] for c in K1142_INVERSE_PRED_FPS])
def test_k1142_e1_excludes_inverse_predicate_fps(fid, M, N, K, hbl_over_tb, note):
    """Both K-1142 inverse-predicate false positives must be EXCLUDED by
    the productionised E1 predicate — they sit inside E1 by N/K but fail
    the M >= 4480 carve-out (K-1142 §5).  Re-admitting either cell
    silently re-introduces the K-1142 paired-n=30 FP and is a regression
    of the K-1151 brief's "no cell regresses >2%" gate (FP1 is the −2.7%
    case)."""
    assert R_K1142_E1_route_to_hbl(M, N, K, torch.bfloat16) is False, (
        f"K-1142 inverse-predicate FP {fid} ({M},{N},{K}) hbl/tb={hbl_over_tb} "
        f"would be re-admitted ({note}) — K-1142 carve-out violated")
    # is_k1121_e1_admit_safe is the symmetric guardrail
    assert is_k1121_e1_admit_safe(M, N, K, torch.bfloat16) is False, (
        f"is_k1121_e1_admit_safe should reject K-1142 FP {fid}")


@pytest.mark.parametrize("cid,M,N,K", K1131_E1_ADMIT_CANDIDATES,
                         ids=[c[0] for c in K1131_E1_ADMIT_CANDIDATES])
def test_k1142_e1_fires_on_k1131_admit_candidates(cid, M, N, K):
    """K-1131 held-out neighbour validation confirmed these cells as
    HBL-WIN at paired n=30; the productionised E1 envelope must continue
    to route them OUT for the K-1142 +3.23% geomean uplift to land."""
    assert R_K1142_E1_route_to_hbl(M, N, K, torch.bfloat16) is True, (
        f"K-1131 E1 admit candidate {cid} ({M},{N},{K}) not routed OUT — "
        f"loses K-1142 K-931 ∩ E1 cohort uplift")


@pytest.mark.parametrize("nid,M,N,K", K931_NEIGHBOURS_OUT_OF_E1,
                         ids=[c[0] for c in K931_NEIGHBOURS_OUT_OF_E1])
def test_k1142_e1_does_not_reach_beyond_envelope(nid, M, N, K):
    """K-931 always-uncovered cells OUTSIDE E1 must NOT be routed by the
    productionised E1 predicate — pins K-1151 to the K-1142 audit scope
    so no unaudited cell silently inherits a route-OUT verdict."""
    assert R_K1142_E1_route_to_hbl(M, N, K, torch.bfloat16) is False, (
        f"K-1142 E1 reached beyond audited envelope on {nid} ({M},{N},{K})")


def test_k1142_e1_is_bf16_only():
    """fp16 / fp32 / int8 must all short-circuit to False — K-1142 audit
    scope is bf16-only (mirrors K-1121 / K-1051 / K-1031 ground truth).
    The K-1093 fp16 cohort is governed by its own strict-equality table."""
    M, N, K = 14208, 2048, 1024  # S29 — would fire under bf16
    assert R_K1142_E1_route_to_hbl(M, N, K, torch.bfloat16) is True
    assert R_K1142_E1_route_to_hbl(M, N, K, torch.float16) is False
    assert R_K1142_E1_route_to_hbl(M, N, K, torch.float32) is False
    assert R_K1142_E1_route_to_hbl(M, N, K, torch.int8) is False


def test_k1142_e1_carveout_floors_match_kb_constants():
    """Pin K-1142's published carve-out floors so a future PR cannot
    silently relax them.  The M-floor is the load-bearing axis (anchor
    minimum 4480 = S24's M); the K-floor is structural (E1's K-axis lower
    bound) and excludes nothing in the K-1142 audit per §5.1."""
    assert K1142_E1_M_FLOOR == 4480
    assert K1142_E1_K_FLOOR == 256
    assert K1142_E1_NS == frozenset({1792, 2048, 3072})
    assert K1142_E1_KS == frozenset({256, 768, 1024})


def test_k1142_e1_supersedes_p6_admit_on_s24_s29():
    """K-1121 pivot — direct hipBLASLt route-OUT supersedes K-1037 P6
    in-kernel admission for the 13-cell MFMA-issue-stall cohort.  Pre-K-1151
    the K-1037 P6 admit gate would intercept S24 (envelope B) and S29
    (envelope A) and dispatch them in-kernel with wpeu=1.  Under K-1151,
    the K-1142 E1 envelope is consulted FIRST and routes both cells OUT
    to hipBLASLt at the K-1121 paired-n=30-confirmed speedup.

    Pin both halves: the P6 admit gate still fires structurally (so its
    NULL-RESULT pin tests above remain valid), but the full-dispatch
    decision routes OUT regardless because E1 short-circuits first."""
    # P6 admit still classifies S24 / S29 as positives at the structural level
    assert R_K1037_P6_admit_wpeu1( 4480, 3072,  768, torch.bfloat16) is True   # S24
    assert R_K1037_P6_admit_wpeu1(14208, 2048, 1024, torch.bfloat16) is True   # S29
    # E1 also fires
    assert R_K1142_E1_route_to_hbl( 4480, 3072,  768, torch.bfloat16) is True
    assert R_K1142_E1_route_to_hbl(14208, 2048, 1024, torch.bfloat16) is True
    # Full dispatch routes OUT (E1 wins over P6 admit per K-1151 ordering)
    assert _k971_route_to_hbl(
         4480, 3072,  768, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is True
    assert _k971_route_to_hbl(
        14208, 2048, 1024, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is True


@pytest.mark.parametrize("sid,M,N,K,src", K1121_ANCHORS_13,
                         ids=[c[0] for c in K1121_ANCHORS_13])
def test_full_dispatch_routes_k1121_anchors_out(sid, M, N, K, src):
    """End-to-end dispatch: every K-1121 anchor routes OUT to hipBLASLt
    via the productionised E1 envelope (regardless of which lower-priority
    predicate would have caught it).  This is the K-1151 functional
    contract — the K-1121 paired-n=30 verdicts are preserved without
    needing the 13-row strict-equality K971_ROUTE_TABLE extension."""
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is True


@pytest.mark.parametrize("fid,M,N,K,hbl_over_tb,note", K1142_INVERSE_PRED_FPS,
                         ids=[c[0] for c in K1142_INVERSE_PRED_FPS])
def test_full_dispatch_does_not_route_k1142_fps(fid, M, N, K, hbl_over_tb, note):
    """End-to-end dispatch: neither K-1142 inverse-predicate FP routes
    OUT — the productionised E1 carve-out keeps them in-kernel where
    paired-n=30 evidence shows tritonblas is not slower (FP1) or wins by
    a tie-band 1.5% margin (FP2).  Failure here means K-1151 has
    regressed the K-1151 brief's "no cell regresses >2%" gate."""
    routed = _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False)
    # FP1/FP2 are NOT in any other route-OUT predicate either (they fail
    # P5 K-floor 512, fail P6 envelopes, are absent from K971_ROUTE_TABLE).
    assert routed is False, (
        f"K-1142 FP {fid} ({M},{N},{K}) routed OUT despite carve-out "
        f"({note}) — K-1151 regression gate violated")


def test_is_k1121_e1_admit_safe_accepts_every_k1121_anchor():
    """Symmetric coverage to test_k1142_e1_excludes_inverse_predicate_fps:
    the helper must return True for every K-1121 anchor so any future
    structural-relaxation PR can use it as a unit-test guard without
    chasing false negatives."""
    for sid, M, N, K, _src in K1121_ANCHORS_13:
        assert is_k1121_e1_admit_safe(M, N, K, torch.bfloat16) is True, (
            f"is_k1121_e1_admit_safe rejected K-1121 anchor {sid} "
            f"({M},{N},{K}) — false positive in the guardrail")


def test_is_k1121_e1_admit_safe_outside_envelope_silent():
    """Cells OUTSIDE E1 are out of K-1142 audit scope and the helper
    must return True (silent) — the helper only flags cells inside E1
    that fail the carve-out, so future productionisers can use it as a
    drop-in admit gate without false rejections elsewhere."""
    # K-984/K-989 LAND anchors outside E1 (e.g. K not in {256,768,1024}).
    assert is_k1121_e1_admit_safe(2304, 2048, 4800, torch.bfloat16) is True  # S04
    assert is_k1121_e1_admit_safe(1024, 2048, 1240, torch.bfloat16) is True  # S40
    # Non-bf16 — out of audit scope.
    assert is_k1121_e1_admit_safe(256, 2048, 256, torch.float16) is True


def test_k1142_e1_does_not_leak_on_k950_land():
    """K-950 LAND set must not leak through the productionised E1
    predicate.  L04 = (768, 2048, 1024, bf16) is the only K-950 cell
    inside the E1 N/K envelope; the M=4480 carve-out (K-1142 §5)
    excludes it cleanly with margin 768 → 4480 = 3712.  This test pins
    that exclusion so any future relaxation of the M-floor immediately
    surfaces in CI (it would re-introduce a K-950 LAND leak)."""
    for cid, M, N, K, dtype, cohort in K950_LAND_CELLS:
        if str(dtype) != "torch.bfloat16":
            continue
        assert R_K1142_E1_route_to_hbl(M, N, K, dtype) is False, (
            f"K-1142 E1 leaked into K-950 LAND cell {cid} "
            f"({M},{N},{K}) {dtype} cohort={cohort}")
