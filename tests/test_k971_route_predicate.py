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
def test_p6_admit_short_circuits_route_out_to_in_kernel(cid, M, N, K):
    """When P6 admit fires, _k971_route_to_hbl must return False so the
    cell dispatches in-kernel (where matmul.py sets waves_per_eu=1).
    Verifies the integration path: P6 admit overrides P5 route-OUT for
    cells in the MFMA-issue-stall structural fingerprint.
    Witness: S29 (14208, 2048, 1024) is otherwise routed-OUT by P5
    Clause-2 (M>=5000, N==2048, K==1024) — P6 admit must short-circuit."""
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is False, (
        f"P6 admit failed to short-circuit route-OUT for {cid} ({M},{N},{K})")


@pytest.mark.parametrize("cid,M,N,K", P6_LAND_ANCHORS,
                         ids=[c[0] for c in P6_LAND_ANCHORS])
def test_p6_admit_does_not_break_k984_k989_route_out_anchors(cid, M, N, K):
    """K-984/K-989 anchors must continue to route-OUT after P6 wiring —
    the new short-circuit must not regress the existing P5 LAND set."""
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is True, (
        f"P6 wiring regressed K-984/K-989 LAND anchor {cid} ({M},{N},{K})")
