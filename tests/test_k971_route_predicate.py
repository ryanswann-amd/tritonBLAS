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


# ===========================================================================
# K-1089 — R_K1037_P6_admit_wpeu1 structural surrogate of K-1037 P6.
#
# The predicate is a calibrated structural surrogate of the K-1037
# 3-clause paired-PMC MFMA-issue-stall classifier (perfect 18/18 PMC truth
# agreement). It is shipped as an offline classifier that downstream
# tickets can wire to a candidate `waves_per_eu`/`enable_streamk`/
# block-size override or any future structural lever; K-1089 v3 deliberately
# does NOT wire the predicate into the dispatch path because both
# candidate levers (`waves_per_eu=1` per K-1051 and `enable_streamk=True`
# per K-1044) failed paired-n=20 + Wilcoxon validation on MI300X
# (slurm 294979). These tests pin the predicate's calibration so a future
# wiring ticket can re-use it without recalibration.
# ===========================================================================

# (cid, M, N, K, expected_admit) drawn from K-1037 paired-PMC truth set
# plus the K-1089 v3 expanded P6+ synthetic cells and K-984/K-989 anchors.
P6_ADMIT_CALIBRATION = [
    # K-1037 paired-PMC P6+ positives (truth-set TPs)
    ("S24",   4480, 3072,  768, True),
    ("S29",  14208, 2048, 1024, True),
    # K-1089 v3 synthetic cells inside Envelope B (extra calibration anchors)
    ("P6_3",  3072, 3072,  768, True),
    ("P6_4",  4096, 2560, 1024, True),
    ("P6_5",  3584, 2304,  512, True),
    # K-984/K-989 LAND anchors — must NOT admit (would break route-OUT)
    ("S25",   6016, 2048, 1024, False),  # below Envelope A M-floor
    ("S27",  10112, 2048, 1024, False),
    ("S28",  12160, 2048, 1024, False),
    ("S18",  5972, 1792,  768,  False),  # above Envelope B maxMN ceiling
    # K-1017 NO-LAND adjacent — must NOT admit (P5 already routes-OUT)
    ("S26",   8064, 2048, 1024, False),
    ("S30",  16256, 2048, 1024, False),  # above Envelope A M-ceiling
    # K-1017 cell S38 — K=200 < 512 floor (Surrogate-C2 MFMA<1% guard)
    ("S38",    384,  128,  200, False),
    # K-950 LAND set — must NOT admit
    ("L01",   1024, 1024,  512, False),
    ("L02",   4096, 4096, 4096, False),  # maxMN=4096 inside, but minMN=4096 too — K is at envelope B's K=1024 ceiling so actually outside
]


@pytest.mark.parametrize("cid,M,N,K,expected", P6_ADMIT_CALIBRATION)
def test_R_K1037_P6_admit_calibration_bf16(cid, M, N, K, expected):
    """Pin K-1037 P6 surrogate verdicts on the K-1089 calibration cohort.

    Locks in the well-calibrated structural surrogate so future wiring
    tickets can rely on the predicate without re-validating against PMC.
    """
    got = R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16)
    assert got is expected, (
        f"P6 surrogate: cell {cid} ({M},{N},{K},bf16) "
        f"admit={got} expected={expected}")


def test_R_K1037_P6_is_bf16_only():
    """Surrogate is calibrated bf16-only (K-1037 / K-1017 / K-1044 truth
    set scope). Other dtypes must NOT admit even on shape matches."""
    M, N, K = 4480, 3072, 768  # would admit at bf16 (Envelope B)
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is True
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.float16) is False
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.float32) is False
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.float8_e4m3fn) is False


def test_R_K1037_P6_K_floor_silences_S38_class():
    """Surrogate-C2 (MFMA_pct >= 1.0%) -> K >= 512 floor. K-1017 cell S38
    (K=200, MFMA%=0.07) and similar launch-overhead-bound cells must NOT
    admit, even if shape would otherwise match an envelope."""
    # Synthetic S38 variants with K below the 512 floor:
    for K in (1, 100, 200, 256, 384, 511):
        assert R_K1037_P6_admit_wpeu1(4480, 3072, K, torch.bfloat16) is False, \
            f"K-floor=512 leaked at K={K}"
    # Boundary: K == 512 inside Envelope B (minMN>=2048, maxMN<=4500)
    assert R_K1037_P6_admit_wpeu1(3584, 2304, 512, torch.bfloat16) is True
    # K == 511 just below the floor on the same shape
    assert R_K1037_P6_admit_wpeu1(3584, 2304, 511, torch.bfloat16) is False


def test_R_K1037_P6_envelope_A_M_bounds():
    """Envelope A: N=2048 AND K=1024 AND 13000 <= M <= 14999.
    Below 13000 = K-984/K-989 anchors (must route-OUT, NOT admit).
    Above 14999 = S30 OCC region (must NOT admit)."""
    for M in (13000, 14000, 14208, 14999):
        assert R_K1037_P6_admit_wpeu1(M, 2048, 1024, torch.bfloat16) is True, \
            f"Envelope A interior leak at M={M}"
    for M in (12999, 12160, 10112, 6016):
        assert R_K1037_P6_admit_wpeu1(M, 2048, 1024, torch.bfloat16) is False, \
            f"Envelope A leak below M-floor at M={M}"
    for M in (15000, 16256, 32768):
        assert R_K1037_P6_admit_wpeu1(M, 2048, 1024, torch.bfloat16) is False, \
            f"Envelope A leak above M-ceiling at M={M}"


def test_R_K1037_P6_envelope_B_anchor_safety():
    """Envelope B: minMN >= 2048 AND maxMN <= 4500 AND 512 <= K <= 1024.
    The maxMN<=4500 ceiling preserves K-984 anchor S18 (maxMN=5972)
    and K-989 anchor S25 (maxMN=6016) which both sit clear above."""
    # K-984/K-989 LAND anchors (maxMN well above 4500): must NOT admit
    for cid, M, N, K in [
        ("S18", 5972, 1792, 768),
        ("S25", 6016, 2048, 1024),
        ("S27", 10112, 2048, 1024),
        ("S28", 12160, 2048, 1024),
    ]:
        assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is False, \
            f"Envelope B LAND-anchor leak: {cid} ({M},{N},{K})"


def test_R_K1037_P6_does_not_change_dispatch_under_K_1089_v3():
    """K-1089 v3 design: predicate is offline-only — it must NOT participate
    in the live `_k971_route_to_hbl` dispatch decision, because both
    candidate admitted levers (wpeu=1, streamk=True) failed Wilcoxon
    paired-n=20 validation. Verifies a P6+ cell that would also be silenced
    by P5 (S29) still routes-OUT under the live dispatch (P5 wins),
    confirming the P6 short-circuit is NOT wired."""
    # S29 = (14208, 2048, 1024) — P6+ AND P5 routes-OUT.
    # If P6 had a wired short-circuit overriding P5, the result would
    # be False (in-kernel). With no wiring, P5 wins -> True (route-OUT).
    M, N, K = 14208, 2048, 1024
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is True
    assert R_K979_P5_route_to_hbl(M, N, K, torch.bfloat16) is True
    routed = _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False)
    assert routed is True, (
        f"P6 surrogate must NOT wire into dispatch in K-1089 v3 "
        f"(both candidate levers failed Wilcoxon validation); "
        f"S29 should continue to route-OUT via P5 — got routed={routed}")


def test_R_K1037_P6_does_not_admit_S24_into_dispatch():
    """S24 is a P6+ admit but P5 does NOT route it. Under K-1089 v3
    (no wiring), the live dispatch verdict must be False (in-kernel
    persistent baseline) — same as it was pre-K-1089."""
    M, N, K = 4480, 3072, 768
    assert R_K1037_P6_admit_wpeu1(M, N, K, torch.bfloat16) is True
    assert R_K979_P5_route_to_hbl(M, N, K, torch.bfloat16) is False
    routed = _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False)
    assert routed is False, (
        f"S24 (P6+, P5-silent) should dispatch in-kernel persistent "
        f"baseline under K-1089 v3 (no wiring) — got routed={routed}")
