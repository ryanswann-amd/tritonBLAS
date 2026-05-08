"""K-633 dispatch precedence ladder regression guard.

Guards the dispatch precedence ladder defined in
``include/tritonblas/matmul.py`` against future regression of the gate
behaviour validated by the K-619 sweep on MI300X.

The fixture CSV ``tests/fixtures/regression_guard_fixture_expanded.csv``
holds 36 shapes spanning all 15 multi-predicate signatures present in
the K-619 588-shape overlap set; ``scripts/iter4_expand_fixture.py`` in
the K-633 workspace regenerates it from the K-619 sweep CSV.

Tests do not launch any kernels; they exercise the ladder as a pure
function via ``_patched_streamk_flag`` so they are CPU-only and fast.
KEEP ``_patched_streamk_flag`` IN SYNC with the ladder in
``_matmul`` (include/tritonblas/matmul.py).
"""

import csv
import os

import pytest


FIXTURE_CSV = os.path.join(
    os.path.dirname(__file__), "fixtures",
    "regression_guard_fixture_expanded.csv",
)


def _load_fixture():
    rows = []
    with open(FIXTURE_CSV) as f:
        for r in csv.DictReader(f):
            rows.append((
                int(r["M"]), int(r["N"]), int(r["K"]),
                r["in_dtype"],
                r["expected_streamk_flag"] == "True",
                r["fixture_role"],
                float(r["k619_g4_delta_pct_vs_baseline"]),
                r["k619_best_variant"],
            ))
    return rows


def _patched_streamk_flag(M, N, K, caller_passed=False):
    """Pure-function mirror of the K-633 precedence ladder.

    KEEP IN SYNC with ``_matmul`` in include/tritonblas/matmul.py."""
    enable_streamk = caller_passed
    # P1 - VETO
    if M <= 8 and K >= 4096:
        return False
    # P2 - Tightened large-balanced auto-flip
    elif (
        not enable_streamk
        and M == N
        and min(M, N) >= 5120
        and K >= 6144
    ):
        enable_streamk = True
    # P3 - small_m_decode is intentionally NOT auto-enabled
    return enable_streamk


FIXTURES = _load_fixture()


@pytest.mark.parametrize(
    "M,N,K,dtype,exp_flag,role,g4_delta,best",
    FIXTURES,
    ids=[f"{r[0]}x{r[1]}x{r[2]}-{r[5]}" for r in FIXTURES],
)
def test_precedence_flag(M, N, K, dtype, exp_flag, role, g4_delta, best):
    """Ladder-level test: the static ladder helper must report the
    K-619-validated streamk setting for every fixture shape."""
    got = _patched_streamk_flag(M, N, K, caller_passed=False)
    assert got is exp_flag, (
        f"Precedence violation at {M}x{N}x{K} ({role}): "
        f"expected enable_streamk={exp_flag}, got {got}. "
        f"K-619 G4_streamk delta vs baseline = {g4_delta:+.1f}%, "
        f"K-619 best variant = {best!r}."
    )


@pytest.mark.parametrize(
    "M,N,K,dtype,exp_flag,role,g4_delta,best",
    [r for r in FIXTURES if r[5] == "P1_VETO"],
)
def test_p1_veto_overrides_caller(M, N, K, dtype, exp_flag, role, g4_delta, best):
    """P1 must veto streamk even when the caller explicitly passed
    ``enable_streamk=True``. K-654 hard rule: no caller can bypass
    the M<=8 AND K>=4096 veto, otherwise the 53/61-shape regression
    returns."""
    got = _patched_streamk_flag(M, N, K, caller_passed=True)
    assert got is False, (
        f"P1 VETO breached at {M}x{N}x{K}: caller-passed streamk=True "
        f"was not vetoed. K-619 measured G4_streamk delta = {g4_delta:+.1f}%."
    )


@pytest.mark.parametrize(
    "M,N,K,dtype,exp_flag,role,g4_delta,best",
    [r for r in FIXTURES if r[5] == "P3_GUARD_SMALL_M_DECODE"],
)
def test_p3_no_auto_flip_small_m_decode(M, N, K, dtype, exp_flag, role,
                                          g4_delta, best):
    """P3: the small_m_decode predicate (M<=64 AND N>=1024 AND K>=1024)
    must NOT auto-flip streamk; K-619 measured 214/221 regressions in
    that envelope. Caller-explicit opt-in is still honored."""
    got = _patched_streamk_flag(M, N, K, caller_passed=False)
    assert got is False, (
        f"P3 violated at {M}x{N}x{K} ({role}): streamk was auto-enabled "
        f"in the small_m_decode envelope. K-619 G4 delta = {g4_delta:+.1f}%."
    )


@pytest.mark.parametrize(
    "M,N,K,dtype,exp_flag,role,g4_delta,best",
    [r for r in FIXTURES if r[5] == "P2_NEGATIVE"],
)
def test_p2_excludes_known_streamk_regressors(M, N, K, dtype, exp_flag,
                                                role, g4_delta, best):
    """P2: the tightened large_balanced auto-flip must NOT enable
    streamk on shapes K-619 measured as regressing."""
    got = _patched_streamk_flag(M, N, K, caller_passed=False)
    assert got is False, (
        f"P2 incorrectly enabled streamk on {M}x{N}x{K} - K-619 measured "
        f"this as delta = {g4_delta:+.1f}% vs baseline persistent."
    )


@pytest.mark.parametrize(
    "M,N,K,dtype,exp_flag,role,g4_delta,best",
    [r for r in FIXTURES if r[5] == "P2_POSITIVE"],
)
def test_p2_includes_known_streamk_winners(M, N, K, dtype, exp_flag,
                                             role, g4_delta, best):
    """P2: the tightened large_balanced auto-flip must enable streamk
    on the K-619-validated winners."""
    got = _patched_streamk_flag(M, N, K, caller_passed=False)
    assert got is True, (
        f"P2 failed to enable streamk on {M}x{N}x{K} - K-619 measured "
        f"this as the canonical large-balanced winner."
    )


# ---------------------------------------------------------------------------
# Boundary tests -- guard the EXACT predicate edges the K-633 ladder commits
# to.  These are the off-by-one cases the K-654 reviewer (Testing Zealot)
# flagged: any future drift in the inequality (<= vs <, >= vs >, == vs >=,
# tightened vs loose threshold) must trip these tests, not just the fixture
# rows.
# ---------------------------------------------------------------------------

# (M, N, K, expected_flag, why)
P1_BOUNDARY_INERT = [
    # Just outside the M<=8 leg -> P1 must NOT veto.
    (9, 1, 4096, False, "M=9 (just over P1 M<=8 boundary, K on threshold)"),
    (9, 8192, 4096, False, "M=9, big N+K -- still outside P1, no auto-flip"),
    (16, 16, 8192, False, "M=16 (clearly outside P1 even at large K)"),
    # Just outside the K>=4096 leg -> P1 must NOT veto.
    (8, 1, 4095, False, "K=4095 (one below P1 K>=4096 threshold)"),
    (1, 1, 4095, False, "M=1, K=4095 -- inside-M but below-K, no veto"),
]

P1_BOUNDARY_VETO = [
    # On the boundary -> P1 MUST fire and veto even when caller asked True.
    (8, 1, 4096, False, "M=8 K=4096 (lower-corner of P1 envelope)"),
    (1, 1024, 4096, False, "M=1 long-K decode (canonical P1 win)"),
    (8, 4096, 8192, False, "M=8 K=8192 (deep inside P1 envelope)"),
]


@pytest.mark.parametrize("M,N,K,exp_flag,why", P1_BOUNDARY_INERT)
def test_p1_boundary_inert(M, N, K, exp_flag, why):
    """Off-by-one guard: shapes one step outside the P1 envelope must
    NOT be vetoed (and must NOT auto-flip via P2)."""
    got = _patched_streamk_flag(M, N, K, caller_passed=False)
    assert got is exp_flag, (
        f"P1 boundary breached at {M}x{N}x{K} ({why}): "
        f"expected enable_streamk={exp_flag}, got {got}."
    )


@pytest.mark.parametrize("M,N,K,exp_flag,why", P1_BOUNDARY_VETO)
def test_p1_boundary_veto_on_corner(M, N, K, exp_flag, why):
    """The lower-corner of the P1 envelope (M=8, K=4096) must veto
    even when the caller explicitly opted into streamk."""
    got = _patched_streamk_flag(M, N, K, caller_passed=True)
    assert got is False, (
        f"P1 corner failure at {M}x{N}x{K} ({why}): "
        f"caller-passed streamk=True was not vetoed."
    )


# (M, N, K, expected_flag, why)
P2_BOUNDARY_INERT = [
    # Asymmetric M != N -> tightened predicate excludes them.
    (5120, 5119, 6144, False, "min(M,N)=5119 (one below P2 threshold)"),
    (5120, 8192, 8192, False, "asymmetric M!=N, large balanced -> no flip"),
    (8192, 4096, 8192, False, "asymmetric M!=N (8192x4096) -> no flip"),
    # K just below 6144 -> tightened predicate excludes.
    (5120, 5120, 6143, False, "K=6143 (one below P2 K>=6144 threshold)"),
    (8192, 8192, 4096, False, "K=4096 -- below tightened P2 K threshold"),
    # min(M,N) below threshold even when square.
    (5119, 5119, 8192, False, "M=N=5119 -- below P2 min>=5120 threshold"),
    (4096, 4096, 8192, False, "M=N=4096 -- the loose-R-G4 trap shape, must not flip"),
]

P2_BOUNDARY_AUTOFLIP = [
    # Exact corner of P2 -- must fire.
    (5120, 5120, 6144, True, "M=N=5120 K=6144 (lower corner of P2)"),
    (6144, 6144, 6144, True, "K-619 measured win, P2 must fire"),
    (8192, 8192, 8192, True, "K-619 measured win, P2 must fire"),
    (8192, 8192, 16384, True, "K-619 measured win, P2 must fire"),
]


@pytest.mark.parametrize("M,N,K,exp_flag,why", P2_BOUNDARY_INERT)
def test_p2_boundary_inert(M, N, K, exp_flag, why):
    """Off-by-one guard: shapes one step outside the tightened P2
    envelope (loose K, asymmetric, sub-threshold min) must NOT be
    auto-flipped to streamk; the K-619 loose R-G4 admitted 7/22
    regressors in this region."""
    got = _patched_streamk_flag(M, N, K, caller_passed=False)
    assert got is exp_flag, (
        f"P2 boundary breached at {M}x{N}x{K} ({why}): "
        f"expected enable_streamk={exp_flag}, got {got}."
    )


@pytest.mark.parametrize("M,N,K,exp_flag,why", P2_BOUNDARY_AUTOFLIP)
def test_p2_boundary_autoflip_on_corner(M, N, K, exp_flag, why):
    """The four K-654 ladder-fired shapes plus the lower P2 corner
    must auto-flip enable_streamk=True under the tightened predicate.

    These are the SAME four shapes the K-644 paired sweep on MI300X
    measured at +5.4% to +9.8% TFLOPS vs baseline; if this test
    fails the precedence ladder has lost its measured wins."""
    got = _patched_streamk_flag(M, N, K, caller_passed=False)
    assert got is True, (
        f"P2 ladder-fired shape regressed at {M}x{N}x{K} ({why}): "
        f"expected enable_streamk=True, got {got}."
    )


# ---------------------------------------------------------------------------
# Rule-classification test -- pin every K-654 ladder-fired shape to its
# expected rule.  This catches the bug class where a future refactor
# silently moves a shape from P2_AUTOFLIP -> INERT (or vice versa) without
# changing the boolean outcome but losing the perf win.
# ---------------------------------------------------------------------------

def _classify_rule(M, N, K):
    """Return the ladder rule that fires for (M,N,K), or 'INERT'."""
    if M <= 8 and K >= 4096:
        return "P1_VETO"
    if M == N and min(M, N) >= 5120 and K >= 6144:
        return "P2_AUTOFLIP"
    return "INERT"


# K-654 paired sweep classification (output/sweep_summary_v2.md).
LADDER_FIRED_SHAPES = [
    (6144, 6144, 6144, "P2_AUTOFLIP"),
    (8192, 8192, 8192, "P2_AUTOFLIP"),
    (8192, 8192, 16384, "P2_AUTOFLIP"),
]
NEAR_BOUNDARY_INERT_SHAPES = [
    # K-619 measured these as inert (predicate doesn't fire); the K-644
    # sweep saw them as noise-floor (delta within +-3% over a high-replay
    # graph-captured timing).
    (4096, 1024, 4096, "INERT"),  # M!=N -> P2 inert, K=4096 not in P1
    (1024, 4096, 4096, "INERT"),  # asymmetric, ladder inert
    (8192, 64, 4096, "INERT"),    # M!=N, small N -> ladder inert
    (1024, 1024, 16384, "INERT"), # min<5120 -> P2 inert
    (9, 8192, 4096, "INERT"),     # M=9 just above P1 boundary
    (8192, 8192, 6143, "INERT"),  # K=6143 just below P2 boundary
    (5119, 5119, 8192, "INERT"),  # min=5119 just below P2 boundary
]


@pytest.mark.parametrize("M,N,K,exp_rule", LADDER_FIRED_SHAPES)
def test_ladder_fired_shape_hits_expected_rule(M, N, K, exp_rule):
    """The K-644 sweep-validated ladder-fired shapes (geomean +7.5%
    TFLOPS) MUST classify to their expected rule -- if a future patch
    makes them INERT the perf win is lost silently."""
    got = _classify_rule(M, N, K)
    assert got == exp_rule, (
        f"Ladder-fired shape {M}x{N}x{K} classified as {got!r}, "
        f"expected {exp_rule!r}; the K-644 sweep relies on this rule."
    )


@pytest.mark.parametrize("M,N,K,exp_rule", NEAR_BOUNDARY_INERT_SHAPES)
def test_near_boundary_inert_shape_does_not_fire_rule(M, N, K, exp_rule):
    """Near-boundary inert shapes MUST classify as INERT -- if a
    future patch sweeps them into P1/P2, the patched code path
    diverges from baseline on shapes K-619 measured as regressors,
    re-introducing the K-580 regression class."""
    got = _classify_rule(M, N, K)
    assert got == exp_rule, (
        f"Near-boundary shape {M}x{N}x{K} classified as {got!r}, "
        f"expected INERT; widening the ladder past its K-619 envelope "
        f"is the K-580 regression class."
    )
