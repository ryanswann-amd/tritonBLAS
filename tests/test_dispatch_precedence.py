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
