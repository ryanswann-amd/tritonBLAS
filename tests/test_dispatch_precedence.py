"""K-633 dispatch precedence ladder regression guard.

These tests EXERCISE the actual ``_matmul`` dispatch path in
``tritonblas.matmul`` (NOT a mirror helper).  The kernel-launch
functions ``streamk_matmul_lt`` and ``persistent_matmul_lt`` plus the
selector factory ``_make_matmul_selector`` are monkey-patched so we can
observe the resolved ``enable_streamk`` flag the ladder produced and
which dispatch branch was taken, without launching any Triton kernels.

This is a deliberate change from the v1 of this file, which compared
against a pure-python ``_patched_streamk_flag`` mirror -- that mirror
could silently drift from the source.  Asserting against the real code
path means any future edit to the ladder in ``include/tritonblas/matmul.py``
will be observed by these tests.

The fixture CSV ``tests/fixtures/regression_guard_fixture_expanded.csv``
holds 36 shapes spanning the 15 multi-predicate signatures present in
the K-619 588-shape overlap set.
"""

import csv
import os
from contextlib import contextmanager
from unittest import mock

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="ladder dispatch is exercised through tritonblas which "
           "requires CUDA at import time",
)

# Importing tritonblas requires CUDA (touches torch.cuda.current_device()
# at module load), so guard the import behind the skipif above.
# NB: ``tritonblas.matmul`` resolves to the function exported via
# ``__init__.py``, shadowing the submodule.  Pull the actual module out of
# ``sys.modules`` to get the namespace where the K-633 ladder lives.
import sys  # noqa: E402
import tritonblas  # noqa: E402,F401
_matmul_mod = sys.modules["tritonblas.matmul"]


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


FIXTURES = _load_fixture()


_DTYPE_MAP = {
    "fp16": torch.float16,
    "f16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp8": getattr(torch, "float8_e4m3fnuz", torch.float16),
    "float8_e4m3fnuz": getattr(torch, "float8_e4m3fnuz", torch.float16),
    "float8_e4m3fn": getattr(torch, "float8_e4m3fn", torch.float16),
}


@contextmanager
def _capture_dispatch():
    """Patch the kernel-launch functions and the selector factory in
    ``tritonblas.matmul`` so a call to ``_matmul`` records:

    * ``selector_streamk`` -- the value of the ``streamk=`` arg passed
      into ``_make_matmul_selector`` AFTER the K-633 ladder ran.
    * ``branch`` -- which kernel-launch path the dispatcher entered.

    No Triton kernels are launched.  ``out`` is returned unmodified by
    the mocked launch functions, so callers see a valid empty tensor."""
    captured = {}

    def fake_make_selector(M, N, K, a_dt, b_dt, c_dt, device, **kw):
        captured["selector_streamk"] = kw.get("streamk", False)
        return mock.MagicMock(name="FakeSelector")

    def fake_streamk(a, b, out, selector, config, **kw):
        captured["branch"] = "streamk"
        return out

    def fake_persistent(a, b, out, selector, config, **kw):
        captured["branch"] = "persistent"
        return out

    # Patch the names as resolved inside _matmul_mod.  matmul_preamble is
    # unaffected (it isn't used unless work_stealing=True).
    with mock.patch.object(_matmul_mod, "_make_matmul_selector",
                           side_effect=fake_make_selector), \
         mock.patch.object(_matmul_mod, "streamk_matmul_lt",
                           side_effect=fake_streamk), \
         mock.patch.object(_matmul_mod, "persistent_matmul_lt",
                           side_effect=fake_persistent):
        yield captured


def _exercise_matmul(M, N, K, dtype_str, *, caller_streamk):
    """Build minimum-size CUDA tensors with the right shape/dtype and
    call the real ``_matmul``; returns the captured dispatch decision."""
    dt = _DTYPE_MAP[dtype_str.lower()]
    # Tiny allocation: the kernels are mocked so size doesn't matter for
    # correctness; we only need a.shape=(M,K), b.shape=(K,N), and a real
    # CUDA device for ``a.new_empty(M, N)``.
    a = torch.empty((M, K), device="cuda", dtype=dt)
    b = torch.empty((K, N), device="cuda", dtype=dt)
    with _capture_dispatch() as cap:
        _matmul_mod._matmul(a, b, enable_streamk=caller_streamk)
    return cap


# ---------------------------------------------------------------------------
# Fixture-driven ladder coverage -- every K-619 multi-predicate signature.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "M,N,K,dtype,exp_flag,role,g4_delta,best",
    FIXTURES,
    ids=[f"{r[0]}x{r[1]}x{r[2]}-{r[5]}" for r in FIXTURES],
)
def test_precedence_dispatched_streamk(M, N, K, dtype, exp_flag, role,
                                         g4_delta, best):
    """The REAL ``_matmul`` dispatch must end up calling the selector
    with the K-619-validated ``streamk`` value AND must enter the
    matching kernel-launch branch."""
    cap = _exercise_matmul(M, N, K, dtype, caller_streamk=False)
    assert cap["selector_streamk"] is exp_flag, (
        f"Precedence violation at {M}x{N}x{K} ({role}): selector got "
        f"streamk={cap['selector_streamk']}, expected {exp_flag}. "
        f"K-619 G4_streamk delta vs baseline = {g4_delta:+.1f}%, "
        f"K-619 best variant = {best!r}."
    )
    assert cap["branch"] == ("streamk" if exp_flag else "persistent"), (
        f"Dispatch branch mismatch at {M}x{N}x{K} ({role}): entered "
        f"{cap['branch']!r} branch, expected "
        f"{'streamk' if exp_flag else 'persistent'!r}."
    )


@pytest.mark.parametrize(
    "M,N,K,dtype,exp_flag,role,g4_delta,best",
    [r for r in FIXTURES if r[5] == "P1_VETO"],
)
def test_p1_veto_overrides_caller(M, N, K, dtype, exp_flag, role,
                                    g4_delta, best):
    """P1 must veto streamk in the REAL dispatch path even when the
    caller explicitly passed ``enable_streamk=True``.  K-654 hard rule:
    no caller can bypass the M<=8 AND K>=4096 veto, otherwise the
    53/61-shape regression returns."""
    cap = _exercise_matmul(M, N, K, dtype, caller_streamk=True)
    assert cap["selector_streamk"] is False, (
        f"P1 VETO breached at {M}x{N}x{K}: selector got "
        f"streamk={cap['selector_streamk']} despite caller-passed "
        f"streamk=True. K-619 G4_streamk delta = {g4_delta:+.1f}%."
    )
    assert cap["branch"] == "persistent", (
        f"P1 VETO breached at {M}x{N}x{K}: dispatch entered "
        f"{cap['branch']!r}, expected 'persistent'."
    )


@pytest.mark.parametrize(
    "M,N,K,dtype,exp_flag,role,g4_delta,best",
    [r for r in FIXTURES if r[5] == "P3_GUARD_SMALL_M_DECODE"],
)
def test_p3_no_auto_flip_small_m_decode(M, N, K, dtype, exp_flag, role,
                                          g4_delta, best):
    """P3: the small_m_decode predicate (M<=64 AND N>=1024 AND K>=1024)
    must NOT auto-flip streamk in the REAL dispatch path; K-619
    measured 214/221 regressions in that envelope."""
    cap = _exercise_matmul(M, N, K, dtype, caller_streamk=False)
    assert cap["selector_streamk"] is False, (
        f"P3 violated at {M}x{N}x{K} ({role}): selector got "
        f"streamk={cap['selector_streamk']} despite the small_m_decode "
        f"envelope. K-619 G4 delta = {g4_delta:+.1f}%."
    )


@pytest.mark.parametrize(
    "M,N,K,dtype,exp_flag,role,g4_delta,best",
    [r for r in FIXTURES if r[5] == "P2_NEGATIVE"],
)
def test_p2_excludes_known_streamk_regressors(M, N, K, dtype, exp_flag,
                                                role, g4_delta, best):
    """P2: the tightened large_balanced auto-flip must NOT enable
    streamk on shapes K-619 measured as regressing."""
    cap = _exercise_matmul(M, N, K, dtype, caller_streamk=False)
    assert cap["selector_streamk"] is False, (
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
    on the K-619-validated winners through the REAL dispatch path."""
    cap = _exercise_matmul(M, N, K, dtype, caller_streamk=False)
    assert cap["selector_streamk"] is True, (
        f"P2 failed to enable streamk on {M}x{N}x{K} - K-619 measured "
        f"this as the canonical large-balanced winner."
    )
    assert cap["branch"] == "streamk", (
        f"P2 winner {M}x{N}x{K} did not enter the streamk branch."
    )


# ---------------------------------------------------------------------------
# Boundary tests -- guard the EXACT predicate edges the K-633 ladder commits
# to.  These are the off-by-one cases the K-654 reviewer (Testing Zealot)
# flagged: any future drift in the inequality (<= vs <, >= vs >, == vs >=,
# tightened vs loose threshold) must trip these tests.  All of them go
# through the REAL ``_matmul`` dispatch path.
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
    cap = _exercise_matmul(M, N, K, "fp16", caller_streamk=False)
    assert cap["selector_streamk"] is exp_flag, (
        f"P1 boundary breached at {M}x{N}x{K} ({why}): selector got "
        f"streamk={cap['selector_streamk']}, expected {exp_flag}."
    )


@pytest.mark.parametrize("M,N,K,exp_flag,why", P1_BOUNDARY_VETO)
def test_p1_boundary_veto_on_corner(M, N, K, exp_flag, why):
    """The lower-corner of the P1 envelope (M=8, K=4096) must veto
    even when the caller explicitly opted into streamk."""
    cap = _exercise_matmul(M, N, K, "fp16", caller_streamk=True)
    assert cap["selector_streamk"] is False, (
        f"P1 corner failure at {M}x{N}x{K} ({why}): caller-passed "
        f"streamk=True surfaced as streamk={cap['selector_streamk']} "
        f"in the real dispatch path."
    )
    assert cap["branch"] == "persistent", (
        f"P1 corner failure at {M}x{N}x{K} ({why}): real dispatch "
        f"entered {cap['branch']!r} branch, expected 'persistent'."
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
    auto-flipped to streamk by the real dispatch path."""
    cap = _exercise_matmul(M, N, K, "fp16", caller_streamk=False)
    assert cap["selector_streamk"] is exp_flag, (
        f"P2 boundary breached at {M}x{N}x{K} ({why}): selector got "
        f"streamk={cap['selector_streamk']}, expected {exp_flag}."
    )


@pytest.mark.parametrize("M,N,K,exp_flag,why", P2_BOUNDARY_AUTOFLIP)
def test_p2_boundary_autoflip_on_corner(M, N, K, exp_flag, why):
    """The four K-654 ladder-fired shapes plus the lower P2 corner
    must auto-flip enable_streamk=True under the tightened predicate
    AND enter the streamk dispatch branch.

    These are the SAME four shapes the K-644 v3 sweep on MI300X
    measured at +5.5% to +8.5% TFLOPS vs baseline; if this test fails
    the precedence ladder has lost its measured wins."""
    cap = _exercise_matmul(M, N, K, "fp16", caller_streamk=False)
    assert cap["selector_streamk"] is True, (
        f"P2 ladder-fired shape regressed at {M}x{N}x{K} ({why}): "
        f"selector got streamk={cap['selector_streamk']}, expected True."
    )
    assert cap["branch"] == "streamk", (
        f"P2 ladder-fired shape {M}x{N}x{K} ({why}) failed to enter "
        f"the streamk dispatch branch (got {cap['branch']!r})."
    )
