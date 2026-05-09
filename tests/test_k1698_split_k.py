"""K-1698: correctness + env-gate tests for the persistent split-K
+ BK=128 + NS=3 prototype kernel.

The kernel itself is a *negative-result* prototype shipped behind the
TBLAS_K1698_SPLIT_K env knob (default 0 = off, no production code path
change).  These tests assert two narrow things so the prototype cannot
silently regress correctness or unintentionally activate:

  1. happy_path:       output matches torch.matmul within bf16 / fp16
                       tolerances on a divisible shape (M=N=K=4096).
  2. edge_undivisible: the eligibility predicate refuses to dispatch
                       when K is NOT divisible by SPLIT_K * BK128, so
                       the prototype falls through to the production
                       persistent_matmul path and still produces a
                       numerically correct answer.
  3. env_gate:         with TBLAS_K1698_SPLIT_K unset / "0", the
                       eligibility predicate returns False for every
                       cell and the production path is taken.

The kernel is launched in a subprocess so the env-knob is read by the
fresh tritonblas import (the env is captured at module import time).
"""
import os
import subprocess
import sys
import textwrap

import pytest

# Each test runs the kernel in a subprocess with the env knob set
# explicitly.  The subprocess prints "OK <key>=<value>" lines on
# success and raises on failure, so the parent test simply asserts
# the subprocess exited 0 and the expected OK markers are in stdout.
_RUNNER = textwrap.dedent("""
    import os, sys, torch
    import tritonblas
    from tritonblas import matmul as _mm

    def _run(M, N, K, dtype, expect_split_k_eligible, label):
        torch.manual_seed(0)
        a = torch.randn(M, K, device='cuda', dtype=dtype) * 0.05
        b = torch.randn(K, N, device='cuda', dtype=dtype) * 0.05
        # Direct probe of the eligibility predicate -- this is what
        # the dispatch sites in matmul.py / matmul_lt / _matmul_out
        # consult before routing through the prototype.
        eligible = _mm._k1698_eligible(M, N, K, dtype, None)
        assert eligible is expect_split_k_eligible, (
            f'{label}: expected _k1698_eligible={expect_split_k_eligible} '
            f'got {eligible}; SPLIT_K={_mm._K1698_SPLIT_K} '
            f'BK={_mm._K1698_BK}')

        c = tritonblas.matmul(a, b)
        ref = (a.float() @ b.float()).to(dtype)
        # Match tritonblas test_matmul_correctness.py tolerances.
        if dtype == torch.bfloat16:
            tol = dict(rtol=5e-2, atol=5e-2)
        else:
            tol = dict(rtol=1e-2, atol=1e-2)
        assert torch.allclose(c, ref, **tol), (
            f'{label}: numerical mismatch '
            f'max_abs={(c.float()-ref.float()).abs().max().item():.4g}')
        print(f'OK {label}')

    case = sys.argv[1]
    if case == 'happy_path_split_k':
        _run(4096, 4096, 4096, torch.bfloat16, True, 'happy_path_split_k')
    elif case == 'edge_undivisible_falls_through':
        # K=4097 is not divisible by SPLIT_K*BK -> ineligible -> fall
        # through to production path; result must still be correct.
        _run(256, 256, 4097, torch.float16, False,
             'edge_undivisible_falls_through')
    elif case == 'env_gate_off':
        # SPLIT_K env not set / 0 -> _k1698_eligible must be False
        # even on a divisible shape.
        _run(4096, 4096, 4096, torch.bfloat16, False, 'env_gate_off')
    else:
        raise SystemExit(f'unknown case: {case}')
""")


def _spawn(env_extra, case):
    env = os.environ.copy()
    # Make sure no inherited TBLAS_K1698_* knobs from the calling shell
    # contaminate the test.
    for k in list(env):
        if k.startswith("TBLAS_K1698_"):
            del env[k]
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", _RUNNER, case],
        env=env, capture_output=True, text=True, timeout=600)


def _gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gpu_available(), reason="K-1698 split-K prototype requires CUDA/HIP GPU")


def test_k1698_happy_path_split_k_correctness():
    """SPLIT_K=4 BK=128 NS=3 on M=N=K=4096 bf16 must agree with torch.matmul."""
    p = _spawn({"TBLAS_K1698_SPLIT_K": "4",
                "TBLAS_K1698_BK": "128",
                "TBLAS_K1698_NS": "3",
                "TBLAS_K1698_BM": "64",
                "TBLAS_K1698_BN": "64"},
               "happy_path_split_k")
    assert p.returncode == 0, (
        f"subprocess failed:\nSTDOUT={p.stdout}\nSTDERR={p.stderr}")
    assert "OK happy_path_split_k" in p.stdout


def test_k1698_edge_K_not_divisible_falls_through():
    """K not divisible by SPLIT_K*BK -> ineligible -> production path."""
    p = _spawn({"TBLAS_K1698_SPLIT_K": "4",
                "TBLAS_K1698_BK": "128",
                "TBLAS_K1698_NS": "3"},
               "edge_undivisible_falls_through")
    assert p.returncode == 0, (
        f"subprocess failed:\nSTDOUT={p.stdout}\nSTDERR={p.stderr}")
    assert "OK edge_undivisible_falls_through" in p.stdout


def test_k1698_env_gate_off_by_default():
    """With TBLAS_K1698_SPLIT_K unset, _k1698_eligible must be False."""
    p = _spawn({}, "env_gate_off")  # explicitly no SPLIT_K env var
    assert p.returncode == 0, (
        f"subprocess failed:\nSTDOUT={p.stdout}\nSTDERR={p.stderr}")
    assert "OK env_gate_off" in p.stdout
