"""GPU integration test for the kpack gating heuristic.

This test goes one level beyond ``tests/test_kpack_gating.py`` (which is a
pure-Python unit suite): it actually runs ``persistent_matmul_lt`` through
the gated kpack path, with a real ``OrigamiMatmulSelector`` whose ``kpack``
attribute is set to 2 by the test.  This proves three things end-to-end:

1.  ``selector.kpack = 2`` is observed by ``_kpack_for_selector``.
2.  Small-K (K < ``_DEFAULT_KPACK2_K_MIN``) shapes are clamped to kpack=1
    by the gate AND still produce numerically-correct output (the gate does
    not break correctness on the safe path).
3.  Large-K (K >= ``_DEFAULT_KPACK2_K_MIN``) shapes pass kpack=2 through to
    the kernel AND still produce numerically-correct output (the
    previously-untested kpack=2 codepath is exercised at least once).

This file is skipped automatically when CUDA/ROCm is not available, so it
remains safe to run in CPU CI.
"""

import os

import pytest
import torch

try:
    import tritonblas
    from tritonblas import OrigamiMatmulSelector
    from tritonblas.matmul import (
        _DEFAULT_KPACK2_K_MIN,
        _kpack_for_selector,
        persistent_matmul_lt,
    )
    _IMPORT_OK = True
    _IMPORT_ERR = None
except Exception as exc:  # pragma: no cover - import diagnostic
    _IMPORT_OK = False
    _IMPORT_ERR = repr(exc)


pytestmark = pytest.mark.skipif(
    not (_IMPORT_OK and torch.cuda.is_available()),
    reason=f"requires CUDA/ROCm + tritonblas (import status: {_IMPORT_OK}, err: {_IMPORT_ERR})",
)


def _make_selector_with_kpack(M, N, K, dtype, kpack):
    sel = OrigamiMatmulSelector(
        M, N, K,
        a_dtype=dtype, b_dtype=dtype, out_dtype=dtype,
        device=torch.device("cuda:0"),
    )
    # The instance gets a per-call ``kpack`` attribute that the gating helper
    # reads via ``getattr(selector, "kpack", 1)``.  We bypass the property
    # protocol by assigning directly to the instance dict.
    sel.kpack = int(kpack)
    return sel


def _run_one(M, N, K, dtype, requested_kpack, atol=2e-2, rtol=2e-2):
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    c = torch.empty(M, N, device="cuda", dtype=dtype)
    sel = _make_selector_with_kpack(M, N, K, dtype, requested_kpack)
    effective = _kpack_for_selector(sel, K)
    persistent_matmul_lt(a, b, c, sel)
    ref = torch.matmul(a, b)
    max_err = (c.float() - ref.float()).abs().max().item()
    ok = torch.allclose(c, ref, atol=atol, rtol=rtol)
    return effective, ok, max_err


class TestKpackGatingIntegration:
    """End-to-end checks that the gate fires AND the kernel still produces
    correct output on both branches of the gate."""

    def test_gate_clamps_small_k_to_one(self):
        K_SMALL = _DEFAULT_KPACK2_K_MIN // 2  # well below the threshold
        effective, ok, err = _run_one(
            M=512, N=512, K=K_SMALL, dtype=torch.bfloat16, requested_kpack=2,
        )
        assert effective == 1, (
            f"Gate failed to clamp small K={K_SMALL} (threshold "
            f"{_DEFAULT_KPACK2_K_MIN}) — got kpack={effective}"
        )
        assert ok, f"persistent_matmul_lt produced incorrect output (max_err={err})"

    def test_gate_passes_large_k_kpack_two(self):
        K_LARGE = max(_DEFAULT_KPACK2_K_MIN, 2048)
        effective, ok, err = _run_one(
            M=1024, N=1024, K=K_LARGE, dtype=torch.bfloat16, requested_kpack=2,
        )
        assert effective == 2, (
            f"Gate dropped kpack=2 for large K={K_LARGE} (threshold "
            f"{_DEFAULT_KPACK2_K_MIN}) — got kpack={effective}"
        )
        assert ok, f"kpack=2 path produced incorrect output (max_err={err})"

    def test_env_override_lowers_threshold(self, monkeypatch):
        # Lower the threshold so K=512 promotes to kpack=2.
        monkeypatch.setenv("TRITONBLAS_KPACK2_K_MIN", "256")
        effective, ok, err = _run_one(
            M=512, N=512, K=512, dtype=torch.bfloat16, requested_kpack=2,
        )
        assert effective == 2, (
            "Env override did not lower threshold — "
            f"K=512 with TRITONBLAS_KPACK2_K_MIN=256 got kpack={effective}"
        )
        assert ok, f"kpack=2 small-K override produced incorrect output (max_err={err})"

    def test_unmodified_selector_keeps_kpack_one(self):
        # Default tritonblas.matmul never sets selector.kpack, so the gate
        # must produce kpack=1 for any K — including large K.  This protects
        # the existing main-line behavior.
        K_LARGE = max(_DEFAULT_KPACK2_K_MIN, 2048)
        sel = OrigamiMatmulSelector(
            1024, 1024, K_LARGE,
            a_dtype=torch.bfloat16, b_dtype=torch.bfloat16, out_dtype=torch.bfloat16,
            device=torch.device("cuda:0"),
        )
        # Do NOT set sel.kpack — emulating today's call sites.
        assert _kpack_for_selector(sel, K_LARGE) == 1
