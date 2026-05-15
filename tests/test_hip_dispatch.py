"""K-7078: unit tests for the HIP fallback dispatch predicate, the
M-padding wrapper used for the gemv-shaped 1x4096x4096 cohort entry,
and the TRITONBLAS_HIP_DISPATCH_LOG audit hook.

Predicate tests are pure-Python and run on any host (no GPU / no .so
needed).  Numerical-correctness tests for `hip_interleaved_matmul` are
skipped when the kernel .so is unavailable or the GPU isn't gfx942.
The dispatch-log tests are host-runnable except for the end-to-end
`_try_hip_fallback` checks, which need CUDA to import `tritonblas.matmul`.
"""
from __future__ import annotations

import pytest
import torch

from tritonblas.hip_kernels import (
    HIP_FALLBACK_SHAPES,
    hip_kernel_available,
    should_use_hip_fallback,
)
from tritonblas.hip_kernels import dispatch as hip_dispatch


# --- predicate tests (run anywhere — no GPU/.so required) -------------

class TestShouldUseHipFallback:
    """K-7078: cover every documented branch of `should_use_hip_fallback`.

    Reviewer (Testing Zealot) flagged that the predicate had only manual
    coverage via verify_routing.py; this class makes it executable in CI.
    """

    # The S-002 cohort the dispatch table targets.  Hard-coded so an
    # accidental table change is caught (don't read from
    # HIP_FALLBACK_SHAPES — that would tautologically pass).
    S002_COHORT = [
        (64, 64, 4096),
        (128, 1024, 8192),
        (1, 4096, 4096),
    ]

    @pytest.mark.parametrize("M,N,K", S002_COHORT)
    def test_cohort_shape_bf16_routes_to_hip(self, M, N, K):
        assert should_use_hip_fallback(M, N, K, torch.bfloat16) is True

    @pytest.mark.parametrize("M,N,K", S002_COHORT)
    def test_cohort_shape_fp16_does_not_route(self, M, N, K):
        # Kernel is BF16-only; predicate must reject other dtypes.
        assert should_use_hip_fallback(M, N, K, torch.float16) is False

    @pytest.mark.parametrize("M,N,K", S002_COHORT)
    def test_cohort_shape_fp32_does_not_route(self, M, N, K):
        assert should_use_hip_fallback(M, N, K, torch.float32) is False

    def test_table_matches_documented_cohort(self):
        # Guard against stale entries leaking back in (Minimalist review).
        documented = {(M, N, K, torch.bfloat16) for (M, N, K) in self.S002_COHORT}
        assert set(HIP_FALLBACK_SHAPES) == documented

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (32, 32, 4096),       # square but not in table
            (64, 64, 1024),       # in-table M,N but K too small
            (256, 256, 4096),     # the dropped K-6953 entry — must NOT route
            (1024, 1024, 4096),   # the other dropped K-6953 entry
            (1, 4096, 8192),      # near-cohort but K differs
        ],
    )
    def test_non_cohort_shapes_do_not_route(self, M, N, K):
        assert should_use_hip_fallback(M, N, K, torch.bfloat16) is False

    def test_gemv_path_accepted_when_n_k_aligned(self):
        # M < BLOCK_M is OK as long as N % 16 == 0 and K % 64 == 0 AND
        # the full (M,N,K,dtype) tuple is in the table.
        assert should_use_hip_fallback(1, 4096, 4096, torch.bfloat16) is True

    def test_gemv_path_rejected_when_not_in_table(self):
        # Even if N/K alignment is satisfied, a shape not in the table
        # must not route — predicate is a table lookup, not a tile check.
        assert should_use_hip_fallback(1, 1024, 4096, torch.bfloat16) is False


# --- numerical correctness for the M-padding wrapper -------------------

# The Testing Zealot specifically called out: "no correctness check that
# padded M=1 output matches reference".  This class provides exactly that
# (and the M=64 baseline as a control).
_SKIP_NO_KERNEL = pytest.mark.skipif(
    not hip_kernel_available(),
    reason="HIP kernel .so unavailable or not gfx942 (skipping numerical tests)",
)


@_SKIP_NO_KERNEL
class TestHipInterleavedMatmulNumerics:
    """Numerical correctness of `hip_interleaved_matmul`, focusing on the
    K-7078 M-padding code path.
    """

    @staticmethod
    def _bf16_tolerance(K: int) -> float:
        # BF16 has 8 mantissa bits; accumulated error grows ~sqrt(K) * eps.
        # Empirically K=4096..8192 lands well under 2.0 absolute on
        # randn(0, 1) inputs.  Match the verify_routing.py tolerance band.
        return 2.0

    def _reference(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # FP32 reference; cast back to BF16 to match the kernel output dtype.
        return (a.to(torch.float32) @ b.to(torch.float32)).to(a.dtype)

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (64, 64, 4096),       # M aligned — control case
            (128, 1024, 8192),    # M aligned — full cohort coverage
        ],
    )
    def test_aligned_m_matches_reference(self, M, N, K):
        torch.manual_seed(0)
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

        c = hip_dispatch.hip_interleaved_matmul(a, b)
        ref = self._reference(a, b)

        assert c.shape == (M, N)
        assert c.dtype == torch.bfloat16
        max_err = (c.float() - ref.float()).abs().max().item()
        assert max_err < self._bf16_tolerance(K), (
            f"M={M} N={N} K={K}: max_abs_err={max_err}"
        )

    def test_padded_m1_matches_reference(self):
        # The K-7078 gemv-shaped cohort entry — the wrapper zero-pads A
        # from (1, K) up to (16, K), launches the 16x16x16 MFMA kernel,
        # and slices output[:1] back.  Padded rows are all zero, so the
        # first output row must equal A_real @ B exactly within BF16
        # accumulator tolerance.
        M, N, K = 1, 4096, 4096
        torch.manual_seed(1)
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

        c = hip_dispatch.hip_interleaved_matmul(a, b)
        ref = self._reference(a, b)

        assert c.shape == (M, N), (
            f"M-padded output must be sliced back to M=1, got {c.shape}"
        )
        max_err = (c.float() - ref.float()).abs().max().item()
        assert max_err < self._bf16_tolerance(K), (
            f"padded M=1 output diverges: max_abs_err={max_err}"
        )

    def test_padded_m4_matches_reference(self):
        # A second padding case: M=4 → pads to 16, slices back to 4.
        # Catches any off-by-one in the slice (e.g. taking output[:M+1]).
        # Not in HIP_FALLBACK_SHAPES — call wrapper directly.
        M, N, K = 4, 16, 64
        torch.manual_seed(2)
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

        c = hip_dispatch.hip_interleaved_matmul(a, b)
        ref = self._reference(a, b)

        assert c.shape == (M, N)
        max_err = (c.float() - ref.float()).abs().max().item()
        assert max_err < self._bf16_tolerance(K), (
            f"M=4 padded output diverges: max_abs_err={max_err}"
        )

    def test_padded_output_into_user_buffer(self):
        # `out=` path: caller-provided output buffer must be filled and
        # returned (mirrors tritonblas.matmul semantics).
        M, N, K = 1, 4096, 4096
        torch.manual_seed(3)
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

        ret = hip_dispatch.hip_interleaved_matmul(a, b, out=out)
        ref = self._reference(a, b)

        assert ret.data_ptr() == out.data_ptr(), (
            "with out=, wrapper must write into and return the user buffer"
        )
        max_err = (out.float() - ref.float()).abs().max().item()
        assert max_err < self._bf16_tolerance(K)


# --- TRITONBLAS_HIP_DISPATCH_LOG audit hook ----------------------------

class TestDispatchLogging:
    """K-7078: protect the K-7054 audit instrumentation.

    Reviewer (Testing Zealot) flagged that without a regression test for
    the `_log_dispatch()` / `TRITONBLAS_HIP_DISPATCH_LOG=1` hook, a future
    edit could silently disable it and re-create the exact bug this PR
    exists to prevent.  These tests pin two invariants:

      1. The env var gate is consulted at *call* time (so a runtime
         toggle works — the audit script and these tests rely on it).
      2. Both the HIP-taken and HIP-skipped branches emit a log line
         tagged with the path taken.

    The direct `_log_dispatch` tests are host-runnable.  The end-to-end
    `_try_hip_fallback` tests need CUDA only because `tritonblas.matmul`
    queries `torch.cuda.current_device()` at import time.
    """

    def test_env_unset_is_silent(self, capsys, monkeypatch):
        monkeypatch.delenv("TRITONBLAS_HIP_DISPATCH_LOG", raising=False)
        hip_dispatch._log_dispatch("any reason", 64, 64, 4096, torch.bfloat16, took_hip=True)
        hip_dispatch._log_dispatch("any reason", 64, 64, 4096, torch.bfloat16, took_hip=False)
        assert capsys.readouterr().out == ""

    def test_env_zero_is_silent(self, capsys, monkeypatch):
        monkeypatch.setenv("TRITONBLAS_HIP_DISPATCH_LOG", "0")
        hip_dispatch._log_dispatch("any reason", 64, 64, 4096, torch.bfloat16, took_hip=True)
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize("truthy", ["1", "true", "yes", "TRUE", "Yes"])
    def test_env_truthy_emits_on_hip_taken(self, capsys, monkeypatch, truthy):
        monkeypatch.setenv("TRITONBLAS_HIP_DISPATCH_LOG", truthy)
        hip_dispatch._log_dispatch("dispatched", 64, 64, 4096, torch.bfloat16, took_hip=True)
        out = capsys.readouterr().out
        assert "-> HIP" in out, out
        assert "M=64 N=64 K=4096" in out
        assert "dispatched" in out

    def test_env_truthy_emits_on_hip_skipped(self, capsys, monkeypatch):
        monkeypatch.setenv("TRITONBLAS_HIP_DISPATCH_LOG", "1")
        hip_dispatch._log_dispatch("not in table", 32, 32, 4096, torch.bfloat16, took_hip=False)
        out = capsys.readouterr().out
        assert "-> TRITON" in out, out
        assert "M=32 N=32 K=4096" in out
        assert "not in table" in out

    def test_env_is_read_per_call(self, capsys, monkeypatch):
        # Toggle the env var between two calls and confirm only the
        # second one emits.  Pins the "no module-level capture" contract.
        monkeypatch.delenv("TRITONBLAS_HIP_DISPATCH_LOG", raising=False)
        hip_dispatch._log_dispatch("first", 64, 64, 4096, torch.bfloat16, took_hip=True)
        monkeypatch.setenv("TRITONBLAS_HIP_DISPATCH_LOG", "1")
        hip_dispatch._log_dispatch("second", 64, 64, 4096, torch.bfloat16, took_hip=True)
        out = capsys.readouterr().out
        assert "first" not in out
        assert "second" in out


# End-to-end audit-hook tests through `_try_hip_fallback`.  These require
# CUDA only because `tritonblas.matmul` does `torch.cuda.current_device()`
# at import time — the *logic* under test is pure Python.
_SKIP_NO_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="tritonblas.matmul cannot be imported without CUDA",
)


@_SKIP_NO_CUDA
class TestTryHipFallbackLogging:
    """K-7078: end-to-end check that `_try_hip_fallback` emits a log line
    on every exit branch (the K-7054 audit hook).  Each branch is reached
    by setting the conditions that gate it; we don't actually need the
    HIP kernel to execute, just to observe the log line for each path.
    """

    @pytest.fixture
    def matmul_module(self):
        # Lazy import: the module pulls in CUDA at import time.
        # `from tritonblas import matmul` would bind the *function*
        # (re-exported in tritonblas/__init__.py); we want the module
        # itself so we can monkey-patch `_HIP_FALLBACK_ENABLED`.
        import importlib
        return importlib.import_module("tritonblas.matmul")

    def _make_inputs(self, M, N, K):
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        return a, b

    def test_logs_on_disabled_by_env_branch(self, capsys, monkeypatch, matmul_module):
        monkeypatch.setenv("TRITONBLAS_HIP_DISPATCH_LOG", "1")
        monkeypatch.setattr(matmul_module, "_HIP_FALLBACK_ENABLED", False)
        a, b = self._make_inputs(64, 64, 4096)  # would otherwise route to HIP
        out = matmul_module._try_hip_fallback(a, b, out=None)
        assert out is None  # disabled → falls through
        captured = capsys.readouterr().out
        assert "-> TRITON" in captured
        assert "disabled by env" in captured

    def test_logs_on_shape_not_eligible_branch(self, capsys, monkeypatch, matmul_module):
        monkeypatch.setenv("TRITONBLAS_HIP_DISPATCH_LOG", "1")
        monkeypatch.setattr(matmul_module, "_HIP_FALLBACK_ENABLED", True)
        a, b = self._make_inputs(32, 32, 4096)  # not in HIP_FALLBACK_SHAPES
        out = matmul_module._try_hip_fallback(a, b, out=None)
        assert out is None
        captured = capsys.readouterr().out
        assert "-> TRITON" in captured
        assert "shape not in HIP_FALLBACK_SHAPES" in captured

    def test_logs_on_so_unavailable_branch(self, capsys, monkeypatch, matmul_module):
        # Force the .so-availability check to report False, even on a
        # gfx942 box where it would normally pass.  Confirms the log
        # fires on this branch too.
        monkeypatch.setenv("TRITONBLAS_HIP_DISPATCH_LOG", "1")
        monkeypatch.setattr(matmul_module, "_HIP_FALLBACK_ENABLED", True)
        monkeypatch.setattr(matmul_module, "hip_kernel_available", lambda: False)
        a, b = self._make_inputs(64, 64, 4096)
        out = matmul_module._try_hip_fallback(a, b, out=None)
        assert out is None
        captured = capsys.readouterr().out
        assert "-> TRITON" in captured
        assert ".so unavailable" in captured or "not gfx942" in captured

    @pytest.mark.skipif(
        not hip_kernel_available(),
        reason="HIP kernel .so unavailable or not gfx942 (skipping HIP-taken log test)",
    )
    def test_logs_on_dispatched_branch(self, capsys, monkeypatch, matmul_module):
        monkeypatch.setenv("TRITONBLAS_HIP_DISPATCH_LOG", "1")
        monkeypatch.setattr(matmul_module, "_HIP_FALLBACK_ENABLED", True)
        a, b = self._make_inputs(64, 64, 4096)
        result = matmul_module._try_hip_fallback(a, b, out=None)
        torch.cuda.synchronize()
        assert result is not None, "HIP path must fire for the cohort shape"
        captured = capsys.readouterr().out
        assert "-> HIP" in captured
        assert "interleaved MFMA kernel" in captured
