# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the atomic-free split-K small-M dispatcher.

Two distinct concerns are exercised here:

  1. **Numerical correctness of the gated path** -- shapes that activate the
     ``persistent_splitk_matmul`` kernel pair must produce results bit-close
     to ``torch.matmul`` (which on ROCm dispatches to hipBLASLt).
  2. **Gate-boundary fall-through** -- shapes that *should not* hit the new
     path must short-circuit out of ``_maybe_dispatch_atomic_free_splitk``
     before any kernel is launched.  This guards against a future refactor
     accidentally widening the gate and silently regressing the larger-M /
     short-K cohorts that the persistent / stream-K paths already handle
     well.

All tests skip cleanly on hosts without a ROCm/CUDA GPU so they can run in
CPU-only CI shards.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a ROCm/CUDA GPU")

import tritonblas  # noqa: E402  (skipped above on CPU-only hosts)
from tritonblas.kernels.persistent_splitk import (  # noqa: E402
    choose_split_k,
    should_use_atomic_free_splitk,
)
from tritonblas.matmul import _maybe_dispatch_atomic_free_splitk  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Pure-Python gate predicate (no GPU required for the predicate itself,
#    but ``choose_split_k`` is a regular ``def`` so we test it here too).
# ---------------------------------------------------------------------------

class TestGatePredicate:
    """Boundary tests for ``should_use_atomic_free_splitk``.

    The gate is intentionally narrow: ``M <= 32 AND N >= 4096 AND K >= 8192
    AND SPLIT_K >= 4``.  Each test below pins exactly one of those terms to
    its boundary (or one step outside) and confirms the predicate flips.
    """

    # In-cohort baseline -- everything that the dispatcher considers a "win".
    @pytest.mark.parametrize("M,N,K,split_k", [
        (16, 4096, 8192, 4),
        (16, 4096, 16384, 8),
        (16, 8192, 8192, 8),
        (16, 8192, 16384, 8),
        (32, 4096, 8192, 4),
        (32, 8192, 16384, 8),
    ])
    def test_in_cohort_enabled(self, M, N, K, split_k):
        assert should_use_atomic_free_splitk(M, N, K, split_k) is True

    # M boundary -- 32 is in, 33 / 64 / 128 are out.
    @pytest.mark.parametrize("M", [33, 64, 128])
    def test_M_boundary_falls_through(self, M):
        assert should_use_atomic_free_splitk(M, 4096, 8192, 8) is False

    # K boundary -- 8192 is in, 4096 / 2048 are out.
    @pytest.mark.parametrize("K", [2048, 4096, 8191])
    def test_K_boundary_falls_through(self, K):
        assert should_use_atomic_free_splitk(16, 4096, K, 8) is False

    # N boundary -- 4096 is in, 1024 / 2048 are out.
    @pytest.mark.parametrize("N", [1024, 2048, 4095])
    def test_N_boundary_falls_through(self, N):
        assert should_use_atomic_free_splitk(16, N, 8192, 8) is False

    # SPLIT_K boundary -- 4 is in, 1 / 2 / 3 are out.
    @pytest.mark.parametrize("split_k", [1, 2, 3])
    def test_SPLIT_K_boundary_falls_through(self, split_k):
        assert should_use_atomic_free_splitk(16, 4096, 8192, split_k) is False


class TestChooseSplitK:
    """``choose_split_k`` returns a sane SPLIT_K (or 1 to disable)."""

    def test_below_K_threshold_returns_1(self):
        # K too small for any split to be useful.
        assert choose_split_k(M=16, K=2048, block_k=64) == 1
        assert choose_split_k(M=16, K=1024, block_k=64) == 1

    def test_block_k_zero_returns_1(self):
        # Defensive: a degenerate block_k must not divide-by-zero.
        assert choose_split_k(M=16, K=8192, block_k=0) == 1

    def test_K_8192_returns_at_least_4(self):
        # K = 8192, block_k = 64 -> 128 K-iters total, room for >= 4 splits
        # of >= 8 iters each.
        assert choose_split_k(M=16, K=8192, block_k=64) >= 4

    def test_K_16384_returns_at_least_4(self):
        assert choose_split_k(M=16, K=16384, block_k=64) >= 4


# ---------------------------------------------------------------------------
# 2. End-to-end dispatcher gating -- requires a GPU because the dispatcher
#    inspects real tensors / dtypes / device location.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_kill_switch(monkeypatch):
    """Make sure no leftover env var from a prior test affects us."""
    monkeypatch.delenv("TBLAS_DISABLE_SPLITK_SMALLM", raising=False)


def _alloc(M, N, K, dtype):
    """Allocate (a, b, c) on the current CUDA/ROCm device."""
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    c = torch.empty(M, N, device="cuda", dtype=dtype)
    return a, b, c


class TestDispatcherGating:
    """End-to-end: ``_maybe_dispatch_atomic_free_splitk`` returns ``c`` only
    when the gate fires, ``None`` otherwise."""

    # ---- in-cohort: dispatcher should run the new kernels ----
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("M,N,K", [
        (16, 4096, 8192),
        (16, 8192, 16384),
        (32, 4096, 16384),
        (32, 8192, 8192),
    ])
    def test_in_cohort_dispatches(self, M, N, K, dtype):
        a, b, c = _alloc(M, N, K, dtype)
        out = _maybe_dispatch_atomic_free_splitk(a, b, c)
        assert out is c, "dispatcher must return the output tensor when gated"
        # Numerical sanity vs torch.matmul (hipBLASLt on ROCm).
        ref = torch.matmul(a, b)
        torch.testing.assert_close(c, ref, rtol=2e-2, atol=2e-2)

    # ---- fall-through cases: dispatcher must return None ----
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("M,N,K,reason", [
        (64,   4096, 8192,  "M > 32"),
        (128,  4096, 8192,  "M > 32"),
        (16,   4096, 4096,  "K < 8192"),
        (16,   4096, 2048,  "K < 8192"),
        (16,   1024, 8192,  "N < 4096"),
        (16,   2048, 8192,  "N < 4096"),
    ])
    def test_out_of_cohort_falls_through(self, M, N, K, reason, dtype):
        a, b, c = _alloc(M, N, K, dtype)
        # Mark c so we can detect accidental writes from an unintended dispatch.
        c.fill_(float("nan"))
        out = _maybe_dispatch_atomic_free_splitk(a, b, c)
        assert out is None, f"expected fall-through ({reason}), got {out!r}"
        # The dispatcher must NOT have touched the output buffer.
        assert torch.isnan(c).all(), \
            f"fall-through case ({reason}) wrote to the output tensor"

    # ---- dtype gate: only fp16/bf16 are eligible ----
    def test_fp32_falls_through(self):
        a, b, c = _alloc(16, 4096, 8192, torch.float32)
        c.fill_(float("nan"))
        assert _maybe_dispatch_atomic_free_splitk(a, b, c) is None
        assert torch.isnan(c).all()

    # ---- kill-switch ----
    def test_env_kill_switch_falls_through(self, monkeypatch):
        monkeypatch.setenv("TBLAS_DISABLE_SPLITK_SMALLM", "1")
        a, b, c = _alloc(16, 4096, 8192, torch.float16)
        c.fill_(float("nan"))
        assert _maybe_dispatch_atomic_free_splitk(a, b, c) is None
        assert torch.isnan(c).all()


# ---------------------------------------------------------------------------
# 3. End-to-end through the public ``tritonblas.matmul`` -- the fall-through
#    cohorts must still produce correct results because they hit the
#    persistent / stream-K paths.  This is the regression guard the
#    Testing Zealot asked for.
# ---------------------------------------------------------------------------

class TestPublicAPIFallThrough:
    """``tritonblas.matmul`` must produce correct results for shapes the
    new dispatcher rejects -- proving the legacy path is still intact."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("M,N,K", [
        # Just outside the M gate -- exercises persistent path
        (64, 4096, 8192),
        # Just outside the K gate -- exercises persistent path
        (16, 4096, 4096),
        # Just outside the N gate -- exercises persistent path
        (16, 2048, 8192),
        # Far outside on every axis (sanity)
        (128, 4096, 2048),
    ])
    def test_fallthrough_path_correct(self, M, N, K, dtype):
        a = torch.randn(M, K, device="cuda", dtype=dtype)
        b = torch.randn(K, N, device="cuda", dtype=dtype)
        c = tritonblas.matmul(a, b)
        ref = torch.matmul(a, b)
        torch.testing.assert_close(c, ref, rtol=2e-2, atol=2e-2)
