"""Tests for the kpack gating heuristic.

The gate in ``include/tritonblas/matmul.py`` exists to protect the small-K
cohort (K=512, M=N in {512..2048}) from a 25-30% regression that prior
kpack=2 deployments produced when an LDS-swizzle prologue ran without
enough K-iterations to amortize.

Layered tests:

  * Unit tests (``TestSelectKpack``) — six pure-Python cases covering
    default behavior, the small-K clamp, the large-K pass-through, the
    invalid-request clamp, the fixed threshold, and selector resolution.
    These are the load-bearing invariants — if they break, the gate is
    not actually doing its job and downstream callers can re-introduce
    the small-K regression.

  * GPU watchdog (``TestKpackGatingGPU``) — exercises the gate
    end-to-end through ``persistent_matmul_lt`` on the small-K cohort
    the gate exists to defend, asserts numerical correctness on both
    sides of the gate, and emits a structured ``PERF_REGRESSION_REPORT``
    line so reviewers can see the live gap between gated (kpack=1) and
    ungated (kpack=2) latency.  Skipped automatically when CUDA / a
    gfx942 device is not available.
"""

from __future__ import annotations

import statistics

import pytest
import torch

from tritonblas.matmul import (
    _DEFAULT_KPACK2_K_MIN,
    _kpack_for_selector,
    select_kpack,
)


# ---------------------------------------------------------------------------
# Unit tests — six cases covering the full decision table.
# ---------------------------------------------------------------------------
class TestSelectKpack:
    def test_default_request_returns_one(self):
        # No caller asks for kpack=2 -> always 1, regardless of K.
        assert select_kpack(K=128) == 1
        assert select_kpack(K=4096) == 1

    @pytest.mark.parametrize("K", [128, 512, _DEFAULT_KPACK2_K_MIN - 1])
    def test_small_k_clamps_kpack_two_to_one(self, K):
        assert select_kpack(K=K, requested_kpack=2) == 1

    @pytest.mark.parametrize("K", [_DEFAULT_KPACK2_K_MIN, 4096, 8192])
    def test_large_k_honors_kpack_two(self, K):
        assert select_kpack(K=K, requested_kpack=2) == 2

    @pytest.mark.parametrize("requested", [0, -1, 3, 8])
    def test_invalid_request_clamps_to_one(self, requested):
        assert select_kpack(K=4096, requested_kpack=requested) == 1

    def test_threshold_is_pinned_at_1024(self):
        # Empirical small-K boundary; bumping this constant requires a
        # follow-up sweep across the K-539 cohort.
        assert _DEFAULT_KPACK2_K_MIN == 1024

    def test_kpack_for_selector_resolves_attribute(self):
        class NoKpack:
            pass

        class WantsTwo:
            kpack = 2

        # Missing attribute defaults to kpack=1.
        assert _kpack_for_selector(NoKpack(), K=8192) == 1
        # Explicit request, gated on K.
        assert _kpack_for_selector(WantsTwo(), K=512) == 1
        assert _kpack_for_selector(WantsTwo(), K=4096) == 2


# ---------------------------------------------------------------------------
# GPU watchdog — runs the gate through a real Triton kernel and emits the
# perf-regression report the PR review asked for.
# ---------------------------------------------------------------------------
try:
    from tritonblas import OrigamiMatmulSelector
    from tritonblas.matmul import persistent_matmul_lt
    _HAVE_TRITONBLAS = True
except Exception as _exc:  # pragma: no cover - import diagnostic
    _HAVE_TRITONBLAS = False
    _IMPORT_ERR = repr(_exc)
else:
    _IMPORT_ERR = None


def _is_gfx942() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
        return "gfx942" in arch
    except Exception:
        return False


_SMALL_K = 512
_MN_SHAPES = [(512, 512), (1024, 1024), (1536, 1536), (2048, 2048)]
_DTYPE = torch.bfloat16


def _make_selector(M, N, K, kpack):
    sel = OrigamiMatmulSelector(
        M, N, K,
        a_dtype=_DTYPE, b_dtype=_DTYPE, out_dtype=_DTYPE,
        device=torch.device("cuda:0"),
    )
    sel.kpack = int(kpack)
    return sel


def _bench(M, N, K, requested_kpack, warmup=10, iters=50):
    a = torch.randn(M, K, device="cuda", dtype=_DTYPE)
    b = torch.randn(K, N, device="cuda", dtype=_DTYPE)
    c = torch.empty(M, N, device="cuda", dtype=_DTYPE)
    sel = _make_selector(M, N, K, requested_kpack)
    eff = _kpack_for_selector(sel, K)
    for _ in range(warmup):
        persistent_matmul_lt(a, b, c, sel)
    torch.cuda.synchronize()
    samples = []
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        s.record()
        persistent_matmul_lt(a, b, c, sel)
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    return eff, statistics.median(samples), c, a, b


@pytest.mark.skipif(
    not (_HAVE_TRITONBLAS and torch.cuda.is_available() and _is_gfx942()),
    reason=f"requires CUDA + tritonblas + gfx942 (import_ok={_HAVE_TRITONBLAS}, err={_IMPORT_ERR})",
)
class TestKpackGatingGPU:
    """End-to-end watchdog: the gate must clamp small-K, both sides of the
    gate must be numerically correct, and the recorded perf gap should
    justify the gate's existence."""

    def test_gate_clamps_and_promotes_correctly(self):
        # `selector.kpack=2` on small K must collapse to effective=1.
        for (M, N) in _MN_SHAPES:
            sel = _make_selector(M, N, _SMALL_K, kpack=2)
            assert _kpack_for_selector(sel, _SMALL_K) == 1, (
                f"gate failed at M={M} N={N} K={_SMALL_K}"
            )
        # `selector.kpack=2` on large K passes through.
        sel_large = _make_selector(1024, 1024, 4096, kpack=2)
        assert _kpack_for_selector(sel_large, 4096) == 2

    def test_perf_regression_report(self):
        """Records the live kpack=1 vs kpack=2 gap on the K=512 cohort.

        Behavioural assertion: numerical output is correct on both sides
        of the gate.  The perf delta itself is recorded via a structured
        PERF_REGRESSION_REPORT line (NOT a hard assertion — the size of
        the regression is a property of the underlying Triton/HW combo
        and a hard threshold would flake on every Triton bump).
        """
        per_shape = []
        for (M, N) in _MN_SHAPES:
            # Baseline: gated path (effective kpack=1).
            eff1, t1, c1, a, b = _bench(M, N, _SMALL_K, requested_kpack=1)
            assert eff1 == 1
            ref = torch.matmul(a, b)
            assert torch.allclose(c1, ref, atol=2e-2, rtol=2e-2), (
                f"kpack=1 numerically wrong at M={M} N={N}"
            )
            # Probe ungated kpack=2 by bypassing the gate at the call site.
            sel2 = _make_selector(M, N, _SMALL_K, kpack=2)
            c2 = torch.empty(M, N, device="cuda", dtype=_DTYPE)

            # Reach past the gate: monkey-patch _kpack_for_selector for
            # this single benchmark loop so we can measure raw kpack=2.
            from tritonblas import matmul as _mm

            orig = _mm._kpack_for_selector
            try:
                _mm._kpack_for_selector = lambda sel, K: 2
                for _ in range(10):
                    persistent_matmul_lt(a, b, c2, sel2)
                torch.cuda.synchronize()
                samples = []
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                for _ in range(50):
                    s.record()
                    persistent_matmul_lt(a, b, c2, sel2)
                    e.record()
                    torch.cuda.synchronize()
                    samples.append(s.elapsed_time(e))
                t2 = statistics.median(samples)
            finally:
                _mm._kpack_for_selector = orig

            assert torch.allclose(c2, ref, atol=2e-2, rtol=2e-2), (
                f"kpack=2 numerically wrong at M={M} N={N}"
            )
            per_shape.append((M, N, t1, t2))
            print(
                f"K={_SMALL_K} M={M} N={N}: kpack=1 {t1:.3f} ms, "
                f"kpack=2 {t2:.3f} ms, delta {(t2 - t1) / t2 * 100:+.2f}%"
            )

        median_gated = statistics.median(t for *_, t, _ in per_shape)
        median_ungated = statistics.median(t for *_, _, t in per_shape)
        speedup = (median_ungated - median_gated) / median_ungated
        print(
            f"PERF_REGRESSION_REPORT cohort=K{_SMALL_K}_MN{{512..2048}} "
            f"median_kpack1_ms={median_gated:.4f} "
            f"median_kpack2_ms={median_ungated:.4f} "
            f"gate_speedup={speedup * 100:+.2f}%"
        )
