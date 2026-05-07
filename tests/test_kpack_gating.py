"""Unit tests for the kpack gating heuristic.

Background
----------
kpack=2 packs two MFMA K-elements per VALU lane and emits an extra
LDS-swizzle/repack sequence in the K-prologue.  The setup cost is amortized
across the K-loop, so it is a net win on large-K shapes but a regression on
small-K shapes (e.g. M,N in {512..2048}, K=512).

The ``select_kpack`` helper guarantees that requests for kpack=2 only succeed
when the K-dimension is large enough to amortize the extra prologue, and that
unrequested or invalid values always collapse to kpack=1 (the safe baseline).
"""

import os

import pytest

from tritonblas.matmul import (
    _DEFAULT_KPACK2_K_MIN,
    _env_kpack2_k_min,
    select_kpack,
)


class TestSelectKpack:
    def test_default_returns_one(self):
        # When the caller does not request kpack=2, K is irrelevant.
        assert select_kpack(K=128) == 1
        assert select_kpack(K=4096) == 1

    @pytest.mark.parametrize("K", [128, 256, 511])
    def test_small_k_clamps_to_one(self, K):
        # Even when kpack=2 is requested, small K must fall back to 1.
        assert select_kpack(K=K, requested_kpack=2) == 1

    @pytest.mark.parametrize("K", [_DEFAULT_KPACK2_K_MIN, _DEFAULT_KPACK2_K_MIN + 1, 8192])
    def test_large_k_honors_request(self, K):
        # K at or above the default threshold honors kpack=2.
        assert select_kpack(K=K, requested_kpack=2) == 2

    def test_default_threshold_value(self):
        # Empirical small-K cohort regression boundary — the default must
        # remain at 1024 unless a follow-up sweep moves it.
        assert _DEFAULT_KPACK2_K_MIN == 1024

    def test_explicit_threshold_override(self):
        # Caller-supplied threshold takes precedence over the default.
        assert select_kpack(K=600, requested_kpack=2, k_min_for_kpack2=512) == 2
        assert select_kpack(K=600, requested_kpack=2, k_min_for_kpack2=1024) == 1

    @pytest.mark.parametrize("requested", [0, -1, 3, 4, 8])
    def test_invalid_request_clamps_to_one(self, requested):
        # Out-of-range kpack requests degrade safely to kpack=1.
        assert select_kpack(K=4096, requested_kpack=requested) == 1


class TestEnvOverride:
    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("TRITONBLAS_KPACK2_K_MIN", raising=False)
        assert _env_kpack2_k_min() == _DEFAULT_KPACK2_K_MIN

    def test_invalid_returns_default(self, monkeypatch):
        monkeypatch.setenv("TRITONBLAS_KPACK2_K_MIN", "not-a-number")
        assert _env_kpack2_k_min() == _DEFAULT_KPACK2_K_MIN

    def test_nonpositive_returns_default(self, monkeypatch):
        monkeypatch.setenv("TRITONBLAS_KPACK2_K_MIN", "0")
        assert _env_kpack2_k_min() == _DEFAULT_KPACK2_K_MIN
        monkeypatch.setenv("TRITONBLAS_KPACK2_K_MIN", "-512")
        assert _env_kpack2_k_min() == _DEFAULT_KPACK2_K_MIN

    def test_valid_override(self, monkeypatch):
        monkeypatch.setenv("TRITONBLAS_KPACK2_K_MIN", "768")
        assert _env_kpack2_k_min() == 768
        # And it should propagate through select_kpack when threshold is None.
        assert select_kpack(K=768, requested_kpack=2) == 2
        assert select_kpack(K=767, requested_kpack=2) == 1


class TestKpackForSelector:
    """``_kpack_for_selector`` reads ``selector.kpack`` and gates it on K."""

    def test_selector_without_kpack_attribute(self):
        from tritonblas.matmul import _kpack_for_selector

        class Sel:  # no kpack attribute -> default to 1
            pass

        assert _kpack_for_selector(Sel(), K=8192) == 1

    def test_selector_with_kpack_two_small_k(self):
        from tritonblas.matmul import _kpack_for_selector

        class Sel:
            kpack = 2

        assert _kpack_for_selector(Sel(), K=512) == 1

    def test_selector_with_kpack_two_large_k(self):
        from tritonblas.matmul import _kpack_for_selector

        class Sel:
            kpack = 2

        assert _kpack_for_selector(Sel(), K=4096) == 2
