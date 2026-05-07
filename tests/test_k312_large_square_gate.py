# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the K-312 large-square cohort gate + override helpers.

These tests do NOT touch the GPU - they exercise the pure-Python predicate
and override logic added in ``tritonblas.matmul``.  The intent is to lock
in three properties, each one corresponding to a reviewer concern from the
K-312 retry round:

  1. ``_k312_large_square_gate`` matches the K-268 envelope and refuses
     small-K shapes (the K-654 anti-regression lesson).
  2. ``_k312_large_square_overrides`` is bit-identical to the input when
     the env var is unset/"0" - this is what makes the K-654 anti-regression
     gate trivially satisfied for every default-path user.
  3. ``_k312_large_square_overrides`` only applies the cohort knob set when
     BOTH the env var is "1" AND the shape gate matches.  Out-of-cohort
     shapes are passed through unchanged even when the env var is enabled.
"""

import os

import pytest

from tritonblas.matmul import (
    _k312_large_square_gate,
    _k312_large_square_overrides,
)


# ─────────────────────────────────────────────────────────────────────────────
# Gate predicate
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "M, N, K",
    [
        (4096, 4096, 4096),
        (4096, 4096, 8192),
        (4096, 4096, 16384),
        (4096, 8192, 4096),
        (4096, 8192, 8192),
        (4096, 8192, 16384),
        (8192, 4096, 4096),
        (8192, 4096, 8192),
        (8192, 4096, 16384),
        (8192, 8192, 4096),
        (8192, 8192, 8192),
        (8192, 8192, 16384),
    ],
)
def test_gate_matches_k268_envelope(M, N, K):
    """Every (M, N, K) in the literal K-268 envelope must trigger the gate."""
    assert _k312_large_square_gate(M, N, K) is True


@pytest.mark.parametrize(
    "M, N, K",
    [
        # K-654 small-K anti-regression cohort (sample): swizzle was harmful here.
        (512, 512, 256),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        # Large M*N but K below the floor - must NOT trigger.
        (4096, 4096, 2048),
        (8192, 8192, 1024),
        # Large K but M*N below the area floor - must NOT trigger.
        (1024, 1024, 16384),
        (2048, 2048, 16384),
    ],
)
def test_gate_rejects_out_of_envelope(M, N, K):
    """Shapes outside the K-268 envelope must NOT trigger the gate.

    This is the K-654 lesson: swizzle-on for small-K shapes regresses 10-20%.
    """
    assert _k312_large_square_gate(M, N, K) is False


# ─────────────────────────────────────────────────────────────────────────────
# Override helper - default-OFF bit-identity
# ─────────────────────────────────────────────────────────────────────────────

def _clear_env(monkeypatch):
    monkeypatch.delenv("TRITONBLAS_K312_LARGE_SQUARE_TILE", raising=False)


def test_overrides_bit_identical_when_env_unset(monkeypatch):
    """With the env var unset, the helper must return its inputs unchanged.

    This property is what makes the K-654 anti-regression gate trivially
    satisfied for every default-path user (including CI).
    """
    _clear_env(monkeypatch)
    # In-cohort shape - even so, no env var ⇒ no override.
    nw, wpe, kp = _k312_large_square_overrides(8192, 8192, 8192, 8, 0, 1)
    assert (nw, wpe, kp) == (8, 0, 1)


def test_overrides_bit_identical_when_env_zero(monkeypatch):
    """Explicit ``"0"`` must also produce bit-identity (no override)."""
    monkeypatch.setenv("TRITONBLAS_K312_LARGE_SQUARE_TILE", "0")
    nw, wpe, kp = _k312_large_square_overrides(8192, 8192, 8192, 8, 0, 1)
    assert (nw, wpe, kp) == (8, 0, 1)


def test_overrides_passthrough_for_out_of_cohort_when_env_on(monkeypatch):
    """Even with the env var enabled, out-of-cohort shapes must be passthrough.

    This locks in the cohort scoping - the override is NOT applied globally
    when the env var is enabled, only inside the K-268 envelope.
    """
    monkeypatch.setenv("TRITONBLAS_K312_LARGE_SQUARE_TILE", "1")
    # K-654 small-K shape, env enabled ⇒ still passthrough.
    nw, wpe, kp = _k312_large_square_overrides(512, 512, 256, 8, 0, 1)
    assert (nw, wpe, kp) == (8, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Override helper - applies the K-312 PRD knob set in-cohort
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "M, N, K",
    [
        (4096, 4096, 4096),
        (4096, 8192, 8192),
        (8192, 8192, 16384),
    ],
)
def test_overrides_apply_in_cohort_when_env_on(monkeypatch, M, N, K):
    """In-cohort + env enabled ⇒ waves_per_eu=2 and kpack=2 are applied."""
    monkeypatch.setenv("TRITONBLAS_K312_LARGE_SQUARE_TILE", "1")
    nw, wpe, kp = _k312_large_square_overrides(M, N, K, 8, 0, 1)
    # waves_per_eu and kpack are clamped to the K-312 PRD set.
    assert wpe == 2
    assert kp == 2
    # num_warps is preserved so the user can still autotune it.
    assert nw == 8


def test_overrides_preserve_user_num_warps(monkeypatch):
    """num_warps is intentionally NOT clamped by the override helper.

    This is so an autotuner that owns num_warps can still vary it inside
    the cohort without fighting the cohort gate.
    """
    monkeypatch.setenv("TRITONBLAS_K312_LARGE_SQUARE_TILE", "1")
    for nw_in in (4, 8, 16):
        nw, _, _ = _k312_large_square_overrides(8192, 8192, 8192, nw_in, 0, 1)
        assert nw == nw_in
