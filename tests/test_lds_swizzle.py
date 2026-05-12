# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Pure-Python tests for the LDS bank-conflict mitigation knob resolver.

These tests do NOT require a GPU — they only exercise
``tritonblas.lds_swizzle.resolve_swizzle`` so the precedence rules can be
validated in CI / on CPU-only build hosts.
"""

import importlib
import os

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "TRITONBLAS_KPACK",
        "TRITONBLAS_MATRIX_INSTR_NONKDIM",
        "TRITONBLAS_WAVES_PER_EU",
        "TRITONBLAS_NUM_WARPS",
        "TRITONBLAS_LDS_SWIZZLE",
        "TRITONBLAS_LDS_AUTO",
        "TRITONBLAS_BLOCK_K",
    ):
        monkeypatch.delenv(var, raising=False)


def _resolve(**kw):
    # Re-import to pick up env-var changes for any module-level caches
    # (resolver itself reads env on each call so this is just defensive).
    from tritonblas.lds_swizzle import resolve_swizzle
    return resolve_swizzle(**kw)


def test_baseline_defaults():
    cfg = _resolve()
    assert cfg.kpack == 1
    assert cfg.matrix_instr_nonkdim == 16
    assert cfg.waves_per_eu == 0
    assert cfg.num_warps == 8


def test_kwarg_overrides_baseline():
    cfg = _resolve(kpack=2, matrix_instr_nonkdim=32)
    assert cfg.kpack == 2
    assert cfg.matrix_instr_nonkdim == 32


def test_env_overrides_baseline(monkeypatch):
    monkeypatch.setenv("TRITONBLAS_KPACK", "2")
    monkeypatch.setenv("TRITONBLAS_MATRIX_INSTR_NONKDIM", "32")
    cfg = _resolve()
    assert cfg.kpack == 2
    assert cfg.matrix_instr_nonkdim == 32


def test_kwarg_overrides_env(monkeypatch):
    monkeypatch.setenv("TRITONBLAS_KPACK", "2")
    cfg = _resolve(kpack=1)
    assert cfg.kpack == 1


def test_global_opt_in_bundle(monkeypatch):
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "1")
    cfg = _resolve()
    assert cfg.kpack == 2  # mitigation bundle picks kpack=2


def test_invalid_kpack_rejected():
    with pytest.raises(ValueError):
        _resolve(kpack=3)


def test_invalid_nonkdim_rejected():
    with pytest.raises(ValueError):
        _resolve(matrix_instr_nonkdim=64)


def test_invalid_num_warps_rejected():
    with pytest.raises(ValueError):
        _resolve(num_warps=3)


def test_as_kwargs_round_trip():
    cfg = _resolve(kpack=2, matrix_instr_nonkdim=32, waves_per_eu=2, num_warps=4)
    assert cfg.as_kwargs() == {
        "kpack": 2,
        "matrix_instr_nonkdim": 32,
        "waves_per_eu": 2,
        "num_warps": 4,
        "block_k_override": None,
    }


def test_cohort_auto_picks_block_k_64():
    # Long-K small-square cohort triggers the BLOCK_K=64 + mfma=32 + kpack=2
    # auto-mitigation that drives SQ_LDS_BANK_CONFLICT to 0 in rocprof.
    cfg = _resolve(M=1024, N=1024, K=16384, BLK_M=64, BLK_N=64)
    assert cfg.block_k_override == 64
    assert cfg.matrix_instr_nonkdim == 32
    assert cfg.kpack == 2


def test_cohort_auto_disabled_by_env(monkeypatch):
    monkeypatch.setenv("TRITONBLAS_LDS_AUTO", "0")
    cfg = _resolve(M=1024, N=1024, K=16384, BLK_M=64, BLK_N=64)
    assert cfg.block_k_override is None
    assert cfg.kpack == 1


def test_cohort_skipped_outside_long_k(monkeypatch):
    cfg = _resolve(M=4096, N=4096, K=4096, BLK_M=128, BLK_N=128)
    assert cfg.block_k_override is None
    assert cfg.kpack == 1


def test_cohort_falls_back_when_blk_not_div_32(monkeypatch):
    # If user-provided BLK_M is not divisible by 32, the auto recipe must
    # downgrade mfma_nonkdim to 16 to keep the kernel launchable.
    cfg = _resolve(M=1024, N=1024, K=16384, BLK_M=16, BLK_N=16)
    assert cfg.matrix_instr_nonkdim == 16


def test_env_block_k_override():
    import os
    os.environ["TRITONBLAS_BLOCK_K"] = "32"
    try:
        cfg = _resolve(M=4096, N=4096, K=4096)
        assert cfg.block_k_override == 32
    finally:
        del os.environ["TRITONBLAS_BLOCK_K"]

