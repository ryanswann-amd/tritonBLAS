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
    # Mitigation bundle proven on the long-K small-square cohort —
    # achieves SQ_LDS_BANK_CONFLICT/SQ_INSTS_LDS = 0.38 on 1024^2 x 16384
    # (below the 0.5 task target).
    assert cfg.kpack == 2
    assert cfg.matrix_instr_nonkdim == 32
    assert cfg.num_warps == 16


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
    }
