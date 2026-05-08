"""K-811 — unit tests for the M=N=2048 large-K split-K=2 routed-override gate.

Pin the predicate so future cohort extensions cannot widen it accidentally
and so the env-disable flag continues to work as the paired-sweep escape
hatch (mirrors K-777's gate-test pattern).
"""
import importlib
import os
import subprocess
import sys

import pytest
import torch

# tritonblas/__init__.py shadows the .matmul submodule with the matmul()
# function via `from .matmul import matmul`, so `import tritonblas.matmul`
# returns the function, not the module. Reach the module directly via
# importlib so we can monkeypatch its module-level constants.
mm = importlib.import_module("tritonblas.matmul")


@pytest.fixture
def gate_on(monkeypatch):
    """Force module-level _K811_ENABLED to True for this test."""
    monkeypatch.setattr(mm, "_K811_ENABLED", True)
    return mm


@pytest.fixture
def gate_off(monkeypatch):
    """Force module-level _K811_ENABLED to False for this test."""
    monkeypatch.setattr(mm, "_K811_ENABLED", False)
    return mm


@pytest.mark.parametrize("K", [4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_k811_gate_fires_on_in_cohort_shapes(K, dtype, gate_on):
    """All 6 in-cohort cells must opt into the split-K route when enabled."""
    assert gate_on._k811_route_split_k(2048, 2048, K, dtype) is True


@pytest.mark.parametrize(
    "M,N",
    [
        (4096, 4096),  # K-707 cohort: kpack/tile changes regress
        (8192, 8192),  # K-707 cohort: ditto
        (1024, 1024),  # K-738 leakage cell
        (1024, 2048),  # rectangular sibling
        (2048, 1024),  # rectangular sibling
        (2048, 4096),  # rectangular sibling
        (4096, 2048),  # rectangular sibling
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_k811_gate_excludes_non_2048_square(M, N, dtype, gate_on):
    """M==N==2048 strict — no widening to other M=N or rectangular shapes."""
    for K in (4096, 8192, 16384):
        assert gate_on._k811_route_split_k(M, N, K, dtype) is False, \
            f"M={M} N={N} K={K} dt={dtype}"


@pytest.mark.parametrize(
    "K",
    [256, 512, 1024, 2048, 3072, 4095, 4097, 6144, 8191, 8193, 12288, 16383, 16385, 32768],
)
def test_k811_gate_excludes_out_of_membership_K(K, gate_on):
    """K must be exactly one of {4096, 8192, 16384}."""
    assert gate_on._k811_route_split_k(2048, 2048, K, torch.float16) is False


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.int8],
)
def test_k811_gate_excludes_non_fp16_bf16_dtypes(dtype, gate_on):
    """fp32 / fp8 / int8 excluded — split-K kernel is fp16/bf16-only."""
    assert gate_on._k811_route_split_k(2048, 2048, 16384, dtype) is False


def test_k811_gate_default_off(gate_off):
    """gate_off must yield False even on in-cohort shapes — escape hatch works.
    Default behaviour (no env var set) is gate OFF — current K-811 default
    is opt-in via TB_K811_ENABLE=1; flips when paired-sweep verifies a +Δpp."""
    assert gate_off._k811_route_split_k(2048, 2048, 16384, torch.float16) is False
    for K in (4096, 8192, 16384):
        for dt in (torch.float16, torch.bfloat16):
            assert gate_off._k811_route_split_k(2048, 2048, K, dt) is False


def test_k811_in_cohort_cardinality_locked(gate_on):
    """Cardinality lock: exactly 6 in-cohort cells fire (3 K's × 2 dtypes)."""
    fires = 0
    for K in (4096, 8192, 16384):
        for dtype in (torch.float16, torch.bfloat16):
            if gate_on._k811_route_split_k(2048, 2048, K, dtype):
                fires += 1
    assert fires == 6, f"expected 6, got {fires}"


def test_k811_module_default_is_off_when_env_unset():
    """Subprocess test: with TB_K811_ENABLE unset, module imports with gate OFF.

    This is the production-default behaviour (the env-toggle is the
    paired-sweep harness escape hatch, not a flip).
    """
    code = (
        "import torch; "
        "from tritonblas.matmul import _k811_route_split_k, _K811_ENABLED;"
        "assert _K811_ENABLED is False, _K811_ENABLED;"
        "assert _k811_route_split_k(2048, 2048, 16384, torch.float16) is False;"
        "print('OK')"
    )
    env = os.environ.copy()
    env.pop("TB_K811_ENABLE", None)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "OK" in r.stdout


def test_k811_module_enabled_when_env_set():
    """Subprocess test: with TB_K811_ENABLE=1, module imports with gate ON."""
    code = (
        "import torch; "
        "from tritonblas.matmul import _k811_route_split_k, _K811_ENABLED;"
        "assert _K811_ENABLED is True, _K811_ENABLED;"
        "assert _k811_route_split_k(2048, 2048, 16384, torch.float16) is True;"
        "assert _k811_route_split_k(4096, 4096, 16384, torch.float16) is False;"
        "print('OK')"
    )
    env = os.environ.copy()
    env["TB_K811_ENABLE"] = "1"
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "OK" in r.stdout
