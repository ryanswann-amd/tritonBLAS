"""K-777 — unit tests for the M=N=2048 large-K square narrow kpack=2 gate.

Pin the predicate so future cohort extensions cannot widen it accidentally.
"""
import os

import pytest
import torch

from tritonblas.matmul import _k777_largeK_square_2048_kpack


@pytest.mark.parametrize("K", [4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_k777_gate_fires_on_in_cohort_shapes(K, dtype):
    """All 6 in-cohort cells must opt into kpack=2."""
    out = _k777_largeK_square_2048_kpack(2048, 2048, K, dtype)
    assert out == 2, f"expected 2, got {out}"


@pytest.mark.parametrize(
    "M,N",
    [
        (4096, 4096),  # K-707 cohort: kpack=2 regressed by −2.9% to −11.0% (K-612)
        (8192, 8192),  # K-707 cohort: ditto
        (1024, 1024),  # K-738 leakage cell: kp=1 wpeu=2 regressed bf16 by −10.59%
        (1024, 2048),  # rectangular sibling
        (2048, 1024),  # rectangular sibling
        (2048, 4096),  # rectangular sibling
        (4096, 2048),  # rectangular sibling
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_k777_gate_excludes_non_2048_square(M, N, dtype):
    """M==N==2048 strict — no widening to other M=N or rectangular shapes."""
    assert _k777_largeK_square_2048_kpack(M, N, 16384, dtype) is None
    assert _k777_largeK_square_2048_kpack(M, N, 8192, dtype) is None
    assert _k777_largeK_square_2048_kpack(M, N, 4096, dtype) is None


@pytest.mark.parametrize("K", [256, 512, 1024, 2048, 3072, 4095, 4097, 6144, 8191, 8193, 12288, 16383, 16385, 32768])
def test_k777_gate_excludes_out_of_membership_K(K):
    """K must be exactly one of {4096, 8192, 16384} — no monotone widening."""
    assert _k777_largeK_square_2048_kpack(2048, 2048, K, torch.float16) is None
    assert _k777_largeK_square_2048_kpack(2048, 2048, K, torch.bfloat16) is None


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.int8,
    ],
)
def test_k777_gate_excludes_non_fp16_bf16(dtype):
    """fp16/bf16 only — fp8 / fp32 / int8 are out of cohort."""
    assert _k777_largeK_square_2048_kpack(2048, 2048, 16384, dtype) is None
    assert _k777_largeK_square_2048_kpack(2048, 2048, 8192, dtype) is None
    assert _k777_largeK_square_2048_kpack(2048, 2048, 4096, dtype) is None


def test_k777_gate_disabled_via_env_subprocess(tmp_path):
    """TB_K777_DISABLE=1 cleanly bypasses the override (paired ON/OFF harness).

    The disable flag is read once at module import (a module-level constant) so
    we exercise it in a fresh Python subprocess to capture the import-time read.
    """
    import subprocess
    import sys

    src = (
        "import os, torch\n"
        "assert os.environ.get('TB_K777_DISABLE') == '1'\n"
        "from tritonblas.matmul import _k777_largeK_square_2048_kpack as g\n"
        "assert g(2048, 2048, 16384, torch.float16) is None\n"
        "assert g(2048, 2048,  4096, torch.bfloat16) is None\n"
        "print('OK')\n"
    )
    env = dict(os.environ)
    env["TB_K777_DISABLE"] = "1"
    out = subprocess.run(
        [sys.executable, "-c", src],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert out.stdout.strip().endswith("OK"), (out.stdout, out.stderr)


def test_k777_in_cohort_size_is_six():
    """Lock the gate's in-cohort size at exactly 6 cells (3 K × 2 dtype)."""
    in_cohort = [
        (2048, 2048, K, dt)
        for K in (4096, 8192, 16384)
        for dt in (torch.float16, torch.bfloat16)
    ]
    n_fires = sum(
        1
        for (M, N, K, dt) in in_cohort
        if _k777_largeK_square_2048_kpack(M, N, K, dt) == 2
    )
    assert n_fires == 6, f"expected 6 in-cohort fires, got {n_fires}"
