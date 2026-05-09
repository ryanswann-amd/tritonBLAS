"""K-1804 unit test: assert the externally-applied N-padding wrapper produces
bit-identical output to unpadded tritonblas.matmul on the K-1795 12-cell cohort.

Also asserts that the production tritonblas.matmul behavior is unchanged
(no env-var hooks consumed, no new code paths triggered) — i.e., this experiment
is opt-in by *caller* construction only, with zero default-on behavior.

Run:
    pytest -xvs scripts/test_n_pad_bit_exact.py
"""
import os
import importlib
import sys

import pytest
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from bench_n_pad import (  # noqa: E402
    COHORT_BF16,
    _pad_n_to_next_wave,
    _gate_triggered,
    tb_padded,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP required"
)


@pytest.mark.parametrize("M,N,K", COHORT_BF16)
def test_padded_path_bit_exact_bf16(M, N, K):
    """The padded-then-sliced output must equal the unpadded output bit-for-bit."""
    dtype = torch.bfloat16
    if not _gate_triggered(M, N, K, dtype):
        pytest.skip(f"cell M={M} N={N} K={K} not gated; padding is a no-op")
    torch.manual_seed(0xC0DE)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    import tritonblas

    out_raw = torch.empty(M, N, device="cuda", dtype=dtype)
    out_pad = torch.empty(M, N, device="cuda", dtype=dtype)
    tritonblas.matmul(a, b, out=out_raw)
    tb_padded(a, b, M, N, K, out_pad)
    torch.cuda.synchronize()
    assert torch.equal(out_raw, out_pad), (
        f"M={M} N={N} K={K}: padded output differs from unpadded "
        f"(max |diff|={(out_raw.float()-out_pad.float()).abs().max().item():.3e})"
    )


def test_pad_unit_extends_to_256_for_n192():
    """Skeptic regression guard: N=192 must pad to 256, not no-op at 64.

    K-1804-r1 found that the v1 gate used _N_PAD_UNIT=64 only, so N=192
    silently no-op'd (192 % 64 == 0). The extended gate must pad N=192 to
    the next-256 multiple instead.
    """
    assert _pad_n_to_next_wave(48) == 64
    assert _pad_n_to_next_wave(80) == 128
    assert _pad_n_to_next_wave(192) == 256, (
        "regression: N=192 must pad to 256 (it is wave-misaligned per K-1795 "
        "even though 192 % 64 == 0); previous gate silently no-op'd."
    )
    # No-op cases — already at a clean tile size
    assert _pad_n_to_next_wave(64) == 64
    assert _pad_n_to_next_wave(128) == 128
    assert _pad_n_to_next_wave(256) == 256
    assert _pad_n_to_next_wave(512) == 512
    assert _pad_n_to_next_wave(1024) == 1024


def test_gate_off_for_aligned_and_outofcohort():
    """The harness gate must NOT trigger on aligned/out-of-cohort cells."""
    bf = torch.bfloat16
    # Aligned N
    assert not _gate_triggered(4096,   64, 4096, bf)
    assert not _gate_triggered(4096,  128, 8192, bf)
    assert not _gate_triggered(4096,  256, 8192, bf)
    # Out-of-cohort M
    assert not _gate_triggered(1024,   48, 4096, bf)
    # Out-of-cohort K
    assert not _gate_triggered(4096,   48, 1024, bf)
    assert not _gate_triggered(4096,   48, 32768, bf)
    # Wrong dtype
    assert not _gate_triggered(4096,   48, 4096, torch.float32)
    # In-cohort wave-misaligned → gate trips
    assert _gate_triggered(4096,   48, 4096, bf)
    assert _gate_triggered(4096,   80, 8192, bf)
    assert _gate_triggered(4096,  192, 4096, bf)


def test_production_matmul_untouched_by_env_var():
    """Verify TRITONBLAS_N_PAD env var is *not* consumed by tritonblas.

    K-1804-r1 Minimalist concern: previous attempt added env-gated dispatch
    logic to the hot matmul() entry. This test guards against that
    regressing: setting the env var must have zero effect on matmul output
    (i.e., production code does not read it).
    """
    dtype = torch.bfloat16
    M, N, K = 4096, 48, 4096
    torch.manual_seed(42)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    import tritonblas

    out_off = torch.empty(M, N, device="cuda", dtype=dtype)
    os.environ.pop("TRITONBLAS_N_PAD", None)
    importlib.reload(tritonblas)  # ensure no cached env consumption
    tritonblas.matmul(a, b, out=out_off)

    out_on = torch.empty(M, N, device="cuda", dtype=dtype)
    os.environ["TRITONBLAS_N_PAD"] = "1"
    importlib.reload(tritonblas)
    tritonblas.matmul(a, b, out=out_on)
    os.environ.pop("TRITONBLAS_N_PAD", None)
    importlib.reload(tritonblas)

    torch.cuda.synchronize()
    assert torch.equal(out_off, out_on), (
        "Production matmul output changed when TRITONBLAS_N_PAD was set. "
        "The env-gated dispatcher gate should NOT exist in production code."
    )
