"""Correctness tests for the small-M split-K fast path.

Exercises the M <= 32 dispatch added in ``include/tritonblas/matmul.py``.
The existing ``test_matmul.py`` only covers M >= 4096, so without these
tests the small-M kernel goes uncovered by CI.

Tolerance: low-precision dtypes use ``atol=2, rtol=1``. The existing
``test_matmul.py`` uses ``atol=1, rtol=1`` for non-split-K shapes; we
allow one extra ulp here because the split-K kernel accumulates
``SPLIT_K`` (up to 16) partial sums via ``tl.atomic_add``, each of which
incurs a cast-to-output rounding (~1 ulp at output magnitude
``sqrt(K)*std``). For K up to 8192, the worst-case extra error is well
below 2.0 in absolute terms.
"""
import pytest
import torch  # type: ignore
import tritonblas  # type: ignore
from tritonblas.kernels.small_m_splitk_gemm import (
    is_small_m_eligible,
    small_m_splitk_matmul_lt,
)


SMALL_M = (8, 16, 32)
NK = (1024, 4096, 8192)


def _atol_for(dtype: torch.dtype) -> float:
    # GEMM noise floors with random N(0,1) inputs:
    #   fp32: O(K*eps) accumulation error; split-K reordering adds a couple of ulps.
    #   bf16/fp16: ~1 ulp at output magnitude per partial; SPLIT_K up to 16 partials.
    if dtype is torch.float32:
        return 0.5
    return 2.0


@pytest.mark.parametrize("M", SMALL_M)
@pytest.mark.parametrize("N", NK)
@pytest.mark.parametrize("K", NK)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_small_m_dispatch_correctness(M, N, K, dtype):
    """matmul() routes to the small-M path and matches torch.matmul."""
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)

    # Sanity: the public entry point should pick the fast path here.
    assert is_small_m_eligible(M, dtype, dtype, dtype, False, False, False)

    out = tritonblas.matmul(a, b)
    ref = torch.matmul(a, b)
    torch.testing.assert_close(out.to(dtype), ref, atol=_atol_for(dtype), rtol=1)


@pytest.mark.parametrize("M", SMALL_M)
@pytest.mark.parametrize("N", NK)
@pytest.mark.parametrize("K", NK)
def test_small_m_kernel_direct_bf16(M, N, K):
    """Direct kernel call — sanity-check the kernel itself, not just dispatch."""
    torch.manual_seed(1)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    out = a.new_empty(M, N)
    small_m_splitk_matmul_lt(a, b, out)
    ref = torch.matmul(a, b)
    torch.testing.assert_close(out, ref, atol=_atol_for(torch.bfloat16), rtol=1)


def test_small_m_eligibility_matrix():
    """Eligibility gate: only routes for M<=32, no streamk/quant/bias, supported dtype."""
    f16 = torch.float16
    f8 = torch.float8_e4m3fn

    # Eligible
    assert is_small_m_eligible(32, f16, f16, f16, False, False, False)
    assert is_small_m_eligible(1, f16, f16, f16, False, False, False)
    # Ineligible: large M
    assert not is_small_m_eligible(33, f16, f16, f16, False, False, False)
    assert not is_small_m_eligible(64, f16, f16, f16, False, False, False)
    # Ineligible: streamk
    assert not is_small_m_eligible(8, f16, f16, f16, True, False, False)
    # Ineligible: quantization
    assert not is_small_m_eligible(8, f16, f16, f16, False, True, False)
    # Ineligible: bias
    assert not is_small_m_eligible(8, f16, f16, f16, False, False, True)
    # Ineligible: unsupported dtype
    assert not is_small_m_eligible(8, f8, f8, f8, False, False, False)


def test_small_m_out_arg_path():
    """matmul(a, b, out=...) variant routes to the same fast path correctly."""
    torch.manual_seed(2)
    M, N, K = 16, 4096, 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    out = a.new_empty(M, N)
    tritonblas.matmul(a, b, out=out)
    ref = torch.matmul(a, b)
    torch.testing.assert_close(out, ref, atol=_atol_for(torch.bfloat16), rtol=1)
