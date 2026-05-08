"""K-513: Split-K small-M decode dispatch + correctness tests.

Two pieces under test:
 1. Dispatch gate — only the 32 in-scope shapes route to the new kernel.
 2. Correctness — when dispatched, the result matches torch.matmul to
    within fp16/bf16 mfma tolerance.
"""

import pytest
import torch

import tritonblas
from tritonblas.kernels.splitk_smallm_gemm import (
    should_dispatch_splitk_smallm,
    splitk_smallm_matmul,
    get_splitk_smallm_config,
)


# ----- gate ---------------------------------------------------------------

GATED_M = [1, 2, 4, 8]
GATED_K = [4096, 8192]
GATED_N = [1024, 2048, 4096, 8192]
GATED_DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("M", GATED_M)
@pytest.mark.parametrize("K", GATED_K)
@pytest.mark.parametrize("N", GATED_N)
@pytest.mark.parametrize("dtype", GATED_DTYPES)
def test_gate_in_scope(M, N, K, dtype):
    assert should_dispatch_splitk_smallm(M, N, K, dtype, dtype, dtype)
    assert get_splitk_smallm_config(M, N, K, dtype) is not None


@pytest.mark.parametrize(
    "M, N, K, dtype",
    [
        # Out of M range
        (16, 4096, 4096, torch.float16),
        (32, 4096, 8192, torch.bfloat16),
        # Out of K range (the very point of the gate — K-484 lesson)
        (4, 4096, 1024, torch.float16),
        (4, 4096, 2048, torch.bfloat16),
        # Out of N range
        (4, 512, 4096, torch.float16),
        # Wrong dtype
        (4, 4096, 4096, torch.float32),
        # Mixed dtype not allowed
    ],
)
def test_gate_out_of_scope(M, N, K, dtype):
    assert not should_dispatch_splitk_smallm(M, N, K, dtype, dtype, dtype)


def test_gate_mixed_dtype_rejected():
    # Mixed input/output dtype — out of scope.
    assert not should_dispatch_splitk_smallm(
        4, 4096, 4096, torch.float16, torch.bfloat16, torch.float16
    )


# ----- correctness --------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("M", [1, 2, 4, 8])
@pytest.mark.parametrize("N", [1024, 4096])
@pytest.mark.parametrize("K", [4096, 8192])
@pytest.mark.parametrize("dtype", GATED_DTYPES)
def test_splitk_smallm_correctness(M, N, K, dtype):
    """Direct kernel call: split-K small-M result matches torch.matmul."""
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=dtype) * 0.1
    b = torch.randn((K, N), device="cuda", dtype=dtype) * 0.1
    c = torch.empty((M, N), device="cuda", dtype=dtype)

    cfg = get_splitk_smallm_config(M, N, K, dtype)
    assert cfg is not None
    split_k, block_n, block_k = cfg

    splitk_smallm_matmul(a, b, c, split_k=split_k, block_n=block_n, block_k=block_k)
    ref = torch.matmul(a.float(), b.float()).to(dtype)

    # bf16/fp16 mfma tolerance for K up to 8192
    atol = 1.0 if dtype == torch.bfloat16 else 0.5
    rtol = 1e-2
    torch.testing.assert_close(c, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("M", [1, 4])
@pytest.mark.parametrize("N", [2048])
@pytest.mark.parametrize("K", [4096, 8192])
@pytest.mark.parametrize("dtype", GATED_DTYPES)
def test_dispatch_through_matmul(M, N, K, dtype):
    """End-to-end: tritonblas.matmul on a gated shape produces correct result."""
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=dtype) * 0.1
    b = torch.randn((K, N), device="cuda", dtype=dtype) * 0.1
    c = torch.empty((M, N), device="cuda", dtype=dtype)

    tritonblas.matmul(a, b, c)
    ref = torch.matmul(a.float(), b.float()).to(dtype)
    atol = 1.0 if dtype == torch.bfloat16 else 0.5
    torch.testing.assert_close(c, ref, atol=atol, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_dispatch_out_of_gate_unaffected():
    """A shape outside the gate must not change behaviour — sanity that the
    gate is tight (K-654 lesson)."""
    M, N, K = 64, 4096, 4096
    dtype = torch.float16
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=dtype) * 0.1
    b = torch.randn((K, N), device="cuda", dtype=dtype) * 0.1
    c = torch.empty((M, N), device="cuda", dtype=dtype)

    # Just check it runs without error and gives a correct result via the
    # original persistent path (no kernel dispatch swap).
    tritonblas.matmul(a, b, c)
    ref = torch.matmul(a.float(), b.float()).to(dtype)
    torch.testing.assert_close(c, ref, atol=1.0, rtol=1e-2)
