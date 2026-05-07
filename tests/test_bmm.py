# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the single-launch batched matmul (``tritonblas.bmm``).

These shapes intentionally cover the cases the prior Python-side per-batch
loop served poorly: small-but-many batches (B=4..32 of 1024^3) and the K-654
top residual shapes (2048^3 fp16, 1024^3 bf16).  Tolerances follow the
existing ``test_matmul.py`` conventions.
"""
import pytest
import torch  # type: ignore
import triton  # type: ignore
import tritonblas  # type: ignore


# Shapes drawn from the K-654 batched residuals + a couple of small/odd cases.
_BMM_SHAPES = [
    (4, 1024, 1024, 1024),
    (8, 1024, 1024, 1024),
    (16, 512, 512, 512),
    (4, 2048, 2048, 2048),
    (2, 4096, 4096, 4096),
    (3, 1024, 768, 1024),     # non-square (rectangular tile)
    (5, 512, 512, 1024),      # K > M,N
]


@pytest.mark.parametrize("b, m, n, k", _BMM_SHAPES)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_bmm_3d(b, m, n, k, in_dtype, out_dtype):
    """Vanilla (B, M, K) x (B, K, N) -> (B, M, N)."""
    torch.manual_seed(0)
    A = torch.randn((b, m, k), device="cuda", dtype=in_dtype)
    B = torch.randn((b, k, n), device="cuda", dtype=in_dtype)
    out = tritonblas.bmm(A, B)
    ref = torch.bmm(A, B).to(out_dtype)
    torch.testing.assert_close(out, ref, atol=1, rtol=1)


@pytest.mark.parametrize("b, m, n, k", [(8, 1024, 1024, 1024), (4, 512, 512, 512)])
def test_bmm_broadcast_b(b, m, n, k):
    """A is rank-3, B is rank-2 -> broadcast B over the batch dim."""
    torch.manual_seed(1)
    A = torch.randn((b, m, k), device="cuda", dtype=torch.float16)
    B = torch.randn((k, n), device="cuda", dtype=torch.float16)
    out = tritonblas.bmm(A, B)
    ref = torch.matmul(A, B)
    torch.testing.assert_close(out, ref, atol=1, rtol=1)


@pytest.mark.parametrize("b, m, n, k", [(8, 1024, 1024, 1024)])
def test_bmm_broadcast_a(b, m, n, k):
    """A is rank-2, B is rank-3 -> broadcast A over the batch dim."""
    torch.manual_seed(2)
    A = torch.randn((m, k), device="cuda", dtype=torch.float16)
    B = torch.randn((b, k, n), device="cuda", dtype=torch.float16)
    out = tritonblas.bmm(A, B)
    ref = torch.matmul(A, B)
    torch.testing.assert_close(out, ref, atol=1, rtol=1)


def test_bmm_rank2_degenerate():
    """Rank-2 x rank-2 is treated as B==1 BMM; result is 3D (1, M, N).

    This contract matches torch.bmm, not torch.matmul: callers asking for
    *batched* matmul of two rank-2 tensors get a leading singleton batch
    dim, which they can squeeze if desired.
    """
    torch.manual_seed(3)
    A = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    B = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    out = tritonblas.bmm(A, B)
    ref = torch.matmul(A, B).unsqueeze(0)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1, rtol=1)


def test_bmm_vector_left():
    """Rank-1 left input (gemv variant): (K,) x (B, K, N) -> (B, N)."""
    torch.manual_seed(4)
    A = torch.randn((1024,), device="cuda", dtype=torch.float16)
    B = torch.randn((4, 1024, 1024), device="cuda", dtype=torch.float16)
    out = tritonblas.bmm(A, B)
    ref = torch.matmul(A, B)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1, rtol=1)


def test_bmm_out_argument():
    """Pre-allocated ``out`` is honoured."""
    torch.manual_seed(5)
    A = torch.randn((4, 512, 512), device="cuda", dtype=torch.float16)
    B = torch.randn((4, 512, 512), device="cuda", dtype=torch.float16)
    out = torch.empty((4, 512, 512), device="cuda", dtype=torch.float16)
    ret = tritonblas.bmm(A, B, out=out)
    assert ret.data_ptr() == out.data_ptr()
    ref = torch.bmm(A, B)
    torch.testing.assert_close(out, ref, atol=1, rtol=1)


def test_bmm_b1_matches_matmul():
    """B==1 BMM must match the non-batched matmul path bit-for-bit-ish."""
    torch.manual_seed(6)
    A = torch.randn((1, 1024, 1024), device="cuda", dtype=torch.float16)
    B = torch.randn((1, 1024, 1024), device="cuda", dtype=torch.float16)
    bmm_out = tritonblas.bmm(A, B)
    mm_out = tritonblas.matmul(A.squeeze(0), B.squeeze(0))
    torch.testing.assert_close(bmm_out.squeeze(0), mm_out, atol=1, rtol=1)
