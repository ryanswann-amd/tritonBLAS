# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for tritonblas.bmm (true batched matmul)."""

import math

import pytest
import torch

import tritonblas


def _ktol(dtype: torch.dtype, K: int, scale: float = 8.0) -> float:
    """K-aware tolerance: scale * sqrt(K) * eps(dtype)."""
    eps = torch.finfo(dtype).eps if dtype.is_floating_point else 1.0
    return scale * math.sqrt(K) * eps


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch,M,N,K",
    [
        (4, 128, 128, 128),
        (4, 1024, 1024, 1024),
        (8, 512, 512, 512),
        (8, 1024, 1024, 1024),
        (4, 2048, 2048, 2048),
        (2, 256, 512, 1024),  # non-square, sanity
    ],
)
def test_bmm_correctness(batch, M, N, K, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    a = torch.randn(batch, M, K, device="cuda", dtype=dtype)
    b = torch.randn(batch, K, N, device="cuda", dtype=dtype)

    out = tritonblas.bmm(a, b)
    ref = torch.bmm(a, b)

    diff = (out.float() - ref.float()).abs().max().item()
    tol = _ktol(dtype, K)
    assert diff < tol, (
        f"bmm correctness failed: shape=({batch},{M},{N},{K}) dtype={dtype} "
        f"max_abs_err={diff} tol={tol}"
    )


def test_bmm_dispatch_through_matmul():
    """tritonblas.matmul should dispatch rank-3 inputs to bmm()."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    a = torch.randn(2, 64, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(2, 128, 64, device="cuda", dtype=torch.float16)
    out = tritonblas.matmul(a, b)
    ref = torch.bmm(a, b)
    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < _ktol(torch.float16, 128), f"matmul rank-3 dispatch failed (err={diff})"


def test_bmm_out_argument():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    a = torch.randn(3, 64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(3, 64, 64, device="cuda", dtype=torch.float16)
    out = torch.empty(3, 64, 64, device="cuda", dtype=torch.float16)
    ret = tritonblas.bmm(a, b, out=out)
    assert ret is out, "bmm should return the supplied out tensor"
    ref = torch.bmm(a, b)
    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < _ktol(torch.float16, 64)
