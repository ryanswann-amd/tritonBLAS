# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the true batched matmul entrypoint."""

import pytest
import torch  # type: ignore
import tritonblas  # type: ignore


def _randn(shape, dtype, device="cuda"):
    # Generate in fp32 then downcast to keep relative error small for low-prec.
    return torch.randn(shape, device=device, dtype=torch.float32).to(dtype)


# Tolerances per dtype tuned to match the rank-2 test_matmul.py philosophy:
# accumulation in fp32, downcast at the epilogue. We compare to torch.bmm /
# torch.matmul which uses the rocBLAS path; small numerical differences are
# expected in the LSBs.
_TOLS = {
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=8e-2, rtol=8e-2),
    torch.float32: dict(atol=1e-3, rtol=1e-3),
}


@pytest.mark.parametrize("batch", [1, 4, 8])
@pytest.mark.parametrize(
    "m, n, k",
    [
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_bmm_matches_torch(batch, m, n, k, dtype):
    """tritonblas.bmm must match torch.bmm for rank-3 inputs."""
    a = _randn((batch, m, k), dtype)
    b = _randn((batch, k, n), dtype)
    out = tritonblas.bmm(a, b)
    ref = torch.bmm(a, b)
    torch.testing.assert_close(out, ref, **_TOLS[dtype])


@pytest.mark.parametrize("batch", [2, 8])
@pytest.mark.parametrize(
    "m, n, k",
    [
        (256, 256, 256),
        (1024, 1024, 1024),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batched_matmul_matches_torch(batch, m, n, k, dtype):
    """tritonblas.batched_matmul must match torch.matmul for rank-3 inputs."""
    a = _randn((batch, m, k), dtype)
    b = _randn((batch, k, n), dtype)
    out = tritonblas.batched_matmul(a, b)
    ref = torch.matmul(a, b)
    torch.testing.assert_close(out, ref, **_TOLS[dtype])


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("m, n, k", [(512, 512, 512)])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_matmul_dispatches_rank3(batch, m, n, k, dtype):
    """tritonblas.matmul must route rank-3 inputs through batched_matmul."""
    a = _randn((batch, m, k), dtype)
    b = _randn((batch, k, n), dtype)
    out = tritonblas.matmul(a, b)
    assert out.shape == (batch, m, n)
    ref = torch.matmul(a, b)
    torch.testing.assert_close(out, ref, **_TOLS[dtype])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batched_matmul_broadcast_b(dtype):
    """A is rank-3, B is rank-2 (shared across the batch)."""
    batch, m, k, n = 4, 256, 256, 256
    a = _randn((batch, m, k), dtype)
    b = _randn((k, n), dtype)
    out = tritonblas.matmul(a, b)
    assert out.shape == (batch, m, n)
    ref = torch.matmul(a, b)
    torch.testing.assert_close(out, ref, **_TOLS[dtype])


@pytest.mark.parametrize("dtype", [torch.float16])
def test_batched_matmul_rank4_leading_dims(dtype):
    """Rank-4 inputs must be flattened on the leading dims and reshaped on output."""
    a = _randn((2, 3, 256, 256), dtype)
    b = _randn((2, 3, 256, 256), dtype)
    out = tritonblas.matmul(a, b)
    assert out.shape == (2, 3, 256, 256)
    ref = torch.matmul(a, b)
    torch.testing.assert_close(out, ref, **_TOLS[dtype])


@pytest.mark.parametrize("dtype", [torch.float16])
def test_batched_matmul_out_param(dtype):
    """User-provided output tensor must be honored."""
    batch, m, k, n = 2, 256, 256, 256
    a = _randn((batch, m, k), dtype)
    b = _randn((batch, k, n), dtype)
    out = torch.empty((batch, m, n), device="cuda", dtype=dtype)
    returned = tritonblas.batched_matmul(a, b, out=out)
    assert returned is out
    ref = torch.matmul(a, b)
    torch.testing.assert_close(out, ref, **_TOLS[dtype])


def test_bmm_rejects_mismatched_batch():
    a = _randn((2, 128, 128), torch.float16)
    b = _randn((4, 128, 128), torch.float16)
    with pytest.raises(ValueError, match="batch"):
        tritonblas.bmm(a, b)


def test_bmm_rejects_rank2():
    a = _randn((128, 128), torch.float16)
    b = _randn((128, 128), torch.float16)
    with pytest.raises(ValueError, match="rank-3"):
        tritonblas.bmm(a, b)
