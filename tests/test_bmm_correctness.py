"""Correctness tests for the true batched matmul (tritonblas.bmm and rank-3
dispatch through tritonblas.matmul)."""

import math
import pytest
import torch

import tritonblas


# Small + medium shapes covering the most launch-overhead-sensitive batched
# matmuls and a rank-3 smoke set.
BMM_SHAPES = [
    # (B, M, N, K)
    (4, 256, 256, 256),
    (8, 512, 512, 512),
    (2, 1024, 1024, 1024),
    # Largest batched residuals where Python-loop launch latency dominated.
    (4, 2048, 2048, 2048),  # fp16 dominant case
    (8, 1024, 1024, 1024),  # bf16 dominant case
    # Rank-3 smoke
    (16, 128, 128, 128),
    (3, 384, 768, 256),
]

DTYPES = [torch.float16, torch.bfloat16]


def _tol(K, dtype):
    """Conservative tolerance: rel_err < 8 * sqrt(K) * eps."""
    eps = torch.finfo(dtype).eps
    return 8.0 * math.sqrt(K) * eps


@pytest.mark.parametrize("B,M,N,K", BMM_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bmm_forward(B, M, N, K, dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm GPU")
    torch.manual_seed(42)
    a = torch.randn(B, M, K, device="cuda", dtype=dtype)
    b = torch.randn(B, K, N, device="cuda", dtype=dtype)

    out = tritonblas.bmm(a, b)
    ref = torch.bmm(a, b)

    # Compute element-wise relative error.
    out_f32 = out.to(torch.float32)
    ref_f32 = ref.to(torch.float32)
    denom = ref_f32.abs().clamp_min(1e-3)
    rel_err = ((out_f32 - ref_f32).abs() / denom).max().item()

    tol = _tol(K, dtype)
    assert rel_err < max(tol, 1e-2), (
        f"bmm rel_err {rel_err:.3e} exceeds tol {tol:.3e} for "
        f"B={B} M={M} N={N} K={K} dtype={dtype}"
    )


@pytest.mark.parametrize("B,M,N,K", [(4, 256, 256, 256), (2, 512, 512, 512)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_matmul_rank3_dispatch(B, M, N, K, dtype):
    """Verify tritonblas.matmul auto-dispatches rank-3 to bmm."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm GPU")
    torch.manual_seed(123)
    a = torch.randn(B, M, K, device="cuda", dtype=dtype)
    b = torch.randn(B, K, N, device="cuda", dtype=dtype)

    out = tritonblas.matmul(a, b)
    ref = torch.bmm(a, b)
    assert out.shape == ref.shape == (B, M, N)

    out_f32 = out.to(torch.float32)
    ref_f32 = ref.to(torch.float32)
    denom = ref_f32.abs().clamp_min(1e-3)
    rel_err = ((out_f32 - ref_f32).abs() / denom).max().item()
    tol = _tol(K, dtype)
    assert rel_err < max(tol, 1e-2), (
        f"matmul(rank3) rel_err {rel_err:.3e} > tol {tol:.3e}"
    )


def _max_rel_err(out, ref):
    out_f32 = out.to(torch.float32)
    ref_f32 = ref.to(torch.float32)
    denom = ref_f32.abs().clamp_min(1e-3)
    return ((out_f32 - ref_f32).abs() / denom).max().item()


@pytest.mark.parametrize("dtype", DTYPES)
def test_bmm_broadcast_a_rank2(dtype):
    """Rank-2 A broadcast across the batch dimension of B (stride-0 path)."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm GPU")
    torch.manual_seed(7)
    B, M, N, K = 4, 256, 256, 256
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(B, K, N, device="cuda", dtype=dtype)
    out = tritonblas.bmm(a, b)
    ref = torch.matmul(a, b)  # broadcasts a over batch
    rel_err = _max_rel_err(out, ref)
    assert rel_err < 1e-1, (
        f"bmm broadcast(a) rel_err {rel_err:.3e} too large"
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_bmm_broadcast_b_rank2(dtype):
    """Rank-3 A × rank-2 B (shared weights × per-batch activations)."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm GPU")
    torch.manual_seed(11)
    B, M, N, K = 4, 256, 256, 256
    a = torch.randn(B, M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    out = tritonblas.bmm(a, b)
    ref = torch.matmul(a, b)
    rel_err = _max_rel_err(out, ref)
    assert rel_err < 1e-1, (
        f"bmm broadcast(b) rel_err {rel_err:.3e} too large"
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_bmm_batch_size_one(dtype):
    """B=1 edge case — grid degenerates to (tiles, 1); batch stride is unused."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm GPU")
    torch.manual_seed(31)
    B, M, N, K = 1, 1024, 1024, 1024
    a = torch.randn(B, M, K, device="cuda", dtype=dtype)
    b = torch.randn(B, K, N, device="cuda", dtype=dtype)
    out = tritonblas.bmm(a, b)
    ref = torch.bmm(a, b)
    rel_err = _max_rel_err(out, ref)
    tol = _tol(K, dtype)
    assert rel_err < max(tol, 1e-2), (
        f"bmm B=1 rel_err {rel_err:.3e} > tol {tol:.3e}"
    )
    # Bit-for-bit equivalent shape to a single rank-2 matmul.
    assert out.shape == (1, M, N)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bmm_noncontiguous_inputs(dtype):
    """Non-contiguous batched inputs: kernel reads strides explicitly so this
    must work without forcing a copy. Construct via permute() so the inner
    dims are non-trivially strided."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm GPU")
    torch.manual_seed(53)
    B, M, N, K = 4, 256, 256, 384
    # Build (B, K, M) then permute to (B, M, K) → non-contiguous.
    a_base = torch.randn(B, K, M, device="cuda", dtype=dtype)
    a = a_base.permute(0, 2, 1)
    assert not a.is_contiguous()
    b = torch.randn(B, K, N, device="cuda", dtype=dtype)
    out = tritonblas.bmm(a, b)
    ref = torch.bmm(a, b)
    rel_err = _max_rel_err(out, ref)
    tol = _tol(K, dtype)
    assert rel_err < max(tol, 1e-2), (
        f"bmm non-contig rel_err {rel_err:.3e} > tol {tol:.3e}"
    )


def test_bmm_mismatched_batch_raises():
    """Mismatched batch dims must raise (caught in the host launcher)."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm GPU")
    a = torch.randn(4, 64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(8, 64, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(AssertionError, match="batch dims"):
        tritonblas.bmm(a, b)


def test_bmm_mismatched_inner_raises():
    """Mismatched inner K must raise."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm GPU")
    a = torch.randn(4, 64, 32, device="cuda", dtype=torch.float16)
    b = torch.randn(4, 64, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(AssertionError, match="inner dims"):
        tritonblas.bmm(a, b)
