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


# ──────────────────────── Degenerate / adversarial edge cases ────────────────────────
# These exercise paths that the 3D-grid offset arithmetic could mis-handle:
# * Empty batch (B=0): zero-size grid, kernel must not be launched.
# * Empty inner dim (K=0): inner reduction over nothing → output must be zeros.
# * Non-monotonic / negative-stride batch: torch.flip / reverse-batch indexing
#   produces a tensor whose stride(0) is negative; the kernel reads bid * stride_ab
#   so a negative stride must work.

@pytest.mark.parametrize("dtype", DTYPES)
def test_bmm_zero_batch(dtype):
    """B=0 — return the empty out tensor without launching a kernel.
    torch.bmm returns an empty (0, M, N) tensor; we must mirror that."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm GPU")
    M, N, K = 64, 64, 64
    a = torch.empty(0, M, K, device="cuda", dtype=dtype)
    b = torch.empty(0, K, N, device="cuda", dtype=dtype)
    out = tritonblas.bmm(a, b)
    assert out.shape == (0, M, N)
    # Reference path
    ref = torch.bmm(a, b)
    assert ref.shape == out.shape


@pytest.mark.parametrize("dtype", DTYPES)
def test_bmm_zero_K(dtype):
    """K=0 — empty inner reduction → output must be all zeros."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm GPU")
    B, M, N, K = 4, 64, 64, 0
    a = torch.empty(B, M, K, device="cuda", dtype=dtype)
    b = torch.empty(B, K, N, device="cuda", dtype=dtype)
    out = tritonblas.bmm(a, b)
    assert out.shape == (B, M, N)
    # All zeros — empty reduction is the additive identity.
    assert out.abs().sum().item() == 0.0
    # Same as torch
    ref = torch.bmm(a, b)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bmm_zero_M_or_N(dtype):
    """Empty M or N output dim — return out unchanged, no kernel launch."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm GPU")
    B, K = 4, 64
    for (M, N) in [(0, 64), (64, 0)]:
        a = torch.empty(B, M, K, device="cuda", dtype=dtype)
        b = torch.empty(B, K, N, device="cuda", dtype=dtype)
        out = tritonblas.bmm(a, b)
        assert out.shape == (B, M, N)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bmm_non_monotonic_batch_stride(dtype):
    """Non-natural batch stride: take every other batch element from a
    larger tensor. This produces ``stride(0) = 2 * M * K`` (not the natural
    ``M * K``) — the kernel must read ``stride(0)`` from the tensor rather
    than assuming the batch slice is packed. Catches a sign-extension /
    "assume contiguous" bug in the 3D-grid offset arithmetic."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm GPU")
    torch.manual_seed(67)
    B, M, N, K = 4, 256, 256, 256
    a_big = torch.randn(2 * B, M, K, device="cuda", dtype=dtype)
    b_big = torch.randn(2 * B, K, N, device="cuda", dtype=dtype)
    # Strided slice: take every other batch. PyTorch returns a non-contiguous
    # view with stride(0) = 2 * stride_natural.
    a = a_big[::2]
    b = b_big[::2]
    assert a.shape == (B, M, K)
    assert a.stride(0) == 2 * a.stride(1) * M, (
        f"expected non-natural stride(0)=2*M*stride(1), got {a.stride()}"
    )
    out = tritonblas.bmm(a, b)
    ref = torch.bmm(a, b)  # torch.bmm handles stride internally
    rel_err = _max_rel_err(out, ref)
    tol = _tol(K, dtype)
    assert rel_err < max(tol, 1e-2), (
        f"bmm non-monotonic-stride rel_err {rel_err:.3e} > tol {tol:.3e}"
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_bmm_b_eq_one_broadcast(dtype):
    """Edge of broadcast: rank-2 A × rank-3 B with B=1. The host launcher
    sets stride_ab=0, the kernel iterates a 1-element batch dim; the result
    must equal a single rank-2 matmul. Caught a corner where the override
    registry was firing on B=1 (it shouldn't — single batch is just rank-2)."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm GPU")
    torch.manual_seed(89)
    M, N, K = 256, 256, 256
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(1, K, N, device="cuda", dtype=dtype)
    out = tritonblas.bmm(a, b)
    assert out.shape == (1, M, N)
    ref = torch.matmul(a, b)
    rel_err = _max_rel_err(out, ref)
    tol = _tol(K, dtype)
    assert rel_err < max(tol, 1e-2), (
        f"bmm b-rank3-B1 rel_err {rel_err:.3e} > tol {tol:.3e}"
    )
