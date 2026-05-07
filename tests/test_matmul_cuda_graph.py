"""Tests for the use_cuda_graph fast path added by K-178."""
import pytest
import torch

import tritonblas


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP required"
)


@pytest.fixture(autouse=True)
def _clear_cache():
    tritonblas.clear_graph_cache()
    yield
    tritonblas.clear_graph_cache()


@pytest.mark.parametrize("M,N,K", [
    (16, 128, 128),
    (16, 256, 256),
    (16, 1024, 1024),
    (8, 128, 128),
    (32, 128, 128),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_use_cuda_graph_matches_normal(M, N, K, dtype):
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(K, N, dtype=dtype, device="cuda")
    ref = tritonblas.matmul(a, b)
    out = tritonblas.matmul(a, b, use_cuda_graph=True)
    # First call captures, output should match the non-graph path bit-exactly
    # (same kernel, same inputs).
    assert torch.allclose(ref, out, atol=0, rtol=0), \
        f"first-call result diverged: max={(ref-out).abs().max().item()}"


def test_replay_with_new_inputs():
    """Subsequent calls must reflect the new input data, not the captured data."""
    M, N, K = 16, 128, 128
    a0 = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b0 = torch.randn(K, N, dtype=torch.float16, device="cuda")
    # First call: capture
    _ = tritonblas.matmul(a0, b0, use_cuda_graph=True)
    # Second call with fresh inputs
    a1 = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b1 = torch.randn(K, N, dtype=torch.float16, device="cuda")
    out = tritonblas.matmul(a1, b1, use_cuda_graph=True)
    ref = tritonblas.matmul(a1, b1)
    assert torch.allclose(out, ref, atol=0, rtol=0)


def test_gate_excludes_large():
    """use_cuda_graph=True on a large shape must NOT capture (gated off)."""
    M, N, K = 1024, 1024, 1024  # 1G work, well above threshold
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    tritonblas.clear_graph_cache()
    _ = tritonblas.matmul(a, b, use_cuda_graph=True)
    # Cache must be empty: gate kept us on the normal path
    from tritonblas.matmul import _graph_cache
    assert len(_graph_cache) == 0, \
        f"expected empty cache for large shape, got {len(_graph_cache)} entries"


def test_gate_includes_tiny():
    """use_cuda_graph=True on a tiny shape MUST populate the cache."""
    M, N, K = 16, 128, 128
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    tritonblas.clear_graph_cache()
    _ = tritonblas.matmul(a, b, use_cuda_graph=True)
    from tritonblas.matmul import _graph_cache
    assert len(_graph_cache) == 1


def test_default_off():
    """Without use_cuda_graph=True, no graph is captured."""
    M, N, K = 16, 128, 128
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    tritonblas.clear_graph_cache()
    _ = tritonblas.matmul(a, b)  # default
    from tritonblas.matmul import _graph_cache
    assert len(_graph_cache) == 0
