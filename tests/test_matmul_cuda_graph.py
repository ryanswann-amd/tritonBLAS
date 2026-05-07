"""Tests for the use_cuda_graph fast path added by K-178."""
import sys

import pytest
import torch

import tritonblas
import tritonblas.matmul  # noqa: F401 -- load submodule
# tritonblas/__init__.py does `from .matmul import matmul`, rebinding
# tritonblas.matmul to the function.  Use sys.modules to reach the module
# itself so we can inspect _graph_cache and _graph_replay_calls.
_matmul_mod = sys.modules["tritonblas.matmul"]


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
    # (same kernel, same inputs).  Note: out aliases the captured static buffer
    # when out= is not provided, so clone before any subsequent matmul call.
    out = out.clone()
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
    out = tritonblas.matmul(a1, b1, use_cuda_graph=True).clone()
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
    assert len(_matmul_mod._graph_cache) == 0, \
        f"expected empty cache for large shape, got {len(_matmul_mod._graph_cache)} entries"


def test_gate_includes_tiny():
    """use_cuda_graph=True on a tiny shape MUST populate the cache."""
    M, N, K = 16, 128, 128
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    tritonblas.clear_graph_cache()
    _ = tritonblas.matmul(a, b, use_cuda_graph=True)
    assert len(_matmul_mod._graph_cache) == 1


def test_default_off():
    """Without use_cuda_graph=True, no graph is captured."""
    M, N, K = 16, 128, 128
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    tritonblas.clear_graph_cache()
    _ = tritonblas.matmul(a, b)  # default
    assert len(_matmul_mod._graph_cache) == 0


def test_gate_bypass_counter_medium_large():
    """Medium and large shapes with use_cuda_graph=True must NOT route into
    _graph_replay_matmul.  We assert via the diagnostic call counter that the
    gate short-circuits *before* any cache lookup or copy.
    """
    tritonblas.clear_graph_cache()
    assert _matmul_mod._graph_replay_calls == 0
    for (M, N, K) in [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]:
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")
        _ = tritonblas.matmul(a, b, use_cuda_graph=True)
    assert _matmul_mod._graph_replay_calls == 0, \
        "gate should have prevented graph-replay entry for medium/large shapes"
    # And tiny shapes flip the counter:
    a = torch.randn(16, 128, dtype=torch.float16, device="cuda")
    b = torch.randn(128, 128, dtype=torch.float16, device="cuda")
    _ = tritonblas.matmul(a, b, use_cuda_graph=True)
    assert _matmul_mod._graph_replay_calls == 1


def test_non_contiguous_falls_back():
    """Non-contiguous inputs must NOT be captured (the captured graph would
    use the wrong stride layout in the static buffers).  They should fall
    through to the normal path."""
    M, N, K = 16, 128, 128
    big = torch.randn(2 * M, K, dtype=torch.float16, device="cuda")
    a_view = big[::2]                             # tiny shape, non-contiguous
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    assert not a_view.is_contiguous()
    tritonblas.clear_graph_cache()
    out = tritonblas.matmul(a_view, b, use_cuda_graph=True)
    # Cache must be empty (gate fell back to normal path):
    assert len(_matmul_mod._graph_cache) == 0
    ref = tritonblas.matmul(a_view, b)
    assert torch.allclose(out, ref, atol=0, rtol=0)


def test_lru_eviction():
    """The graph cache is LRU-bounded; exceeding capacity evicts the LRU entry."""
    saved = _matmul_mod._GRAPH_CACHE_MAXSIZE
    _matmul_mod._GRAPH_CACHE_MAXSIZE = 3
    try:
        tritonblas.clear_graph_cache()
        # 4 distinct tiny shapes -> 4 distinct keys -> oldest must be evicted.
        shapes = [(16, 128, 128), (16, 256, 256), (16, 512, 512), (32, 128, 128)]
        for (M, N, K) in shapes:
            a = torch.randn(M, K, dtype=torch.float16, device="cuda")
            b = torch.randn(K, N, dtype=torch.float16, device="cuda")
            _ = tritonblas.matmul(a, b, use_cuda_graph=True)
        assert len(_matmul_mod._graph_cache) == 3, \
            f"expected cache bounded to 3, got {len(_matmul_mod._graph_cache)}"
    finally:
        _matmul_mod._GRAPH_CACHE_MAXSIZE = saved


def test_alias_output_overwritten_on_next_call():
    """Documented contract: when out= is not provided the returned tensor
    aliases the captured static buffer.  A subsequent replay overwrites it.
    Callers that need to retain the result must clone()."""
    M, N, K = 16, 128, 128
    a0 = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b0 = torch.randn(K, N, dtype=torch.float16, device="cuda")
    out0 = tritonblas.matmul(a0, b0, use_cuda_graph=True)
    snap = out0.clone()
    a1 = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b1 = torch.randn(K, N, dtype=torch.float16, device="cuda")
    out1 = tritonblas.matmul(a1, b1, use_cuda_graph=True)
    # out0 is the same storage as out1 (alias semantics).
    assert out0.data_ptr() == out1.data_ptr()
    # And it has been overwritten with the new result, no longer matching snap.
    assert not torch.equal(out0, snap)


def test_out_argument_does_not_alias():
    """When the caller provides out=, the result is copied into it; the
    returned tensor IS the user's out tensor and is NOT overwritten by the
    next replay."""
    M, N, K = 16, 128, 128
    a0 = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b0 = torch.randn(K, N, dtype=torch.float16, device="cuda")
    out0 = torch.empty(M, N, dtype=torch.float16, device="cuda")
    r0 = tritonblas.matmul(a0, b0, out=out0, use_cuda_graph=True)
    assert r0.data_ptr() == out0.data_ptr()
    snap = out0.clone()
    a1 = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b1 = torch.randn(K, N, dtype=torch.float16, device="cuda")
    out1 = torch.empty(M, N, dtype=torch.float16, device="cuda")
    _ = tritonblas.matmul(a1, b1, out=out1, use_cuda_graph=True)
    # out0 was NOT touched by the second call.
    assert torch.equal(out0, snap)
