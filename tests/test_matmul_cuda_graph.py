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


def test_capture_then_replay_matches_normal():
    """Capture (first call) and replay (second call with new inputs) both
    return the same result as the non-graph path."""
    M, N, K = 16, 128, 128
    a0 = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b0 = torch.randn(K, N, dtype=torch.float16, device="cuda")
    # First call -> captures the graph; result must match non-graph path.
    out0 = tritonblas.matmul(a0, b0, use_cuda_graph=True)
    ref0 = tritonblas.matmul(a0, b0)
    assert torch.allclose(ref0, out0, atol=0, rtol=0), \
        f"capture-call diverged: max={(ref0-out0).abs().max().item()}"

    # Second call -> replays with NEW inputs; must reflect the new data.
    a1 = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b1 = torch.randn(K, N, dtype=torch.float16, device="cuda")
    out1 = tritonblas.matmul(a1, b1, use_cuda_graph=True)
    ref1 = tritonblas.matmul(a1, b1)
    assert torch.allclose(ref1, out1, atol=0, rtol=0)
    # Output is a fresh allocation, NOT an alias of the static buffer:
    # out0 must survive a subsequent replay unchanged.
    assert torch.allclose(ref0, out0, atol=0, rtol=0), \
        "out0 was mutated by the second call -- output must not alias static buffer"


def test_gate_bypass_counter_medium_large():
    """Medium and large shapes with use_cuda_graph=True must NOT route into
    _graph_replay_matmul.  Asserted via the diagnostic call counter so we know
    the gate short-circuits *before* any cache lookup or copy."""
    assert _matmul_mod._graph_replay_calls == 0
    for (M, N, K) in [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]:
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")
        _ = tritonblas.matmul(a, b, use_cuda_graph=True)
    assert _matmul_mod._graph_replay_calls == 0, \
        "gate should have prevented graph-replay entry for medium/large shapes"
    assert len(_matmul_mod._graph_cache) == 0
    # And tiny shapes flip the counter:
    a = torch.randn(16, 128, dtype=torch.float16, device="cuda")
    b = torch.randn(128, 128, dtype=torch.float16, device="cuda")
    _ = tritonblas.matmul(a, b, use_cuda_graph=True)
    assert _matmul_mod._graph_replay_calls == 1


def test_default_off():
    """Without use_cuda_graph=True, no graph is captured."""
    a = torch.randn(16, 128, dtype=torch.float16, device="cuda")
    b = torch.randn(128, 128, dtype=torch.float16, device="cuda")
    _ = tritonblas.matmul(a, b)  # default
    assert len(_matmul_mod._graph_cache) == 0
    assert _matmul_mod._graph_replay_calls == 0


def test_non_contiguous_falls_back():
    """Non-contiguous inputs would invalidate the captured stride/pointer
    assumptions in the static buffers, so they must fall through to the
    normal path."""
    M, N, K = 16, 128, 128
    big = torch.randn(2 * M, K, dtype=torch.float16, device="cuda")
    a_view = big[::2]                             # tiny shape, non-contiguous
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    assert not a_view.is_contiguous()
    out = tritonblas.matmul(a_view, b, use_cuda_graph=True)
    assert len(_matmul_mod._graph_cache) == 0, "should have fallen back to normal path"
    ref = tritonblas.matmul(a_view, b)
    assert torch.allclose(out, ref, atol=0, rtol=0)


def test_lru_eviction():
    """The graph cache is LRU-bounded; exceeding capacity evicts the LRU entry."""
    saved = _matmul_mod._GRAPH_CACHE_MAXSIZE
    _matmul_mod._GRAPH_CACHE_MAXSIZE = 3
    try:
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
