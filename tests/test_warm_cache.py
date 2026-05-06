"""Tests for warm-cache pre-population (K-495)."""
import math

import pytest
import torch

import tritonblas
from tritonblas.warm_cache import (
    KNNSelectorCache,
    _ConfigSnapshot,
    _dtype_key,
    get_cache,
    lookup_warm_selector,
    warmup,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    cache = get_cache()
    cache.clear()
    cache.set_enabled(True)
    yield
    cache.clear()
    cache.set_enabled(True)


# ---------------------------------------------------------------------------
# Pure-Python tests (no GPU): registry behaviour
# ---------------------------------------------------------------------------


def _snap(bm=128, bn=128, bk=64, gm=8, xcd=8, ns=2):
    return _ConfigSnapshot(
        block_m=bm, block_n=bn, block_k=bk,
        group_m=gm, num_xcds=xcd, n_cu=304, active_cu=304,
        waves_per_eu=0, num_stages=ns,
    )


def test_knn_cache_exact_match():
    cache = KNNSelectorCache()
    key = _dtype_key(torch.bfloat16, torch.bfloat16, torch.bfloat16, 0, False, 2)
    cache.insert(4096, 4096, 4096, key, _snap())
    res = cache.lookup(4096, 4096, 4096, key)
    assert res is not None
    config, dist, kind = res
    assert kind == "exact"
    assert dist < 0.001
    assert config.block_m == 128


def test_knn_cache_near_match_via_vote():
    cache = KNNSelectorCache(k=3)
    key = _dtype_key(torch.bfloat16, torch.bfloat16, torch.bfloat16, 0, False, 2)
    # Three close neighbours all picking 256x256x64
    cache.insert(2048, 2048, 2048, key, _snap(bm=256, bn=256, bk=64))
    cache.insert(4096, 4096, 4096, key, _snap(bm=256, bn=256, bk=64))
    cache.insert(8192, 8192, 8192, key, _snap(bm=256, bn=256, bk=64))
    # And one far neighbour picking 64x64x32
    cache.insert(64, 64, 64, key, _snap(bm=64, bn=64, bk=32))
    res = cache.lookup(3000, 3000, 3000, key)
    assert res is not None
    config, _dist, kind = res
    assert kind == "knn"
    # Vote winner is the 256x256x64 cluster
    assert (config.block_m, config.block_n, config.block_k) == (256, 256, 64)


def test_knn_cache_far_lookup_returns_none():
    cache = KNNSelectorCache()
    key = _dtype_key(torch.bfloat16, torch.bfloat16, torch.bfloat16, 0, False, 2)
    cache.insert(4096, 4096, 4096, key, _snap())
    # 16x16x16 is way outside log-distance 1.0 from 4096^3
    res = cache.lookup(16, 16, 16, key)
    assert res is None


def test_knn_cache_dtype_isolation():
    cache = KNNSelectorCache()
    bf = _dtype_key(torch.bfloat16, torch.bfloat16, torch.bfloat16, 0, False, 2)
    fp = _dtype_key(torch.float16, torch.float16, torch.float16, 0, False, 2)
    cache.insert(4096, 4096, 4096, bf, _snap(bm=128))
    res = cache.lookup(4096, 4096, 4096, fp)
    assert res is None  # different dtype bucket


def test_knn_cache_disable():
    cache = KNNSelectorCache()
    key = _dtype_key(torch.bfloat16, torch.bfloat16, torch.bfloat16, 0, False, 2)
    cache.insert(4096, 4096, 4096, key, _snap())
    cache.set_enabled(False)
    assert cache.lookup(4096, 4096, 4096, key) is None
    cache.set_enabled(True)
    assert cache.lookup(4096, 4096, 4096, key) is not None


def test_knn_cache_reinsertion_overwrites():
    cache = KNNSelectorCache()
    key = _dtype_key(torch.bfloat16, torch.bfloat16, torch.bfloat16, 0, False, 2)
    cache.insert(4096, 4096, 4096, key, _snap(bm=64))
    cache.insert(4096, 4096, 4096, key, _snap(bm=256))
    res = cache.lookup(4096, 4096, 4096, key)
    assert res is not None
    config, _, _ = res
    assert config.block_m == 256
    assert cache.size() == 1


# ---------------------------------------------------------------------------
# GPU tests: end-to-end correctness with the real selector
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_warmup_populates_cache():
    device = torch.device("cuda", 0)
    n = warmup(
        shapes=[(2048, 2048, 2048)],
        dtypes=[(torch.bfloat16, torch.bfloat16, torch.bfloat16)],
        device=device,
        compile_kernels=False,
    )
    assert n == 1
    assert tritonblas.is_warm()
    assert get_cache().size() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("m,n,k", [(2048, 2048, 2048), (4096, 4096, 4096)])
def test_warm_path_matches_origami_exact(m, n, k):
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    warmup(
        shapes=[(m, n, k)],
        dtypes=[(dtype, dtype, dtype)],
        device=device,
        compile_kernels=False,
    )
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    c_warm = tritonblas.matmul(a, b)
    get_cache().set_enabled(False)
    c_orig = tritonblas.matmul(a, b)
    get_cache().set_enabled(True)
    assert torch.equal(c_warm, c_orig), \
        f"warm and origami diverged: max diff = {(c_warm-c_orig).abs().max().item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_warm_path_knn_interpolated_correct():
    """K-NN-interpolated config must still produce numerically valid matmul."""
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    warmup(
        shapes=[(2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)],
        dtypes=[(dtype, dtype, dtype)],
        device=device,
        compile_kernels=False,
    )
    # 3072 sits in the K-NN neighbourhood of 2048/4096
    m = n = k = 3072
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    c_warm = tritonblas.matmul(a, b)
    c_ref = (a.to(torch.float32) @ b.to(torch.float32)).to(dtype)
    torch.testing.assert_close(c_warm, c_ref, atol=2.0, rtol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_warm_disabled_falls_back_to_origami():
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    warmup(
        shapes=[(2048, 2048, 2048)],
        dtypes=[(dtype, dtype, dtype)],
        device=device,
        compile_kernels=False,
    )
    cache = get_cache()
    cache.set_enabled(False)
    a = torch.randn(2048, 2048, device=device, dtype=dtype)
    b = torch.randn(2048, 2048, device=device, dtype=dtype)
    # Should not crash, should match plain torch within bf16 tol
    c = tritonblas.matmul(a, b)
    c_ref = (a.to(torch.float32) @ b.to(torch.float32)).to(dtype)
    torch.testing.assert_close(c, c_ref, atol=2.0, rtol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_lookup_warm_selector_far_shape_returns_none():
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    warmup(
        shapes=[(4096, 4096, 4096)],
        dtypes=[(dtype, dtype, dtype)],
        device=device,
        compile_kernels=False,
    )
    # log2 distance from (4096^3) to (16, 4096, 16) is huge
    sel = lookup_warm_selector(16, 4096, 16, dtype, dtype, dtype, device)
    assert sel is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_warmup_with_dtype_isolation():
    """bf16 warmup must not satisfy fp16 lookups."""
    device = torch.device("cuda", 0)
    warmup(
        shapes=[(2048, 2048, 2048)],
        dtypes=[(torch.bfloat16, torch.bfloat16, torch.bfloat16)],
        device=device,
        compile_kernels=False,
    )
    # Same shape, different dtype: should miss the bf16 cache.
    sel = lookup_warm_selector(2048, 2048, 2048, torch.float16, torch.float16, torch.float16, device)
    assert sel is None
