"""Unit tests for the @functools.lru_cache on _make_matmul_selector.

Regression test guarding the K-198 / K-231 finding that selector construction
(~280us via Origami) dominates per-call overhead at small M, where the kernel
itself runs in <1ms. Disabling the cache makes the heuristic re-run on every
matmul call and tanks the small-M cohort. These tests assert:

  1. Repeated calls with the same (M, N, K, dtype, device, ...) return the
     SAME selector instance (cache hit).
  2. Calls with different shapes return DIFFERENT instances (cache by key).
  3. The lru_cache decorator is actually applied (cache_info() is callable
     and bookkeeping increments).

These tests do NOT require a GPU — they import the matmul module and exercise
the cache wrapper. Origami selector construction itself does need a device,
so we use a CPU-only stand-in by mocking OrigamiMatmulSelector.
"""

from __future__ import annotations

import importlib
from unittest import mock

import pytest
import torch

# `from .matmul import matmul` in tritonblas/__init__.py overrides the
# submodule attribute with the function, so `import tritonblas.matmul`
# returns the function in some import orders. Use importlib to always
# reach the submodule.
matmul_mod = importlib.import_module("tritonblas.matmul")


@pytest.fixture
def fake_selector(monkeypatch):
    """Replace OrigamiMatmulSelector with a stub that doesn't need a GPU.

    Each call to the stub returns a fresh sentinel object so we can verify by
    object identity whether the cache returned the same instance or built a
    new one. We also clear the lru_cache between tests so state from earlier
    cases doesn't leak in.
    """
    matmul_mod._make_matmul_selector.cache_clear()
    counter = {"n": 0}

    def _stub(*args, **kwargs):
        counter["n"] += 1
        return mock.MagicMock(name=f"selector_call_{counter['n']}")

    monkeypatch.setattr(matmul_mod, "OrigamiMatmulSelector", _stub)
    yield counter
    matmul_mod._make_matmul_selector.cache_clear()


def _key():
    """Common key tuple — only the values matter, the device need not exist."""
    return dict(
        M=16,
        N=4096,
        K=4096,
        a_dtype=torch.bfloat16,
        b_dtype=torch.bfloat16,
        c_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        mx_block_size=0,
        streamk=False,
        num_stages=2,
    )


def test_lru_cache_decorator_present():
    """_make_matmul_selector must be lru_cache-wrapped (cache_info exposed)."""
    assert hasattr(matmul_mod._make_matmul_selector, "cache_info"), (
        "lru_cache must be enabled on _make_matmul_selector — disabling it "
        "regresses the small-M cohort by >10x (see K-198 / K-231)."
    )
    assert hasattr(matmul_mod._make_matmul_selector, "cache_clear")


def test_repeated_call_returns_same_instance(fake_selector):
    """Same (M,N,K,dtype,device) -> cache hit, same object."""
    sel1 = matmul_mod._make_matmul_selector(**_key())
    sel2 = matmul_mod._make_matmul_selector(**_key())
    assert sel1 is sel2, "lru_cache must return the same selector instance for repeated keys"
    assert fake_selector["n"] == 1, "OrigamiMatmulSelector must be constructed exactly once"


def test_different_shape_yields_different_instance(fake_selector):
    """Different M -> different cache slot, new construction."""
    sel_m16 = matmul_mod._make_matmul_selector(**{**_key(), "M": 16})
    sel_m32 = matmul_mod._make_matmul_selector(**{**_key(), "M": 32})
    assert sel_m16 is not sel_m32
    assert fake_selector["n"] == 2


def test_different_dtype_yields_different_instance(fake_selector):
    """Different a_dtype -> different cache slot."""
    sel_bf16 = matmul_mod._make_matmul_selector(**{**_key(), "a_dtype": torch.bfloat16})
    sel_fp16 = matmul_mod._make_matmul_selector(**{**_key(), "a_dtype": torch.float16})
    assert sel_bf16 is not sel_fp16
    assert fake_selector["n"] == 2


def test_cache_hit_count_grows(fake_selector):
    """100 calls with same key -> 1 miss + 99 hits (small-M perf path)."""
    matmul_mod._make_matmul_selector.cache_clear()
    for _ in range(100):
        matmul_mod._make_matmul_selector(**_key())
    info = matmul_mod._make_matmul_selector.cache_info()
    assert info.misses == 1, f"expected 1 miss, got {info.misses}"
    assert info.hits == 99, f"expected 99 hits, got {info.hits}"
    assert fake_selector["n"] == 1, "Origami heuristic must run only once for a repeated shape"
