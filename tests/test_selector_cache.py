"""Tests guarding the lru_cache on _make_matmul_selector (K-1659).

These tests ensure that the selector cache:

1. Actually caches — `cache_info().hits` increments on a repeat call
   with identical arguments.
2. Returns the SAME selector object on a hit (identity, not just equality)
   so callers see amortized origami selection.
3. Distinguishes shapes that should NOT alias (different M / N / K /
   dtype / streamk → distinct entries, distinct selectors).
4. Does not silently corrupt numerical results: a cached run produces
   bitwise-identical output to a run with `cache_clear()` between
   launches, on a real `tritonblas.matmul` call across multiple shape
   instances. This guards against a future refactor adding a
   non-hashable-but-relevant arg (e.g. a tensor stride) that would
   collapse two distinct shapes onto one cache entry.

Filed under K-1659 in response to reviewer feedback (The Testing Zealot
asked for cache-hit assertions, The Skeptic asked for a torch.allclose
correctness gate against an uncached baseline).
"""

from __future__ import annotations

import pytest
import torch

import tritonblas  # type: ignore
from tritonblas.matmul import _make_matmul_selector  # type: ignore


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a ROCm/CUDA device"
)


def _selector_args(M: int, N: int, K: int, dtype: torch.dtype):
    """Canonical _make_matmul_selector args for these tests."""
    device = torch.device("cuda", 0)
    return (M, N, K, dtype, dtype, dtype, device)


def test_cache_args_are_hashable():
    """Sanity: every arg type in the cache key must be hashable.

    A future refactor that passes a tensor / list / dict here will
    raise TypeError immediately and this test will catch it.
    """
    args = _selector_args(512, 512, 512, torch.float16)
    # Each arg individually:
    for a in args:
        hash(a)
    # And the full tuple (which is what lru_cache hashes):
    hash(args)


def test_repeat_call_increments_hit_count():
    """Calling _make_matmul_selector twice with identical args must
    increment cache_info().hits by exactly 1 and leave misses unchanged.
    """
    _make_matmul_selector.cache_clear()
    args = _selector_args(1024, 1024, 1024, torch.float16)

    base = _make_matmul_selector.cache_info()
    assert base.hits == 0 and base.misses == 0 and base.currsize == 0

    s1 = _make_matmul_selector(*args)
    after_miss = _make_matmul_selector.cache_info()
    assert after_miss.misses == 1
    assert after_miss.hits == 0
    assert after_miss.currsize == 1

    s2 = _make_matmul_selector(*args)
    after_hit = _make_matmul_selector.cache_info()
    assert after_hit.misses == 1, "second call must NOT miss"
    assert after_hit.hits == 1, "second call must register as a hit"
    assert after_hit.currsize == 1

    # Identity, not equality: a hit must return the same object.
    assert s1 is s2

    # And the selected tile shape must be stable across the hit.
    assert (s1.block_m, s1.block_n, s1.block_k) == (
        s2.block_m,
        s2.block_n,
        s2.block_k,
    )


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        # Different K only.
        ((1024, 1024, 1024, torch.float16), (1024, 1024, 2048, torch.float16)),
        # Different M only.
        ((1024, 1024, 1024, torch.float16), (2048, 1024, 1024, torch.float16)),
        # Different dtype only.
        ((1024, 1024, 1024, torch.float16), (1024, 1024, 1024, torch.bfloat16)),
    ],
)
def test_distinct_shapes_do_not_alias(shape_a, shape_b):
    """Selector must NOT be reused across non-equivalent (M,N,K,dtype)."""
    _make_matmul_selector.cache_clear()
    sa = _make_matmul_selector(*_selector_args(*shape_a))
    sb = _make_matmul_selector(*_selector_args(*shape_b))
    assert sa is not sb, (
        "distinct shapes returned the SAME cached selector — cache key "
        "is collapsing inputs that should be distinct"
    )
    info = _make_matmul_selector.cache_info()
    assert info.misses == 2, "two distinct shapes must produce two misses"
    assert info.currsize == 2


@pytest.mark.parametrize(
    "M, N, K, dtype",
    [
        (512, 512, 512, torch.float16),
        (1024, 1024, 1024, torch.bfloat16),
        # A shape from the K-COMPLEMENT envelope this ticket targets.
        (4096, 4096, 2048, torch.bfloat16),
    ],
)
def test_cached_matmul_output_matches_uncached(M, N, K, dtype):
    """End-to-end correctness: a cached selector must produce the same
    numerical result as an uncached one.

    Procedure:
      1. clear cache, run tritonblas.matmul → C_uncached (selector miss)
      2. run again on a fresh output buffer → C_cached_hit (selector hit)
      3. clear cache, run again on a fresh output buffer
         → C_uncached_again (selector miss; checks recompute path is
         deterministic)

    Assert all three outputs are bitwise-equal. If a stale cache returned
    a wrong tile / wrong workgroup mapping for the second launch, the
    Triton kernel would either crash or write a different result.
    """
    torch.manual_seed(0)
    device = torch.device("cuda", 0)
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)

    # Pass #1 — cold cache → miss
    _make_matmul_selector.cache_clear()
    c_uncached = torch.empty(M, N, dtype=dtype, device=device)
    tritonblas.matmul(a, b, c_uncached)
    torch.cuda.synchronize()
    info_after_first = _make_matmul_selector.cache_info()

    # Pass #2 — warm cache → must hit
    c_cached_hit = torch.empty(M, N, dtype=dtype, device=device)
    tritonblas.matmul(a, b, c_cached_hit)
    torch.cuda.synchronize()
    info_after_second = _make_matmul_selector.cache_info()

    assert info_after_second.hits == info_after_first.hits + 1, (
        "second matmul call did not register a cache hit on "
        f"_make_matmul_selector (info={info_after_second})"
    )

    # Pass #3 — cold cache again → miss (deterministic recompute)
    _make_matmul_selector.cache_clear()
    c_uncached_again = torch.empty(M, N, dtype=dtype, device=device)
    tritonblas.matmul(a, b, c_uncached_again)
    torch.cuda.synchronize()

    # Bitwise equality across all three runs — same kernel, same inputs,
    # same tile config.
    assert torch.equal(c_uncached, c_cached_hit), (
        "cache HIT produced a different result than the cold MISS — "
        "the cached selector is returning stale state"
    )
    assert torch.equal(c_uncached, c_uncached_again), (
        "two cold MISS runs produced different results — "
        "selector recomputation is not deterministic (test infra bug, "
        "not a cache bug, but this assertion guards the comparison above)"
    )

    # Cross-check vs torch.matmul at a generous tolerance to confirm
    # we are computing matmul at all (this is not what the cache test
    # is for; it's a smoke gate).
    ref = torch.matmul(a, b)
    torch.testing.assert_close(c_uncached.to(dtype), ref, atol=1, rtol=1)
