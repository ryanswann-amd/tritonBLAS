"""Unit tests for the persistent autotune cache (K-536) and the K-591
boundary-tolerance fall-back.

These tests do not require a GPU: they exercise the file-format / lookup
behavior with a temporary cache directory.  GPU-bearing tests live in the
existing ``test_matmul.py`` suite and are unaffected.
"""

from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture
def tmp_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "1")
    import tritonblas.autotune_cache as ac
    importlib.reload(ac)
    ac.reset_registry()
    yield tmp_path
    ac.reset_registry()


def _sample_params():
    return {
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "workgroup_mapping": 8,
        "xcc_workgroup_mapping": 8,
        "counters_per_xcd": 4,
        "grid": 304,
    }


# ---------------------------------------------------------------------------
# K-536 baseline: persistence, isolation, invalidation
# ---------------------------------------------------------------------------


def test_lookup_miss_returns_none(tmp_cache_dir):
    from tritonblas.autotune_cache import get_cache, make_cache_key

    cache = get_cache("gfx942", 304)
    key = make_cache_key(8192, 8192, 8192, "f16", "f16", "f16", 0, False, 2, 304)
    assert cache.lookup(key) is None
    assert len(cache) == 0


def test_store_then_lookup_roundtrip(tmp_cache_dir):
    """Exact-key path is the K-536 fast path and must continue to work
    identically — the K-591 nearest fall-back never touches it."""
    from tritonblas.autotune_cache import get_cache, make_cache_key

    cache = get_cache("gfx942", 304)
    key = make_cache_key(8192, 8192, 8192, "f16", "f16", "f16", 0, False, 2, 304)
    cache.store(key, _sample_params())
    assert cache.lookup(key) == _sample_params()


def test_persistence_across_processes(tmp_cache_dir):
    """A fresh registry instance must read the on-disk cache."""
    from tritonblas.autotune_cache import (
        get_cache,
        make_cache_key,
        reset_registry,
    )

    cache = get_cache("gfx942", 304)
    key = make_cache_key(4096, 4096, 4096, "bf16", "bf16", "bf16", 0, True, 2, 304)
    cache.store(key, _sample_params())

    reset_registry()
    cache2 = get_cache("gfx942", 304)
    assert cache2 is not cache
    assert cache2.lookup(key) == _sample_params()


def test_arch_isolation(tmp_cache_dir):
    from tritonblas.autotune_cache import get_cache, make_cache_key

    c942 = get_cache("gfx942", 304)
    c950 = get_cache("gfx950", 256)
    key = make_cache_key(1024, 1024, 1024, "f16", "f16", "f16", 0, False, 2, 304)
    c942.store(key, _sample_params())
    assert c950.lookup(key) is None


def test_version_mismatch_invalidates(tmp_cache_dir):
    from tritonblas.autotune_cache import get_cache, make_cache_key, reset_registry

    cache = get_cache("gfx942", 304)
    key = make_cache_key(2048, 2048, 2048, "f16", "f16", "f16", 0, False, 2, 304)
    cache.store(key, _sample_params())

    raw = json.loads(cache.path.read_text())
    raw["tritonblas_version"] = "git:DEADBEEF-not-a-real-hash"
    cache.path.write_text(json.dumps(raw))

    reset_registry()
    cache2 = get_cache("gfx942", 304)
    assert cache2.lookup(key) is None


def test_disabled_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "0")
    import tritonblas.autotune_cache as ac
    importlib.reload(ac)
    ac.reset_registry()

    cache = ac.get_cache("gfx942", 304)
    assert cache.enabled is False
    key = ac.make_cache_key(1, 1, 1, "f16", "f16", "f16", 0, False, 2, 304)
    cache.store(key, _sample_params())
    assert cache.lookup(key) is None
    assert not any(p.suffix == ".json" for p in tmp_path.iterdir())


def test_store_rejects_partial_params(tmp_cache_dir):
    from tritonblas.autotune_cache import get_cache, make_cache_key

    cache = get_cache("gfx942", 304)
    key = make_cache_key(8, 8, 8, "f16", "f16", "f16", 0, False, 2, 304)
    with pytest.raises(ValueError):
        cache.store(key, {"block_m": 16, "block_n": 16})


def test_make_cache_key_is_stable():
    from tritonblas.autotune_cache import make_cache_key

    a = make_cache_key(1024, 2048, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    b = make_cache_key(1024, 2048, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    assert a == b
    c = make_cache_key(1024, 2048, 4096, "f16", "f16", "f16", 0, True, 2, 304)
    assert a != c, "streamk flag must be part of the key"
    d = make_cache_key(1024, 2048, 4096, "bf16", "bf16", "bf16", 0, False, 2, 304)
    assert a != d, "dtype must be part of the key"


# ---------------------------------------------------------------------------
# K-591: boundary-tolerance fall-back.  Four core tests:
#   1. happy path — jittered shape inside the band hits the cached entry
#   2. negative path — shape outside the per-axis band correctly misses
#   3. suffix mismatch — dtype / streamk / num_stages / n_cu must match exactly
#   4. drift guard — nearest-match results are NOT persisted (K-588 chain risk)
# ---------------------------------------------------------------------------


def test_lookup_nearest_hits_within_tolerance(tmp_cache_dir):
    """K-588 cold-miss workload distilled to one shape: canonical
    (4096,4096,4096) is stored, +/-1 jitters on every axis must hit."""
    from tritonblas.autotune_cache import get_cache, make_cache_key

    cache = get_cache("gfx942", 304)
    canon = make_cache_key(4096, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    cache.store(canon, _sample_params())

    for dM, dN, dK in [(+1, +1, +1), (-1, -1, -1), (+1, 0, -1), (0, +1, 0)]:
        jit = make_cache_key(
            4096 + dM, 4096 + dN, 4096 + dK,
            "f16", "f16", "f16", 0, False, 2, 304,
        )
        assert cache.lookup(jit) is None, (
            f"sanity: +/-1 jitter {(dM, dN, dK)} must miss exact lookup"
        )
        near = cache.lookup_nearest(jit)
        assert near is not None, (
            f"+/-1 jitter {(dM, dN, dK)} on a tile-boundary shape MUST "
            f"hit boundary-tolerance lookup (this is the K-591 fix)"
        )
        params, matched = near
        assert matched == canon
        assert params == _sample_params()


def test_lookup_nearest_outside_tolerance_misses(tmp_cache_dir):
    """A shape whose deviation on at least one axis exceeds ``BOUNDARY_TOLERANCE``
    must NOT borrow the cached entry — the caller must fall through to a
    full Origami tune.  Silently borrowing a far entry is the worst failure
    mode (regress to a poor default)."""
    from tritonblas.autotune_cache import get_cache, make_cache_key

    cache = get_cache("gfx942", 304)
    canon = make_cache_key(4096, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    cache.store(canon, _sample_params())

    # Each row breaks the 5% band on exactly one axis.
    for M, N, K in [
        (4096 + 256, 4096, 4096),  # M off ~6.25%
        (4096, 4096 + 256, 4096),  # N off ~6.25%
        (4096, 4096, 4096 + 256),  # K off ~6.25%
        (4096, 4096, 8192),         # K doubled
    ]:
        far = make_cache_key(M, N, K, "f16", "f16", "f16", 0, False, 2, 304)
        assert cache.lookup(far) is None
        assert cache.lookup_nearest(far) is None, (
            f"shape {(M, N, K)} is outside 5% per-axis tolerance and "
            f"must NOT borrow the canonical entry"
        )


def test_lookup_nearest_requires_suffix_match(tmp_cache_dir):
    """K-588 F9: the suffix (dtypes, streamk, num_stages, n_cu) must match
    exactly so the cached tile is LDS-safe for the substituted shape.
    Crossing dtype, streamk mode, or num_stages is a hard miss."""
    from tritonblas.autotune_cache import get_cache, make_cache_key

    cache = get_cache("gfx942", 304)
    canon = make_cache_key(4096, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    cache.store(canon, _sample_params())

    near_shape = (4097, 4096, 4096)
    for label, key in [
        ("dtype", make_cache_key(*near_shape, "bf16", "bf16", "bf16", 0, False, 2, 304)),
        ("streamk", make_cache_key(*near_shape, "f16", "f16", "f16", 0, True, 2, 304)),
        ("num_stages", make_cache_key(*near_shape, "f16", "f16", "f16", 0, False, 3, 304)),
    ]:
        assert cache.lookup_nearest(key) is None, (
            f"near-neighbor with mismatched {label} must NOT borrow"
        )


def test_lookup_nearest_does_not_persist_synthetic_entries(tmp_cache_dir):
    """K-591 Devil's Advocate: nearest-match results must NOT enter the
    on-disk cache, otherwise a chain A -> A' -> A'' could drift arbitrarily
    from the canonical tuned entry."""
    from tritonblas.autotune_cache import get_cache, make_cache_key

    cache = get_cache("gfx942", 304)
    canon = make_cache_key(2048, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    cache.store(canon, _sample_params())
    assert len(cache) == 1

    jit = make_cache_key(2049, 4097, 4090, "f16", "f16", "f16", 0, False, 2, 304)
    assert cache.lookup_nearest(jit) is not None  # hits — but we don't store

    assert len(cache) == 1
    assert canon in cache
    assert jit not in cache

    raw = json.loads(cache.path.read_text())
    assert list(raw["entries"].keys()) == [canon]


def test_make_cache_key_rejects_active_cu_arg():
    """K-591 Minimalist guard: ``active_cu`` was removed from the cache
    API entirely (rather than deprecated in place), so a refactor that
    silently re-adds it as a positional or keyword argument must surface
    as a hard TypeError instead of as a silently-ignored knob.

    Including a sub-CU mask in the key would re-introduce the K-588
    "near-miss -> 0% hit" failure mode: canonical entries are stored
    without an explicit sub-CU mask while queries may supply one,
    silently splitting the suffix bucket.
    """
    import inspect

    from tritonblas import autotune_cache as ac

    sig = inspect.signature(ac.make_cache_key)
    assert "active_cu" not in sig.parameters, (
        "active_cu must not reappear in make_cache_key; fold sub-CU "
        "context into n_cu instead (see autotune_cache module docstring)."
    )

    # Calling with the old positional active_cu must raise loudly.
    with pytest.raises(TypeError):
        ac.make_cache_key(
            2048, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304, 256
        )


def test_boundary_tolerance_is_module_constant():
    """K-591 Minimalist: ``BOUNDARY_TOLERANCE`` is a module constant, not
    an env-var knob.  A module constant is auditable in code review and
    avoids silent per-process variation in cache behaviour that would
    make production debugging harder."""
    from tritonblas import autotune_cache as ac

    assert hasattr(ac, "BOUNDARY_TOLERANCE")
    assert isinstance(ac.BOUNDARY_TOLERANCE, float)
    assert 0.0 < ac.BOUNDARY_TOLERANCE <= 0.5, (
        "tolerance must be a small positive fraction (5% by default; >50%"
        " would let near-miss serve arbitrary shapes)."
    )
