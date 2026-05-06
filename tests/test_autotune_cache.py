"""Unit tests for the persistent autotune cache (K-536).

These tests do not require a GPU: they exercise the file-format / lookup
behavior with a temporary cache directory.  GPU-bearing tests live in the
existing ``test_matmul.py`` suite and are unaffected.
"""

from __future__ import annotations

import json
import os
import importlib

import pytest


@pytest.fixture
def tmp_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "1")
    # Reload the cache module so registry / env vars are re-read.
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


def test_lookup_miss_returns_none(tmp_cache_dir):
    from tritonblas.autotune_cache import get_cache, make_cache_key

    cache = get_cache("gfx942", 304)
    key = make_cache_key(8192, 8192, 8192, "f16", "f16", "f16", 0, False, 2, 304, None)
    assert cache.lookup(key) is None
    assert len(cache) == 0


def test_store_then_lookup_roundtrip(tmp_cache_dir):
    from tritonblas.autotune_cache import get_cache, make_cache_key

    cache = get_cache("gfx942", 304)
    key = make_cache_key(8192, 8192, 8192, "f16", "f16", "f16", 0, False, 2, 304, None)
    cache.store(key, _sample_params())

    got = cache.lookup(key)
    assert got is not None
    assert got["block_m"] == 128
    assert got["block_n"] == 128
    assert got["block_k"] == 32
    assert got["workgroup_mapping"] == 8
    assert got["counters_per_xcd"] == 4


def test_persistence_across_processes(tmp_cache_dir):
    """A fresh registry instance must read the on-disk cache."""
    from tritonblas.autotune_cache import (
        get_cache,
        make_cache_key,
        reset_registry,
    )

    cache = get_cache("gfx942", 304)
    key = make_cache_key(4096, 4096, 4096, "bf16", "bf16", "bf16", 0, True, 2, 304, None)
    cache.store(key, _sample_params())

    # Simulate a fresh process: drop the in-memory registry and re-load.
    reset_registry()
    cache2 = get_cache("gfx942", 304)
    assert cache2 is not cache
    got = cache2.lookup(key)
    assert got is not None
    assert got == _sample_params()


def test_arch_isolation(tmp_cache_dir):
    """gfx950 must not see entries written under gfx942."""
    from tritonblas.autotune_cache import get_cache, make_cache_key

    c942 = get_cache("gfx942", 304)
    c950 = get_cache("gfx950", 256)
    key = make_cache_key(1024, 1024, 1024, "f16", "f16", "f16", 0, False, 2, 304, None)
    c942.store(key, _sample_params())
    assert c950.lookup(key) is None


def test_version_mismatch_invalidates(tmp_cache_dir):
    from tritonblas import autotune_cache as ac
    from tritonblas.autotune_cache import get_cache, make_cache_key, reset_registry

    cache = get_cache("gfx942", 304)
    key = make_cache_key(2048, 2048, 2048, "f16", "f16", "f16", 0, False, 2, 304, None)
    cache.store(key, _sample_params())

    # Tamper the on-disk file so the version key no longer matches.
    raw = json.loads(cache.path.read_text())
    raw["tritonblas_version"] = "git:DEADBEEF-not-a-real-hash"
    cache.path.write_text(json.dumps(raw))

    reset_registry()
    cache2 = get_cache("gfx942", 304)
    assert cache2.lookup(key) is None, "stale-version entries must be ignored"


def test_schema_mismatch_invalidates(tmp_cache_dir):
    from tritonblas.autotune_cache import get_cache, make_cache_key, reset_registry

    cache = get_cache("gfx942", 304)
    key = make_cache_key(512, 512, 512, "f16", "f16", "f16", 0, False, 2, 304, None)
    cache.store(key, _sample_params())

    raw = json.loads(cache.path.read_text())
    raw["schema_version"] = -999
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
    key = ac.make_cache_key(1, 1, 1, "f16", "f16", "f16", 0, False, 2, 304, None)
    cache.store(key, _sample_params())
    # Disabled cache should never expose entries.
    assert cache.lookup(key) is None
    # And no file should be written.
    assert not any(p.suffix == ".json" for p in tmp_path.iterdir())


def test_store_rejects_partial_params(tmp_cache_dir):
    from tritonblas.autotune_cache import get_cache, make_cache_key

    cache = get_cache("gfx942", 304)
    key = make_cache_key(8, 8, 8, "f16", "f16", "f16", 0, False, 2, 304, None)
    bad = {"block_m": 16, "block_n": 16}  # missing required fields
    with pytest.raises(ValueError):
        cache.store(key, bad)


def test_make_cache_key_is_stable():
    from tritonblas.autotune_cache import make_cache_key

    a = make_cache_key(1024, 2048, 4096, "f16", "f16", "f16", 0, False, 2, 304, None)
    b = make_cache_key(1024, 2048, 4096, "f16", "f16", "f16", 0, False, 2, 304, None)
    assert a == b
    c = make_cache_key(1024, 2048, 4096, "f16", "f16", "f16", 0, True, 2, 304, None)
    assert a != c, "streamk flag must be part of the key"
    d = make_cache_key(1024, 2048, 4096, "bf16", "bf16", "bf16", 0, False, 2, 304, None)
    assert a != d, "dtype must be part of the key"
