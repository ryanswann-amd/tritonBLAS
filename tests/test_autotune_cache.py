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
    key = make_cache_key(8192, 8192, 8192, "f16", "f16", "f16", 0, False, 2, 304)
    assert cache.lookup(key) is None
    assert len(cache) == 0


def test_store_then_lookup_roundtrip(tmp_cache_dir):
    from tritonblas.autotune_cache import get_cache, make_cache_key

    cache = get_cache("gfx942", 304)
    key = make_cache_key(8192, 8192, 8192, "f16", "f16", "f16", 0, False, 2, 304)
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
    key = make_cache_key(4096, 4096, 4096, "bf16", "bf16", "bf16", 0, True, 2, 304)
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
    key = make_cache_key(1024, 1024, 1024, "f16", "f16", "f16", 0, False, 2, 304)
    c942.store(key, _sample_params())
    assert c950.lookup(key) is None


def test_version_mismatch_invalidates(tmp_cache_dir):
    from tritonblas import autotune_cache as ac
    from tritonblas.autotune_cache import get_cache, make_cache_key, reset_registry

    cache = get_cache("gfx942", 304)
    key = make_cache_key(2048, 2048, 2048, "f16", "f16", "f16", 0, False, 2, 304)
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
    key = make_cache_key(512, 512, 512, "f16", "f16", "f16", 0, False, 2, 304)
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
    key = ac.make_cache_key(1, 1, 1, "f16", "f16", "f16", 0, False, 2, 304)
    cache.store(key, _sample_params())
    # Disabled cache should never expose entries.
    assert cache.lookup(key) is None
    # And no file should be written.
    assert not any(p.suffix == ".json" for p in tmp_path.iterdir())


def test_store_rejects_partial_params(tmp_cache_dir):
    from tritonblas.autotune_cache import get_cache, make_cache_key

    cache = get_cache("gfx942", 304)
    key = make_cache_key(8, 8, 8, "f16", "f16", "f16", 0, False, 2, 304)
    bad = {"block_m": 16, "block_n": 16}  # missing required fields
    with pytest.raises(ValueError):
        cache.store(key, bad)


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
# Boundary-tolerance lookup (K-591 / K-588 follow-up)
# ---------------------------------------------------------------------------


def test_parse_cache_key_roundtrip():
    from tritonblas.autotune_cache import make_cache_key, parse_cache_key

    key = make_cache_key(2048, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    parsed = parse_cache_key(key)
    assert parsed is not None
    m, n, k, suffix = parsed
    assert (m, n, k) == (2048, 4096, 4096)
    # Suffix must round-trip exactly so two keys differing only in shape
    # still share a suffix-index bucket.
    other = make_cache_key(2049, 4097, 4090, "f16", "f16", "f16", 0, False, 2, 304)
    other_parsed = parse_cache_key(other)
    assert other_parsed is not None
    assert other_parsed[3] == suffix


def test_parse_cache_key_rejects_garbage():
    from tritonblas.autotune_cache import parse_cache_key

    assert parse_cache_key("") is None
    assert parse_cache_key("not-a-key") is None
    assert parse_cache_key("1x2|suffix") is None  # missing K
    assert parse_cache_key("axbxc|suffix") is None  # non-numeric
    assert parse_cache_key("0x4096x4096|suffix") is None  # zero dim
    assert parse_cache_key(None) is None  # type-safe


def test_lookup_nearest_hit_within_tolerance(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "1")
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "0.05")
    import tritonblas.autotune_cache as ac
    importlib.reload(ac)
    ac.reset_registry()

    cache = ac.get_cache("gfx942", 304)
    canon_key = ac.make_cache_key(
        2048, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304
    )
    cache.store(canon_key, _sample_params())

    # +/- 1 jitter on every axis (K-510 dynamic-batch pattern).
    jit_key = ac.make_cache_key(
        2049, 4097, 4090, "f16", "f16", "f16", 0, False, 2, 304
    )
    assert cache.lookup(jit_key) is None, "exact lookup must miss the jittered key"

    near = cache.lookup_nearest(jit_key)
    assert near is not None, "jittered shape within 5% must hit boundary-tolerance lookup"
    params, matched_key = near
    assert matched_key == canon_key
    assert params == _sample_params()


def test_lookup_nearest_skips_when_tolerance_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "1")
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "0")
    import tritonblas.autotune_cache as ac
    importlib.reload(ac)
    ac.reset_registry()

    cache = ac.get_cache("gfx942", 304)
    assert cache.tolerance == 0.0
    canon_key = ac.make_cache_key(
        2048, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304
    )
    cache.store(canon_key, _sample_params())

    jit_key = ac.make_cache_key(
        2049, 4097, 4090, "f16", "f16", "f16", 0, False, 2, 304
    )
    assert cache.lookup_nearest(jit_key) is None, "tolerance=0 must disable fall-back"


def test_lookup_nearest_outside_tolerance_misses(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "1")
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "0.05")
    import tritonblas.autotune_cache as ac
    importlib.reload(ac)
    ac.reset_registry()

    cache = ac.get_cache("gfx942", 304)
    canon_key = ac.make_cache_key(
        2048, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304
    )
    cache.store(canon_key, _sample_params())

    # M shifted ~17% — far outside the 5% band on the M axis.
    far_key = ac.make_cache_key(
        2400, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304
    )
    assert cache.lookup_nearest(far_key) is None


def test_lookup_nearest_keeps_dtype_and_streamk_exact(tmp_path, monkeypatch):
    """K-588 F9: the suffix (dtypes, streamk, num_stages, n_cu) must
    match exactly so the cached tile is LDS-safe for the substituted
    shape. Crossing dtypes or streamk modes is a hard miss even when the
    shape is a near-neighbor."""
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "1")
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "0.05")
    import tritonblas.autotune_cache as ac
    importlib.reload(ac)
    ac.reset_registry()

    cache = ac.get_cache("gfx942", 304)
    canon_key = ac.make_cache_key(
        2048, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304
    )
    cache.store(canon_key, _sample_params())

    # Same shape, different dtype — must not borrow.
    diff_dtype = ac.make_cache_key(
        2049, 4096, 4096, "bf16", "bf16", "bf16", 0, False, 2, 304
    )
    assert cache.lookup_nearest(diff_dtype) is None

    # Same shape, different streamk flag — must not borrow.
    diff_sk = ac.make_cache_key(
        2049, 4096, 4096, "f16", "f16", "f16", 0, True, 2, 304
    )
    assert cache.lookup_nearest(diff_sk) is None

    # Same shape, different num_stages — must not borrow.
    diff_ns = ac.make_cache_key(
        2049, 4096, 4096, "f16", "f16", "f16", 0, False, 3, 304
    )
    assert cache.lookup_nearest(diff_ns) is None


def test_lookup_nearest_picks_closest_in_log_l1(tmp_path, monkeypatch):
    """When two candidates are both within tolerance the lookup must
    return the one with the smaller log-L1 distance."""
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "1")
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "0.10")
    import tritonblas.autotune_cache as ac
    importlib.reload(ac)
    ac.reset_registry()

    cache = ac.get_cache("gfx942", 304)
    near_key = ac.make_cache_key(
        2050, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304
    )
    far_key = ac.make_cache_key(
        2200, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304
    )
    near_params = dict(_sample_params(), block_m=128)  # distinguishable
    far_params = dict(_sample_params(), block_m=64)
    cache.store(near_key, near_params)
    cache.store(far_key, far_params)

    query = ac.make_cache_key(
        2049, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304
    )
    near = cache.lookup_nearest(query)
    assert near is not None
    params, matched = near
    assert matched == near_key
    assert params["block_m"] == 128


def test_lookup_nearest_skips_exact_key(tmp_path, monkeypatch):
    """If the requested key is itself present, lookup() should already
    have returned it — lookup_nearest must skip it to avoid masking the
    bug where someone accidentally calls only the fall-back."""
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "1")
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "0.05")
    import tritonblas.autotune_cache as ac
    importlib.reload(ac)
    ac.reset_registry()

    cache = ac.get_cache("gfx942", 304)
    key = ac.make_cache_key(
        2048, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304
    )
    cache.store(key, _sample_params())
    # Only entry is the exact key -> nearest must miss.
    assert cache.lookup_nearest(key) is None


def test_lookup_nearest_persists_across_reload(tmp_path, monkeypatch):
    """The suffix index must be rebuilt from disk so a fresh process
    still sees boundary-tolerance hits."""
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "1")
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "0.05")
    import tritonblas.autotune_cache as ac
    importlib.reload(ac)
    ac.reset_registry()

    cache = ac.get_cache("gfx942", 304)
    canon = ac.make_cache_key(
        4096, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304
    )
    cache.store(canon, _sample_params())

    # Simulate a fresh process.
    ac.reset_registry()
    cache2 = ac.get_cache("gfx942", 304)
    assert cache2 is not cache
    near = cache2.lookup_nearest(
        ac.make_cache_key(4097, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    )
    assert near is not None
    params, matched = near
    assert matched == canon
    assert params == _sample_params()


def test_make_cache_suffix_rejects_active_cu_arg():
    """K-591 Minimalist guard: ``active_cu`` was removed from the suffix
    API entirely (rather than deprecated in place), so a refactor that
    silently re-adds it as a positional or keyword argument must surface
    as a hard TypeError instead of as a silently-ignored knob.

    Including ``active_cu`` in the suffix would re-introduce the K-588
    "near-miss -> 0% hit" failure mode, because canonical entries are
    stored without an explicit sub-CU mask while queries may supply one.
    """
    import inspect

    import tritonblas.autotune_cache as ac

    sig = inspect.signature(ac.make_cache_suffix)
    assert "active_cu" not in sig.parameters, (
        "active_cu must not reappear in make_cache_suffix; fold sub-CU "
        "context into n_cu instead (see autotune_cache module docstring)."
    )
    sig_key = inspect.signature(ac.make_cache_key)
    assert "active_cu" not in sig_key.parameters, (
        "active_cu must not reappear in make_cache_key either."
    )

    # And calling with the old positional active_cu must raise loudly.
    with pytest.raises(TypeError):
        ac.make_cache_suffix("f16", "f16", "f16", 0, False, 2, 304, 256)
    with pytest.raises(TypeError):
        ac.make_cache_key(
            2048, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304, 256
        )


def test_lookup_nearest_does_not_propagate_synthetic_entries(
    tmp_path, monkeypatch
):
    """K-591 Devil's Advocate: nearest-match results must NOT be persisted
    to the on-disk cache.  Only true Origami-tuned ``store()`` calls
    populate it; the in-memory nearest lookup is recomputed each time.
    Otherwise an interpolated entry could be picked as the source for a
    further interpolation, drifting arbitrarily from any tuned config.
    """
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "1")
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "0.05")
    import tritonblas.autotune_cache as ac
    importlib.reload(ac)
    ac.reset_registry()

    cache = ac.get_cache("gfx942", 304)
    canon = ac.make_cache_key(
        2048, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304
    )
    cache.store(canon, _sample_params())
    assert len(cache) == 1

    # Hit nearest-lookup — but do not call store() on the new key.
    jit = ac.make_cache_key(
        2049, 4097, 4090, "f16", "f16", "f16", 0, False, 2, 304
    )
    near = cache.lookup_nearest(jit)
    assert near is not None

    # The cache must still hold exactly one entry: the canonical store.
    assert len(cache) == 1
    assert canon in cache
    assert jit not in cache

    # On-disk file must mirror that — only the canonical entry is persisted.
    raw = json.loads(cache.path.read_text())
    assert list(raw["entries"].keys()) == [canon]


def test_tolerance_env_parsing(tmp_path, monkeypatch):
    """Resolution of TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE: explicit
    overrides, off-toggles, and out-of-range values are all handled."""
    import tritonblas.autotune_cache as ac

    # Numeric override
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "0.10")
    assert ac._configured_tolerance() == 0.10

    # Off toggles -> 0
    for off in ("0", "off", "false", "no", "none", ""):
        monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", off)
        assert ac._configured_tolerance() == 0.0, off

    # Negative -> 0
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "-0.5")
    assert ac._configured_tolerance() == 0.0

    # Garbage -> default
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "not-a-float")
    assert ac._configured_tolerance() == ac._DEFAULT_TOLERANCE

    # Above ceiling -> clamped
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "10.0")
    assert ac._configured_tolerance() == ac._MAX_TOLERANCE

    # Unset -> default
    monkeypatch.delenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", raising=False)
    assert ac._configured_tolerance() == ac._DEFAULT_TOLERANCE


# ---------------------------------------------------------------------------
# K-591 Testing-Zealot guards: explicit happy-path / negative-path tests for
# the per-axis boundary tolerance, named so they are unambiguously the
# "core" feature tests.  Stronger-than-the-existing-tests on two fronts:
#   * +/- 1 jitter on every axis (the dynamic-batch padding pattern that
#     K-588 root-caused) — proves the smallest nontrivial perturbation
#     hits the cached tile-boundary entry.
#   * a sharply over-tolerance shape on one axis confirms the lookup
#     correctly returns ``None`` (does NOT silently borrow a far entry).
# ---------------------------------------------------------------------------


def test_K591_boundary_tolerance_plus_minus_one_jitter_hits_cache(
    tmp_path, monkeypatch
):
    """Happy path: a +/- 1 jitter on every axis (M, N, K) must hit the
    cached entry written for the canonical tile-boundary shape.

    This is the K-588 cold-miss workload distilled to a single shape:
    canonical (4096, 4096, 4096) is stored, and the dynamic-batch jitter
    (4097, 4095, 4097) — a relative perturbation of ~0.024% on each axis,
    well inside the 5% default tolerance — must be served from cache.
    """
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "1")
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "0.05")
    import tritonblas.autotune_cache as ac
    importlib.reload(ac)
    ac.reset_registry()

    cache = ac.get_cache("gfx942", 304)
    canon = ac.make_cache_key(
        4096, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304
    )
    cache.store(canon, _sample_params())

    # Independent +/-1 jitters on every axis.
    for dM, dN, dK in [
        (+1, +1, +1),
        (-1, -1, -1),
        (+1, -1, +1),
        (-1, +1, -1),
        (+1, 0, 0),
        (0, +1, 0),
        (0, 0, -1),
    ]:
        jit_key = ac.make_cache_key(
            4096 + dM, 4096 + dN, 4096 + dK,
            "f16", "f16", "f16", 0, False, 2, 304,
        )
        assert cache.lookup(jit_key) is None, (
            f"sanity: +/-1 jitter {(dM, dN, dK)} must miss exact lookup"
        )
        near = cache.lookup_nearest(jit_key)
        assert near is not None, (
            f"+/-1 jitter {(dM, dN, dK)} on a tile-boundary shape MUST "
            f"hit boundary-tolerance lookup (this is the K-591 fix)"
        )
        params, matched = near
        assert matched == canon
        assert params == _sample_params()


def test_K591_boundary_tolerance_outside_band_correctly_misses(
    tmp_path, monkeypatch
):
    """Negative path: a shape whose deviation on at least one axis exceeds
    the configured tolerance must NOT borrow the cached entry.  We must
    return ``None`` so the caller falls through to the full Origami path
    and tunes a fresh, correct config — silently borrowing a far entry
    would be the worst possible failure mode (regress to a poor default).
    """
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "1")
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "0.05")
    import tritonblas.autotune_cache as ac
    importlib.reload(ac)
    ac.reset_registry()

    cache = ac.get_cache("gfx942", 304)
    canon = ac.make_cache_key(
        4096, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304
    )
    cache.store(canon, _sample_params())

    # Each row breaks tolerance on exactly one axis (>= 6%) while the other
    # two are perfectly aligned — proves the per-axis filter, not just the
    # aggregate distance, is what gates the lookup.
    over_tol_shapes = [
        (4096 + 256, 4096, 4096),  # M off by ~6.25%
        (4096, 4096 + 256, 4096),  # N off by ~6.25%
        (4096, 4096, 4096 + 256),  # K off by ~6.25%
        (4096 + 1024, 4096, 4096),  # M off by 25%
        (4096, 4096, 8192),         # K doubled (100% off)
    ]
    for M, N, K in over_tol_shapes:
        far_key = ac.make_cache_key(
            M, N, K, "f16", "f16", "f16", 0, False, 2, 304,
        )
        assert cache.lookup(far_key) is None
        assert cache.lookup_nearest(far_key) is None, (
            f"shape {(M, N, K)} is outside the 5% per-axis tolerance and "
            f"MUST NOT borrow the canonical entry; got a hit instead"
        )
