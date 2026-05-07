"""Unit tests for the persistent autotune cache (K-536) and the K-591
boundary-tolerance fall-back.

These tests do not require a GPU: they exercise the file-format / lookup
behavior with a temporary cache directory.  GPU-bearing tests live in the
existing ``test_matmul.py`` suite and are unaffected.

Layout:
    1. K-536 baseline — persistence, isolation, schema/version invalidation,
       partial-store rejection, key stability.
    2. K-591 fall-back — happy path, negative path, suffix isolation,
       no-drift, tie-breaking, exact-key skip, reload-survives.
    3. K-591 env-var contract — resolution rules + end-to-end override.
"""

from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture
def tmp_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "1")
    # Default tolerance for the fall-back tests.  Individual tests that
    # need a different value override it.
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "0.05")
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

    got = cache.lookup(key)
    assert got is not None
    assert got == _sample_params()
    # All required fields round-tripped (regression guard for the JSON
    # _coerce_jsonable path).
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

    reset_registry()
    cache2 = get_cache("gfx942", 304)
    assert cache2 is not cache
    assert cache2.lookup(key) == _sample_params()


def test_arch_isolation(tmp_cache_dir):
    """gfx950 must not see entries written under gfx942."""
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
    e = make_cache_key(1024, 2048, 4096, "f16", "f16", "f16", 0, False, 3, 304)
    assert a != e, "num_stages must be part of the key"


# ---------------------------------------------------------------------------
# K-591: parse_cache_key contract — needed by the boundary lookup AND by
# external sweep / debug tools.
# ---------------------------------------------------------------------------


def test_parse_cache_key_roundtrip():
    from tritonblas.autotune_cache import make_cache_key, parse_cache_key

    key = make_cache_key(2048, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    parsed = parse_cache_key(key)
    assert parsed is not None
    m, n, k, suffix = parsed
    assert (m, n, k) == (2048, 4096, 4096)

    # Two keys that differ only in shape MUST share a suffix string —
    # that's the bucketing invariant the boundary-tolerance lookup uses.
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


# ---------------------------------------------------------------------------
# K-591: boundary-tolerance fall-back — happy / negative / suffix / drift /
# tie-break / exact-key skip / reload.
# ---------------------------------------------------------------------------


def test_lookup_nearest_hits_within_tolerance(tmp_cache_dir):
    """K-588 cold-miss workload distilled to one shape: canonical
    (4096,4096,4096) is stored; jitters on every axis must hit."""
    from tritonblas.autotune_cache import get_cache, make_cache_key

    cache = get_cache("gfx942", 304)
    canon = make_cache_key(4096, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    cache.store(canon, _sample_params())

    for dM, dN, dK in [
        (+1, +1, +1),
        (-1, -1, -1),
        (+1, -1, +1),
        (-1, +1, -1),
        (+1, 0, 0),
        (0, +1, 0),
        (0, 0, -1),
    ]:
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
    """A shape whose deviation on at least one axis exceeds the tolerance
    must NOT borrow the cached entry — silently borrowing a far entry
    would be the worst failure mode (regress to a poor default)."""
    from tritonblas.autotune_cache import get_cache, make_cache_key

    cache = get_cache("gfx942", 304)
    canon = make_cache_key(4096, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    cache.store(canon, _sample_params())

    # Each row breaks the 5% band on exactly one axis.
    for M, N, K in [
        (4096 + 256, 4096, 4096),  # M off ~6.25%
        (4096, 4096 + 256, 4096),  # N off ~6.25%
        (4096, 4096, 4096 + 256),  # K off ~6.25%
        (4096 + 1024, 4096, 4096),  # M off 25%
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


def test_lookup_nearest_skips_when_tolerance_zero(tmp_path, monkeypatch):
    """Setting the env-var to ``0`` is the explicit kill switch and must
    disable the fall-back even though entries are present."""
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "1")
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "0")
    import tritonblas.autotune_cache as ac
    importlib.reload(ac)
    ac.reset_registry()

    cache = ac.get_cache("gfx942", 304)
    assert cache.tolerance == 0.0
    canon = ac.make_cache_key(2048, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    cache.store(canon, _sample_params())

    jit = ac.make_cache_key(2049, 4097, 4090, "f16", "f16", "f16", 0, False, 2, 304)
    assert cache.lookup_nearest(jit) is None, "tolerance=0 must disable fall-back"


def test_lookup_nearest_picks_closest_in_log_l1(tmp_path, monkeypatch):
    """When two candidates are within tolerance the lookup must return the
    one with the smaller log-L1 distance (scale-free tie break)."""
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "1")
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "0.10")
    import tritonblas.autotune_cache as ac
    importlib.reload(ac)
    ac.reset_registry()

    cache = ac.get_cache("gfx942", 304)
    near_key = ac.make_cache_key(2050, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    far_key = ac.make_cache_key(2200, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    near_params = dict(_sample_params(), block_m=128)  # distinguishable
    far_params = dict(_sample_params(), block_m=64)
    cache.store(near_key, near_params)
    cache.store(far_key, far_params)

    query = ac.make_cache_key(2049, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    near = cache.lookup_nearest(query)
    assert near is not None
    params, matched = near
    assert matched == near_key
    assert params["block_m"] == 128


def test_lookup_nearest_skips_exact_key(tmp_cache_dir):
    """If the requested key is itself present, ``lookup_nearest`` must
    skip it — otherwise nearest-only callers would silently mask the
    exact-lookup-was-skipped bug."""
    from tritonblas.autotune_cache import get_cache, make_cache_key

    cache = get_cache("gfx942", 304)
    key = make_cache_key(2048, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    cache.store(key, _sample_params())
    assert cache.lookup_nearest(key) is None


def test_lookup_nearest_persists_across_reload(tmp_cache_dir):
    """The fall-back must work after a registry reset: the on-disk file
    is the source of truth, no in-memory side-index is required."""
    from tritonblas.autotune_cache import (
        get_cache,
        make_cache_key,
        reset_registry,
    )

    cache = get_cache("gfx942", 304)
    canon = make_cache_key(4096, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    cache.store(canon, _sample_params())

    reset_registry()
    cache2 = get_cache("gfx942", 304)
    assert cache2 is not cache
    near = cache2.lookup_nearest(
        make_cache_key(4097, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    )
    assert near is not None
    params, matched = near
    assert matched == canon
    assert params == _sample_params()


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


# ---------------------------------------------------------------------------
# K-591 env-var contract: tolerance is operator-tunable.  The Skeptic
# explicitly required (a) the env-var override path, and (b) an end-to-end
# test proving an override actually changes lookup behaviour.
# ---------------------------------------------------------------------------


def test_boundary_tolerance_default_is_module_constant():
    """The compiled-in default is auditable as a module constant; the
    env-var only overrides it.  Tolerance is bounded to a sane fraction so
    operators cannot accidentally let near-miss serve arbitrary shapes."""
    from tritonblas import autotune_cache as ac

    assert hasattr(ac, "BOUNDARY_TOLERANCE")
    assert isinstance(ac.BOUNDARY_TOLERANCE, float)
    assert 0.0 < ac.BOUNDARY_TOLERANCE <= 0.5
    assert ac._DEFAULT_TOLERANCE == ac.BOUNDARY_TOLERANCE
    assert 0.0 < ac._MAX_TOLERANCE <= 1.0


def test_tolerance_env_parsing(tmp_path, monkeypatch):
    """Resolution of ``TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE``: explicit
    overrides, off-toggles, garbage and out-of-range values are all
    handled deterministically."""
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    import tritonblas.autotune_cache as ac
    importlib.reload(ac)

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


def test_tolerance_env_override_changes_lookup_end_to_end(
    tmp_path, monkeypatch
):
    """Skeptic: the env var must actually change observable lookup
    behaviour, not just the resolver return value.

    A shape that lies just outside the 5% default band but inside a 10%
    band must miss with the default tolerance and hit when the operator
    widens it via the env var — proving that ``lookup_nearest`` re-reads
    the env var per call and that the override path is wired all the way
    through.  Then flipping the env var to the kill-switch value must
    disable the fall-back without rebuilding the cache.
    """
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE", "1")
    # Start at the compiled-in default.
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "0.05")
    import tritonblas.autotune_cache as ac
    importlib.reload(ac)
    ac.reset_registry()

    cache = ac.get_cache("gfx942", 304)
    canon = ac.make_cache_key(4096, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    cache.store(canon, _sample_params())

    # ~6.25% off on the M axis — outside the default 5% band.
    jit = ac.make_cache_key(4352, 4096, 4096, "f16", "f16", "f16", 0, False, 2, 304)
    assert cache.tolerance == 0.05
    assert cache.lookup_nearest(jit) is None, (
        "default 5% tolerance must reject a 6.25% jitter"
    )

    # Operator widens the band to 10% — same cache, no rebuild — must hit.
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "0.10")
    assert cache.tolerance == 0.10
    near = cache.lookup_nearest(jit)
    assert near is not None, (
        "after env-var widens tolerance to 10%, the same shape must hit"
    )
    params, matched = near
    assert matched == canon
    assert params == _sample_params()

    # Operator pulls the kill switch — same cache, no rebuild — must miss.
    monkeypatch.setenv("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE", "off")
    assert cache.tolerance == 0.0
    assert cache.lookup_nearest(jit) is None, (
        "kill switch (TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE=off) must "
        "disable fall-back at runtime"
    )
