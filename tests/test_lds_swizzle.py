# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the LDS bank-conflict mitigation knob (K-580).

These tests cover the pure-Python plumbing — env-var modes, the medium-K
residual gate, persistent cache round-trips and autotune-callback wiring —
without requiring a GPU.  GPU-side correctness is covered by the existing
test_matmul_correctness suite once the kernel call sites pick up the new
config.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))

from tritonblas.lds_swizzle import (  # noqa: E402
    BASELINE_CONFIG,
    SMALL_K_GATE_THRESHOLD,
    SWIZZLED_CONFIG,
    LDSSwizzleConfig,
    PersistentSwizzleCache,
    get_mode,
    is_medium_k_residual,
    reset_cache_for_testing,
    select_lds_config,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Each test gets a clean env + a private cache file."""
    monkeypatch.delenv("TRITONBLAS_LDS_SWIZZLE", raising=False)
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE_CACHE", str(tmp_path / "cache.json"))
    reset_cache_for_testing()
    yield
    reset_cache_for_testing()


# ──────────────────────────────────────────────────────────────────────────────
# Mode resolution
# ──────────────────────────────────────────────────────────────────────────────


def test_default_mode_is_auto():
    assert get_mode() == "auto"


def test_unknown_mode_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "garbage")
    assert get_mode() == "auto"


@pytest.mark.parametrize("mode", ["off", "auto", "on", "autotune"])
def test_recognized_modes(monkeypatch, mode):
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", mode)
    assert get_mode() == mode


# ──────────────────────────────────────────────────────────────────────────────
# Medium-K residual gate
# ──────────────────────────────────────────────────────────────────────────────


def test_medium_k_residual_in_band():
    assert is_medium_k_residual(4096, 4096, 1024, block_k=64)
    assert is_medium_k_residual(4096, 4096, 1536, block_k=64)
    assert is_medium_k_residual(8192, 8192, 2048, block_k=64)


def test_medium_k_residual_small_k_excluded():
    """Small K (< SMALL_K_GATE_THRESHOLD) is excluded — the K-loop is too
    short to amortize the wider-load VGPR pressure of kpack=2."""
    assert not is_medium_k_residual(4096, 4096, 256, block_k=64)
    assert not is_medium_k_residual(4096, 4096, 512, block_k=64)
    assert not is_medium_k_residual(4096, 4096, 768, block_k=64)
    # Boundary: exactly the threshold IS in-band.
    assert is_medium_k_residual(4096, 4096, SMALL_K_GATE_THRESHOLD, block_k=64)
    assert not is_medium_k_residual(
        4096, 4096, SMALL_K_GATE_THRESHOLD - 1, block_k=64
    )


def test_medium_k_residual_out_of_band_low_k():
    assert not is_medium_k_residual(4096, 4096, 128, block_k=64)


def test_medium_k_residual_out_of_band_high_k():
    assert not is_medium_k_residual(4096, 4096, 4096, block_k=64)


def test_medium_k_residual_small_mn_excluded():
    assert not is_medium_k_residual(512, 512, 1024, block_k=64)


def test_medium_k_residual_large_block_k_excluded():
    assert not is_medium_k_residual(4096, 4096, 1024, block_k=128)


# ──────────────────────────────────────────────────────────────────────────────
# Mode → config mapping
# ──────────────────────────────────────────────────────────────────────────────


def _select(**overrides):
    kwargs = dict(
        M=4096, N=4096, K=1024,
        a_dtype="f16", b_dtype="f16", c_dtype="f16",
        block_m=128, block_n=128, block_k=64,
    )
    kwargs.update(overrides)
    return select_lds_config(**kwargs)


def test_off_mode_returns_baseline(monkeypatch):
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "off")
    assert _select() == BASELINE_CONFIG


def test_on_mode_returns_swizzled(monkeypatch):
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "on")
    assert _select() == SWIZZLED_CONFIG


# ──────────────────────────────────────────────────────────────────────────────
# Small-K runtime guard — the K-539 cohort regression fix.
#
# These tests pin the small-K guard against regression. The guard runs
# BEFORE the cache lookup and BEFORE the "on" mode override so neither a
# stale cache entry nor a forced "on" can demote a small-K shape into the
# regressing kpack=2 config.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("small_k", [64, 128, 256, 512, 768, SMALL_K_GATE_THRESHOLD - 1])
def test_small_k_guard_overrides_on_mode(monkeypatch, small_k):
    """Even mode=on must not pick swizzle for K below the threshold."""
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "on")
    assert _select(K=small_k) == BASELINE_CONFIG


@pytest.mark.parametrize("small_k", [64, 256, 512, 768, SMALL_K_GATE_THRESHOLD - 1])
def test_small_k_guard_overrides_cached_swizzle(monkeypatch, tmp_path, small_k):
    """A stale cached SWIZZLED_CONFIG for a small-K shape must be ignored."""
    cache_path = tmp_path / "stale.json"
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE_CACHE", str(cache_path))
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "auto")
    reset_cache_for_testing()

    # Pre-populate cache with SWIZZLED for the small-K shape (simulating a
    # stale entry from before the guard was added).
    cache = PersistentSwizzleCache(path=cache_path)
    sk = "ds"
    ws = "nows"
    key = (
        f"2048x2048x{small_k}|f16|f16|f16|"
        f"128x128x64|{sk}|{ws}"
    )
    cache.set(key, SWIZZLED_CONFIG)

    reset_cache_for_testing()  # force singleton to re-read on disk
    cfg = select_lds_config(
        2048, 2048, small_k,
        "f16", "f16", "f16",
        128, 128, 64,
        streamk=False, work_stealing=False,
    )
    # Even though the cache says SWIZZLED, the guard demotes to BASELINE.
    assert cfg == BASELINE_CONFIG


def test_small_k_guard_preserves_off_mode(monkeypatch):
    """mode=off short-circuits before the guard — both still return BASELINE."""
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "off")
    assert _select(K=512) == BASELINE_CONFIG
    assert _select(K=2048) == BASELINE_CONFIG


def test_threshold_boundary_small_k_guard(monkeypatch):
    """K == threshold passes the guard; K == threshold-1 is gated."""
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "on")
    # Exactly at the threshold: guard does NOT demote.
    assert _select(K=SMALL_K_GATE_THRESHOLD) == SWIZZLED_CONFIG
    # One below: guard demotes.
    assert _select(K=SMALL_K_GATE_THRESHOLD - 1) == BASELINE_CONFIG


def test_k539_cohort_kpack_is_one(monkeypatch):
    """The K-539 small-K cohort (M,N=512..2048, K=512) must dispatch kpack=1.

    This is the pinned acceptance test for the K-681 fix: every shape in the
    cohort must take the baseline (kpack=1) config regardless of mode.
    """
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "on")
    for mn in (512, 1024, 1536, 2048):
        cfg = _select(M=mn, N=mn, K=512)
        assert cfg.kpack == 1, (
            f"K-539 cohort regression: M=N={mn}, K=512 dispatched kpack={cfg.kpack}, "
            "expected 1 (small-K gate must demote)."
        )


@pytest.mark.parametrize(
    "mode,K,cache_seed,expected_kpack,case",
    [
        # 1. mode=off short-circuits FIRST — wins over everything.
        ("off",  512, None,            1, "mode=off  | K<gate          | empty cache  -> BASELINE"),
        ("off",  2048, None,           1, "mode=off  | K>=gate         | empty cache  -> BASELINE"),
        ("off",  2048, SWIZZLED_CONFIG, 1, "mode=off  | K>=gate         | cache=SWIZ   -> BASELINE (off wins)"),
        # 2. K<gate short-circuits SECOND — wins over mode=on AND cache.
        ("on",   512, None,            1, "mode=on   | K<gate          | empty cache  -> BASELINE (gate wins)"),
        ("auto", 512, SWIZZLED_CONFIG, 1, "mode=auto | K<gate          | cache=SWIZ   -> BASELINE (gate wins)"),
        ("on",   SMALL_K_GATE_THRESHOLD - 1, None, 1,
                                          "mode=on   | K=gate-1        | empty cache  -> BASELINE (gate wins)"),
        # 3. mode=on short-circuits THIRD — wins over cache (only at K>=gate).
        ("on",   2048, None,            2, "mode=on   | K>=gate         | empty cache  -> SWIZZLED"),
        ("on",   2048, BASELINE_CONFIG, 2, "mode=on   | K>=gate         | cache=BASE   -> SWIZZLED (on wins)"),
        ("on",   SMALL_K_GATE_THRESHOLD, None, 2,
                                          "mode=on   | K=gate          | empty cache  -> SWIZZLED"),
        # 4. cache lookup runs LAST (auto mode, K>=gate, populated cache).
        ("auto", 2048, SWIZZLED_CONFIG, 2, "mode=auto | K>=gate         | cache=SWIZ   -> SWIZZLED (cache hit)"),
        ("auto", 2048, BASELINE_CONFIG, 1, "mode=auto | K>=gate         | cache=BASE   -> BASELINE (cache hit)"),
        ("auto", 2048, None,            1, "mode=auto | K>=gate         | empty cache  -> BASELINE (safe default)"),
    ],
)
def test_select_lds_config_ordering_invariant(
    monkeypatch, tmp_path, mode, K, cache_seed, expected_kpack, case
):
    """Pin the four-step ordering invariant of ``select_lds_config``.

    The decision tree MUST be evaluated in this exact order:

        1. mode == "off"       -> BASELINE  (user opt-out wins absolutely)
        2. K  <  SMALL_K_GATE  -> BASELINE  (small-K guard; pre-empts mode=on AND cache)
        3. mode == "on"        -> SWIZZLED  (user force, only above threshold)
        4. cache.get(key)      -> cached    (autotuned winner; otherwise BASELINE)

    Reordering any step silently re-introduces the K-580 -27% K-539 cohort
    regression. This parametrized test is the load-bearing CI gate that
    enforces the invariant beyond the inline comment block in the source.
    """
    cache_path = tmp_path / "ordering.json"
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE_CACHE", str(cache_path))
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", mode)
    reset_cache_for_testing()

    if cache_seed is not None:
        # Seed the cache with the explicit (BASELINE | SWIZZLED) entry. The
        # key MUST match what _cache_key would produce for the test shape.
        cache = PersistentSwizzleCache(path=cache_path)
        key = (
            f"2048x2048x{K}|f16|f16|f16|"
            f"128x128x64|ds|nows"
        )
        cache.set(key, cache_seed)
        reset_cache_for_testing()

    cfg = select_lds_config(
        2048, 2048, K,
        "f16", "f16", "f16",
        128, 128, 64,
        streamk=False, work_stealing=False,
    )
    assert cfg.kpack == expected_kpack, f"ordering invariant broken: {case}"


def test_auto_mode_defaults_to_baseline_when_cache_empty():
    """auto is safe: with no cache entry it picks baseline (kpack=1).

    The medium-K heuristic exists as advice for the autotuner; auto itself
    must never regress shapes that haven't been measured yet.
    """
    # Medium-K shape with empty cache — still baseline.
    assert _select(M=4096, N=4096, K=1024, block_k=64) == BASELINE_CONFIG
    # Out-of-band shapes — also baseline.
    assert _select(K=128) == BASELINE_CONFIG
    assert _select(K=4096) == BASELINE_CONFIG
    assert _select(block_k=128) == BASELINE_CONFIG


# ──────────────────────────────────────────────────────────────────────────────
# Persistent cache
# ──────────────────────────────────────────────────────────────────────────────


def test_persistent_cache_roundtrip(tmp_path):
    cache = PersistentSwizzleCache(path=tmp_path / "rt.json")
    cache.set("shape-A", LDSSwizzleConfig(kpack=2, num_warps=4))
    # Reload from disk.
    cache2 = PersistentSwizzleCache(path=tmp_path / "rt.json")
    cfg = cache2.get("shape-A")
    assert cfg == LDSSwizzleConfig(kpack=2, num_warps=4)


def test_persistent_cache_handles_corrupt_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    cache = PersistentSwizzleCache(path=p)
    assert cache.get("anything") is None
    # Still writable after recovery.
    cache.set("k", LDSSwizzleConfig(kpack=2))
    assert cache.get("k") == LDSSwizzleConfig(kpack=2, num_warps=8)


def test_cache_overrides_heuristic(monkeypatch, tmp_path):
    """A cached entry takes precedence over the heuristic."""
    cache_path = tmp_path / "override.json"
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE_CACHE", str(cache_path))
    reset_cache_for_testing()

    # Pre-populate cache: medium-K shape forced to BASELINE.
    cache = PersistentSwizzleCache(path=cache_path)
    key = (
        "4096x4096x1024|torch.float16|torch.float16|torch.float16|"
        "128x128x64|ds|nows"
    )
    cache.set(key, BASELINE_CONFIG)

    reset_cache_for_testing()  # force singleton to re-read on disk
    cfg = select_lds_config(
        4096, 4096, 1024,
        "torch.float16", "torch.float16", "torch.float16",
        128, 128, 64,
        streamk=False, work_stealing=False,
    )
    # Heuristic would have said SWIZZLED_CONFIG; cache wins.
    assert cfg == BASELINE_CONFIG


# ──────────────────────────────────────────────────────────────────────────────
# Autotune callback
# ──────────────────────────────────────────────────────────────────────────────


def test_autotune_picks_faster_and_persists(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto.json"
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE_CACHE", str(cache_path))
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "autotune")
    reset_cache_for_testing()

    # Synthetic timing: swizzled is faster.
    def timing(cfg: LDSSwizzleConfig) -> float:
        return 1.0 if cfg == SWIZZLED_CONFIG else 2.0

    cfg = select_lds_config(
        4096, 4096, 1024,
        "f16", "f16", "f16",
        128, 128, 64,
        streamk=False, work_stealing=False,
        autotune_fn=timing,
    )
    assert cfg == SWIZZLED_CONFIG

    # Cache file written.
    saved = json.loads(cache_path.read_text())
    assert any(v.get("kpack") == 2 for v in saved.values())


def test_autotune_picks_baseline_when_faster(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE_CACHE", str(tmp_path / "auto2.json"))
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "autotune")
    reset_cache_for_testing()

    def timing(cfg: LDSSwizzleConfig) -> float:
        return 1.0 if cfg == BASELINE_CONFIG else 5.0

    cfg = select_lds_config(
        4096, 4096, 1024,
        "f16", "f16", "f16",
        128, 128, 64,
        autotune_fn=timing,
    )
    assert cfg == BASELINE_CONFIG


def test_autotune_falls_back_on_callback_error(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE_CACHE", str(tmp_path / "err.json"))
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "autotune")
    reset_cache_for_testing()

    def boom(_):
        raise RuntimeError("kernel failed")

    # Falls through to auto; cache empty → safe baseline (no regression risk).
    cfg = select_lds_config(
        4096, 4096, 1024,
        "f16", "f16", "f16",
        128, 128, 64,
        autotune_fn=boom,
    )
    assert cfg == BASELINE_CONFIG
