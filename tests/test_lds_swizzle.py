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
    # K=1024, BK=64 → mainloop_iters=16 (>=KPACK2_MAINLOOP_ITERS_MIN). Tile check
    # skipped when block_m/n omitted, so the legacy 2-arg form still admits.
    assert is_medium_k_residual(4096, 4096, 1024, block_k=64)
    # K=2048, BK=64 → mainloop_iters=32 (== KPACK2_MAINLOOP_ITERS_MAX, still
    # admitted since the upper bound is inclusive).
    assert is_medium_k_residual(8192, 8192, 2048, block_k=64)


def test_medium_k_residual_out_of_band_low_k():
    assert not is_medium_k_residual(4096, 4096, 128, block_k=64)


def test_medium_k_residual_out_of_band_high_k():
    assert not is_medium_k_residual(4096, 4096, 4096, block_k=64)


def test_medium_k_residual_small_mn_excluded():
    assert not is_medium_k_residual(512, 512, 1024, block_k=64)


def test_medium_k_residual_large_block_k_excluded():
    assert not is_medium_k_residual(4096, 4096, 1024, block_k=128)


def test_medium_k_residual_amortization_floor_excludes_small_mainloop_iters():
    """K-654 K-539 cohort: K=512, BK=64 → 8 iters < 16 → reject."""
    # In-band by K (256<=K<=2048), in-band by M/N (>=1024), block_k<=64,
    # but mainloop_iters=8 below the amortization floor (=16).
    assert not is_medium_k_residual(2048, 2048, 512, block_k=64)
    assert not is_medium_k_residual(1024, 1024, 512, block_k=64)
    # K=256 with BK=32 → 8 iters → reject under new floor.
    assert not is_medium_k_residual(2048, 2048, 256, block_k=32)


def test_medium_k_residual_tile_count_upper_bound_excludes_large_grids():
    """K-646 PRD-guard losers: tiles > 128 → reject (L2 working-set spill)."""
    # 4096x4096 at BM=BN=128 → 1024 tiles → reject regardless of K.
    assert not is_medium_k_residual(
        4096, 4096, 1024, block_k=64, block_m=128, block_n=128
    )
    # 8192x4096 at BM=BN=256 → 32*16=512 tiles → reject.
    assert not is_medium_k_residual(
        8192, 4096, 2048, block_k=64, block_m=256, block_n=256
    )


def test_medium_k_residual_admits_k580_winner_envelope():
    """K-580 winner cohort: K/BK=32 AND tiles=128 → admit."""
    # All three K-580 winner shapes at the autotuner-selected BM=BN=256, BK=64.
    for M, N, K in [(4096, 2048, 2048), (8192, 1024, 2048), (2048, 4096, 2048)]:
        assert is_medium_k_residual(
            M, N, K, block_k=64, block_m=256, block_n=256
        ), f"K-580 winner {M}x{N}x{K} must remain admitted"


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


def test_on_mode_returns_swizzled_only_inside_envelope(monkeypatch):
    """K-683: mode=on respects the envelope (K-646/K-654 lessons).

    Pre-K-683, mode=on returned SWIZZLED unconditionally — that contract
    landed the K-580 PR despite K-646 PRD-guard regressions and K-654
    K-539-cohort regressions (all triggered when downstream consumers
    forced mode=on, bypassing the gate). Mode=on must now match the
    autotuned `auto`-mode envelope.
    """
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "on")
    # In-envelope K-580 winner shape → swizzled.
    assert _select(M=4096, N=2048, K=2048, block_m=256, block_n=256, block_k=64) \
        == SWIZZLED_CONFIG
    # Out-of-envelope K-539 small-K cohort → baseline (was buggy SWIZZLED pre-K-683).
    assert _select(M=2048, N=2048, K=512, block_m=256, block_n=256, block_k=64) \
        == BASELINE_CONFIG
    # Out-of-envelope K-646 large-square loser → baseline.
    assert _select(M=4096, N=4096, K=4096, block_m=128, block_n=128, block_k=64) \
        == BASELINE_CONFIG


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

    # Use an in-envelope K-580 winner shape (BM=BN=256, BK=64, K=2048) so the
    # autotuner is allowed to consider the swizzled candidate at all.
    cfg = select_lds_config(
        4096, 2048, 2048,
        "f16", "f16", "f16",
        256, 256, 64,
        streamk=False, work_stealing=False,
        autotune_fn=timing,
    )
    assert cfg == SWIZZLED_CONFIG

    # Cache file written.
    saved = json.loads(cache_path.read_text())
    assert any(v.get("kpack") == 2 for v in saved.values())


def test_autotune_skips_swizzled_outside_envelope(monkeypatch, tmp_path):
    """K-683: out-of-envelope shapes must never time-and-cache kpack=2.

    The K-654 root cause was an autotune cache poisoned with kpack=2
    entries on shapes outside the K-646 envelope; this test pins the new
    contract that autotune restricts its candidate set to BASELINE for
    such shapes.
    """
    cache_path = tmp_path / "auto_outside.json"
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE_CACHE", str(cache_path))
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "autotune")
    reset_cache_for_testing()

    # Even if the synthetic timing claims SWIZZLED is faster, autotune
    # must not pick or persist it for an out-of-envelope shape.
    def timing(cfg: LDSSwizzleConfig) -> float:
        return 1.0 if cfg == SWIZZLED_CONFIG else 2.0

    # K-539 cohort: K=512 → mainloop_iters=8 < amortization floor.
    cfg = select_lds_config(
        2048, 2048, 512,
        "f16", "f16", "f16",
        256, 256, 64,
        autotune_fn=timing,
    )
    assert cfg == BASELINE_CONFIG
    saved = json.loads(cache_path.read_text())
    # The persisted entry, if any, must be kpack=1.
    assert all(v.get("kpack") == 1 for v in saved.values())


def test_cached_kpack2_dropped_for_out_of_envelope_shape(monkeypatch, tmp_path):
    """K-683: stale kpack=2 cache entries on out-of-envelope shapes are
    dropped to baseline at lookup time, not served.

    This protects against silent regressions when the envelope tightens
    between releases (the K-654 mechanism).
    """
    cache_path = tmp_path / "stale.json"
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE_CACHE", str(cache_path))
    reset_cache_for_testing()

    # Manually pre-populate the cache as if a previous (looser-gate) release
    # had timed kpack=2 as the winner on a now-out-of-envelope shape.
    cache = PersistentSwizzleCache(path=cache_path)
    key = (
        "2048x2048x512|f16|f16|f16|"
        "256x256x64|ds|nows"
    )
    cache.set(key, SWIZZLED_CONFIG)
    reset_cache_for_testing()

    cfg = select_lds_config(
        2048, 2048, 512,
        "f16", "f16", "f16",
        256, 256, 64,
        streamk=False, work_stealing=False,
    )
    assert cfg == BASELINE_CONFIG, (
        "stale kpack=2 cache entry on K-539 cohort shape must drop to baseline"
    )


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
