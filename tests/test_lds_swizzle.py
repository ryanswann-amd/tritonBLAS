# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the LDS bank-conflict mitigation knob.

These tests cover the pure-Python plumbing — env-var modes, the
:class:`KPack2Envelope` routing predicate, persistent cache round-trips
and autotune-callback wiring — without requiring a GPU. GPU-side
correctness is covered by the existing ``test_matmul_correctness`` suite
once the kernel call sites pick up the new config.

Cohort labels (``small_k``, ``benefit``, ``large_square``) refer to
empirical regression / benefit cohorts identified in the pre-merge
sweep; see the ``KPack2Envelope`` docstring for the calibration
rationale per bound.
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
    DEFAULT_KPACK2_ENVELOPE,
    KPack2Envelope,
    LDSSwizzleConfig,
    PersistentSwizzleCache,
    SWIZZLED_CONFIG,
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
# KPack2Envelope (medium-K residual gate)
# ──────────────────────────────────────────────────────────────────────────────


def test_envelope_default_is_singleton():
    """The module-level default envelope is shared across callers."""
    assert isinstance(DEFAULT_KPACK2_ENVELOPE, KPack2Envelope)
    # Default values match the published calibration.
    assert DEFAULT_KPACK2_ENVELOPE.k_min == 256
    assert DEFAULT_KPACK2_ENVELOPE.k_max == 2048
    assert DEFAULT_KPACK2_ENVELOPE.min_mn == 1024
    assert DEFAULT_KPACK2_ENVELOPE.max_block_k == 64
    assert DEFAULT_KPACK2_ENVELOPE.mainloop_iters_min == 16
    assert DEFAULT_KPACK2_ENVELOPE.mainloop_iters_max == 32
    assert DEFAULT_KPACK2_ENVELOPE.tiles_max == 128


def test_envelope_admits_in_band():
    # K=1024, BK=64 → mainloop_iters=16 (>= mainloop_iters_min). Tile check
    # skipped when block_m/n omitted, so the legacy 2-arg form still admits.
    assert is_medium_k_residual(4096, 4096, 1024, block_k=64)
    # K=2048, BK=64 → mainloop_iters=32 (== mainloop_iters_max, still admitted
    # since the upper bound is inclusive).
    assert is_medium_k_residual(8192, 8192, 2048, block_k=64)


def test_envelope_rejects_out_of_band_low_k():
    assert not is_medium_k_residual(4096, 4096, 128, block_k=64)


def test_envelope_rejects_out_of_band_high_k():
    assert not is_medium_k_residual(4096, 4096, 4096, block_k=64)


def test_envelope_rejects_small_mn():
    assert not is_medium_k_residual(512, 512, 1024, block_k=64)


def test_envelope_rejects_large_block_k():
    assert not is_medium_k_residual(4096, 4096, 1024, block_k=128)


def test_envelope_rejects_small_mainloop_iters():
    """Small-K cohort: K=512, BK=64 → 8 iters < 16 → reject.

    This protects shapes where the kpack=2 codegen prologue and
    cache-lookup cost are not amortized over enough MFMA iterations.
    """
    # In-band by K (256<=K<=2048), in-band by M/N (>=1024), block_k<=64,
    # but mainloop_iters=8 below the amortization floor (=16).
    assert not is_medium_k_residual(2048, 2048, 512, block_k=64)
    assert not is_medium_k_residual(1024, 1024, 512, block_k=64)
    # K=256 with BK=32 → 8 iters → reject under amortization floor.
    assert not is_medium_k_residual(2048, 2048, 256, block_k=32)


def test_envelope_rejects_large_grids():
    """Large-square cohort: tiles > 128 → reject (L2 working-set spill)."""
    # 4096x4096 at BM=BN=128 → 1024 tiles → reject regardless of K.
    assert not is_medium_k_residual(
        4096, 4096, 1024, block_k=64, block_m=128, block_n=128
    )
    # 8192x4096 at BM=BN=256 → 32*16=512 tiles → reject.
    assert not is_medium_k_residual(
        8192, 4096, 2048, block_k=64, block_m=256, block_n=256
    )


def test_envelope_admits_benefit_cohort():
    """Mainline benefit cohort: K/BK=32 AND tiles<=128 → admit."""
    # All three benefit-cohort shapes at the autotuner-selected BM=BN=256, BK=64.
    for M, N, K in [(4096, 2048, 2048), (8192, 1024, 2048), (2048, 4096, 2048)]:
        assert is_medium_k_residual(
            M, N, K, block_k=64, block_m=256, block_n=256
        ), f"benefit shape {M}x{N}x{K} must remain admitted"


def test_custom_envelope_overrides_default():
    """A caller-supplied envelope replaces the module default."""
    relaxed = KPack2Envelope(mainloop_iters_min=4, tiles_max=4096)
    # Shape that the default envelope rejects on the amortization floor.
    assert not is_medium_k_residual(2048, 2048, 512, block_k=64)
    # Same shape under a relaxed envelope is admitted.
    assert is_medium_k_residual(2048, 2048, 512, block_k=64, envelope=relaxed)


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
    """mode=on respects the envelope; out-of-envelope shapes get baseline.

    A pre-fix bug returned ``SWIZZLED`` unconditionally on ``mode=on``,
    bypassing the envelope. Downstream consumers that force ``mode=on``
    (e.g., microbenchmark harnesses) inherited this regression for any
    out-of-envelope shape. ``mode=on`` must now match the autotuned
    ``auto``-mode envelope so a single routing contract holds.
    """
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "on")
    # In-envelope benefit shape → swizzled.
    assert _select(M=4096, N=2048, K=2048, block_m=256, block_n=256, block_k=64) \
        == SWIZZLED_CONFIG
    # Out-of-envelope small-K cohort → baseline (was buggy SWIZZLED pre-fix).
    assert _select(M=2048, N=2048, K=512, block_m=256, block_n=256, block_k=64) \
        == BASELINE_CONFIG
    # Out-of-envelope large-square loser → baseline.
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
    """A cached entry takes precedence over the heuristic (in-envelope)."""
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
# Cache-hit gating (root-cause coverage)
# ──────────────────────────────────────────────────────────────────────────────


def test_cached_kpack2_dropped_for_out_of_envelope_shape_auto(monkeypatch, tmp_path):
    """Stale ``kpack=2`` cache entries on out-of-envelope shapes are
    dropped to baseline at lookup time, not served (auto mode).

    This protects against silent regressions when the envelope tightens
    between releases — the empirical regression mechanism for the
    small-K cohort.
    """
    cache_path = tmp_path / "stale_auto.json"
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
        "stale kpack=2 cache entry on small-K cohort shape must drop to baseline"
    )


def test_cached_kpack2_dropped_for_out_of_envelope_shape_on_mode(monkeypatch, tmp_path):
    """Same root-cause coverage but with ``mode=on`` — verifies the
    envelope check fires structurally on every dispatch (not just on the
    cold-miss / cache-empty path).

    Pre-fix, ``mode=on`` short-circuited to ``SWIZZLED_CONFIG`` without
    consulting the cache or the envelope. This test pins the new
    contract: even if a cached kpack=2 entry exists, the envelope is
    re-checked and rejects the cached choice for an out-of-envelope
    shape.
    """
    cache_path = tmp_path / "stale_on.json"
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE_CACHE", str(cache_path))
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "on")
    reset_cache_for_testing()

    # Prime the cache with SWIZZLED for a small-K shape that the envelope
    # rejects (mainloop_iters=8 at K=512/BK=64).
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
        "mode=on must re-check envelope on every dispatch including cache hits"
    )


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

    # Use an in-envelope benefit shape (BM=BN=256, BK=64, K=2048) so the
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
    """Out-of-envelope shapes must never time-and-cache ``kpack=2``.

    The empirical root cause for the small-K cohort regression was an
    autotune cache poisoned with ``kpack=2`` entries on shapes outside
    the envelope; this test pins the new contract that autotune
    restricts its candidate set to ``BASELINE`` for such shapes.
    """
    cache_path = tmp_path / "auto_outside.json"
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE_CACHE", str(cache_path))
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "autotune")
    reset_cache_for_testing()

    # Even if the synthetic timing claims SWIZZLED is faster, autotune
    # must not pick or persist it for an out-of-envelope shape.
    def timing(cfg: LDSSwizzleConfig) -> float:
        return 1.0 if cfg == SWIZZLED_CONFIG else 2.0

    # Small-K cohort: K=512 → mainloop_iters=8 < amortization floor.
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
