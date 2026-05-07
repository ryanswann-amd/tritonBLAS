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
    batched_skip,
    get_mode,
    is_medium_k_residual,
    reset_cache_for_testing,
    select_lds_config,
    small_k_guard,
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


# ──────────────────────────────────────────────────────────────────────────────
# K-707 sub-predicates: small_k_guard / batched_skip / medium_k_residual
# ──────────────────────────────────────────────────────────────────────────────


def test_small_k_guard_rejects_k539_cohort():
    """K-539 cohort (M, N ∈ {512, 1024, 2048}, K=512) — rejected by floor.

    The K-654 PRD-guard sweep showed this cohort regressing −27..−28%.
    The K-707 ``small_k_guard`` predicate enforces ``K <= 512`` OR
    ``M*N <= 2048*2048`` as a structural floor that fires regardless
    of whether ``block_k`` is supplied. This test pins the floor on
    every member of the cohort independently, so a future regression
    on any single shape is caught.
    """
    for M in (512, 1024, 2048):
        for N in (512, 1024, 2048):
            assert small_k_guard(M, N, 512), (
                f"K-539 cohort shape M={M} N={N} K=512 must be rejected by "
                f"small_k_guard (K<=512 floor)"
            )
            # Independent of block_k — the floor still fires when block_k is None.
            assert not is_medium_k_residual(M, N, 512), (
                f"K-539 cohort shape M={M} N={N} K=512 (no block_k) must NOT "
                f"be admitted to the envelope; got admit=True"
            )
            assert not is_medium_k_residual(M, N, 512, block_k=64), (
                f"K-539 cohort shape M={M} N={N} K=512 (block_k=64) must NOT "
                f"be admitted to the envelope; got admit=True"
            )


def test_small_k_guard_floors_independently_of_block_k():
    """K=512 floor is enforced even when block_k would yield admittance.

    Pre-fix, a caller that passed ``block_k=32`` for K=512 would yield
    ``mainloop_iters=16`` which slipped through the legacy
    amortization-floor check. ``small_k_guard``'s K-floor closes that
    loophole structurally.
    """
    assert small_k_guard(4096, 4096, 512)
    # Even with block_k=32 → mainloop_iters=16, the K-floor still rejects.
    assert not is_medium_k_residual(4096, 4096, 512, block_k=32)


def test_small_k_guard_rejects_small_problem_area():
    """``M*N <= 2048*2048`` floor rejects under-sized launches.

    Catches shapes that pass the per-axis ``min_mn`` check but whose
    total M*N is still launch-bound (kpack=2 mainloop savings cannot
    amortize the prologue/epilogue cost).
    """
    # M=N=2048 → M*N == 2048*2048 (boundary, inclusive reject).
    assert small_k_guard(2048, 2048, 1024)
    # M=1024, N=4096 → M*N = 4M = 2048² (boundary, inclusive reject).
    assert small_k_guard(1024, 4096, 1024)
    # Shrunk to area boundary even though K is well in band.
    assert not is_medium_k_residual(2048, 2048, 1024, block_k=64)


def test_small_k_guard_admits_k580_cohort():
    """K-580 medium-K benefit cohort must NOT be flagged by small_k_guard.

    All three benefit shapes have K=2048 (>512) AND M*N >> 2048*2048,
    so the small-K floor does not fire and the kpack=2 routing remains
    available.
    """
    for M, N, K in [(4096, 2048, 2048), (8192, 1024, 2048), (2048, 4096, 2048)]:
        assert not small_k_guard(M, N, K), (
            f"K-580 benefit shape M={M} N={N} K={K} must NOT trip the "
            f"small_k_guard (would erase the medium-K kpack=2 lift)"
        )


def test_batched_skip_rejects_large_grid():
    """``batched_skip`` rejects launches whose tile-count > tiles_max.

    Replays the K-580 PRD-cohort large-square loser shape: 8K x 8K at
    BM=BN=128 → 64*64 = 4096 tiles >> 128 → reject (TCC miss frac
    +36..+62% under wider LDS).
    """
    assert batched_skip(8192, 8192, block_m=128, block_n=128)
    assert batched_skip(4096, 4096, block_m=128, block_n=128)  # 1024 tiles


def test_batched_skip_admits_in_band_grid():
    """In-band benefit shapes have grid <= tiles_max → not skipped."""
    # 4096 x 2048 at BM=BN=256 → 16*8 = 128 tiles == tiles_max (admitted).
    assert not batched_skip(4096, 2048, block_m=256, block_n=256)
    # 2048 x 2048 at BM=BN=256 → 8*8 = 64 tiles.
    assert not batched_skip(2048, 2048, block_m=256, block_n=256)


def test_batched_skip_safe_default_when_block_unknown():
    """When tile dims are unknown (legacy callers) batched_skip defers.

    The cache-side / autotune-side gating provides the load-bearing
    safety in that path; ``batched_skip`` returning False here just
    means it has no opinion to contribute, NOT that the shape is
    admitted.
    """
    assert not batched_skip(8192, 8192)  # no block_m/block_n
    assert not batched_skip(8192, 8192, block_m=None, block_n=128)


def test_admits_decomposes_into_subpredicates():
    """``admits`` is exactly the conjunction of the named sub-predicates.

    This pins the contract that the god-function refactor is faithful
    — any future change to ``admits`` must keep the equivalence
    ``admits == NOT small_k_guard AND NOT batched_skip AND medium_k_residual``.
    """
    cases = [
        # (M, N, K, BK, BM, BN, expected)
        # K-539 cohort — small_k_guard rejects.
        (2048, 2048, 512, 64, 256, 256, False),
        # K-580 benefit shape — all three pass.
        (4096, 2048, 2048, 64, 256, 256, True),
        # Large-square loser — batched_skip rejects.
        (8192, 8192, 1024, 64, 128, 128, False),
        # Out-of-band high K — medium_k_residual rejects.
        (4096, 4096, 4096, 64, 128, 128, False),
    ]
    env = DEFAULT_KPACK2_ENVELOPE
    for M, N, K, BK, BM, BN, expected in cases:
        composed = (
            (not env.small_k_guard(M, N, K))
            and (not env.batched_skip(M, N, block_m=BM, block_n=BN))
            and env.medium_k_residual(M, N, K, block_k=BK)
        )
        assert composed == env.admits(M, N, K, block_k=BK, block_m=BM, block_n=BN), (
            f"admits() decomposition mismatch on M={M} N={N} K={K} "
            f"BK={BK} BM={BM} BN={BN}"
        )
        assert composed == expected, (
            f"composed predicate disagrees with expected on "
            f"M={M} N={N} K={K} BK={BK} BM={BM} BN={BN}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# K-707 cohort pinning: end-to-end select_lds_config decisions
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "M,N,K",
    [
        (M, N, 512)
        for M in (512, 1024, 2048)
        for N in (512, 1024, 2048)
    ],
)
@pytest.mark.parametrize("mode", ["off", "auto", "on"])
def test_k539_cohort_never_swizzles(monkeypatch, M, N, K, mode):
    """End-to-end: every K-539 cohort shape returns BASELINE on every mode.

    This is the load-bearing K-707 invariant: K<=512 OR M*N<=2048²
    must NEVER receive kpack=2 routing under ANY mode (off / auto /
    on), regardless of cache state. Pinned per-(shape, mode) so a
    regression on any single combination is caught individually.
    """
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", mode)
    cfg = select_lds_config(
        M, N, K,
        "f16", "f16", "f16",
        block_m=256, block_n=256, block_k=64,
        streamk=False, work_stealing=False,
    )
    assert cfg == BASELINE_CONFIG, (
        f"K-539 cohort shape M={M} N={N} K={K} on mode={mode} got "
        f"{cfg}; must be BASELINE_CONFIG (kpack=1)"
    )


@pytest.mark.parametrize(
    "M,N,K",
    [
        (4096, 2048, 2048),
        (8192, 1024, 2048),
        (2048, 4096, 2048),
    ],
)
def test_k580_medium_k_cohort_swizzles_on_mode_on(monkeypatch, M, N, K):
    """K-580 medium-K benefit cohort returns SWIZZLED on mode=on.

    Companion to ``test_k539_cohort_never_swizzles``: the K-580 lift
    is preserved (the K-707 floor does not over-reject the benefit
    cohort).
    """
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "on")
    cfg = select_lds_config(
        M, N, K,
        "f16", "f16", "f16",
        block_m=256, block_n=256, block_k=64,
        streamk=False, work_stealing=False,
    )
    assert cfg == SWIZZLED_CONFIG, (
        f"K-580 benefit shape M={M} N={N} K={K} on mode=on got "
        f"{cfg}; must be SWIZZLED_CONFIG (kpack=2)"
    )


def test_k539_floor_holds_against_poisoned_cache(monkeypatch, tmp_path):
    """Even a hand-poisoned ``kpack=2`` cache entry on a K-539 shape
    must be dropped to baseline at lookup time.

    Regression mechanism documented in K-654: a cache populated by an
    older / looser-gate release can carry a stale ``kpack=2`` entry on
    a now-out-of-envelope shape. The K-707 small_k_guard floor and the
    structural assertion in ``select_lds_config`` jointly guarantee
    the cached choice is rejected on every dispatch.
    """
    cache_path = tmp_path / "k539_poisoned.json"
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE_CACHE", str(cache_path))
    reset_cache_for_testing()

    cache = PersistentSwizzleCache(path=cache_path)
    for M in (512, 1024, 2048):
        for N in (512, 1024, 2048):
            key = (
                f"{M}x{N}x512|f16|f16|f16|"
                f"256x256x64|ds|nows"
            )
            cache.set(key, SWIZZLED_CONFIG)
    reset_cache_for_testing()

    for M in (512, 1024, 2048):
        for N in (512, 1024, 2048):
            cfg = select_lds_config(
                M, N, 512,
                "f16", "f16", "f16",
                256, 256, 64,
                streamk=False, work_stealing=False,
            )
            assert cfg == BASELINE_CONFIG, (
                f"K-539 cohort shape M={M} N={N} K=512 returned {cfg} from "
                f"poisoned cache; small_k_guard floor must drop to baseline"
            )




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


def test_cached_kpack2_preserved_for_in_envelope_shape_on_mode(monkeypatch, tmp_path):
    """Symmetric companion to ``..._dropped_for_out_of_envelope_shape_on_mode``.

    Pins the positive side of the gating contract: an in-envelope shape
    on ``mode=on`` must return ``SWIZZLED_CONFIG`` whether or not a cache
    entry exists. Together with the drop-case test this locks the
    two-sided behavior so regressions in either direction are caught.
    """
    cache_path = tmp_path / "preserved_on.json"
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE_CACHE", str(cache_path))
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "on")
    reset_cache_for_testing()

    # Mainline-benefit shape: K=2048, BK=64 → 32 iters; tiles = 8*8 = 64;
    # M=N=2048 ≥ min_mn — fully inside the envelope.
    cache = PersistentSwizzleCache(path=cache_path)
    key = (
        "2048x2048x2048|f16|f16|f16|"
        "256x256x64|ds|nows"
    )
    cache.set(key, SWIZZLED_CONFIG)
    reset_cache_for_testing()

    cfg = select_lds_config(
        2048, 2048, 2048,
        "f16", "f16", "f16",
        256, 256, 64,
        streamk=False, work_stealing=False,
    )
    assert cfg == SWIZZLED_CONFIG, (
        "mode=on must preserve kpack=2 routing for in-envelope shapes "
        "on every dispatch including warm cache hits"
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


def test_warm_dispatch_is_consistent_after_autotune_cache_write(monkeypatch, tmp_path):
    """Memoization invariants: warm dispatch returns the same config
    on repeated calls, and an autotune-driven cache mutation invalidates
    the memo so the next call reflects the newly persisted entry.
    """
    cache_path = tmp_path / "memo.json"
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE_CACHE", str(cache_path))
    reset_cache_for_testing()

    # Cold call (auto, empty cache) → baseline.
    args = dict(
        M=4096, N=2048, K=2048,
        a_dtype="f16", b_dtype="f16", c_dtype="f16",
        block_m=256, block_n=256, block_k=64,
    )
    first = select_lds_config(**args)
    second = select_lds_config(**args)
    assert first == BASELINE_CONFIG
    assert second == BASELINE_CONFIG

    # Run autotune (env switched mid-process). It must persist SWIZZLED
    # for this in-envelope shape and the next auto call must observe it,
    # proving the memo invalidates on cache mutation.
    monkeypatch.setenv("TRITONBLAS_LDS_SWIZZLE", "autotune")
    select_lds_config(autotune_fn=lambda c: 1.0 if c == SWIZZLED_CONFIG else 5.0, **args)

    monkeypatch.delenv("TRITONBLAS_LDS_SWIZZLE")  # back to default 'auto'
    after = select_lds_config(**args)
    assert after == SWIZZLED_CONFIG, (
        "memoized auto-mode decision must reflect the autotune-written cache entry"
    )


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
