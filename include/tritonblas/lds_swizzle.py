# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
LDS bank-conflict mitigation for medium-K GEMM residuals.

The Triton AMD backend stages A and B operand tiles through LDS (shared memory)
using a swizzled (xor-based) shared-memory encoding selected by the compiler.
The width of each LDS load and the K-packing factor (``kpack``) directly
control the bank-distribution pattern: ``kpack=2`` packs two k-step MFMA
operands into a single 128-bit LDS access, which redistributes addresses
across the 32 LDS banks and eliminates the 2-way bank conflicts that show up
on medium-K (256 <= K <= 2048) residual shapes whose tile_k <= 64.

This module exposes a single autotunable knob — ``LDSSwizzleConfig`` — and a
persistent JSON cache so the autotune cost is amortized across runs.

Modes (``TRITONBLAS_LDS_SWIZZLE`` env var):
    off       Always use the baseline (kpack=1) config — preserves prior behavior.
    auto      (default) Consult the persistent cache; on miss fall back to the
              baseline.  This is **safe** — auto can never regress: any shape
              not yet timed by the autotuner stays on the baseline kpack=1
              config that has been the historical default.
    on        Force the swizzled config for every shape.  Useful for
              microbenchmarks but can regress shapes outside the medium-K
              band, so do not use as a global default.
    autotune  Time-measure both candidates and persist the winner to the
              cache.  This is the recommended way to populate the cache for
              new shapes — the per-shape decision avoids the regressions
              ``on`` exhibits on, e.g., very large K.

The cache file lives at ``$TRITONBLAS_LDS_SWIZZLE_CACHE`` if set, else
``~/.cache/tritonblas/lds_swizzle.json``.

Each kernel call site (``persistent_matmul_lt`` and ``streamk_matmul_lt``)
calls :func:`select_lds_config` to retrieve the (kpack, num_warps) tuple to
forward to the Triton kernel launch — keeping the change isolated and
backwards-compatible.
"""

from __future__ import annotations

import functools
import json
import os
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Configuration knob
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LDSSwizzleConfig:
    """Compile-time knobs forwarded to the Triton kernel launch.

    These two parameters together determine the LDS bank-distribution
    pattern emitted by the AMD Triton backend:

    * ``kpack`` — number of k-step MFMA operands packed per LDS access.
      ``kpack=1`` is the historical baseline (one MFMA per LDS read).
      ``kpack=2`` is the swizzled / wide-load configuration that
      eliminates 2-way bank conflicts on medium-K residual tiles.
    * ``num_warps`` — workgroup width.  The default of 8 matches the
      legacy hardcoded value; the autotuner only varies ``kpack``.
    """

    kpack: int = 1
    num_warps: int = 8

    def as_kwargs(self) -> dict:
        return {"kpack": self.kpack, "num_warps": self.num_warps}


#: Baseline config — preserves prior behavior (kpack=1).
BASELINE_CONFIG = LDSSwizzleConfig(kpack=1, num_warps=8)

#: Swizzled config — wide LDS loads / xor-distributed banks (kpack=2).
SWIZZLED_CONFIG = LDSSwizzleConfig(kpack=2, num_warps=8)


# ──────────────────────────────────────────────────────────────────────────────
# Medium-K residual gate
# ──────────────────────────────────────────────────────────────────────────────


# Upper bound on K-iteration count beyond which kpack=2's longer per-LDS-issue
# wait dominates the bank-conflict reduction (cluster rocprof confirmed:
# per_lds_wait +54..+96%, mfma_active_frac -10..-18% with no offsetting bank
# conflict at the kpack=1 baseline). At BK=64 this admits K up to 2048,
# matching the original design envelope's upper bound.
KPACK2_MAINLOOP_ITERS_MAX = 32

# Lower bound on mainloop iterations needed to amortize the per-launch fixed
# overhead of the wider LDS path (autotune cache miss + kpack=2 codegen prologue).
# Below this the small-K cohort regresses (regression sweep: K=512 with
# M,N in {512,1024,2048} regressed -27..-28% under unconditional kpack=2).
# At BK=64 this rejects K < 1024.
KPACK2_MAINLOOP_ITERS_MIN = 16

# Upper bound on grid tile-count beyond which kpack=2's L2 working-set spill
# becomes the secondary regression mechanism (TCC miss frac +36..+62% on the
# two largest PRD-guard shapes at 8K/16K-square cohorts).
KPACK2_TILES_MAX = 128


def _amortizes_kpack2_overhead(K: int, block_k: int) -> bool:
    """Return True iff mainloop_iters = ceil(K/BK) lands in the amortization band.

    The small-K regression cohort (M,N in {512,1024,2048}, K=512, BK=64 -> 8 iters)
    regressed -27..-28% vs the kpack=1 baseline because the per-launch fixed
    cost of the kpack=2 codegen prologue + autotune-cache lookup dominates
    the (small) bank-conflict win when the mainloop runs <16 iters.

    The large-K regression cohort (4096-16384 square at BK=64 -> 64-256 iters)
    regressed -2.9..-11% vs the kpack=1 baseline because the per-LDS-issue
    dependency chain starves the MFMA pipeline (per_lds_wait +54..+96%,
    mfma_active_frac -10..-18%) once the mainloop exceeds 32 iters.

    Boundaries are inclusive on both sides: K=1024 / BK=64 = 16 iters admits;
    K=2048 / BK=64 = 32 iters admits; K=512 / BK=64 = 8 iters and
    K=4096 / BK=64 = 64 iters reject.
    """
    if block_k <= 0:
        return False
    if block_k > 64:
        # Larger BK tiles already have favorable LDS access widths -- the
        # bank-conflict win kpack=2 chases doesn't exist at this BK, so the
        # overhead is pure cost.
        return False
    iters = (K + block_k - 1) // block_k
    return KPACK2_MAINLOOP_ITERS_MIN <= iters <= KPACK2_MAINLOOP_ITERS_MAX


def _fits_kpack2_working_set(M: int, N: int, block_m: int, block_n: int) -> bool:
    """Return True iff the launch's tile count keeps the L2 working set in budget.

    The PRD-guard cohort (8K/16K-square at BM=BN=256 -> 256-1024 tiles)
    regressed because the wider kpack=2 LDS path inflates the per-CU
    operand-tile working set and spills L2 (TCC miss frac +36..+62%). The
    original winner cohort sits at <=128 tiles; that boundary is the
    calibration.

    Also rejects launch-bound tail shapes (M < 1024 or N < 1024) where the
    grid is too small for the kpack=2 prologue overhead to amortize, even if
    the K dimension would otherwise pass _amortizes_kpack2_overhead().
    """
    if block_m <= 0 or block_n <= 0:
        return False
    if M < 1024 or N < 1024:
        return False
    tiles = ((M + block_m - 1) // block_m) * ((N + block_n - 1) // block_n)
    return tiles <= KPACK2_TILES_MAX


@functools.lru_cache(maxsize=4096)
def _kpack2_admissible(M: int, N: int, K: int,
                       block_m: int, block_n: int, block_k: int) -> bool:
    """Memoized envelope check -- composed from the two named predicates.

    Hot path: invoked from select_lds_config() on every kernel launch.
    The LRU cache keeps this O(1) per shape after first call (a few
    hundred ns of dict lookup vs ~33us kernel runtime on the small-K
    cohort, where launch-overhead Python tax was a real concern).
    """
    return (
        _amortizes_kpack2_overhead(K, block_k)
        and _fits_kpack2_working_set(M, N, block_m, block_n)
    )


def is_medium_k_residual(
    M: int,
    N: int,
    K: int,
    block_k: Optional[int] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
) -> bool:
    """Composite envelope predicate: K admissibility AND working-set fit.

    Public stable name retained for backwards compatibility with existing
    autotune call sites and unit tests. Internally this is the composition
    of :func:`_amortizes_kpack2_overhead` and :func:`_fits_kpack2_working_set`,
    routed through a memoizing wrapper for hot-path performance.

    Calibrated against three disjoint shape sources:

    * The original winner cohort (medium-K, in-envelope).
    * The PRD-guard regression cohort (large-K, large-tile-count).
    * The post-landing small-K regression cohort (low mainloop iters).

    Block dims are optional only for legacy callers that predate the
    composite gate (such callers fall back to the K-band approximation
    ``256 <= K <= 2048`` and skip the working-set check). All in-tree call
    sites (``select_lds_config``) pass full block dims and therefore get
    the strict composite check; legacy advisory callers are intentionally
    permissive because the load-bearing enforcement happens at
    :func:`select_lds_config`, not here.
    """
    # Strict path: full block dims supplied -> route through the composite gate.
    if block_k is not None and block_m is not None and block_n is not None:
        return _kpack2_admissible(M, N, K, block_m, block_n, block_k)

    # Legacy/advisory path: block dims missing -> approximate with the
    # original K-band shipped in the initial landing. select_lds_config
    # never takes this path, so the missing tile-count guard cannot leak
    # into codegen.
    if not (256 <= K <= 2048):
        return False
    if M < 1024 or N < 1024:
        return False
    if block_k is not None:
        if not _amortizes_kpack2_overhead(K, block_k):
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Persistent cache
# ──────────────────────────────────────────────────────────────────────────────


_CACHE_LOCK = threading.Lock()
_CACHE_INSTANCE: Optional["PersistentSwizzleCache"] = None


class PersistentSwizzleCache:
    """JSON-file backed cache mapping shape→best LDSSwizzleConfig.

    Keys are stable string tuples derived from the problem shape and dtypes
    (see :func:`_cache_key`).  The cache is read once on construction and
    written atomically on every update.
    """

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = Path(
                os.environ.get(
                    "TRITONBLAS_LDS_SWIZZLE_CACHE",
                    str(Path.home() / ".cache" / "tritonblas" / "lds_swizzle.json"),
                )
            )
        self._path = path
        self._data: dict = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        if self._path.exists():
            try:
                with self._path.open("r") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                # Corrupt or unreadable cache → start fresh; do not crash.
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            with tmp.open("w") as f:
                json.dump(self._data, f, indent=2, sort_keys=True)
            os.replace(tmp, self._path)
        except OSError:
            # Best-effort persistence — caller should still see the in-memory cache.
            pass

    def get(self, key: str) -> Optional[LDSSwizzleConfig]:
        entry = self._data.get(key)
        if entry is None:
            return None
        try:
            return LDSSwizzleConfig(
                kpack=int(entry["kpack"]),
                num_warps=int(entry.get("num_warps", 8)),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def set(self, key: str, cfg: LDSSwizzleConfig) -> None:
        self._data[key] = asdict(cfg)
        self._save()

    def clear(self) -> None:
        self._data = {}
        self._save()


def get_cache() -> PersistentSwizzleCache:
    """Process-wide singleton accessor for the persistent cache."""
    global _CACHE_INSTANCE
    with _CACHE_LOCK:
        if _CACHE_INSTANCE is None:
            _CACHE_INSTANCE = PersistentSwizzleCache()
        return _CACHE_INSTANCE


def reset_cache_for_testing() -> None:
    """Used by unit tests to drop the singleton without touching the JSON file.

    Also evicts the in-process LRU cache that memoizes the envelope predicate
    -- some tests parametrize across many shapes and would otherwise see
    stale answers from a previous test's cohort.
    """
    global _CACHE_INSTANCE
    with _CACHE_LOCK:
        _CACHE_INSTANCE = None
    _kpack2_admissible.cache_clear()


# ──────────────────────────────────────────────────────────────────────────────
# Mode resolution
# ──────────────────────────────────────────────────────────────────────────────


_VALID_MODES = ("off", "auto", "on", "autotune")


def get_mode() -> str:
    """Resolve the current swizzle mode from the env var."""
    mode = os.environ.get("TRITONBLAS_LDS_SWIZZLE", "auto").strip().lower()
    if mode not in _VALID_MODES:
        return "auto"
    return mode


def _cache_key(
    M: int,
    N: int,
    K: int,
    a_dtype: str,
    b_dtype: str,
    c_dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    streamk: bool,
    work_stealing: bool,
) -> str:
    """Stable, human-readable cache key for one (shape, dtypes, tile, mode)."""
    sk = "sk" if streamk else "ds"
    ws = "ws" if work_stealing else "nows"
    return (
        f"{M}x{N}x{K}|{a_dtype}|{b_dtype}|{c_dtype}|"
        f"{block_m}x{block_n}x{block_k}|{sk}|{ws}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────


def select_lds_config(
    M: int,
    N: int,
    K: int,
    a_dtype: str,
    b_dtype: str,
    c_dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    streamk: bool = False,
    work_stealing: bool = False,
    autotune_fn: Optional[Callable[[LDSSwizzleConfig], float]] = None,
) -> LDSSwizzleConfig:
    """Resolve the LDS swizzle config for one kernel launch.

    The mode env var, the persistent cache and the medium-K residual gate
    are consulted in that order.  In ``autotune`` mode (or when forced via
    ``autotune_fn``), both candidates are timed and the winner persisted.

    Args:
        M, N, K: Problem dimensions.
        a_dtype, b_dtype, c_dtype: Dtype strings (e.g. "f16", "bf16", "f8").
        block_m, block_n, block_k: Selected macro-tile sizes.
        streamk: Whether the launch is StreamK (affects cache key).
        work_stealing: Whether the launch is work-stealing (affects cache key).
        autotune_fn: Optional callback that runs the kernel with a candidate
            config and returns elapsed time (ms). Required when mode is
            "autotune" — without it autotune falls back to the cached/auto
            decision.

    Returns:
        ``LDSSwizzleConfig`` to forward to the Triton kernel launch.
    """
    mode = get_mode()

    if mode == "off":
        return BASELINE_CONFIG

    # Single envelope decision -- O(1) after first call per shape via lru_cache.
    # All three downstream branches (mode=on, mode=autotune, cache hit) consult
    # this one boolean, replacing three separate envelope checks that had
    # drifted apart in earlier revisions.
    in_envelope = _kpack2_admissible(M, N, K, block_m, block_n, block_k)

    def _enforce_envelope(config: LDSSwizzleConfig) -> LDSSwizzleConfig:
        """Drop any non-baseline config to BASELINE when the shape is out of envelope.

        Load-bearing enforcement of the gate contract: kpack=2 is *never*
        returned for out-of-envelope shapes, regardless of how the caller
        arrived at the candidate (mode=on override, cache hit, or autotune
        winner). Three earlier regression cycles (static analysis, PRD-guard
        large-square cohort, post-landing small-K cohort) all traced
        regressions to "predicate present, predicate ignored" on one of
        these three paths -- factoring them into a single helper prevents
        the same bug class from recurring.
        """
        if config.kpack == BASELINE_CONFIG.kpack:
            return config
        return config if in_envelope else BASELINE_CONFIG

    if mode == "on":
        return _enforce_envelope(SWIZZLED_CONFIG)

    cache = get_cache()
    key = _cache_key(
        M, N, K, a_dtype, b_dtype, c_dtype,
        block_m, block_n, block_k, streamk, work_stealing,
    )

    # autotune mode: time both candidates and persist the winner. Restrict
    # the candidate set to the baseline outside the envelope so the cache
    # cannot be poisoned by a regressing kpack=2 entry on out-of-envelope
    # shapes (this was the small-K cohort's regression root cause).
    if mode == "autotune" and autotune_fn is not None:
        candidates = (BASELINE_CONFIG, SWIZZLED_CONFIG) if in_envelope else (BASELINE_CONFIG,)
        try:
            timings = [(autotune_fn(c), c) for c in candidates]
        except Exception:
            # Fall through to auto on autotune error to preserve correctness.
            mode = "auto"
            timings = None
        else:
            timings.sort(key=lambda t: t[0])
            winner = _enforce_envelope(timings[0][1])
            cache.set(key, winner)
            return winner

    # auto mode (and autotune fall-through): cache -> baseline.
    # auto is intentionally safe -- it never picks a non-baseline config without
    # an empirical timing in the cache. Cache hits are also routed through
    # _enforce_envelope so a stale kpack=2 entry on an out-of-envelope shape
    # is silently downgraded rather than served, preventing regressions when
    # cohort definitions tighten between releases.
    cached = cache.get(key)
    if cached is not None:
        return _enforce_envelope(cached)

    return BASELINE_CONFIG


__all__ = [
    "LDSSwizzleConfig",
    "BASELINE_CONFIG",
    "SWIZZLED_CONFIG",
    "PersistentSwizzleCache",
    "KPACK2_MAINLOOP_ITERS_MIN",
    "KPACK2_MAINLOOP_ITERS_MAX",
    "KPACK2_TILES_MAX",
    "_amortizes_kpack2_overhead",
    "_fits_kpack2_working_set",
    "get_cache",
    "get_mode",
    "is_medium_k_residual",
    "select_lds_config",
    "reset_cache_for_testing",
]
