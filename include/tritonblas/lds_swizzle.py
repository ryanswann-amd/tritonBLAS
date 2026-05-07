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
# wait dominates the bank-conflict reduction (cluster rocprof confirmed: see
# project_context lessons "K-646 iter-2/3"). At BK=64 this admits K up to 2048,
# matching the original design envelope's upper bound.
KPACK2_MAINLOOP_ITERS_MAX = 32

# Lower bound on mainloop iterations needed to amortize the per-launch fixed
# overhead of the wider LDS path (autotune cache miss + kpack=2 codegen prologue).
# Below this the small-K cohort regresses (see K-654 sweep: K=512 cohort regressed
# −27..−28% under unconditional kpack=2). At BK=64 this rejects K < 1024.
KPACK2_MAINLOOP_ITERS_MIN = 16

# Upper bound on grid tile-count beyond which kpack=2's L2 working-set spill
# becomes the secondary regression mechanism (K-646 iter-2: TCC miss frac
# +36..+62% on the two largest PRD-guard shapes).
KPACK2_TILES_MAX = 128


def is_medium_k_residual(
    M: int,
    N: int,
    K: int,
    block_k: Optional[int] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
) -> bool:
    """Return True if the shape sits in the medium-K residual band.

    Two-sided gate calibrated against three disjoint shape sources (K-580
    winners, K-646 PRD-guard losers, K-654 K-539 small-K losers):

    * **Lower K bound (K-654)**: ``mainloop_iters = ceil(K/BK) >= 16`` —
      reject the K-539 small-K cohort (M,N ∈ {512,1024,2048}, K=512) where
      kpack=2 codegen + cache-lookup overhead dominates over the (small)
      bank-conflict win. Empirically observed −27..−28% regression vs the
      kpack=1 baseline before this guard.
    * **Upper K bound (K-646)**: ``mainloop_iters <= 32`` — reject large-K
      shapes whose per-LDS-issue dependency chain starves the MFMA pipeline
      (rocprof: ``per_lds_wait`` +54..+96%, ``mfma_active_frac`` −10..−18%
      at kpack=2 with ``SQ_LDS_BANK_CONFLICT == 0`` at the baseline → pure
      overhead, no offsetting benefit).
    * **Tile-count upper bound (K-646)**: ``(M//BM) * (N//BN) <= 128`` —
      reject high-occupancy launches whose L2 working-set spills under the
      wider LDS path (TCC miss frac +36..+62% on 8K/16K-square shapes).
    * **Original lower bound (K-550/K-552)**: ``M, N >= 1024`` and
      ``block_k <= 64`` are kept to exclude launch-bound tail shapes and
      naturally-wide-LDS tiles respectively.

    All four conditions must hold. Block dims default to None for legacy
    callers; when unknown the tile-count check is skipped (consult the
    cache-side autotune as the load-bearing safety in that path).
    """
    if not (256 <= K <= 2048):
        return False
    if M < 1024 or N < 1024:
        return False
    if block_k is not None and block_k > 64:
        # Larger block_k tiles already have favorable LDS access widths.
        return False

    # K-654 amortization-floor guard (lower bound on mainloop iters).
    if block_k is not None:
        mainloop_iters = (K + block_k - 1) // block_k
        if mainloop_iters < KPACK2_MAINLOOP_ITERS_MIN:
            return False
        if mainloop_iters > KPACK2_MAINLOOP_ITERS_MAX:
            return False

    # K-646 tile-count upper bound — only enforceable when block dims are known.
    if block_m is not None and block_n is not None and block_m > 0 and block_n > 0:
        tiles = ((M + block_m - 1) // block_m) * ((N + block_n - 1) // block_n)
        if tiles > KPACK2_TILES_MAX:
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
    """Used by unit tests to drop the singleton without touching the JSON file."""
    global _CACHE_INSTANCE
    with _CACHE_LOCK:
        _CACHE_INSTANCE = None


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

    # K-683: structural guard — kpack=2 may only be returned for shapes that
    # pass the calibrated medium-K residual gate. This is enforced **regardless
    # of mode or cache state** because (per K-646 iter-3) the K-580 PR landed
    # despite passing both Criterion-A (winners) and Criterion-B (PRD guards)
    # yet still regressed 5 large-square shapes under forced mode=on, and
    # (per K-654) regressed 53/61 shapes including the K-539 small-K cohort
    # by −27..−28%. The fix must live at the codegen layer, not behind the
    # mode flag, because downstream consumers do force mode=on. The
    # block-aware predicate is only enforceable when block dims are known;
    # all in-tree call sites supply them.
    in_envelope = is_medium_k_residual(
        M, N, K, block_k=block_k, block_m=block_m, block_n=block_n
    )

    if mode == "on":
        return SWIZZLED_CONFIG if in_envelope else BASELINE_CONFIG

    cache = get_cache()
    key = _cache_key(
        M, N, K, a_dtype, b_dtype, c_dtype,
        block_m, block_n, block_k, streamk, work_stealing,
    )

    # autotune mode: time both candidates and persist the winner. Restrict
    # the candidate set to the baseline outside the envelope so the cache
    # cannot be poisoned by a regressing kpack=2 entry on out-of-envelope
    # shapes (the K-654 root cause for the K-539 cohort).
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
            winner = timings[0][1]
            cache.set(key, winner)
            return winner

    # auto mode (and autotune fall-through): cache → baseline.
    # auto is intentionally safe — it never picks a non-baseline config without
    # an empirical timing in the cache. Cache hits are still gated by the
    # envelope: a stale kpack=2 entry on an out-of-envelope shape is dropped
    # to baseline rather than served, preventing K-654-style regressions when
    # cohort definitions tighten between releases.
    cached = cache.get(key)
    if cached is not None:
        if cached.kpack == BASELINE_CONFIG.kpack or in_envelope:
            return cached
        # Cached non-baseline config but shape is out of envelope → baseline.
        return BASELINE_CONFIG

    return BASELINE_CONFIG


__all__ = [
    "LDSSwizzleConfig",
    "BASELINE_CONFIG",
    "SWIZZLED_CONFIG",
    "PersistentSwizzleCache",
    "get_cache",
    "get_mode",
    "is_medium_k_residual",
    "select_lds_config",
    "reset_cache_for_testing",
]
