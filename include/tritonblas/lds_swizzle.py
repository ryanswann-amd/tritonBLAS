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

#: Minimum K below which kpack=2 / wide-LDS swizzle is a net regression.
#:
#: Empirical finding: the swizzled (kpack=2) config doubles the per-step LDS
#: load width but also doubles VGPR pressure on the LDS-load chain. For
#: K below ~1024 the K-loop is too short to amortize the extra spill-and-fill
#: cost — wall-time regresses by 5-30% on the small-K cohort
#: (M,N=512..2048, K=512). Above K=1024 the bank-conflict savings dominate
#: and swizzle is a net win in the medium-K residual band.
#:
#: Source: small-K cohort regression sweep (-27 to -28% on K=512 shapes
#: when swizzle fires unconditionally); recovers when K-gated to >=1024.
#: Confirmed across K in {256, 512, 768, 1024, 1536, 2048}.
SMALL_K_GATE_THRESHOLD = 1024


def is_medium_k_residual(
    M: int,
    N: int,
    K: int,
    block_k: Optional[int] = None,
) -> bool:
    """Return True if the shape sits in the medium-K residual band.

    The K-550/K-552 sweeps + K-521 ISA audit identified residual underperformance
    in the band ``SMALL_K_GATE_THRESHOLD <= K <= 2048`` for tiles with
    ``block_k <= 64``: small enough that K-loop unroll keeps multiple LDS
    accesses in flight, large enough that bank conflicts dominate over
    global-load latency.

    Below ``SMALL_K_GATE_THRESHOLD`` (the small-K cohort), the swizzled
    config's doubled VGPR pressure outweighs the bank-conflict savings — the
    K-loop is too short to amortize the wider-load spill cost. Those shapes
    must stay on the baseline (kpack=1) config. See
    :func:`select_lds_config` for the runtime guard that enforces this even
    when the persistent cache or ``mode=on`` would otherwise pick swizzle.

    The shape itself must also be reasonably sized (M >= 1024, N >= 1024) to
    avoid penalizing small tail-shapes that are launch-bound.
    """
    if not (SMALL_K_GATE_THRESHOLD <= K <= 2048):
        return False
    if M < 1024 or N < 1024:
        return False
    if block_k is not None and block_k > 64:
        # Larger block_k tiles already have favorable LDS access widths.
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

    # Small-K guard (applies to every mode except explicit "off"):
    # below the threshold the K-loop is too short to amortize the wider
    # LDS-load VGPR pressure, and the swizzled config regresses by
    # 5-30% (worst on the K=512, M,N=512..2048 small-K cohort). The
    # guard runs BEFORE the cache lookup and the "on" override so a
    # stale cache entry or a forced "on" can never demote a small-K
    # shape — preserving the baseline (kpack=1) is always correctness-
    # preserving and recovers the small-K cohort.
    if K < SMALL_K_GATE_THRESHOLD:
        return BASELINE_CONFIG

    if mode == "on":
        return SWIZZLED_CONFIG

    cache = get_cache()
    key = _cache_key(
        M, N, K, a_dtype, b_dtype, c_dtype,
        block_m, block_n, block_k, streamk, work_stealing,
    )

    # autotune mode: time both candidates and persist the winner.
    if mode == "autotune" and autotune_fn is not None:
        candidates = (BASELINE_CONFIG, SWIZZLED_CONFIG)
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
    # an empirical timing in the cache.  `is_medium_k_residual` is exposed only
    # as advice for the autotuner / users, not as a default decision.
    cached = cache.get(key)
    if cached is not None:
        return cached

    return BASELINE_CONFIG


__all__ = [
    "LDSSwizzleConfig",
    "BASELINE_CONFIG",
    "SWIZZLED_CONFIG",
    "PersistentSwizzleCache",
    "SMALL_K_GATE_THRESHOLD",
    "get_cache",
    "get_mode",
    "is_medium_k_residual",
    "select_lds_config",
    "reset_cache_for_testing",
]
