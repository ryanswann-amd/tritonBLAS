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
    on        Force the swizzled config for shapes inside the
              :class:`KPack2Envelope` (regressing shapes still get the
              baseline). Useful for microbenchmarks; the envelope guarantees
              no out-of-band shape ever pays the kpack=2 penalty regardless
              of mode.
    autotune  Time-measure both candidates (only inside the envelope) and
              persist the winner. The envelope restricts the candidate set so
              the cache cannot be poisoned with a regressing kpack=2 entry on
              shapes that are dominated by overhead rather than bank conflicts.

The cache file lives at ``$TRITONBLAS_LDS_SWIZZLE_CACHE`` if set, else
``~/.cache/tritonblas/lds_swizzle.json``.

Each kernel call site (``persistent_matmul_lt`` and ``streamk_matmul_lt``)
calls :func:`select_lds_config` to retrieve the (kpack, num_warps) tuple to
forward to the Triton kernel launch — keeping the change isolated and
backwards-compatible.

Routing chokepoint
------------------
:func:`select_lds_config` is the single entry point that returns the config
to forward to the Triton launch. Every code path — mode resolution, cache
hit, cache miss, autotune — is funneled through one call to
:meth:`KPack2Envelope.admits` so the routing contract cannot drift between
call sites. Callers must NOT branch on the cached value themselves; the
envelope is enforced inside the lookup.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional


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
# kpack=2 routing envelope (single source of truth)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KPack2Envelope:
    """Calibrated bounds for when ``kpack=2`` is profitable.

    Each bound corresponds to a regression mechanism observed during
    pre-merge sweeps on MI300X / gfx942 (cluster rocprof + 61-shape
    PRD-guard sweep). The envelope is the **single source of truth**
    for whether a shape is allowed to receive ``kpack=2`` — every code
    path in :func:`select_lds_config` (mode resolution, cache hit,
    cache miss, autotune) calls :meth:`admits` rather than re-deriving
    the predicate.

    Bounds (defaults are calibrated for MI300X bf16/fp16):

    * ``k_min`` / ``k_max`` — outer K-axis admittance window. Below
      ``k_min`` the bank-conflict win is too small to matter; above
      ``k_max`` the per-LDS-issue dependency chain dominates.
    * ``min_mn`` — minimum M and N. Smaller launches are tail-shaped
      and launch-bound, so kpack changes are noise.
    * ``max_block_k`` — block_k ceiling. Tiles wider than this already
      have favorable LDS access widths; kpack=2 is redundant.
    * ``mainloop_iters_min`` — amortization floor on
      ``ceil(K / block_k)``. Below this the kpack=2 codegen prologue
      and cache-lookup cost are not amortized over enough MFMA
      iterations. (The default is calibrated empirically — a
      regression sweep on the small-K cohort observed −27..−28% on
      ``M, N ∈ {512, 1024, 2048}, K=512`` because at ``block_k=64``
      that cohort spends only 8 iterations in the mainloop and the
      fixed kpack=2 prologue/epilogue dominates. The amortization
      floor must be high enough that the bank-conflict savings on
      the *mainloop body* exceed those fixed overheads — empirically
      ``mainloop_iters >= 16`` is the smallest power-of-two round
      number that does so on this hardware. ``9`` would suffice to
      reject the worst-offending cohort but still admits cases like
      ``K=576/BK=64 → 9 iters`` which were observed flat-or-worse;
      ``16`` provides a safety margin without rejecting any winner
      shapes from the mainline benefit cohort, which are all at
      ``K=2048/BK=64 → 32 iters``.)
    * ``mainloop_iters_max`` — upper bound on mainloop iterations.
      Beyond this the per-LDS-issue wait dominates the bank-conflict
      reduction (rocprof: ``per_lds_wait`` +54..+96%,
      ``mfma_active_frac`` −10..−18% at ``kpack=2`` once
      ``SQ_LDS_BANK_CONFLICT == 0`` already holds at the baseline →
      pure overhead, no offsetting benefit).
    * ``tiles_max`` — upper bound on grid tile-count.
      ``(M/BM) * (N/BN) > tiles_max`` triggers L2 working-set spill
      under the wider LDS path (TCC miss frac +36..+62% on the
      8K/16K-square shapes in the PRD-guard cohort).
    """

    k_min: int = 256
    k_max: int = 2048
    min_mn: int = 1024
    max_block_k: int = 64
    mainloop_iters_min: int = 16
    mainloop_iters_max: int = 32
    tiles_max: int = 128

    def admits(
        self,
        M: int,
        N: int,
        K: int,
        block_k: Optional[int] = None,
        block_m: Optional[int] = None,
        block_n: Optional[int] = None,
    ) -> bool:
        """Return True if the shape is inside the envelope.

        All conditions must hold. Block dims default to ``None`` for
        legacy callers; when the tile dim is unknown that specific
        bound is conservatively skipped (the autotune-side cache
        gating is the load-bearing safety in that path).
        """
        if not (self.k_min <= K <= self.k_max):
            return False
        if M < self.min_mn or N < self.min_mn:
            return False
        if block_k is not None and block_k > self.max_block_k:
            return False

        if block_k is not None:
            mainloop_iters = (K + block_k - 1) // block_k
            if mainloop_iters < self.mainloop_iters_min:
                return False
            if mainloop_iters > self.mainloop_iters_max:
                return False

        if block_m is not None and block_n is not None and block_m > 0 and block_n > 0:
            tiles = ((M + block_m - 1) // block_m) * ((N + block_n - 1) // block_n)
            if tiles > self.tiles_max:
                return False

        return True


#: Process-wide singleton envelope. There is exactly one in-tree caller
#: (``select_lds_config``); other callers must consume the predicate
#: through :func:`is_medium_k_residual` rather than constructing their own
#: envelope so the routing contract stays centralized.
DEFAULT_KPACK2_ENVELOPE = KPack2Envelope()


def is_medium_k_residual(
    M: int,
    N: int,
    K: int,
    block_k: Optional[int] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
) -> bool:
    """Return True if (M, N, K, tile) is admitted to the kpack=2 envelope.

    Thin wrapper around the singleton :data:`DEFAULT_KPACK2_ENVELOPE` so
    callers and tests have a single, side-effect-free predicate to
    consult. There is intentionally no ``envelope=`` override — the
    routing contract is process-wide and shared by every dispatch.
    """
    return DEFAULT_KPACK2_ENVELOPE.admits(
        M, N, K, block_k=block_k, block_m=block_m, block_n=block_n
    )


# ──────────────────────────────────────────────────────────────────────────────
# Persistent cache
# ──────────────────────────────────────────────────────────────────────────────


_CACHE_LOCK = threading.Lock()
_CACHE_INSTANCE: Optional["PersistentSwizzleCache"] = None

# Bumped whenever the persistent cache is mutated (set/clear) or replaced;
# the per-shape memoization in :func:`select_lds_config` keys on this so
# warm dispatches are an O(1) dict lookup but newly-cached entries from
# autotune are picked up on the next call.
_CACHE_VERSION = 0


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
        _bump_cache_version()

    def clear(self) -> None:
        self._data = {}
        self._save()
        _bump_cache_version()


def _bump_cache_version() -> None:
    """Invalidate the per-shape decision memo. Called on every cache mutation."""
    global _CACHE_VERSION
    with _CACHE_LOCK:
        _CACHE_VERSION += 1
    _decide_memo.cache_clear()


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
    _bump_cache_version()


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
# Public entry point — single chokepoint for kpack routing
# ──────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=4096)
def _decide_memo(
    mode: str,
    cache_version: int,
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
) -> LDSSwizzleConfig:
    """Resolve the routing decision for the no-autotune-callback path.

    Pure function of (mode, cache snapshot version, shape, dtypes, tile,
    schedule). The cache_version arg is the invalidation hook —
    :func:`_bump_cache_version` clears the lru_cache whenever the
    persistent cache is mutated, so this memo stays consistent with
    the JSON-file backing store while keeping warm dispatches O(1).
    """
    if mode == "off":
        return BASELINE_CONFIG

    in_envelope = DEFAULT_KPACK2_ENVELOPE.admits(
        M, N, K, block_k=block_k, block_m=block_m, block_n=block_n
    )

    if mode == "on":
        return SWIZZLED_CONFIG if in_envelope else BASELINE_CONFIG

    # auto mode (and autotune-without-callback fall-through): cache → baseline.
    cache = get_cache()
    key = _cache_key(
        M, N, K, a_dtype, b_dtype, c_dtype,
        block_m, block_n, block_k, streamk, work_stealing,
    )
    cached = cache.get(key)
    if cached is not None:
        if cached.kpack == BASELINE_CONFIG.kpack or in_envelope:
            return cached
        # Cached non-baseline config but shape is out of envelope → baseline.
        return BASELINE_CONFIG
    return BASELINE_CONFIG


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

    Routes mode, cache, and autotune through one envelope check so the
    contract that "no out-of-envelope shape ever sees ``kpack=2``" is
    enforced structurally rather than by convention. Callers MUST funnel
    every dispatch through this function.

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

    # autotune mode runs the actual GPU timing callback and is intentionally
    # NOT memoized — every call is a real measurement. All other modes
    # (off / on / auto) are pure functions of (mode, cache version, shape,
    # tile) and go through the lru_cache for O(1) warm-dispatch lookup.
    if mode == "autotune" and autotune_fn is not None:
        in_envelope = DEFAULT_KPACK2_ENVELOPE.admits(
            M, N, K, block_k=block_k, block_m=block_m, block_n=block_n
        )
        # Restrict the candidate set to the baseline outside the envelope so
        # the cache cannot be poisoned by a regressing kpack=2 entry on an
        # out-of-envelope shape (the observed root cause for the small-K
        # cohort regression in pre-merge sweeps).
        candidates = (BASELINE_CONFIG, SWIZZLED_CONFIG) if in_envelope else (BASELINE_CONFIG,)
        try:
            timings = [(autotune_fn(c), c) for c in candidates]
        except Exception:
            # Fall through to auto on autotune error to preserve correctness.
            pass
        else:
            timings.sort(key=lambda t: t[0])
            winner = timings[0][1]
            cache = get_cache()
            key = _cache_key(
                M, N, K, a_dtype, b_dtype, c_dtype,
                block_m, block_n, block_k, streamk, work_stealing,
            )
            cache.set(key, winner)  # bumps _CACHE_VERSION → memo invalidated
            return winner
        # autotune callback raised → fall through to the memoized auto path.
        mode = "auto"

    return _decide_memo(
        mode, _CACHE_VERSION,
        M, N, K, a_dtype, b_dtype, c_dtype,
        block_m, block_n, block_k, streamk, work_stealing,
    )


__all__ = [
    "LDSSwizzleConfig",
    "BASELINE_CONFIG",
    "SWIZZLED_CONFIG",
    "KPack2Envelope",
    "DEFAULT_KPACK2_ENVELOPE",
    "PersistentSwizzleCache",
    "get_cache",
    "get_mode",
    "is_medium_k_residual",
    "select_lds_config",
    "reset_cache_for_testing",
]
