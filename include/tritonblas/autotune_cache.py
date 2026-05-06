"""
Persistent on-disk autotune cache for tritonBLAS heuristic selection.

Eliminates cross-process / cross-run cold starts by serializing the best
selector parameters per (M, N, K, dtype, arch, mode) shape to a JSON file
and re-loading them in subsequent processes.

Cache layout (one JSON file per architecture):

    {
        "schema_version": <int>,
        "tritonblas_version": <str>,    # invalidation key
        "arch": "gfx942",
        "n_cu": 304,
        "entries": {
            "<key>": {
                "block_m": 256,
                "block_n": 256,
                "block_k": 64,
                "workgroup_mapping": 8,
                "xcc_workgroup_mapping": 8,
                "counters_per_xcd": 4,
                "grid": 304
            }, ...
        }
    }

The cache is invalidated (treated as empty) when the schema version or the
``tritonblas_version`` (commit hash if available, else package version)
changes.

Disable persistence by setting ``TRITONBLAS_AUTOTUNE_CACHE=0``.
Override directory with ``TRITONBLAS_AUTOTUNE_CACHE_DIR``.

Boundary-tolerance lookup (K-591 / K-588 follow-up):
----------------------------------------------------
The cache key encodes the exact ``(M, N, K)`` shape, which means a single
+/- 1 jitter on any dimension is a full miss even when the cached entry is
an obvious near-neighbor (e.g. dynamic-batch padding around 4096).  K-588
root-caused this and showed (F9, F10) that within a relative tolerance of
~5-10% the analytical Origami selector picks the same tile/group/grid for
~80%+ of jittered shapes; LDS-capacity safety is inherited because it
depends only on the tile + dtype + num_stages, not on the shape.

After an exact-key miss we therefore perform a *nearest-neighbor* fall-back:
candidate cached entries that match the non-shape suffix of the key exactly
(dtype-a, dtype-b, dtype-c, mx_block_size, streamk, num_stages, n_cu) and
whose (M', N', K') is within a configurable per-axis relative tolerance of
(M, N, K) are scored by log-L1 distance, and the closest is returned.  The
default tolerance is 0.05 (5%) per axis; override with
``TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE`` (set to ``0`` to disable).

Nearest-match entries are **not** persisted under the requesting key.  Only
the original Origami-tuned configs (the ``store()`` calls in the cold path)
ever populate the on-disk cache, which keeps the cache from drifting:
without this guard a chain of near-neighbors A -> A' -> A'' could each be
within tolerance of their predecessor while A'' lies arbitrarily far from
the actually-tuned canonical entry.  See the K-591 reviewer feedback
(devil's-advocate concern about self-reinforcing cache drift).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

# Schema version — bump if the cached entry layout changes.
SCHEMA_VERSION = 1

# Default location.  ``state/tritonblas_autotune`` (relative to cwd) is the
# canonical location requested by the task spec; we fall back to the user
# cache dir if that location is unwritable or unset.
_DEFAULT_REL_DIR = "state/tritonblas_autotune"
_USER_FALLBACK_DIR = os.path.expanduser("~/.cache/tritonblas/autotune")

_REQUIRED_FIELDS = (
    "block_m",
    "block_n",
    "block_k",
    "workgroup_mapping",
    "xcc_workgroup_mapping",
    "counters_per_xcd",
    "grid",
)

# Default per-axis relative tolerance for the boundary-tolerance lookup.
# 5% covers the typical M+/-{1..7} dynamic-batch jitter and the K-588/K-510
# held-out workload while staying conservative enough that Origami's tile
# selection rarely changes inside the band (K-588 F10).  Override via
# ``TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE`` (set to "0" / "off" to disable).
_DEFAULT_TOLERANCE = 0.05

# Hard ceiling on the configured tolerance.  Values >50% would let the cache
# return shapes nowhere near the request and is almost certainly a typo.
_MAX_TOLERANCE = 0.50


def _is_cache_enabled() -> bool:
    val = os.environ.get("TRITONBLAS_AUTOTUNE_CACHE", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def _configured_tolerance() -> float:
    """Resolve the per-axis relative tolerance for boundary-tolerance lookups.

    ``TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE`` overrides the default; values
    that fail to parse, are negative, or look like an explicit "off" toggle
    yield ``0.0`` which disables the fall-back and preserves K-536's
    exact-match semantics.
    """
    raw = os.environ.get("TRITONBLAS_AUTOTUNE_CACHE_TOLERANCE")
    if raw is None:
        return _DEFAULT_TOLERANCE
    s = raw.strip().lower()
    if s in ("", "off", "false", "no", "none"):
        return 0.0
    try:
        v = float(s)
    except ValueError:
        return _DEFAULT_TOLERANCE
    if not math.isfinite(v) or v <= 0.0:
        return 0.0
    if v > _MAX_TOLERANCE:
        return _MAX_TOLERANCE
    return v


def _cache_dir() -> Path:
    """Resolve the cache directory, preferring an explicit override."""
    env = os.environ.get("TRITONBLAS_AUTOTUNE_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    # Prefer the canonical relative dir if it already exists in cwd
    rel = Path(_DEFAULT_REL_DIR)
    if rel.exists():
        return rel
    return Path(_USER_FALLBACK_DIR)


def _cache_path(arch: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in (arch or "unknown"))
    return _cache_dir() / f"{safe}.json"


def _git_commit_hash() -> Optional[str]:
    """Return the tritonblas package's git HEAD sha if available, else None."""
    try:
        pkg_dir = Path(__file__).resolve().parent
        # Walk upward looking for a .git directory
        for parent in [pkg_dir, *pkg_dir.parents]:
            if (parent / ".git").exists():
                out = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(parent),
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
                return out.decode().strip() or None
    except Exception:
        pass
    return None


def _package_version() -> str:
    try:
        from importlib.metadata import version  # type: ignore

        return version("tritonblas")
    except Exception:
        return "0.0.0"


def _version_key() -> str:
    """Combined invalidation key: prefer git commit, fall back to package version."""
    commit = _git_commit_hash()
    if commit:
        return f"git:{commit}"
    return f"pkg:{_package_version()}"


def make_cache_suffix(
    a_dtype_str: str,
    b_dtype_str: str,
    out_dtype_str: str,
    mx_block_size: int,
    streamk: bool,
    num_stages: int,
    n_cu: int,
    active_cu: Optional[int],  # accepted for API symmetry; intentionally
    # NOT included in the suffix (see note below).
) -> str:
    """Return the non-shape ("suffix") portion of a cache key.

    Two cache keys with identical suffixes share the same dtype / mode /
    architecture context and only differ in their (M, N, K) shape.  The
    boundary-tolerance lookup uses this to restrict the candidate set to
    entries that are LDS-safe (K-588 F9) and grid-compatible without having
    to re-run any heuristic.

    NOTE on ``active_cu``: a per-call sub-CU mask (e.g. running on a
    partition of the device) does not change the LDS-safety of a cached
    tile or the dtype/streamk/num_stages context, and on a given
    architecture ``n_cu`` already pins the hardware identity.  We deliberately
    omit ``active_cu`` from the suffix so that an entry stored with
    ``active_cu=None`` (i.e. ``n_cu``) is reusable for a near-neighbor query
    that might supply an explicit ``active_cu`` value — otherwise we would
    silently re-introduce the exact "near-miss -> 0% hit" failure mode that
    K-588 root-caused.  ``active_cu`` is still threaded through the public
    API so we can re-introduce it (or fold it into ``n_cu``) without an API
    break if a future kernel is shown to be sensitive to it.
    """
    _ = active_cu  # consumed but intentionally not part of the key
    return (
        f"{a_dtype_str}|{b_dtype_str}|{out_dtype_str}"
        f"|mx{mx_block_size}|sk{int(bool(streamk))}|ns{num_stages}"
        f"|cu{n_cu}"
    )


def make_cache_key(
    M: int,
    N: int,
    K: int,
    a_dtype_str: str,
    b_dtype_str: str,
    out_dtype_str: str,
    mx_block_size: int,
    streamk: bool,
    num_stages: int,
    n_cu: int,
    active_cu: Optional[int],
) -> str:
    """Stable string key for an autotune entry."""
    suffix = make_cache_suffix(
        a_dtype_str,
        b_dtype_str,
        out_dtype_str,
        mx_block_size,
        streamk,
        num_stages,
        n_cu,
        active_cu,
    )
    return f"{M}x{N}x{K}|{suffix}"


def parse_cache_key(key: str) -> Optional[Tuple[int, int, int, str]]:
    """Inverse of ``make_cache_key``.

    Returns ``(M, N, K, suffix)`` if ``key`` matches the canonical layout,
    or ``None`` for any malformed / non-numeric prefix.  This is used by the
    boundary-tolerance lookup to filter candidates without re-keying the
    on-disk file.
    """
    if not isinstance(key, str):
        return None
    head, sep, suffix = key.partition("|")
    if not sep:
        return None
    parts = head.split("x")
    if len(parts) != 3:
        return None
    try:
        m = int(parts[0])
        n = int(parts[1])
        k = int(parts[2])
    except ValueError:
        return None
    if m <= 0 or n <= 0 or k <= 0:
        return None
    return m, n, k, suffix


def _load_raw(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with tmp.open("w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        # Best-effort: if we can't write the cache, silently drop the update.
        try:
            tmp.unlink()
        except OSError:
            pass


class PersistentAutotuneCache:
    """In-memory mirror of the on-disk autotune cache for one architecture.

    Multiple instances may target the same architecture (e.g. across processes);
    the on-disk file is read once at construction.  Writes are performed
    eagerly (atomic ``os.replace``) on each ``store`` so that a fresh process
    started immediately afterwards observes the new entry.
    """

    def __init__(self, arch: str, n_cu: int) -> None:
        self.arch = arch or "unknown"
        self.n_cu = int(n_cu)
        self._lock = threading.Lock()
        self._enabled = _is_cache_enabled()
        self._tolerance = _configured_tolerance() if self._enabled else 0.0
        self._path = _cache_path(self.arch)
        self._version = _version_key()
        self._entries: Dict[str, Dict[str, Any]] = {}
        # Index from non-shape suffix -> list of (M, N, K, key) tuples for
        # constant-time narrowing during nearest-neighbor lookup.  Rebuilt
        # whenever ``_entries`` is mutated.
        self._suffix_index: Dict[str, list] = {}
        if self._enabled:
            self._load_from_disk()
            self._rebuild_suffix_index_locked()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> Path:
        return self._path

    @property
    def version(self) -> str:
        return self._version

    @property
    def tolerance(self) -> float:
        """Per-axis relative tolerance used by ``lookup_nearest``."""
        return self._tolerance

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def lookup(self, key: str) -> Optional[Dict[str, Any]]:
        if not self._enabled:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            # Defensive copy (callers must not mutate)
            return dict(entry)

    def lookup_nearest(
        self,
        key: str,
        tolerance: Optional[float] = None,
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        """Find the nearest cached entry to ``key`` within ``tolerance``.

        Two filters are applied in order:

        1. The candidate's *suffix* (the non-shape part of the key — dtypes,
           streamk, num_stages, n_cu, active_cu) must match exactly.  This
           preserves K-588 F9's LDS-safety guarantee: the cached tile was
           validated for the same dtype/num_stages and is therefore valid
           for any (M', N', K') we substitute in.
        2. Each shape axis must be within the relative tolerance:
           ``|M - M'| / max(M, M') <= tolerance``, and likewise for N and K.

        Among the surviving candidates, the one minimizing the log-L1
        distance ``|log(M/M')| + |log(N/N')| + |log(K/K')|`` is returned.
        The metric is scale-free, monotone in proportional jitter, and
        matches Origami's tile-selection sensitivity (K-588 F8).  Ties are
        broken deterministically by lexicographic key order.

        Returns ``(params, matched_key)`` on hit, ``None`` on miss / when
        tolerance is 0 / when the key is malformed.
        """
        if not self._enabled:
            return None
        tol = self._tolerance if tolerance is None else float(tolerance)
        if tol <= 0.0:
            return None
        parsed = parse_cache_key(key)
        if parsed is None:
            return None
        m, n, k, suffix = parsed

        best_key: Optional[str] = None
        best_score = math.inf
        with self._lock:
            bucket = self._suffix_index.get(suffix)
            if not bucket:
                return None
            for cm, cn, ck, cand_key in bucket:
                if cand_key == key:
                    # The exact key was already tried by lookup() — skip.
                    continue
                # Symmetric relative-error filter; safe for any positive M.
                if abs(cm - m) / max(cm, m) > tol:
                    continue
                if abs(cn - n) / max(cn, n) > tol:
                    continue
                if abs(ck - k) / max(ck, k) > tol:
                    continue
                # Log-L1 distance (scale-free).  log(a/b) handles every
                # positive ratio without divide-by-zero hazards.
                score = (
                    abs(math.log(cm / m))
                    + abs(math.log(cn / n))
                    + abs(math.log(ck / k))
                )
                if score < best_score or (
                    score == best_score
                    and (best_key is None or cand_key < best_key)
                ):
                    best_score = score
                    best_key = cand_key
            if best_key is None:
                return None
            return dict(self._entries[best_key]), best_key

    def store(self, key: str, params: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        # Validate schema before persisting to keep the file usable.
        for field in _REQUIRED_FIELDS:
            if field not in params:
                raise ValueError(f"PersistentAutotuneCache: missing field '{field}' in params")
        clean = {k: _coerce_jsonable(params[k]) for k in _REQUIRED_FIELDS}
        with self._lock:
            existing = self._entries.get(key)
            if existing == clean:
                return  # no-op
            self._entries[key] = clean
            self._add_to_suffix_index_locked(key)
            self._flush_locked()

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        raw = _load_raw(self._path)
        if not isinstance(raw, dict):
            return
        if raw.get("schema_version") != SCHEMA_VERSION:
            return
        if raw.get("tritonblas_version") != self._version:
            return
        # Architecture mismatch is a hard miss.
        if raw.get("arch") and raw.get("arch") != self.arch:
            return
        entries = raw.get("entries", {})
        if not isinstance(entries, dict):
            return
        sane: Dict[str, Dict[str, Any]] = {}
        for k, v in entries.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                continue
            if not all(field in v for field in _REQUIRED_FIELDS):
                continue
            sane[k] = {field: v[field] for field in _REQUIRED_FIELDS}
        self._entries = sane

    def _flush_locked(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "tritonblas_version": self._version,
            "arch": self.arch,
            "n_cu": self.n_cu,
            "entries": self._entries,
        }
        _atomic_write(self._path, payload)

    def _rebuild_suffix_index_locked(self) -> None:
        """(Re)build the suffix -> [(M, N, K, key), ...] index from scratch."""
        index: Dict[str, list] = {}
        for k in self._entries.keys():
            parsed = parse_cache_key(k)
            if parsed is None:
                continue
            m, n, kk, suffix = parsed
            index.setdefault(suffix, []).append((m, n, kk, k))
        self._suffix_index = index

    def _add_to_suffix_index_locked(self, key: str) -> None:
        """Incrementally add (or refresh) a single entry's index slot."""
        parsed = parse_cache_key(key)
        if parsed is None:
            return
        m, n, kk, suffix = parsed
        bucket = self._suffix_index.setdefault(suffix, [])
        # Replace any pre-existing tuple for the same (M, N, K) shape so a
        # rewrite of the same key doesn't double-index.
        for i, entry in enumerate(bucket):
            if entry[3] == key:
                bucket[i] = (m, n, kk, key)
                return
        bucket.append((m, n, kk, key))


def _coerce_jsonable(value: Any) -> Any:
    """Convert numpy / torch scalars to plain Python ints / floats."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    return int(value)


# ----------------------------------------------------------------------
# Process-wide cache registry
# ----------------------------------------------------------------------

_REGISTRY_LOCK = threading.Lock()
_REGISTRY: Dict[Tuple[str, int], PersistentAutotuneCache] = {}


def get_cache(arch: str, n_cu: int) -> PersistentAutotuneCache:
    """Return the per-architecture cache, constructing it on first call."""
    key = (arch or "unknown", int(n_cu))
    with _REGISTRY_LOCK:
        cache = _REGISTRY.get(key)
        if cache is None:
            cache = PersistentAutotuneCache(*key)
            _REGISTRY[key] = cache
        return cache


def reset_registry() -> None:
    """Drop all in-process caches (test helper)."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
