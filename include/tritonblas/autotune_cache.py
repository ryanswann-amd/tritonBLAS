"""
Persistent on-disk autotune cache for tritonBLAS heuristic selection.

Eliminates cross-process / cross-run cold starts by serializing the best
selector parameters per (M, N, K, dtype, arch, mode) shape to a JSON file
and re-loading them in subsequent processes.

Disable with ``TRITONBLAS_AUTOTUNE_CACHE=0``; override directory with
``TRITONBLAS_AUTOTUNE_CACHE_DIR``.

Boundary-tolerance fall-back (K-591):
   After an exact-key miss we scan cached entries that share the same
   non-shape suffix (dtypes, mx_block_size, streamk, num_stages, n_cu) and
   whose (M', N', K') is within ``BOUNDARY_TOLERANCE`` (5%) on every axis,
   and return the closest by log-L1 distance.  K-588 root-caused that
   Origami picks the same tile within this band and that the cached tile
   is LDS-safe whenever the suffix matches.  Nearest-match results are NOT
   persisted under the requesting key — only true Origami-tuned entries
   populate the on-disk file, which prevents drift through chained
   interpolations.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Schema version — bump if the cached entry layout changes.
SCHEMA_VERSION = 1

# Per-axis relative tolerance for the boundary-tolerance lookup.  5% covers
# the typical M+/-{1..7} dynamic-batch jitter and the K-588 / K-510 held-out
# workload while staying inside the band where Origami's tile selection is
# stable (K-588 F10).  Module constant — no env-var knob (K-591 Minimalist).
BOUNDARY_TOLERANCE = 0.05

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


def _is_cache_enabled() -> bool:
    val = os.environ.get("TRITONBLAS_AUTOTUNE_CACHE", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def _cache_dir() -> Path:
    env = os.environ.get("TRITONBLAS_AUTOTUNE_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    rel = Path(_DEFAULT_REL_DIR)
    if rel.exists():
        return rel
    return Path(_USER_FALLBACK_DIR)


def _cache_path(arch: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in (arch or "unknown"))
    return _cache_dir() / f"{safe}.json"


def _git_commit_hash() -> Optional[str]:
    try:
        pkg_dir = Path(__file__).resolve().parent
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
    commit = _git_commit_hash()
    if commit:
        return f"git:{commit}"
    return f"pkg:{_package_version()}"


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
) -> str:
    """Stable string key for an autotune entry.

    The non-shape suffix (dtypes / mx_block_size / streamk / num_stages /
    n_cu) is what the boundary-tolerance lookup matches exactly, so the
    cached tile remains LDS-safe for the substituted shape (K-588 F9).

    A per-call sub-CU mask is intentionally NOT part of the key: on a given
    arch ``n_cu`` already pins the hardware, and including a separate
    ``active_cu`` field would silently split the suffix bucket so canonical
    entries (stored with the default full-CU value) miss every near-neighbor
    query supplying a smaller value — re-introducing the K-588 cold-miss.
    """
    return (
        f"{M}x{N}x{K}|"
        f"{a_dtype_str}|{b_dtype_str}|{out_dtype_str}"
        f"|mx{mx_block_size}|sk{int(bool(streamk))}|ns{num_stages}"
        f"|cu{n_cu}"
    )


def _split_key(key: str) -> Optional[Tuple[int, int, int, str]]:
    """Return ``(M, N, K, suffix)`` for a canonical key, else ``None``."""
    if not isinstance(key, str):
        return None
    head, sep, suffix = key.partition("|")
    if not sep:
        return None
    parts = head.split("x")
    if len(parts) != 3:
        return None
    try:
        m, n, k = int(parts[0]), int(parts[1]), int(parts[2])
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
        try:
            tmp.unlink()
        except OSError:
            pass


class PersistentAutotuneCache:
    """In-memory mirror of the on-disk autotune cache for one architecture."""

    def __init__(self, arch: str, n_cu: int) -> None:
        self.arch = arch or "unknown"
        self.n_cu = int(n_cu)
        self._lock = threading.Lock()
        self._enabled = _is_cache_enabled()
        self._path = _cache_path(self.arch)
        self._version = _version_key()
        self._entries: Dict[str, Dict[str, Any]] = {}
        if self._enabled:
            self._load_from_disk()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> Path:
        return self._path

    @property
    def version(self) -> str:
        return self._version

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
            return dict(entry) if entry is not None else None

    def lookup_nearest(
        self, key: str
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        """Return the closest cached entry to ``key`` within
        ``BOUNDARY_TOLERANCE`` per axis, or ``None``.

        Filtering: candidates must share the non-shape suffix exactly
        (preserves LDS-safety, K-588 F9) and every axis must satisfy
        ``|x - x'| / max(x, x') <= BOUNDARY_TOLERANCE``.  Ties are broken
        by log-L1 distance (scale-free, matches Origami's tile sensitivity)
        then lexicographic key order.
        """
        if not self._enabled:
            return None
        parsed = _split_key(key)
        if parsed is None:
            return None
        m, n, k, suffix = parsed
        tol = BOUNDARY_TOLERANCE

        best_key: Optional[str] = None
        best_score = math.inf
        with self._lock:
            for cand_key in self._entries:
                if cand_key == key:
                    continue
                cparsed = _split_key(cand_key)
                if cparsed is None or cparsed[3] != suffix:
                    continue
                cm, cn, ck, _ = cparsed
                if abs(cm - m) / max(cm, m) > tol:
                    continue
                if abs(cn - n) / max(cn, n) > tol:
                    continue
                if abs(ck - k) / max(ck, k) > tol:
                    continue
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
        for field in _REQUIRED_FIELDS:
            if field not in params:
                raise ValueError(
                    f"PersistentAutotuneCache: missing field '{field}' in params"
                )
        clean = {k: _coerce_jsonable(params[k]) for k in _REQUIRED_FIELDS}
        with self._lock:
            if self._entries.get(key) == clean:
                return
            self._entries[key] = clean
            self._flush_locked()

    def _load_from_disk(self) -> None:
        raw = _load_raw(self._path)
        if not isinstance(raw, dict):
            return
        if raw.get("schema_version") != SCHEMA_VERSION:
            return
        if raw.get("tritonblas_version") != self._version:
            return
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


def _coerce_jsonable(value: Any) -> Any:
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


_REGISTRY_LOCK = threading.Lock()
_REGISTRY: Dict[Tuple[str, int], PersistentAutotuneCache] = {}


def get_cache(arch: str, n_cu: int) -> PersistentAutotuneCache:
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
