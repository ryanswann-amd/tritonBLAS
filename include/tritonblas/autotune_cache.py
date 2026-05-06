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
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


def _is_cache_enabled() -> bool:
    val = os.environ.get("TRITONBLAS_AUTOTUNE_CACHE", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


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
    ac = active_cu if active_cu is not None else n_cu
    return (
        f"{M}x{N}x{K}|{a_dtype_str}|{b_dtype_str}|{out_dtype_str}"
        f"|mx{mx_block_size}|sk{int(bool(streamk))}|ns{num_stages}"
        f"|cu{n_cu}|ac{ac}"
    )


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
        self._path = _cache_path(self.arch)
        self._version = _version_key()
        self._entries: Dict[str, Dict[str, Any]] = {}
        if self._enabled:
            self._load_from_disk()

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
