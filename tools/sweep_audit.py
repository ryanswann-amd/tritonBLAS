#!/usr/bin/env python3
"""
sweep_audit.py — audit tritonBLAS sweep workspaces for stale / zombie runs.

A long sweep over GEMM shapes (e.g. ``tile_sweep.py`` or ``sweep_grid.py``)
typically writes per-run logs and a final ``summary.csv`` (or
``manifest.yaml``) into a per-run workspace directory. When a sweep
crashes, gets killed by Slurm, or is forgotten by an operator, the
workspace is left behind and downstream tooling has no easy way to know
which runs are still live, which finished cleanly, and which are
duplicates of an already-complete run on the same shape.

This tool walks a workspace tree of sweep runs and classifies each
sub-directory into one of:

    ACTIVE        newest file mtime < 2 h
    STALE         mtime > 24 h **and** no manifest / summary present
    ZOMBIE        mtime > 72 h **and** no recent log activity
    DUPLICATE     shape config matches an already-complete run
    COMPLETE      manifest / summary present (treated as finished)

The tool is read-only: it prints a table and writes a CSV. It does not
delete anything. Operators can use the CSV to drive whatever cleanup
policy they want (e.g. ``rm -rf`` zombies, requeue stale runs, skip
duplicates in the next sweep).

Usage
-----
    python3 tools/sweep_audit.py --root /path/to/sweep_workspaces
    python3 tools/sweep_audit.py --root . --out audit.csv
    python3 tools/sweep_audit.py --root . --json   # machine-readable

Each sweep run directory is expected to optionally contain:

* ``config.json`` or ``run.json`` with at least ``M``, ``N``, ``K``
  (used for duplicate detection — tolerated absent)
* ``manifest.yaml`` / ``manifest.json`` / ``summary.csv`` (any of these
  marks the run COMPLETE)
* ``*.log`` / ``*.out`` / ``*.txt`` files (used as the "recent activity"
  signal for ZOMBIE classification)

The tool is conservative: a run is only marked DUPLICATE when *another*
run with the same ``(M, N, K, dtype)`` tuple is already COMPLETE.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

NOW = time.time()
TWO_HOURS = 2 * 3600
ONE_DAY = 24 * 3600
THREE_DAYS = 72 * 3600

MANIFEST_NAMES = (
    "manifest.yaml",
    "manifest.yml",
    "manifest.json",
    "summary.csv",
    "summary.json",
)
CONFIG_NAMES = ("config.json", "run.json", "shape.json")
LOG_GLOBS = ("*.log", "*.out", "*.txt")


def newest_mtime(path: Path) -> float:
    """Return the newest mtime of any file under ``path`` (recursively)."""
    newest = path.stat().st_mtime
    for sub in path.rglob("*"):
        try:
            m = sub.stat().st_mtime
        except OSError:
            continue
        if m > newest:
            newest = m
    return newest


def has_recent_log_activity(run: Path, threshold_s: float) -> bool:
    cutoff = NOW - threshold_s
    for pattern in LOG_GLOBS:
        for cand in run.rglob(pattern):
            try:
                if cand.stat().st_mtime > cutoff:
                    return True
            except OSError:
                pass
    return False


def find_manifest(run: Path) -> Optional[Path]:
    for name in MANIFEST_NAMES:
        p = run / name
        if p.exists():
            return p
    return None


def find_config(run: Path) -> Optional[Path]:
    for name in CONFIG_NAMES:
        p = run / name
        if p.exists():
            return p
    return None


def shape_key(cfg_path: Optional[Path]) -> Optional[tuple]:
    """Return a comparable shape tuple ``(M, N, K, dtype)`` or ``None``."""
    if cfg_path is None:
        return None
    try:
        data = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    keys_lower = {k.lower(): k for k in data.keys()}

    def get(name: str):
        return data.get(keys_lower.get(name, name))

    m, n, k = get("m"), get("n"), get("k")
    if m is None or n is None or k is None:
        return None
    dtype = get("dtype") or get("a_dtype") or "?"
    return (int(m), int(n), int(k), str(dtype))


def classify(run: Path) -> dict:
    manifest = find_manifest(run)
    cfg = find_config(run)
    mtime = newest_mtime(run)
    age_s = NOW - mtime
    age_h = round(age_s / 3600, 1)
    has_recent_log = has_recent_log_activity(run, THREE_DAYS)
    shape = shape_key(cfg)

    if manifest is not None:
        bucket = "COMPLETE"
        reason = f"manifest={manifest.name}, age {age_h}h"
    elif age_s < TWO_HOURS:
        bucket = "ACTIVE"
        reason = f"workspace mtime {age_h}h ago"
    elif age_s > THREE_DAYS and not has_recent_log:
        bucket = "ZOMBIE"
        reason = f"workspace {age_h}h old, no log activity in 72h"
    elif age_s > ONE_DAY:
        bucket = "STALE"
        reason = f"workspace {age_h}h old, no manifest"
    else:
        bucket = "ACTIVE"  # 2h–24h, treat as still-running
        reason = f"mtime {age_h}h ago, no manifest yet"

    return {
        "path": str(run),
        "bucket": bucket,
        "age_h": age_h,
        "manifest": bool(manifest),
        "shape": shape,
        "reason": reason,
    }


def detect_duplicates(rows: list[dict]) -> None:
    """Promote a run to DUPLICATE if a COMPLETE run shares its shape key."""
    completed_by_shape: dict[tuple, str] = {}
    for r in rows:
        if r["bucket"] == "COMPLETE" and r["shape"] is not None:
            completed_by_shape.setdefault(r["shape"], r["path"])

    for r in rows:
        if r["bucket"] in {"ACTIVE", "COMPLETE"}:
            # never reclassify a still-running or already-complete run
            continue
        if r["shape"] is None:
            continue
        match = completed_by_shape.get(r["shape"])
        if match and match != r["path"]:
            r["bucket"] = "DUPLICATE"
            r["reason"] += f"; same shape as complete {match}"


def discover_runs(root: Path) -> list[Path]:
    """A sweep run is any direct child directory of ``root``.

    If ``root`` itself looks like a single run (has a manifest or config),
    return ``[root]`` so users can also point at one run.
    """
    if find_manifest(root) is not None or find_config(root) is not None:
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="directory containing per-run sweep subdirectories")
    ap.add_argument("--out", type=Path, default=None,
                    help="CSV output path (default: <root>/sweep_audit.csv)")
    ap.add_argument("--json", action="store_true",
                    help="print machine-readable JSON to stdout")
    args = ap.parse_args()

    if not args.root.is_dir():
        print(f"error: --root {args.root} is not a directory", file=sys.stderr)
        return 2

    runs = discover_runs(args.root)
    if not runs:
        print(f"[sweep_audit] no run dirs under {args.root}", file=sys.stderr)
        return 0

    rows = [classify(r) for r in runs]
    detect_duplicates(rows)

    out = args.out or (args.root / "sweep_audit.csv")
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "path", "bucket", "age_h", "manifest", "shape", "reason",
        ])
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["bucket"], x["path"])):
            row = dict(r)
            row["shape"] = "" if r["shape"] is None else "x".join(
                str(v) for v in r["shape"]
            )
            w.writerow(row)
    print(f"[sweep_audit] wrote {out}", file=sys.stderr)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1

    if args.json:
        print(json.dumps({"counts": counts, "rows": rows}, indent=2,
                         default=str))
    else:
        print()
        print("Sweep audit summary")
        print("-------------------")
        print(f"Root: {args.root}")
        print(f"Runs scanned: {len(rows)}")
        for k in ("ACTIVE", "COMPLETE", "STALE", "ZOMBIE", "DUPLICATE"):
            print(f"  {k:<10} {counts.get(k, 0)}")
        print(f"Detail: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
