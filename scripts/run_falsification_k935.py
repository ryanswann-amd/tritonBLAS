#!/usr/bin/env python3
"""K-935 — paired falsification of the cohort-A LDS-pressure mitigation.

Runs the K-935 gate (kpack=2, num_warps=8 forced for M=N=1024 × K=16384,
fp16+bf16) through the K-883/K-901 guarded-override harness:

  - In-cohort: 2 cells (1024,1024,16384 × {fp16,bf16})
  - Out-of-cohort A: K-790 92-shape gap-profile sample (json or built-in defaults)
  - Out-of-cohort B: K-832 top-traffic production sample (csv, top-N by share)

Per-cell paired n>=30 HIP-graph kernel-only ON/OFF/REF sampling, round-robin
order within each round (K-867 / K-901 M1).  Writes per_cell.csv, routing.csv,
verdict.json, REPORT.md to ``--out``.  Exit code 0 on LAND, 1 on NO-LAND.

Acceptance gates (K-883 §5, tightened by K-901 V2 OOC-bleed CI95-upper<0):

  - V1 in-cohort geomean speedup >= 1.05x AND >= 80% cells SPEEDUP class
  - V2 OOC bleed: 0 cells with classifier=REGRESS AND CI95-upper<0
  - V3 routing trace: per-key fires == cohort enumeration; 0 OOC fires

Usage::

    PYTHONPATH=$REPO/include python scripts/run_falsification_k935.py \\
        --shapes-k790 path/to/shapes_k790_fp16bf16.json \\
        --shapes-k832 path/to/k832_ehsan_top100.csv --k832-n 50 \\
        --out output/k935

If neither --shapes-k790 nor --shapes-k832 is provided, falls back to a
small built-in OOC sample so the script still produces a verdict for
smoke-testing.  Production gates require both files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List


_BUILTIN_OOC = [
    # K-905 cohort-A neighbours (must NOT regress; gate must NOT fire on them).
    {"M": 1024, "N": 1024, "K": 8192,  "dtype": "fp16", "batch": 1, "marker": "neighbour-K"},
    {"M": 1024, "N": 1024, "K": 8192,  "dtype": "bf16", "batch": 1, "marker": "neighbour-K"},
    {"M": 1024, "N": 1024, "K": 4096,  "dtype": "fp16", "batch": 1, "marker": "neighbour-K"},
    {"M": 1024, "N": 1024, "K": 4096,  "dtype": "bf16", "batch": 1, "marker": "neighbour-K"},
    {"M": 1024, "N": 1024, "K": 2048,  "dtype": "fp16", "batch": 1, "marker": "neighbour-K"},
    {"M": 1024, "N": 1024, "K": 2048,  "dtype": "bf16", "batch": 1, "marker": "neighbour-K"},
    # K-905 cohort-A above-K (extension cells; gate intentionally not extended)
    {"M": 1024, "N": 1024, "K": 32768, "dtype": "fp16", "batch": 1, "marker": "above-K"},
    {"M": 1024, "N": 1024, "K": 32768, "dtype": "bf16", "batch": 1, "marker": "above-K"},
    # K-882 cohort cell — must NOT fire (different cohort).
    {"M": 2048, "N": 2048, "K": 16384, "dtype": "fp16", "batch": 1, "marker": "K-882-cohort"},
    {"M": 2048, "N": 2048, "K": 16384, "dtype": "bf16", "batch": 1, "marker": "K-882-cohort"},
    # Mid-rect (K-836/K-854/K-864 NO-LAND zones — kpack=2 known-harmful).
    {"M": 2048, "N": 2048, "K": 4096,  "dtype": "fp16", "batch": 1, "marker": "mid-rect"},
    {"M": 2048, "N": 2048, "K": 4096,  "dtype": "bf16", "batch": 1, "marker": "mid-rect"},
    # Production hot-shape proxies.
    {"M": 4096, "N": 4096, "K": 4096,  "dtype": "bf16", "batch": 1, "marker": "prod-hot"},
    {"M": 8192, "N": 8192, "K": 1024,  "dtype": "bf16", "batch": 1, "marker": "prod-hot"},
    {"M": 1024, "N": 4096, "K": 4096,  "dtype": "fp16", "batch": 1, "marker": "prod-hot"},
    {"M": 4096, "N": 1024, "K": 4096,  "dtype": "bf16", "batch": 1, "marker": "prod-hot"},
]


def _load_k790_shapes(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def _load_k832_shapes(path: str, n: int = 50) -> List[Dict[str, Any]]:
    """Top-N by share_pct from the K-832 production-taxonomy CSV.

    Each unique (M,N,K) row contributes one fp16 and one bf16 entry (mirrors
    the K-790 catalog convention).  Skips rows with non-supported dtype.
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
            except (KeyError, ValueError):
                continue
            for dt in ("fp16", "bf16"):
                key = (M, N, K, dt)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"M": M, "N": N, "K": K, "dtype": dt, "batch": 1,
                            "marker": f"K-832-#{row.get('rank', '?')}"})
                if len(out) >= n:
                    return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes-k790", type=str, required=False)
    ap.add_argument("--shapes-k832", type=str, required=False)
    ap.add_argument("--k832-n", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-warm", type=int, default=5)
    ap.add_argument("--n-capture", type=int, default=30)
    ap.add_argument("--n-rounds", type=int, default=30,
                    help="Paired (OFF,ON,REF) rounds; n>=30 enforces K-901 PRD.")
    ap.add_argument("--no-ref", action="store_true",
                    help="Skip reference (hipBLASLt via torch.matmul).")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.n_rounds < 30:
        print(f"WARN: --n-rounds={args.n_rounds} < 30; K-901 PRD requires "
              f"n>=30 paired samples per cell.", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)

    # Trigger override registration via package import.
    import tritonblas  # noqa: F401
    from tritonblas.guarded_override import OverrideRegistry, run_falsification

    # K-935 self-registers on `import tritonblas.overrides`.  Locate it.
    k935 = None
    for g in OverrideRegistry.all():
        if g.ticket == "K-935":
            k935 = g
            break
    if k935 is None:
        print("ERROR: K-935 gate not registered; check tritonblas/overrides/.",
              file=sys.stderr)
        sys.exit(2)

    # ---- Build the shape list ------------------------------------------------
    shapes: List[Dict[str, Any]] = []
    # In-cohort first (so progress log shows them at the top).
    for K in (16384,):
        for dt in ("fp16", "bf16"):
            shapes.append({"M": 1024, "N": 1024, "K": K, "dtype": dt,
                           "batch": 1, "marker": "K-935 cohort"})

    if args.shapes_k790:
        shapes.extend(_load_k790_shapes(args.shapes_k790))
    if args.shapes_k832:
        shapes.extend(_load_k832_shapes(args.shapes_k832, n=args.k832_n))
    # Always tack on the built-in safety neighbours.
    shapes.extend(_BUILTIN_OOC)

    # De-dup on (M,N,K,dtype) keeping first occurrence.
    seen = set()
    deduped = []
    for sh in shapes:
        key = (int(sh["M"]), int(sh["N"]), int(sh["K"]), sh["dtype"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sh)
    shapes = deduped

    if args.limit:
        shapes = shapes[:args.limit]

    print(f"[K-935] gate={k935.ticket} cohort_size={len(k935.cohort_keys)} "
          f"shapes={len(shapes)} n_rounds={args.n_rounds}")

    verdict = run_falsification(
        override=k935,
        shapes=shapes,
        out_dir=args.out,
        n_warm=args.n_warm,
        n_capture=args.n_capture,
        n_rounds=args.n_rounds,
        measure_ref=(not args.no_ref),
    )
    print(verdict.report)
    sys.exit(0 if verdict.land else 1)


if __name__ == "__main__":
    main()
