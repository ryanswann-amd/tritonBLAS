#!/usr/bin/env python3
"""K-901 — run K-882 grid-cap override through the K-883/K-901 guarded-override
harness against the K-790 92-shape gap profile and a K-832 production sample.

Outputs per_cell.csv, routing.csv, verdict.json, REPORT.md to ``--out``.
Verdict (LAND or NO-LAND) drives the K-901 acceptance gate.

Usage::

    PYTHONPATH=$REPO/include python scripts/run_falsification_k882.py \\
        --shapes-k790 path/to/shapes_k790_fp16bf16.json \\
        --shapes-k832 path/to/k832_ehsan_top100.csv --k832-n 30 \\
        --out output/k901_k882

The script is single-process / single-GPU (K-867 protocol).  No torch.distributed,
no multi-process workers — paired ON/OFF correctness depends on shared
allocator/JIT-cache state.
"""
import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, List


def _load_k790_shapes(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def _load_k832_shapes(path: str, n: int = 30) -> List[Dict[str, Any]]:
    """Return up to ``n`` (M,N,K,dtype) rows from the K-832 production taxonomy.

    Sampled by descending share_pct (top-N most-trafficked shapes).  Any rows
    with unsupported dtype are skipped.  Each unique row contributes one fp16
    + one bf16 entry (mirroring K-790 catalog convention).
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
    ap.add_argument("--shapes-k790", type=str, required=False,
                    help="Path to K-790 shapes JSON (e.g. shapes_k790_fp16bf16.json).")
    ap.add_argument("--shapes-k832", type=str, required=False,
                    help="Path to K-832 production-taxonomy CSV (e.g. k832_ehsan_top100.csv).")
    ap.add_argument("--k832-n", type=int, default=30,
                    help="Number of K-832 shapes to sample by traffic share.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--n-warm", type=int, default=5)
    ap.add_argument("--n-capture", type=int, default=30)
    ap.add_argument("--n-rounds", type=int, default=3)
    ap.add_argument("--no-ref", action="store_true",
                    help="Skip reference (hipBLASLt) measurement.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit total shapes (debug / smoke).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Import after argparse so --help works without CUDA.
    import tritonblas  # noqa: F401  (forces override registration)
    from tritonblas.guarded_override import OverrideRegistry, run_falsification

    # The K-882 override is registered automatically on tritonblas import.
    gate = next((g for g in OverrideRegistry.all() if g.ticket == "K-882"), None)
    if gate is None:
        print("FATAL: K-882 gate not registered.  Did overrides/__init__.py import?",
              file=sys.stderr)
        sys.exit(2)

    # Build the shape list.
    shapes: List[Dict[str, Any]] = []
    # Always include the 6 in-cohort cells first so V1 (in-cohort lift) gets
    # measured even if --shapes-* are missing.
    DT = ("fp16", "bf16")
    for K in (4096, 8192, 16384):
        for dt in DT:
            shapes.append({"M": 2048, "N": 2048, "K": K, "dtype": dt,
                           "batch": 1, "marker": "K-882 cohort"})
    # Then K-790 and K-832 OOC.
    if args.shapes_k790:
        for sh in _load_k790_shapes(args.shapes_k790):
            shapes.append({**sh, "marker": sh.get("marker", "K-790")})
    if args.shapes_k832:
        shapes.extend(_load_k832_shapes(args.shapes_k832, n=args.k832_n))

    # De-dup (M,N,K,dtype) keeping first.
    seen = set()
    deduped = []
    for sh in shapes:
        k = (int(sh["M"]), int(sh["N"]), int(sh["K"]), sh["dtype"])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(sh)
    shapes = deduped

    if args.limit:
        shapes = shapes[:args.limit]

    print(f"[K-901] gate={gate.ticket}  cohort_size={len(gate.cohort_keys)}")
    print(f"[K-901] running {len(shapes)} shapes "
          f"(in-cohort-by-key first; OOC after)")
    verdict = run_falsification(
        override=gate,
        shapes=shapes,
        out_dir=args.out,
        n_warm=args.n_warm,
        n_capture=args.n_capture,
        n_rounds=args.n_rounds,
        measure_ref=(not args.no_ref),
    )
    print(verdict.report)

    # Exit 0 on LAND, 1 on NO-LAND, so CI can gate on this directly.
    sys.exit(0 if verdict.land else 1)


if __name__ == "__main__":
    main()
