#!/usr/bin/env python3
"""K-901 -- run K-882 grid-cap override prototype through the K-883/K-901
guarded-override harness against the K-790 92-shape gap profile and a K-832
production sample, ON c42/MI300X.

The K-882 prototype is defined INLINE here (not in the production
``tritonblas`` tree) because its harness verdict is NO-LAND -- shipping
NO-LAND code "just to test the tool" is the dead-code anti-pattern called
out by K-901 review.  Instead, this script and ``tests/test_guarded_override.py``
are the canonical place that exercises K-882: the script for GPU paired-CI
falsification, the test for CPU-only structural guarantees.

Outputs per_cell.csv, routing.csv, verdict.json, REPORT.md to ``--out``.
Exit code 0 on LAND, 1 on NO-LAND (so CI can gate directly).

Per K-883 §5 / K-901 PRD: paired n>=30 HIP-graph kernel-only ON/OFF
benchmarking against the composite-14 baseline AND hipBLASLt, with
95% paired-CI deltas reported per cell.

Usage::

    PYTHONPATH=$REPO/include python scripts/run_falsification_k882.py \\
        --shapes-k790 path/to/shapes_k790_fp16bf16.json \\
        --shapes-k832 path/to/k832_ehsan_top100.csv --k832-n 30 \\
        --out output/k901_k882

The script is single-process / single-GPU (K-867 protocol).  No
torch.distributed, no multi-process workers -- paired ON/OFF correctness
depends on shared allocator / JIT-cache state.
"""
import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List


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


def _build_k882_gate():
    """K-882 grid-cap prototype, defined inline.  This is a TEST FIXTURE --
    not a production override -- because its harness verdict is NO-LAND.
    """
    import torch
    from tritonblas.guarded_override import GuardedOverride
    return GuardedOverride(
        ticket="K-882",
        env_var="TB_K882_ENABLE",
        cohort_keys=[
            (2048, 2048, 4096, torch.float16),
            (2048, 2048, 4096, torch.bfloat16),
            (2048, 2048, 8192, torch.float16),
            (2048, 2048, 8192, torch.bfloat16),
            (2048, 2048, 16384, torch.float16),
            (2048, 2048, 16384, torch.bfloat16),
        ],
        dtype_allowlist=(torch.float16, torch.bfloat16),
        override_fn=lambda M, N, K, dt: {
            "grid_cap_to_n_cu": True,
            "num_stages": 2,
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes-k790", type=str, required=False,
                    help="Path to K-790 shapes JSON (e.g. shapes_k790_fp16bf16.json).")
    ap.add_argument("--shapes-k832", type=str, required=False,
                    help="Path to K-832 production-taxonomy CSV.")
    ap.add_argument("--k832-n", type=int, default=30,
                    help="Number of K-832 shapes to sample by traffic share.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--n-warm", type=int, default=5)
    ap.add_argument("--n-capture", type=int, default=30)
    ap.add_argument("--n-rounds", type=int, default=30,
                    help="Paired (OFF,ON,REF) rounds; n>=30 enforces K-901 PRD.")
    ap.add_argument("--no-ref", action="store_true",
                    help="Skip reference (hipBLASLt) measurement.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit total shapes (debug / smoke).")
    args = ap.parse_args()

    if args.n_rounds < 30:
        print(f"WARN: --n-rounds={args.n_rounds} < 30; K-901 PRD requires "
              f"n>=30 paired samples per cell.  CI estimates will widen.",
              file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)

    # Import after argparse so --help works without CUDA.
    import tritonblas  # noqa: F401
    from tritonblas.guarded_override import OverrideRegistry, run_falsification

    # K-882 is NOT in the production registry by design; we register the
    # inline fixture for the duration of this script only.
    gate = _build_k882_gate()
    OverrideRegistry.register(gate)

    # Build the shape list.
    shapes: List[Dict[str, Any]] = []
    DT = ("fp16", "bf16")
    for K in (4096, 8192, 16384):
        for dt in DT:
            shapes.append({"M": 2048, "N": 2048, "K": K, "dtype": dt,
                           "batch": 1, "marker": "K-882 cohort"})
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

    print(f"[K-901] gate={gate.ticket} (test-fixture, NOT in production registry)")
    print(f"[K-901] cohort_size={len(gate.cohort_keys)} shapes={len(shapes)}")
    print(f"[K-901] paired n_rounds={args.n_rounds} (>=30 per PRD)")
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
    OverrideRegistry.unregister("K-882")
    sys.exit(0 if verdict.land else 1)


if __name__ == "__main__":
    main()
