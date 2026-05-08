#!/usr/bin/env python3
"""K-935 — paired falsification of the cohort-A LDS-pressure mitigation.

The K-935 prototype is defined INLINE here (not in the production
``tritonblas`` tree) per K-901 M2 (fail-closed semantics require
package-deletion, not disable-flags).  Run this script to produce a
LAND/NO-LAND verdict; only LAND verdicts justify shipping the gate as a
production override.

Mechanism (full math in the gate's docstring below): for cohort A
(M=N=1024 × K=16384, fp16/bf16) the K-905 PMC trace shows ds_read2_b64
saturating LDS issue slots (8.39 M cyc/disp SQ_LDS_BANK_CONFLICT +
38.22 M cyc/disp SQ_WAIT_INST_LDS = 41× hipBLASLt).  Setting kpack=2
asks the Triton-AMD codegen to widen the LDS access pattern to b128
(halving the round-trip count for the same fragment bytes), with
num_warps=8 pinned to the production default for self-containment.

Validation protocol (K-867 / K-901):
  - Per-cell paired n>=30 HIP-graph kernel-only ON/OFF/REF rounds
  - Round-robin order within each round (K-901 M1 cancels thermal drift)
  - Reference impl: hipBLASLt via torch.matmul on AMD ROCm
  - Verdict per K-883 §5 with K-901 V2 tightening (CI95-upper<0)

Acceptance gates:
  - V1: in-cohort geomean ON/OFF >= 1.05x AND >= 80% cells SPEEDUP
  - V2: 0 OOC cells with classifier=REGRESS AND CI95-upper<0
  - V3: per-key fires == cohort enumeration; 0 OOC fires

Usage::

    PYTHONPATH=$REPO/include python scripts/run_falsification_k935.py \\
        --shapes-k790 path/to/shapes_k790_fp16bf16.json \\
        --shapes-k832 path/to/k832_ehsan_top100.csv --k832-n 50 \\
        --out output/k935

Either flag may be omitted; a small built-in OOC sample is always added
so the script produces a verdict for smoke-testing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List


_BUILTIN_OOC = [
    # K-905 cohort-A near-neighbours that MUST NOT fire (K-883 R1 strict eq).
    {"M": 1024, "N": 1024, "K": 8192,  "dtype": "fp16", "batch": 1, "marker": "neighbour-K"},
    {"M": 1024, "N": 1024, "K": 8192,  "dtype": "bf16", "batch": 1, "marker": "neighbour-K"},
    {"M": 1024, "N": 1024, "K": 4096,  "dtype": "fp16", "batch": 1, "marker": "neighbour-K"},
    {"M": 1024, "N": 1024, "K": 4096,  "dtype": "bf16", "batch": 1, "marker": "neighbour-K"},
    {"M": 1024, "N": 1024, "K": 2048,  "dtype": "fp16", "batch": 1, "marker": "neighbour-K"},
    {"M": 1024, "N": 1024, "K": 2048,  "dtype": "bf16", "batch": 1, "marker": "neighbour-K"},
    # K-905 cohort-A above-K cells (intentionally NOT in the gate -- the
    # mechanism does not generalize past 16384 without a cohort-extension
    # study; this validates the gate's K-bound).
    {"M": 1024, "N": 1024, "K": 32768, "dtype": "fp16", "batch": 1, "marker": "above-K"},
    {"M": 1024, "N": 1024, "K": 32768, "dtype": "bf16", "batch": 1, "marker": "above-K"},
    # K-882 cohort cell -- different cohort, must not fire.
    {"M": 2048, "N": 2048, "K": 16384, "dtype": "fp16", "batch": 1, "marker": "K-882-cohort"},
    {"M": 2048, "N": 2048, "K": 16384, "dtype": "bf16", "batch": 1, "marker": "K-882-cohort"},
    # Mid-rect (K-836/K-854/K-864 NO-LAND zones; kpack=2 known-harmful here).
    {"M": 2048, "N": 2048, "K": 4096,  "dtype": "fp16", "batch": 1, "marker": "mid-rect"},
    {"M": 2048, "N": 2048, "K": 4096,  "dtype": "bf16", "batch": 1, "marker": "mid-rect"},
    # Production hot-shape proxies.
    {"M": 4096, "N": 4096, "K": 4096,  "dtype": "bf16", "batch": 1, "marker": "prod-hot"},
    {"M": 8192, "N": 8192, "K": 1024,  "dtype": "bf16", "batch": 1, "marker": "prod-hot"},
    {"M": 1024, "N": 4096, "K": 4096,  "dtype": "fp16", "batch": 1, "marker": "prod-hot"},
    {"M": 4096, "N": 1024, "K": 4096,  "dtype": "bf16", "batch": 1, "marker": "prod-hot"},
]


def build_k935_gate():
    """K-935 cohort-A LDS-pressure mitigation prototype (TEST FIXTURE).

    Defined here, not in ``include/tritonblas/overrides/``, because the
    harness verdict so far is NO-LAND (in-cohort geomean ~0.99x on
    c42/MI300X gfx942: kpack=2 alone is insufficient -- the Triton-AMD
    codegen does not emit ds_read_b128 even when kpack=2 widens the
    fragment, K-905 §6/K-878 §3 codegen-gap analysis).  Per K-901 M2 the
    gate ships only as a fixture; only a LAND verdict justifies moving
    it into the production tree.

    LDS bandwidth math (cohort-A geometry, BLOCK_M=BLOCK_N=128, BK=64,
    fp16/bf16, 2 B/elem):

      A-fragment per-warp:  128 * 64 * 2 B = 16 384 B
      B-fragment per-warp:  128 * 64 * 2 B = 16 384 B

      kpack=1: ds_read2_b64 -> 1024 B/instr -> 32 instr/K-tile
      kpack=2: ds_read_b128 -> 2048 B/instr -> 16 instr/K-tile

    With K=16384 / BK=64 = 256 K-tiles per output tile, ~8 output tiles
    per CU at this geometry, the issue-slot pressure drops by ~50% --
    IF codegen actually emits b128 (which the K-905 ISA dump shows it
    does not on Triton-AMD today).  The gate is the cleanest experiment
    for that codegen path; if a future Triton release closes the gap,
    re-running this script may flip the verdict to LAND.
    """
    import torch
    from tritonblas.guarded_override import GuardedOverride

    return GuardedOverride(
        ticket="K-935",
        env_var="TB_K935_ENABLE",
        cohort_keys=[
            (1024, 1024, 16384, torch.float16),
            (1024, 1024, 16384, torch.bfloat16),
        ],
        dtype_allowlist=(torch.float16, torch.bfloat16),
        override_fn=lambda M, N, K, dt: {
            "kpack": 2,
            "num_warps": 8,
            "num_stages": 2,
        },
        cohort_size_cap=4,
    )


def _load_k790_shapes(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def _load_k832_shapes(path: str, n: int = 50) -> List[Dict[str, Any]]:
    """Top-N by share_pct from the K-832 production-taxonomy CSV."""
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

    import tritonblas  # noqa: F401
    from tritonblas.guarded_override import OverrideRegistry, run_falsification

    gate = build_k935_gate()
    OverrideRegistry.register(gate)

    # Build the shape list (in-cohort first so logs show them at the top).
    shapes: List[Dict[str, Any]] = []
    for K in (16384,):
        for dt in ("fp16", "bf16"):
            shapes.append({"M": 1024, "N": 1024, "K": K, "dtype": dt,
                           "batch": 1, "marker": "K-935 cohort"})

    if args.shapes_k790:
        shapes.extend(_load_k790_shapes(args.shapes_k790))
    if args.shapes_k832:
        shapes.extend(_load_k832_shapes(args.shapes_k832, n=args.k832_n))
    shapes.extend(_BUILTIN_OOC)

    # De-dup on (M,N,K,dtype) keeping first.
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

    print(f"[K-935] gate={gate.ticket} (test-fixture, NOT in production registry)")
    print(f"[K-935] cohort_size={len(gate.cohort_keys)} shapes={len(shapes)}")
    print(f"[K-935] paired n_rounds={args.n_rounds}")

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
    OverrideRegistry.unregister("K-935")
    sys.exit(0 if verdict.land else 1)


if __name__ == "__main__":
    main()
