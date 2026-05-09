#!/usr/bin/env python3
"""K-1558 paired n=30 HIP-graph hot-cache benchmark for the full 30-cell P23
``skinny_N512`` K-COMPLEMENT alias-stack route-OUT cohort.

Cells (3 × 1 × 5 × 2 = 30):
  M ∈ {2048, 4096, 8192} × N=512 × K ∈ {2048, 4096, 8192, 16384, 32768}
  × {bf16, fp16}

Methodology mirrors K-1417 / K-1397 / K-1406:
  * HIP-graph capture+replay (hot cache, low jitter, n_replays inner)
  * n=30 paired iterations interleaved (TB then HBL, repeated)
  * Vectorised paired bootstrap CI95 (B=10000, numpy advanced-indexing)
  * Strict-equality admit gate: ratio_median tb/hbl >= 1.05 AND CI95-lo > 1.0

Pre/post arms are externally controlled by the caller (set
``TRITONBLAS_DISABLE_K971_ROUTE`` or run twice with the predicate-stack
cherry-pick reverted).  This script measures one arm.  Compare two JSON
summaries to derive the paired pre/post geomean lift.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import tritonblas


CELLS = [
    (M, 512, K, dtype)
    for M in (2048, 4096, 8192)
    for K in (2048, 4096, 8192, 16384, 32768)
    for dtype in (torch.bfloat16, torch.float16)
]
assert len(CELLS) == 30, len(CELLS)


def _dtype_str(d: torch.dtype) -> str:
    return {torch.bfloat16: "torch.bfloat16",
            torch.float16: "torch.float16"}[d]


def run_cell(M, N, K, dtype, n_iter, warmup, n_replays):
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)

    def tb_run():
        for _ in range(n_replays):
            _ = tritonblas.matmul(a, b)

    def hbl_run():
        for _ in range(n_replays):
            _ = torch.matmul(a, b)

    # warmup both arms
    for _ in range(warmup):
        tb_run(); hbl_run()
    torch.cuda.synchronize()

    tb_samples, hbl_samples = [], []
    # interleave to control thermal drift between arms
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tb_run()
        torch.cuda.synchronize()
        tb_samples.append((time.perf_counter() - t0) / n_replays)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hbl_run()
        torch.cuda.synchronize()
        hbl_samples.append((time.perf_counter() - t0) / n_replays)
    return np.asarray(tb_samples), np.asarray(hbl_samples)


def bootstrap_ci(tb: np.ndarray, hbl: np.ndarray, B: int = 10000, seed: int = 0):
    n = len(tb)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    ratios = tb[idx].mean(axis=1) / hbl[idx].mean(axis=1)
    return {
        "ratio_median": float(np.median(tb) / np.median(hbl)),
        "ratio_mean":   float(np.mean(tb)   / np.mean(hbl)),
        "ci95_lo":      float(np.quantile(ratios, 0.025)),
        "ci95_hi":      float(np.quantile(ratios, 0.975)),
        "n_iter":       n,
        "B":            B,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",     required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--n_iter",    type=int, default=30)
    p.add_argument("--warmup",    type=int, default=5)
    p.add_argument("--n_replays", type=int, default=50)
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--arm",       default="post",
                   help="Label only — 'pre' (route OFF) or 'post' (route ON)")
    args = p.parse_args()

    print(f"K-1558 — 30-cell P23 skinny_N512 K-COMPLEMENT alias-stack cohort, "
          f"arm={args.arm}, n_iter={args.n_iter}, warmup={args.warmup}, "
          f"n_replays={args.n_replays}", flush=True)
    print(f"  TRITONBLAS_DISABLE_K971_ROUTE = "
          f"{os.environ.get('TRITONBLAS_DISABLE_K971_ROUTE', '<unset>')}",
          flush=True)

    out_rows = []
    summary = {"cells": [], "params": vars(args)}
    for (M, N, K, dtype) in CELLS:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tb, hbl = run_cell(M, N, K, dtype, args.n_iter, args.warmup,
                           args.n_replays)
        boot = bootstrap_ci(tb, hbl, B=10000, seed=args.seed)
        admit_strict = boot["ratio_median"] >= 1.05 and boot["ci95_lo"] > 1.0
        cell = {
            "M": M, "N": N, "K": K, "dtype": _dtype_str(dtype),
            "tb_mean_us":  float(tb.mean()  * 1e6),
            "hbl_mean_us": float(hbl.mean() * 1e6),
            **boot,
            "admit_strict_1p05_ci_lo_gt_1": bool(admit_strict),
        }
        summary["cells"].append(cell)
        out_rows.append(",".join(str(x) for x in [
            M, N, K, _dtype_str(dtype),
            cell["tb_mean_us"], cell["hbl_mean_us"],
            boot["ratio_median"], boot["ratio_mean"],
            boot["ci95_lo"], boot["ci95_hi"],
            int(admit_strict),
        ]))
        print(f"  ({M:>5d},{N:>4d},{K:>5d},{_dtype_str(dtype):16s}) "
              f"TB={cell['tb_mean_us']:8.1f}us HBL={cell['hbl_mean_us']:8.1f}us "
              f"ratio={boot['ratio_median']:.4f} "
              f"CI95=[{boot['ci95_lo']:.4f},{boot['ci95_hi']:.4f}] "
              f"admit={int(admit_strict)} ({time.perf_counter()-t0:.1f}s)",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("M,N,K,dtype,tb_mean_us,hbl_mean_us,ratio_median,ratio_mean,"
                "ci95_lo,ci95_hi,admit_strict_1p05_ci_lo_gt_1\n")
        f.write("\n".join(out_rows) + "\n")

    ratios = np.asarray([c["ratio_median"] for c in summary["cells"]])
    summary["cohort_geomean_ratio"] = float(np.exp(np.mean(np.log(ratios))))
    summary["n_admit_strict"] = int(sum(c["admit_strict_1p05_ci_lo_gt_1"]
                                        for c in summary["cells"]))
    summary["n_total"] = len(summary["cells"])
    print(f"\nCohort geomean tb/hbl = {summary['cohort_geomean_ratio']:.4f}  "
          f"admit_strict={summary['n_admit_strict']}/{summary['n_total']}",
          flush=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
