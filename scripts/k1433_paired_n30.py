#!/usr/bin/env python3
"""K-1433 P16 skinny_N1024 K-COMPLEMENT BASE paired benchmark.

Mirrors K-1409 P15 N=512 BASE protocol exactly with N axis bumped 512 -> 1024:

- 18-cell cohort: M in {2048,4096,8192} x N=1024 x K in {4096,8192,16384} x {bf16,fp16}.
- Paired n=30 HIP-graph hot-cache; alternating tb-first / hbl-first ordering
  to defeat ordering bias.
- 50 graph replays per timed iteration; mean across replays = one paired sample.
- B=10000 NumPy-vectorised paired bootstrap on per-iteration ratio array -> CI95.
- ROUTED arm  = tritonblas.matmul (this cohort N=1024 K in BASE band, no prior
                                   stacked predicate fires; native triton path).
- HBL arm     = torch.matmul (PyTorch dispatches to hipBLASLt).
- ratio r     = tb_us / hbl_us ; r > 1 => hipBLASLt is faster => candidate route-OUT.

Output JSON: per-cell median ratio, CI95-lo/hi, raw 30-sample arrays.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from itertools import product

import numpy as np
import torch

import tritonblas  # noqa: F401


M_VALUES = (2048, 4096, 8192)
N_VALUE = 1024
K_VALUES = (4096, 8192, 16384)
DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


@dataclass
class CellResult:
    M: int
    N: int
    K: int
    dtype: str
    n: int
    tb_med_us: float
    hbl_med_us: float
    ratio_median: float
    ratio_mean: float
    ratio_ci95_lo: float
    ratio_ci95_hi: float
    tb_us_samples: list
    hbl_us_samples: list
    ratio_samples: list


REPLAYS_PER_ITER = 50
WARMUP = 5
N_PAIRED = 30


def time_graph(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / replays


def capture_graph(call_fn) -> torch.cuda.CUDAGraph:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call_fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call_fn()
    return g


def measure_cell(M: int, N: int, K: int, dtype_key: str, *,
                 n: int = N_PAIRED, replays: int = REPLAYS_PER_ITER,
                 warmup: int = WARMUP, seed: int = 0,
                 verify: bool = True) -> CellResult:
    dtype = DTYPES[dtype_key]
    torch.manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    out_tb = torch.empty(M, N, device="cuda", dtype=dtype)
    out_hbl = torch.empty(M, N, device="cuda", dtype=dtype)

    def tb_call():
        out_tb.copy_(tritonblas.matmul(a, b))

    def hbl_call():
        torch.matmul(a, b, out=out_hbl)

    for _ in range(warmup):
        tb_call()
        hbl_call()
    torch.cuda.synchronize()

    if verify:
        tb_ref = tritonblas.matmul(a, b)
        hbl_ref = torch.matmul(a, b)
        diff = (tb_ref.float() - hbl_ref.float()).abs().max().item()
        rel = diff / max(hbl_ref.float().abs().max().item(), 1e-9)
        assert rel < 5e-2, f"Numerical mismatch {M}x{N}x{K} {dtype_key} rel={rel}"

    g_tb = capture_graph(tb_call)
    g_hbl = capture_graph(hbl_call)

    tb_us, hbl_us = [], []
    for i in range(n):
        if i % 2 == 0:
            t1 = time_graph(g_tb, replays)
            t2 = time_graph(g_hbl, replays)
        else:
            t2 = time_graph(g_hbl, replays)
            t1 = time_graph(g_tb, replays)
        tb_us.append(t1)
        hbl_us.append(t2)

    tb_arr = np.asarray(tb_us, dtype=np.float64)
    hbl_arr = np.asarray(hbl_us, dtype=np.float64)
    ratio = tb_arr / hbl_arr

    rng = np.random.default_rng(0xC0FFEE)
    B = 10000
    idx = rng.integers(0, n, size=(B, n))
    boots = ratio[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])

    return CellResult(
        M=M, N=N, K=K, dtype=dtype_key, n=n,
        tb_med_us=float(np.median(tb_arr)),
        hbl_med_us=float(np.median(hbl_arr)),
        ratio_median=float(np.median(ratio)),
        ratio_mean=float(ratio.mean()),
        ratio_ci95_lo=float(lo),
        ratio_ci95_hi=float(hi),
        tb_us_samples=tb_arr.tolist(),
        hbl_us_samples=hbl_arr.tolist(),
        ratio_samples=ratio.tolist(),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--smoke", action="store_true",
                   help="single (2048,1024,4096,bf16) cell only")
    p.add_argument("--n", type=int, default=N_PAIRED)
    p.add_argument("--replays", type=int, default=REPLAYS_PER_ITER)
    args = p.parse_args()

    device = torch.cuda.get_device_name(0)
    print(f"[host] device={device}", file=sys.stderr, flush=True)

    cells = [(2048, 1024, 4096, "bf16")] if args.smoke else [
        (m, N_VALUE, k, d) for m, k, d in product(M_VALUES, K_VALUES, DTYPES.keys())
    ]
    print(f"[plan] {len(cells)} cells, n={args.n}, replays={args.replays}",
          file=sys.stderr, flush=True)

    results = []
    t0 = time.time()
    for i, (M, N, K, dt) in enumerate(cells, 1):
        ts = time.time()
        r = measure_cell(M, N, K, dt, n=args.n, replays=args.replays)
        results.append(r)
        print(f"[{i}/{len(cells)}] M={M} N={N} K={K} {dt} "
              f"tb={r.tb_med_us:.1f}us hbl={r.hbl_med_us:.1f}us "
              f"r={r.ratio_median:.3f} CI95=[{r.ratio_ci95_lo:.3f},"
              f"{r.ratio_ci95_hi:.3f}] ({time.time()-ts:.1f}s)",
              file=sys.stderr, flush=True)

    print(f"[done] {len(results)} cells in {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)

    payload = {
        "device": device,
        "n_per_cell": args.n,
        "replays_per_iter": args.replays,
        "cells": [asdict(r) for r in results],
    }
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)

    fields = ["M", "N", "K", "dtype", "n", "tb_med_us", "hbl_med_us",
              "ratio_median", "ratio_mean", "ratio_ci95_lo", "ratio_ci95_hi"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fields}
            w.writerow(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
