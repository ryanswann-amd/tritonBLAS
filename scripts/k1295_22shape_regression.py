#!/usr/bin/env python3
"""K-1295 22-shape cohort regression script — verify K-1441 P16 30-cell stack
introduces no slowdown on the 22-shape K-1295 cohort.

Methodology mirrors `scripts/k1429_paired_n30.py` (paired n=30 HIP-graph hot
cache; B=10000 vectorised paired bootstrap CI95).  The 22 shapes are the
canonical K-1295 production cohort (K-1295 anchors used to validate the
P12 / P13 / P14 / P15 stack — sourced from
`project_context/tritonblas/cohorts/k1295_22.yaml` if present, otherwise
falls back to the inlined K-1295 anchors).

For K-1441 (which only adds a 10th-position predicate at N=1024), the 22
shapes are EXPECTED to route identically before and after — geomean delta
should be within ±2% (graph-replay noise floor) of 1.000.

Run:
    python3 scripts/k1295_22shape_regression.py \
        --out-json /home/ryaswann/mc2-workspaces/K-1441/output/k1295_regression.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

import tritonblas  # noqa: F401


# K-1295 22-shape canonical cohort (matches K-1295 anchor set used to gate
# every productionisation in the K-1361 → K-1441 lineage).  Each tuple is
# (M, N, K, dtype_key).
K1295_22_SHAPES = [
    # K-905/K-971 anchor cluster (square-ish mid)
    (1024, 1024, 16384, "bf16"),
    (1024, 1024, 16384, "fp16"),
    (2048, 2048, 16384, "fp16"),
    # K-984 / K-989 anchors (shipped P5 closed-form)
    (2304, 2048, 4800, "bf16"),
    (3072, 4096, 7168, "bf16"),
    # K-1175 stacked-predicate validation cells
    (4096, 4096, 4096, "bf16"),
    (4096, 4096, 4096, "fp16"),
    (2048, 2048, 2048, "bf16"),
    (2048, 2048, 2048, "fp16"),
    # P13 N=128 / N=256 sibling siblings (K-1367 / K-1397 anchors)
    (2048,  128, 4096, "bf16"),
    (2048,  128, 8192, "bf16"),
    (4096,  256, 4096, "bf16"),
    (4096,  256, 8192, "bf16"),
    # P15 N=512 (K-1417 / K-1425) anchors
    (4096,  512, 4096, "bf16"),
    (4096,  512, 8192, "bf16"),
    (4096,  512, 16384, "bf16"),
    # K-COMPLEMENT EXTREMES sibling pins (N=128 / N=256 EXTREMES)
    (4096,  128, 32768, "bf16"),
    (4096,  256, 32768, "bf16"),
    # Rectangular off-cohort guardrails (must NOT regress)
    (4096, 2048, 4096, "bf16"),
    (8192, 2048, 32768, "fp16"),
    (1024, 1024, 4096, "bf16"),
    (16384, 1024, 4096, "bf16"),
]
assert len(K1295_22_SHAPES) == 22


REPLAYS = 50
WARMUP = 5
N_PAIRED = 30
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass
class Cell:
    M: int
    N: int
    K: int
    dtype: str
    n: int
    tb_us_med: float
    hbl_us_med: float
    ratio_median: float
    ratio_ci95_lo: float
    ratio_ci95_hi: float


def time_graph(g, replays):
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(replays):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / replays


def capture(call_fn):
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


def measure(M, N, K, dt_key):
    dt = DTYPES[dt_key]
    a = torch.randn(M, K, device="cuda", dtype=dt)
    b = torch.randn(K, N, device="cuda", dtype=dt)
    out_tb = torch.empty(M, N, device="cuda", dtype=dt)
    out_hbl = torch.empty(M, N, device="cuda", dtype=dt)
    tb_call = lambda: out_tb.copy_(tritonblas.matmul(a, b))
    hbl_call = lambda: torch.matmul(a, b, out=out_hbl)
    for _ in range(WARMUP):
        tb_call(); hbl_call()
    torch.cuda.synchronize()
    g_tb, g_hbl = capture(tb_call), capture(hbl_call)
    tb_us, hbl_us = [], []
    for i in range(N_PAIRED):
        if i % 2 == 0:
            t1 = time_graph(g_tb, REPLAYS); t2 = time_graph(g_hbl, REPLAYS)
        else:
            t2 = time_graph(g_hbl, REPLAYS); t1 = time_graph(g_tb, REPLAYS)
        tb_us.append(t1); hbl_us.append(t2)
    tb_arr = np.asarray(tb_us); hbl_arr = np.asarray(hbl_us)
    ratio = tb_arr / hbl_arr
    rng = np.random.default_rng(0xC0FFEE)
    idx = rng.integers(0, N_PAIRED, size=(10000, N_PAIRED))
    boots = ratio[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return Cell(M, N, K, dt_key, N_PAIRED,
                float(np.median(tb_arr)), float(np.median(hbl_arr)),
                float(np.median(ratio)), float(lo), float(hi))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-csv", default="")
    args = p.parse_args()

    device = torch.cuda.get_device_name(0)
    print(f"[host] device={device}", file=sys.stderr, flush=True)
    print(f"[plan] {len(K1295_22_SHAPES)} K-1295 cohort cells, n={N_PAIRED}",
          file=sys.stderr, flush=True)

    rows = []
    t0 = time.time()
    for i, (M, N, K, dt) in enumerate(K1295_22_SHAPES, 1):
        ts = time.time()
        r = measure(M, N, K, dt)
        rows.append(r)
        print(f"[{i:2d}/{len(K1295_22_SHAPES)}] M={M} N={N} K={K} {dt} "
              f"tb={r.tb_us_med:.1f}us hbl={r.hbl_us_med:.1f}us "
              f"r={r.ratio_median:.3f} CI95=[{r.ratio_ci95_lo:.3f},"
              f"{r.ratio_ci95_hi:.3f}] ({time.time()-ts:.1f}s)",
              file=sys.stderr, flush=True)

    medians = np.array([r.ratio_median for r in rows])
    geomean = float(np.exp(np.log(medians).mean()))
    print(f"[done] cohort geomean(tb/hbl) over 22 K-1295 shapes = {geomean:.4f}",
          file=sys.stderr, flush=True)

    with open(args.out_json, "w") as f:
        json.dump({
            "device": device,
            "n_per_cell": N_PAIRED,
            "cohort_geomean_tb_over_hbl": geomean,
            "cells": [asdict(r) for r in rows],
        }, f, indent=2)
    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
