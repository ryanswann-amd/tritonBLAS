"""K-1835 P34 pre-stack (TB-native, route-OUT ablated) vs post-stack (HBL-routed).

Measures the cohort speedup the patch unlocks by running the SAME 18 P34 cells
twice: once with TRITONBLAS_DISABLE_K971=1 forcing the TB-native path, once
with the live post-patch routing oracle.  Output: per-cell pre/post timings
and the speedup the patch realises (post / pre HBL/TB ratio).

Usage:
    PYTHONPATH=$(pwd)/include python3 scripts/pre_vs_post.py
"""
from __future__ import annotations

import csv
import math
import os
import statistics
import sys

import torch

TB_INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "include"))
if os.path.isdir(TB_INCLUDE):
    sys.path.insert(0, TB_INCLUDE)

import tritonblas
from tritonblas._route_predicate import _P34_SKINNY_N288_KCOMPL_VERIFIED_WIN_18


def hipgraph_capture(fn, n_inner=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_inner):
            fn()
    torch.cuda.synchronize()
    def replay(): g.replay()
    return replay, n_inner


def time_replay(replay, n_inner, n_outer=30):
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(n_outer):
        torch.cuda.synchronize()
        start.record()
        replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / n_inner)
    return statistics.median(times)


def bench_one(M, N, K, dtype, fn_factory):
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    out = torch.empty(M, N, device="cuda", dtype=dtype)
    fn = fn_factory(a, b, out)
    rep, ni = hipgraph_capture(fn)
    return time_replay(rep, ni)


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


torch.backends.cuda.preferred_blas_library("hipblaslt")

cells = sorted(_P34_SKINNY_N288_KCOMPL_VERIFIED_WIN_18)

print("device:", torch.cuda.get_device_name(0))
print(f"Pre vs post on {len(cells)} P34 cells")
print()
results = []
speedups = []
for (M, N, K, dt) in cells:
    dtype = torch.bfloat16 if dt == "torch.bfloat16" else torch.float16
    # PRE: TB-native (route-OUT ablated via env var)
    os.environ["TRITONBLAS_DISABLE_K971"] = "1"
    t_pre = bench_one(M, N, K, dtype,
                      lambda a, b, out: lambda: tritonblas.matmul(a, b, out=out))
    # POST: live oracle (route-OUT engaged → P34 fires → HBL via torch.matmul)
    os.environ["TRITONBLAS_DISABLE_K971"] = "0"
    t_post = bench_one(M, N, K, dtype,
                       lambda a, b, out: lambda: tritonblas.matmul(a, b, out=out))
    # Reference HBL (always torch.matmul)
    t_hbl = bench_one(M, N, K, dtype,
                      lambda a, b, out: lambda: torch.matmul(a, b, out=out))
    speedup = t_pre / t_post  # >1 means post is faster
    speedups.append(speedup)
    results.append({
        "M": M, "N": N, "K": K, "dtype": dt,
        "t_pre_us": round(t_pre * 1000, 3),
        "t_post_us": round(t_post * 1000, 3),
        "t_hbl_us": round(t_hbl * 1000, 3),
        "speedup_pre_over_post": round(speedup, 4),
        "post_over_hbl": round(t_post / t_hbl, 4),
    })
    print(f"  M={M:5d} N={N} K={K:5d} {dt[6:]:9s}  pre={t_pre*1000:7.2f}us post={t_post*1000:7.2f}us hbl={t_hbl*1000:7.2f}us  speedup={speedup:.3f}x")

print(f"\ncohort gmean speedup (pre/post) = {gmean(speedups):.4f}x")
print(f"cohort min  speedup (pre/post) = {min(speedups):.4f}x")
print(f"cohort max  speedup (pre/post) = {max(speedups):.4f}x")
print(f"cells with speedup >= 1.05 = {sum(1 for s in speedups if s >= 1.05)}/{len(speedups)}")

out_dir = os.environ.get("OUT_DIR", "/tmp/K-1835/output")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "p34_pre_vs_post_18cell.csv")
with open(out_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
print(f"\nWrote {out_path}")
