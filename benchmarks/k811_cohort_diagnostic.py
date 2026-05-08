# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""K-811 cohort diagnostic — paired tritonblas vs hipBLASLt sweep.

Standalone diagnostic harness that measures the M=N=2048 large-K
fp16/bf16 cohort (and a guard set) against hipBLASLt (`torch.matmul`)
using kernel-only hot-cache HIP-graph replay.

Background
----------
The 6 cells (M=N=2048, K∈{4096, 8192, 16384}, dtype∈{fp16, bf16}) are
the residual band where tritonblas trails hipBLASLt by the largest
margin. Investigation tickets (K-760, K-783, K-784, K-791, K-795,
K-811) attribute the gap to backend mechanisms not currently expressible
through the user-facing Triton kernel:

  (A) MIWT4_7 chained MFMAs (28 MFMAs per LDS load),
  (B) LDSB1 row-padded LDS swizzle (zero LDS bank conflicts),
  (C) GSU=2 (split-K) dispatch.

A user-space split-K=2 routed override (K-811 prototype, K-795 kernel)
was paired-tested against the standard non-splitK kernel on c42/MI300X
(n=50 hot-cache CUDA-graph) and regressed every in-cohort cell by
24-50 percentage points on/hbl ratio because mechanisms (A) and (B)
remained unaddressed and the FP32 atomic-add cost of (C) alone could
not be amortised. The override was therefore NOT landed; this harness
is the instrumentation that produced the NULL-result evidence and is
preserved for re-evaluation when a Triton-AMD backend successor that
exposes MIWT chaining + LDSB1-equivalent swizzle lands.

Usage
-----
    python benchmarks/k811_cohort_diagnostic.py --out cohort.csv
    python benchmarks/k811_cohort_diagnostic.py --shapes guard --out guard.csv

The 12-cell guard set (M=N∈{4096, 1024} mirrors at K∈{4096, 8192,
16384}, both dtypes) verifies no regression outside the in-cohort band
and is useful for tracking whether future tritonblas changes
inadvertently widen or narrow the gap.
"""
import argparse
import csv
import math
import statistics
import sys

import torch


def parse_shapes(name):
    in_cohort = []
    for K in (4096, 8192, 16384):
        for dt in (torch.float16, torch.bfloat16):
            in_cohort.append((2048, 2048, K, dt))

    guard = []
    for M in (4096, 1024):
        for K in (4096, 8192, 16384):
            for dt in (torch.float16, torch.bfloat16):
                guard.append((M, M, K, dt))

    if name == "in_cohort":
        return in_cohort
    if name == "guard":
        return guard
    if name == "all":
        return in_cohort + guard
    raise ValueError(f"unknown shape set {name!r}")


def make_inputs(M, N, K, dtype, device="cuda"):
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    out = torch.empty(M, N, device=device, dtype=dtype)
    return a, b, out


def time_graph(fn, n_capture, n_warm, n_rounds):
    # Warm Triton cache + first compile.
    for _ in range(2):
        fn()
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_capture):
            fn()

    for _ in range(n_warm):
        g.replay()
    torch.cuda.synchronize()

    rounds = []
    for _ in range(n_rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        rounds.append(start.elapsed_time(end) / n_capture)
    return rounds


def run_cell(M, N, K, dtype, backend, n_capture, n_warm, n_rounds):
    a, b, out = make_inputs(M, N, K, dtype)
    if backend == "hbl":
        def fn():
            torch.matmul(a, b, out=out)
    elif backend == "tb":
        import tritonblas

        def fn():
            tritonblas.matmul(a, b, out=out)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return time_graph(fn, n_capture, n_warm, n_rounds)


def geomean(xs):
    xs = [x for x in xs if x > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default="in_cohort",
                    choices=["in_cohort", "guard", "all"])
    ap.add_argument("--n-capture", type=int, default=10)
    ap.add_argument("--n-warm", type=int, default=5)
    ap.add_argument("--n-rounds", type=int, default=50)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        sys.exit(2)

    shapes = parse_shapes(args.shapes)
    rows = []
    for (M, N, K, dt) in shapes:
        tb_rounds = run_cell(M, N, K, dt, "tb",
                             args.n_capture, args.n_warm, args.n_rounds)
        hbl_rounds = run_cell(M, N, K, dt, "hbl",
                              args.n_capture, args.n_warm, args.n_rounds)
        tb_ms = statistics.median(tb_rounds)
        hbl_ms = statistics.median(hbl_rounds)
        rows.append((M, N, K, str(dt).split(".")[-1], tb_ms, hbl_ms))
        print(f"  {M:5d}×{N:5d}×{K:5d} {str(dt).split('.')[-1]:8s}  "
              f"tb={tb_ms:.4f} ms  hbl={hbl_ms:.4f} ms  "
              f"hbl/tb={hbl_ms / tb_ms:.4f}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "N", "K", "dtype", "tb_ms", "hbl_ms", "hbl_over_tb"])
        ratios = []
        for (M, N, K, dt, tb_ms, hbl_ms) in rows:
            r = hbl_ms / tb_ms if tb_ms > 0 else float("nan")
            ratios.append(r)
            w.writerow([M, N, K, dt,
                        f"{tb_ms:.6f}", f"{hbl_ms:.6f}", f"{r:.4f}"])
    print()
    print(f"=== {args.shapes} geomean (n={len(ratios)}) ===")
    print(f"  hbl/tb geomean = {geomean(ratios):.4f}")


if __name__ == "__main__":
    main()
