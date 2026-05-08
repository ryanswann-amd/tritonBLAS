"""K-967 single-process kernel-only HIP-graph paired benchmark.

Runs ONE backend (tb or torch), with TB_K967_PROTO already set in env at
process launch (avoids the K-888 methodology gotcha — Triton's selector
caches the kernel signature on first compile so flipping env vars
mid-process leaves the cached kernel stale).

Usage:
  python3 bench_one.py M N K dtype backend rounds_warm rounds_capture rounds_n out_csv
    backend in {tb, torch}
    dtype   in {bf16, fp16}
"""
import os
import sys
import json
import time
import csv

import torch
import triton

import tritonblas


def make_inputs(M, N, K, dtype):
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    c = torch.empty((M, N), device="cuda", dtype=dtype)
    return a, b, c


def time_with_hip_graph(callable_fn, n_warm=10, n_capture=20, n_rounds=30):
    # Warm
    s = torch.cuda.Stream()
    for _ in range(n_warm):
        callable_fn()
    torch.cuda.synchronize()

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_capture):
            callable_fn()
    torch.cuda.synchronize()

    # Replay rounds
    times_ms = []
    for _ in range(n_rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0 / n_capture)
    return times_ms


def main():
    M = int(sys.argv[1]); N = int(sys.argv[2]); K = int(sys.argv[3])
    dt = sys.argv[4]; backend = sys.argv[5]
    n_warm = int(sys.argv[6]); n_cap = int(sys.argv[7]); n_rounds = int(sys.argv[8])
    out_csv = sys.argv[9]

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dt]
    a, b, c = make_inputs(M, N, K, dtype)

    if backend == "tb":
        def fn():
            tritonblas.matmul(a, b, c)
    elif backend == "torch":
        def fn():
            torch.matmul(a, b, out=c)
    else:
        raise ValueError(backend)

    times = time_with_hip_graph(fn, n_warm=n_warm, n_capture=n_cap, n_rounds=n_rounds)

    proto = os.environ.get("TB_K967_PROTO", "0")
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        for i, t in enumerate(times):
            w.writerow([M, N, K, dt, backend, proto, i, f"{t:.6f}"])

    times.sort()
    p50 = times[len(times) // 2]
    print(f"{M}x{N}x{K} {dt} backend={backend} TB_K967_PROTO={proto} "
          f"p50={p50*1000:.3f}us min={min(times)*1000:.3f}us max={max(times)*1000:.3f}us n={len(times)}")


if __name__ == "__main__":
    main()
