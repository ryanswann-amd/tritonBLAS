"""Benchmark the K-539 small-K cohort with kpack gating.

For each (M,N,K) and each kpack request in {1, 2}, we measure
``persistent_matmul_lt`` end-to-end latency on MI300X.  The gate guarantees
that a request for kpack=2 only takes effect when K >= TRITONBLAS_KPACK2_K_MIN
(default 1024).  We use a wide K-sweep so the boundary is visible.

For every shape we report:
  - kpack=1 latency (always-safe baseline that the gate guarantees for small K)
  - kpack=2 *requested* (effective kpack after gating)
  - speedup of (kpack=2 requested) vs (kpack=1) — at small K the gate clamps
    to 1, so this ratio is ~1.0 (no regression) instead of -25..-30%.

Output: CSV to stdout AND to --csv path.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch

import tritonblas
from tritonblas import OrigamiMatmulSelector
from tritonblas.matmul import _kpack_for_selector, persistent_matmul_lt


def _make_selector(M, N, K, dtype, kpack):
    sel = OrigamiMatmulSelector(
        M, N, K,
        a_dtype=dtype, b_dtype=dtype, out_dtype=dtype,
        device=torch.device("cuda:0"),
    )
    sel.kpack = int(kpack)
    return sel


def _bench_one(M, N, K, dtype, requested_kpack, warmup=10, iters=50):
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    c = torch.empty(M, N, device="cuda", dtype=dtype)
    sel = _make_selector(M, N, K, dtype, requested_kpack)
    effective = _kpack_for_selector(sel, K)

    # Warmup (also triggers JIT compile)
    for _ in range(warmup):
        persistent_matmul_lt(a, b, c, sel)
    torch.cuda.synchronize()

    # Timed run with cuda events for accuracy
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        persistent_matmul_lt(a, b, c, sel)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters

    # Numerical check — small-K cohort uses bf16, ref uses fp32 promotion
    ref = torch.matmul(a, b)
    ok = torch.allclose(c, ref, atol=2e-2, rtol=2e-2)
    return effective, ms, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/dev/stdout")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    # K-539 small-K cohort: M,N in {512..2048}, K=512.
    # We also include K in {768, 1024, 1536, 2048} so the boundary
    # the task asked us to confirm (1024) is visible in the data.
    mn_sweep = [512, 1024, 1536, 2048]
    k_sweep = [512, 768, 1024, 1536, 2048]
    dtype = torch.bfloat16

    rows = []
    print(
        "M,N,K,kpack_req,kpack_effective,latency_ms,gflops,correct",
        flush=True,
    )

    for M in mn_sweep:
        for N in mn_sweep:
            for K in k_sweep:
                for req in (1, 2):
                    eff, ms, ok = _bench_one(
                        M, N, K, dtype, req,
                        warmup=args.warmup, iters=args.iters,
                    )
                    gflops = 2.0 * M * N * K / (ms * 1e-3) / 1e9
                    line = (
                        f"{M},{N},{K},{req},{eff},{ms:.4f},{gflops:.1f},{int(ok)}"
                    )
                    print(line, flush=True)
                    rows.append(line)

    # Persist
    if args.csv != "/dev/stdout":
        with open(args.csv, "w") as f:
            f.write("M,N,K,kpack_req,kpack_effective,latency_ms,gflops,correct\n")
            f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
