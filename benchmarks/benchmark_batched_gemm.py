# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark batched GEMM: single-launch batched kernel vs Python for-loop.

Reports median TFLOPS for both paths and a ratio (loop-time / batched-time).
A ratio > 1.0 means the new batched kernel is faster than the legacy loop.
"""

import argparse
import json
import statistics

import torch
import tritonblas


def _bench_callable(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return statistics.median(times) / 1000.0  # seconds


def _tflops(batch, m, n, k, seconds):
    flops = 2.0 * batch * m * n * k
    return flops / seconds / 1e12


def _python_loop(a, b):
    """Legacy dispatch: one tritonblas.matmul per batch slice."""
    outs = [tritonblas.matmul(a[i], b[i]) for i in range(a.shape[0])]
    return torch.stack(outs, dim=0)


def _torch_bmm(a, b):
    return torch.bmm(a, b)


def _benchmark_shape(batch, m, n, k, dtype, warmup, iters):
    a = torch.randn(batch, m, k, device="cuda", dtype=torch.float32).to(dtype)
    b = torch.randn(batch, k, n, device="cuda", dtype=torch.float32).to(dtype)

    # Warm up Triton compile cache for both paths.
    _ = tritonblas.bmm(a, b)
    _ = _python_loop(a, b)
    torch.cuda.synchronize()

    t_batched = _bench_callable(lambda: tritonblas.bmm(a, b), warmup, iters)
    t_loop = _bench_callable(lambda: _python_loop(a, b), warmup, iters)
    t_torch = _bench_callable(lambda: _torch_bmm(a, b), warmup, iters)

    # Correctness check (fp32 reference).
    ref = torch.matmul(a.float(), b.float())
    out = tritonblas.bmm(a, b).float()
    rel_err = (out - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)

    return {
        "batch": batch, "m": m, "n": n, "k": k, "dtype": str(dtype),
        "batched_s": t_batched,
        "loop_s": t_loop,
        "torch_bmm_s": t_torch,
        "batched_tflops": _tflops(batch, m, n, k, t_batched),
        "loop_tflops": _tflops(batch, m, n, k, t_loop),
        "torch_bmm_tflops": _tflops(batch, m, n, k, t_torch),
        "ratio_loop_over_batched": t_loop / t_batched,
        "ratio_batched_over_torch": t_torch / t_batched,
        "rel_err_max": rel_err,
    }


# Targets from the K-678 problem statement: top-2 batched residuals from K-654
# plus the K-545 reference shape (rank-2; included as a launch-overhead control).
DEFAULT_SHAPES = [
    # batch, M, N, K, dtype
    (8, 2048, 2048, 2048, torch.float16),    # 2048^3 fp16 batch (K-654 top-1)
    (8, 1024, 1024, 1024, torch.bfloat16),   # 1024^3 bf16 batch (K-654 top-2)
    (4, 1024, 8192, 8192, torch.bfloat16),   # K-545 reference shape, batched
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out", default=None,
                   help="Write JSONL results to this path.")
    p.add_argument("--shapes", default=None,
                   help="Optional path to a JSON list of [batch, M, N, K, dtype_str].")
    args = p.parse_args()

    if args.shapes:
        with open(args.shapes) as f:
            raw = json.load(f)
        shapes = []
        for entry in raw:
            batch, m, n, k, dt = entry
            dtype = getattr(torch, dt) if isinstance(dt, str) else dt
            shapes.append((batch, m, n, k, dtype))
    else:
        shapes = DEFAULT_SHAPES

    results = []
    for batch, m, n, k, dtype in shapes:
        r = _benchmark_shape(batch, m, n, k, dtype, args.warmup, args.iters)
        results.append(r)
        print(
            f"[batch={r['batch']:3d} M={r['m']:5d} N={r['n']:5d} K={r['k']:5d} "
            f"{r['dtype']:>20}] "
            f"batched={r['batched_tflops']:7.2f} TFLOPS  "
            f"loop={r['loop_tflops']:7.2f} TFLOPS  "
            f"torch.bmm={r['torch_bmm_tflops']:7.2f} TFLOPS  "
            f"loop/batched={r['ratio_loop_over_batched']:5.2f}x  "
            f"rel_err={r['rel_err_max']:.2e}"
        )

    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
