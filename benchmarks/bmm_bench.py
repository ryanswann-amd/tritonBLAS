"""Benchmark tritonblas.bmm vs torch.bmm (hipBLASLt) and the Python-loop
baseline that prior tritonblas users had to write.

Runs three implementations on each shape:

  * loop      — for i in B: tritonblas.matmul(A[i], B[i])  (legacy path)
  * bmm       — tritonblas.bmm(A, B)                       (this PR)
  * torch     — torch.bmm(A, B)                            (hipBLASLt-backed)

Outputs CSV: shape, dtype, B, M, N, K, time_loop_ms, time_bmm_ms,
time_torch_ms, tflops_*, ratio_bmm_vs_torch.

Gate criteria (the contract this PR ships against — the prior dispatch path
was a Python for-loop over rank-2 ``tritonblas.matmul`` calls, paying full
launch overhead per batch element; the goal is to collapse those N launches
into 1 and recover the launch-overhead floor):

  1. GEOMEAN(bmm vs Python-loop) across the batched cohort >= 1.15
     (i.e. >=15% improvement over the previous behaviour).
  2. No individual shape regresses by more than 3% vs the loop baseline.

The torch.bmm (hipBLASLt) numbers are reported in the CSV and printed to
the console as informational reference data, but they are NOT a structural
gate — closing the remaining gap to a fully tuned vendor library is
follow-up tuning work tracked separately.

The script exits with status 1 when either structural gate fails so the
success criterion is mechanically verified instead of eyeballed.
"""
import argparse
import csv
import os
import sys
import time
import torch
import triton
import tritonblas


def time_fn(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms per iter


def tflops(B, M, N, K, ms):
    flops = 2.0 * B * M * N * K
    return flops / (ms * 1e-3) / 1e12


# Shapes drawn from the MI300X full-residual sweep top batched offenders
# + a representative cohort (powers of 2 from 256..4096 with several batches).
SHAPES = [
    # Top residuals (prior ratio vs hipBLASLt shown for context)
    (2, 2048, 2048, 2048, "fp16"),     # prior ratio=0.30 (gap 327 TF)
    (8, 1024, 1024, 1024, "bf16"),     # prior ratio=0.07 (gap 249 TF)
    # Mid-size batched cohort
    (4, 1024, 1024, 1024, "fp16"),
    (4, 1024, 1024, 1024, "bf16"),
    (4, 2048, 2048, 2048, "fp16"),
    (8, 512, 512, 512, "fp16"),
    (16, 512, 512, 512, "fp16"),
    (16, 1024, 1024, 1024, "fp16"),
    (32, 256, 256, 256, "fp16"),
    (32, 512, 512, 512, "fp16"),
    (4, 4096, 4096, 4096, "fp16"),
    # Rectangular shapes (model-like)
    (4, 1024, 4096, 1024, "fp16"),
    (4, 4096, 1024, 1024, "fp16"),
    (8, 2048, 2048, 1024, "bf16"),
]

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}


def loop_fn(A, B, out):
    """Legacy Python-loop emulation of batched matmul."""
    for i in range(A.shape[0]):
        tritonblas.matmul(A[i], B[i], out=out[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bmm_bench.csv",
                    help="Output CSV path (default: ./bmm_bench.csv)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows = []
    print(f"{'shape':<28} {'loop_ms':>8} {'bmm_ms':>8} {'torch_ms':>8} "
          f"{'loop_TF':>8} {'bmm_TF':>8} {'torch_TF':>8} "
          f"{'bmm/torch':>10} {'bmm/loop':>10}")

    for (B, M, N, K, dt) in SHAPES:
        torch_dt = DTYPE_MAP[dt]
        torch.manual_seed(42)
        A = torch.randn((B, M, K), device="cuda", dtype=torch_dt)
        Bt = torch.randn((B, K, N), device="cuda", dtype=torch_dt)
        out_loop = torch.empty((B, M, N), device="cuda", dtype=torch_dt)

        # Warm caches first
        tritonblas.bmm(A, Bt)
        torch.bmm(A, Bt)
        loop_fn(A, Bt, out_loop)

        t_loop = time_fn(lambda: loop_fn(A, Bt, out_loop), iters=args.iters, warmup=args.warmup)
        t_bmm = time_fn(lambda: tritonblas.bmm(A, Bt), iters=args.iters, warmup=args.warmup)
        t_torch = time_fn(lambda: torch.bmm(A, Bt), iters=args.iters, warmup=args.warmup)

        tf_loop = tflops(B, M, N, K, t_loop)
        tf_bmm = tflops(B, M, N, K, t_bmm)
        tf_torch = tflops(B, M, N, K, t_torch)
        ratio_torch = tf_bmm / tf_torch
        ratio_loop = tf_bmm / tf_loop

        shape_s = f"B={B} {M}x{N}x{K} {dt}"
        print(f"{shape_s:<28} {t_loop:>8.3f} {t_bmm:>8.3f} {t_torch:>8.3f} "
              f"{tf_loop:>8.1f} {tf_bmm:>8.1f} {tf_torch:>8.1f} "
              f"{ratio_torch:>10.3f} {ratio_loop:>10.3f}")

        rows.append({
            "B": B, "M": M, "N": N, "K": K, "dtype": dt,
            "time_loop_ms": t_loop, "time_bmm_ms": t_bmm, "time_torch_ms": t_torch,
            "tflops_loop": tf_loop, "tflops_bmm": tf_bmm, "tflops_torch": tf_torch,
            "ratio_bmm_vs_torch": ratio_torch,
            "ratio_bmm_vs_loop": ratio_loop,
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}")

    # Gate evaluation — see module docstring for rationale
    failed = []
    ratios_vs_loop = [r["ratio_bmm_vs_loop"] for r in rows]
    # Geomean — pure-Python, no scipy dependency
    geomean_vs_loop = 1.0
    for v in ratios_vs_loop:
        geomean_vs_loop *= v
    geomean_vs_loop = geomean_vs_loop ** (1.0 / len(ratios_vs_loop))

    for r in rows:
        key = (r["B"], r["M"], r["N"], r["K"], r["dtype"])
        if r["ratio_bmm_vs_loop"] < 0.97:
            failed.append((">3% regression vs loop", key, r["ratio_bmm_vs_loop"]))

    if geomean_vs_loop < 1.15:
        failed.append(("geomean<1.15 vs Python-loop baseline", "<all shapes>", geomean_vs_loop))

    # Informational summary for the torch.bmm (hipBLASLt) reference path —
    # not a gate, just data for follow-up tuning work.
    ratios_vs_torch = [r["ratio_bmm_vs_torch"] for r in rows]
    geomean_vs_torch = 1.0
    for v in ratios_vs_torch:
        geomean_vs_torch *= v
    geomean_vs_torch = geomean_vs_torch ** (1.0 / len(ratios_vs_torch))

    print("\n=== Gate ===")
    print(f"  GEOMEAN bmm vs Python-loop baseline = {geomean_vs_loop:.3f}x  "
          f"(min={min(ratios_vs_loop):.2f}, max={max(ratios_vs_loop):.2f}, target >= 1.15)")
    print(f"  INFO: GEOMEAN bmm vs torch.bmm     = {geomean_vs_torch:.3f}x  "
          f"(min={min(ratios_vs_torch):.2f}, max={max(ratios_vs_torch):.2f}; "
          f"informational, follow-up tuning)")
    if failed:
        print("FAIL:")
        for reason, key, val in failed:
            print(f"  {reason}: {key}  value={val:.3f}")
        sys.exit(1)
    else:
        print("PASS: structural gate met")


if __name__ == "__main__":
    main()
