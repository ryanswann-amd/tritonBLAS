"""
Benchmark: fused (bias + GEMM + activation) vs unfused (3-call sequence).

Measures the saving K-491 attributed to epilogue fusion: the unfused tritonblas
sequence (matmul -> bias -> activation as 3 kernels) vs the new fused
``tritonblas.addmm(..., activation=ACT)`` (single kernel, activation in
registers).

Outputs a CSV table to stdout (and to ``--out`` if given) with columns:
    name, M, N, K, dtype, activation, t_unfused_us, t_fused_us,
    fused_speedup, throughput_TFLOPS_fused

Usage:
    python3 benchmarks/addmm_activation_fusion.py
    python3 benchmarks/addmm_activation_fusion.py --activations gelu silu --out results.csv
"""
import argparse
import csv
import sys
import torch
import torch.nn.functional as F

import tritonblas


# Workload-realistic shapes (transformer FFN / attention-out / BERT/T5).
# These mirror the K-491 evidence table at workspaces/K-491/output/epilogue_gap_table.csv.
SHAPES = [
    # name,                         M,    N,     K
    ("llama2-7b FFN1 prefill BT=2048", 2048, 11008, 4096),
    ("llama2-7b FFN1 prefill BT=512",  512,  11008, 4096),
    ("llama2-7b FFN1 prefill BT=128",  128,  11008, 4096),
    ("llama2-7b FFN2 prefill BT=2048", 2048,  4096, 11008),
    ("llama2-7b attn-out BT=2048",     2048,  4096,  4096),
    ("BERT-base FFN1 BT=512",          512,   3072,   768),
    ("BERT-base FFN2 BT=512",          512,    768,  3072),
    ("T5-large FFN1 BT=512",           512,   4096,  1024),
]

# Activations we ship.  "gelu" = tanh-approx (most common in inference).
DEFAULT_ACTIVATIONS = ["relu", "gelu", "silu"]


def _torch_act(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return lambda x: F.gelu(x, approximate="tanh")
    if activation == "gelu_exact":
        return F.gelu
    if activation == "sigmoid":
        return torch.sigmoid
    if activation in ("silu", "swish"):
        return F.silu
    raise ValueError(activation)


def _bench(fn, warmup=10, rep=50):
    """Median-of-rep timer in microseconds (CUDA graphs would be cleaner but we
    keep it simple; rep is enough to wash out launch jitter)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Use cuda events for low-overhead timing.
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    times_ms.sort()
    median = times_ms[len(times_ms) // 2]
    return median * 1000.0  # -> us


def _bench_unfused(a, b, bias, act_fn):
    """Three calls: matmul -> add bias -> activation."""
    def run():
        c = tritonblas.matmul(a, b)
        c = c + bias
        c = act_fn(c)
        return c
    return _bench(run)


def _bench_fused(a, b, bias, activation):
    """Single fused call."""
    def run():
        return tritonblas.addmm(bias, a, b, activation=activation)
    return _bench(run)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--activations", nargs="+", default=DEFAULT_ACTIVATIONS,
                   choices=sorted(tritonblas.SUPPORTED_ACTIVATIONS - {"none"}))
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    p.add_argument("--out", default=None, help="Optional CSV output path")
    p.add_argument("--shapes", default=None,
                   help="Optional CSV of M,N,K triples; default = transformer suite")
    args = p.parse_args(argv)

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    shapes = SHAPES
    if args.shapes is not None:
        shapes = []
        for line in args.shapes.split(";"):
            m, n, k = (int(x) for x in line.split(","))
            shapes.append((f"custom_{m}x{n}x{k}", m, n, k))

    rows = []
    fieldnames = ["name", "M", "N", "K", "dtype", "activation",
                  "t_unfused_us", "t_fused_us", "fused_speedup",
                  "throughput_TFLOPS_fused"]
    print(",".join(fieldnames))

    for name, M, N, K in shapes:
        a = torch.randn(M, K, device='cuda', dtype=dtype)
        b = torch.randn(K, N, device='cuda', dtype=dtype)
        bias = torch.randn(N, device='cuda', dtype=dtype)

        for activation in args.activations:
            act_fn = _torch_act(activation)
            t_unfused = _bench_unfused(a, b, bias, act_fn)
            t_fused = _bench_fused(a, b, bias, activation)

            flops = 2.0 * M * N * K
            throughput = flops / (t_fused * 1e-6) / 1e12

            row = dict(
                name=name, M=M, N=N, K=K, dtype=args.dtype,
                activation=activation,
                t_unfused_us=round(t_unfused, 2),
                t_fused_us=round(t_fused, 2),
                fused_speedup=round(t_unfused / t_fused, 3),
                throughput_TFLOPS_fused=round(throughput, 2),
            )
            rows.append(row)
            print(",".join(str(row[f]) for f in fieldnames))

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"# wrote {len(rows)} rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
