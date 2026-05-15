#!/usr/bin/env python3
"""K-6830 tritonblas benchmark.

Bench tritonblas.matmul (the production GEMM kernel) vs torch.matmul
(hipBLASLt) on a sweep of small-N shapes that K-6779/K-6792/K-6723
called out as the lgkmcnt-stalled regime.

Writes JSON keyed by --label so unpatched/patched runs can be diffed.
"""
import argparse
import glob
import json
import os
import sys
import tempfile

import torch
import triton
import tritonblas


def _full_triton_version():
    """Pip metadata version (e.g. '3.7.0+git3c71c5f3'); falls back to triton.__version__."""
    try:
        from importlib.metadata import version
        return version("triton")
    except Exception:
        return triton.__version__


def longest_mfma_run(src):
    longest, cur = 0, 0
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("v_mfma"):
            cur += 1
            longest = max(longest, cur)
        elif s.startswith("ds_read") or s.startswith("ds_write") or s.startswith("global_load"):
            cur = 0
    return longest


def lds_mfma_transitions(src):
    prev, n = None, 0
    for line in src.splitlines():
        s = line.strip()
        kind = "lds" if s.startswith("ds_read") else ("mfma" if s.startswith("v_mfma") else None)
        if kind and prev and kind != prev:
            n += 1
        if kind:
            prev = kind
    return n


def bench_shape(M, N, K, dtype, reps, warmup):
    A = torch.randn((M, K), device="cuda", dtype=dtype)
    B = torch.randn((K, N), device="cuda", dtype=dtype)

    # tritonblas.matmul
    def run_tb():
        return tritonblas.matmul(A, B)

    # warmup + numerics
    Ctb = run_tb()
    torch.cuda.synchronize()
    ref = (A.float() @ B.float()).to(dtype)
    err_tb = (Ctb - ref).abs().max().item()

    ms_tb = triton.testing.do_bench(run_tb, warmup=warmup, rep=reps)
    flops = 2.0 * M * N * K
    tflops_tb = flops / (ms_tb * 1e-3) / 1e12

    # torch.matmul (hipBLASLt)
    def run_th():
        return torch.matmul(A, B)

    run_th()
    torch.cuda.synchronize()
    ms_th = triton.testing.do_bench(run_th, warmup=warmup, rep=reps)
    tflops_th = flops / (ms_th * 1e-3) / 1e12

    return {
        "M": M, "N": N, "K": K,
        "dtype": str(dtype).split(".")[-1],
        "tritonblas_us": ms_tb * 1e3,
        "tritonblas_tflops": tflops_tb,
        "hipblaslt_us": ms_th * 1e3,
        "hipblaslt_tflops": tflops_th,
        "max_abs_err_tritonblas": err_tb,
        "speedup_vs_hipblaslt": tflops_tb / tflops_th,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = tempfile.mkdtemp(prefix=f"k6830_tb_{args.label}_")
    os.environ["TRITON_CACHE_DIR"] = cache_dir

    arch = torch.cuda.get_device_properties(0).gcnArchName
    print(f"label={args.label}  arch={arch}  triton={_full_triton_version()}  torch={torch.__version__}")

    # K-6779/K-6723 lgkmcnt-stalled regime: small N, big K, BF16
    shapes = [
        (64, 64, 4096),
        (64, 64, 8192),
        (64, 64, 16384),
        (128, 128, 4096),
        (256, 256, 4096),
        (1024, 1024, 4096),
    ]

    results = []
    for (M, N, K) in shapes:
        try:
            r = bench_shape(M, N, K, torch.bfloat16, args.reps, args.warmup)
            results.append(r)
            print(f"M={M:5d} N={N:5d} K={K:6d}  tb={r['tritonblas_tflops']:7.2f} TFLOPS "
                  f"({r['tritonblas_us']:7.2f}us)  hbl={r['hipblaslt_tflops']:7.2f} TFLOPS "
                  f"({r['hipblaslt_us']:7.2f}us)  speedup={r['speedup_vs_hipblaslt']:.3f}  "
                  f"max_err={r['max_abs_err_tritonblas']:.4f}")
        except Exception as e:
            print(f"M={M} N={N} K={K} FAILED: {e}")
            results.append({"M": M, "N": N, "K": K, "error": str(e)})

    # ISA dump for first cached kernel
    isa_summary = []
    amdgcns = sorted(glob.glob(os.path.join(cache_dir, "**/*.amdgcn"), recursive=True))
    print(f"found {len(amdgcns)} amdgcn dumps in {cache_dir}")
    for p in amdgcns[:6]:  # cap at 6 to keep JSON small
        with open(p) as f:
            src = f.read()
        ds_read = src.count("ds_read")
        v_mfma = src.count("v_mfma")
        ds_write = src.count("ds_write")
        longest = longest_mfma_run(src)
        trans = lds_mfma_transitions(src)
        isa_summary.append({
            "name": os.path.basename(p),
            "ds_read": ds_read, "v_mfma": v_mfma, "ds_write": ds_write,
            "longest_mfma_run": longest, "lds_mfma_transitions": trans,
        })
        print(f"isa {os.path.basename(p)}: ds_read={ds_read} v_mfma={v_mfma} "
              f"ds_write={ds_write} longest_mfma_run={longest} transitions={trans}")

    out = {
        "label": args.label,
        "arch": arch,
        "triton_version": _full_triton_version(),
        "torch_version": torch.__version__,
        "results": results,
        "isa": isa_summary,
        "cache_dir": cache_dir,
    }
    out_path = os.path.join(args.out_dir, f"bench_tb_{args.label}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
