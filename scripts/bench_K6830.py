#!/usr/bin/env python3
"""K-6830 benchmark + ISA probe.

Compiles a small BF16 GEMM (default 64x64x4096) with whatever Triton is
installed in the env, dumps the AMDGCN ISA, counts ds_read / v_mfma /
ds_write occurrences and the longest run of contiguous v_mfma's
(higher = more clustering, lower = more interleaving), and benchmarks
TFLOPS against torch.matmul (hipBLASLt on ROCm).

JSON output is written to <out_dir>/bench_<label>.json so two runs
(unpatched / patched) can be diffed.
"""
import argparse
import glob
import json
import os
import re
import sys
import tempfile
import time

import torch
import triton
import triton.language as tl


def _full_triton_version():
    """Pip metadata version (e.g. '3.7.0+git3c71c5f3'); falls back to triton.__version__."""
    try:
        from importlib.metadata import version
        return version("triton")
    except Exception:
        return triton.__version__


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    sa_m, sa_k, sb_k, sb_n, sc_m, sc_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k
    b_ptrs = B_ptr + offs_k[:, None] * sb_k + offs_n[None, :] * sb_n
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * sa_k
        b_ptrs += BLOCK_K * sb_k
    c = acc.to(A_ptr.dtype.element_ty)
    c_ptrs = C_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
    tl.store(c_ptrs, c)


def longest_mfma_run(src: str) -> int:
    """Walk ISA line by line, return longest stretch of consecutive v_mfma
    instructions uninterrupted by ds_read/ds_write/global_load."""
    longest = 0
    cur = 0
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("v_mfma"):
            cur += 1
            longest = max(longest, cur)
        elif s.startswith("ds_read") or s.startswith("ds_write") or s.startswith("global_load"):
            cur = 0
    return longest


def count_interleave_pairs(src: str) -> int:
    """Adjacent (ds_read -> v_mfma) or (v_mfma -> ds_read) transitions in ISA.
    More transitions = more interleaving."""
    prev_kind = None
    transitions = 0
    for line in src.splitlines():
        s = line.strip()
        kind = None
        if s.startswith("ds_read"):
            kind = "lds"
        elif s.startswith("v_mfma"):
            kind = "mfma"
        if kind is not None and prev_kind is not None and kind != prev_kind:
            transitions += 1
        if kind is not None:
            prev_kind = kind
    return transitions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--ns", type=int, default=4)
    ap.add_argument("--block_k", type=int, default=64)
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = tempfile.mkdtemp(prefix=f"k6830_{args.label}_")
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"

    arch = torch.cuda.get_device_properties(0).gcnArchName
    print(f"label={args.label}  arch={arch}  triton={triton.__version__}  torch={torch.__version__}")
    print(f"shape=M{args.M}xN{args.N}xK{args.K} bf16  num_stages={args.ns}  block_k={args.block_k}")
    print(f"cache_dir={cache_dir}")

    M, N, K = args.M, args.N, args.K
    dt = torch.bfloat16
    torch.manual_seed(0)
    A = torch.randn((M, K), device="cuda", dtype=dt)
    B = torch.randn((K, N), device="cuda", dtype=dt)
    C = torch.empty((M, N), device="cuda", dtype=dt)

    BLOCK_M, BLOCK_N, BLOCK_K = M, N, args.block_k
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    def run_triton():
        matmul_kernel[grid](
            A, B, C, M, N, K,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, NUM_STAGES=args.ns,
        )
        return C

    # Warmup + numerical check
    run_triton()
    torch.cuda.synchronize()
    ref = (A.float() @ B.float()).to(dt)
    err_tri = (C - ref).abs().max().item()
    print(f"max_abs_err_triton={err_tri:.4f}")

    # Bench Triton
    ms_tri = triton.testing.do_bench(run_triton, warmup=args.warmup, rep=args.reps)
    flops = 2.0 * M * N * K
    tflops_tri = flops / (ms_tri * 1e-3) / 1e12
    print(f"triton:    {ms_tri*1e3:8.2f} us  {tflops_tri:6.2f} TFLOPS")

    # Bench torch.matmul (hipBLASLt baseline on ROCm)
    def run_torch():
        return torch.matmul(A, B)

    run_torch()
    torch.cuda.synchronize()
    ms_th = triton.testing.do_bench(run_torch, warmup=args.warmup, rep=args.reps)
    tflops_th = flops / (ms_th * 1e-3) / 1e12
    print(f"hipblaslt: {ms_th*1e3:8.2f} us  {tflops_th:6.2f} TFLOPS")

    # ISA dump check
    amdgcns = sorted(glob.glob(os.path.join(cache_dir, "**/*.amdgcn"), recursive=True))
    isa_summary = []
    for p in amdgcns:
        with open(p) as f:
            src = f.read()
        ds_read = src.count("ds_read")
        v_mfma = src.count("v_mfma")
        ds_write = src.count("ds_write")
        longest = longest_mfma_run(src)
        transitions = count_interleave_pairs(src)
        entry = {
            "path": p,
            "ds_read": ds_read,
            "v_mfma": v_mfma,
            "ds_write": ds_write,
            "longest_mfma_run": longest,
            "lds_mfma_transitions": transitions,
        }
        isa_summary.append(entry)
        print(f"isa {os.path.basename(p)}: ds_read={ds_read} v_mfma={v_mfma} "
              f"ds_write={ds_write} longest_mfma_run={longest} transitions={transitions}")

    out = {
        "label": args.label,
        "arch": arch,
        "triton_version": _full_triton_version(),
        "torch_version": torch.__version__,
        "shape": {"M": M, "N": N, "K": K, "dtype": "bf16",
                  "num_stages": args.ns, "block_k": args.block_k},
        "max_abs_err": err_tri,
        "triton_us": ms_tri * 1e3,
        "triton_tflops": tflops_tri,
        "hipblaslt_us": ms_th * 1e3,
        "hipblaslt_tflops": tflops_th,
        "isa": isa_summary,
    }
    out_path = os.path.join(args.out_dir, f"bench_{args.label}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
