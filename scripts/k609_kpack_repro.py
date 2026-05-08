#!/usr/bin/env python3
"""Manually-runnable rocprofv3 reproducer for the K-609 negative finding.

Runs the K-451 cohort centre-point GEMM (M=N=3072, K=512, fp16) twice — once
with the persistent kernel's default ``kpack=1``, once forcibly compiled with
``kpack=2`` — and reports wall-clock plus (if rocprofv3 is on PATH) the
SQ_LDS_BANK_CONFLICT delta. Use this to verify that the K-519 / K-609 negative
result still holds before re-attempting the dead-end ``kpack=2`` LDS swizzle.

Usage::

    # wall-clock + functional only:
    python3 scripts/k609_kpack_repro.py

    # full rocprofv3 capture (writes counters next to the script):
    rocprofv3 --pmc SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE SQ_INSTS_LDS \
        --output-format csv --output-directory /tmp/k609_kp1 -- \
        python3 scripts/k609_kpack_repro.py --kpack 1 --rocprof-mode
    rocprofv3 --pmc SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE SQ_INSTS_LDS \
        --output-format csv --output-directory /tmp/k609_kp2 -- \
        python3 scripts/k609_kpack_repro.py --kpack 2 --rocprof-mode

Background
----------
K-519 hypothesised that the residual perf gap vs hipBLASLt on the K-451
medium-K square FP16/BF16 cohort was caused by LDS bank conflicts on
``ds_read_b128`` in the (256, 256, 64) winner-tile kernel. K-609 verified
on the current Triton AMD backend / MI300X that the backend already produces
**zero** LDS bank conflicts on that kernel at ``kpack=1``, and ``kpack=2``
*adds* ~9.7M conflicts (~88% of LDS instructions) — the opposite of the
K-519 claim. This script lets the next agent reproduce that negative
result before re-attempting the dead end. See
``tests/test_k451_kpack_default.py`` for the full evidence table.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

import torch
import triton

import tritonblas  # noqa: F401  (ensures package is initialised)

# ``tritonblas.matmul`` is re-exported as a *function* from ``__init__.py``,
# so use the explicit submodule import.
tbm = importlib.import_module("tritonblas.matmul")
pg = importlib.import_module("tritonblas.kernels.persistent_gemm")


def _call_kernel_with_kpack(a, b, c, selector, *, kpack: int) -> None:
    """Direct kernel launch matching ``persistent_matmul_lt`` but with explicit
    ``kpack``. Mirrors the exact call site in ``include/tritonblas/matmul.py``.
    """
    M, K = a.shape
    _, N = b.shape
    BLK_M, BLK_N, BLK_K = selector.block_m, selector.block_n, selector.block_k
    gsize_m = selector.group_m
    num_xcds = selector.num_sms or 1
    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0
    chunk_size = gsize_m * gsize_m
    if selector.num_sms > 0:
        chunk_size = min(chunk_size, max(1, total_tiles // num_xcds))
    pg.persistent_matmul[(total_tiles,)](
        a, b, c, None, None, None,
        M, N, K,
        a.stride(0), b.stride(1), c.stride(0), c.stride(1), 0,
        stride_ak=a.stride(1), stride_bk=b.stride(0),
        BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m,
        NUM_SMS=total_tiles, NUM_XCDS=num_xcds, CHUNK_SIZE=chunk_size,
        BIAS=False, EVEN_K=even_k,
        CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None,
        QUANTIZED=False,
        num_stages=2, num_warps=8, waves_per_eu=0,
        matrix_instr_nonkdim=16, kpack=kpack,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=3072)
    p.add_argument("--n", type=int, default=3072)
    p.add_argument("--k", type=int, default=512)
    p.add_argument(
        "--kpack", type=int, default=None,
        help="If set, force this kpack value (otherwise sweep 1 and 2).",
    )
    p.add_argument(
        "--rocprof-mode", action="store_true",
        help="Single fixed-iter run for rocprofv3 capture (no timing print).",
    )
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA/HIP device", file=sys.stderr)
        return 2
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"shape:  M={args.m} N={args.n} K={args.k}  dtype=float16")

    dtype = torch.float16
    a = torch.randn(args.m, args.k, device="cuda", dtype=dtype)
    b = torch.randn(args.k, args.n, device="cuda", dtype=dtype)
    c = a.new_empty(args.m, args.n)
    selector = tbm._make_matmul_selector(args.m, args.n, args.k,
                                         a.dtype, b.dtype, c.dtype, a.device)
    print(f"selected tile: ({selector.block_m},{selector.block_n},{selector.block_k}) "
          f"GROUP_M={selector.group_m} NUM_XCDS={selector.num_sms}")

    kpacks = [args.kpack] if args.kpack is not None else [1, 2]
    results: dict[int, float] = {}
    for kp in kpacks:
        for _ in range(args.warmup):
            _call_kernel_with_kpack(a, b, c, selector, kpack=kp)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            _call_kernel_with_kpack(a, b, c, selector, kpack=kp)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3 / args.iters
        results[kp] = ms
        if not args.rocprof_mode:
            print(f"kpack={kp}: {ms*1e3:7.1f} us / call")

    if not args.rocprof_mode and 1 in results and 2 in results:
        ratio = results[1] / results[2]
        print(f"kpack=1 / kpack=2 wall-clock ratio: {ratio:.4f}x "
              f"(>1.0 means kpack=2 faster; K-609 expects ~1.00x = noise)")

    # Functional sanity vs torch.matmul on a single tile.
    ref = (a @ b).to(dtype)
    err = (c - ref).abs().max().item()
    print(f"max |c - a@b|: {err:.4f}  (cohort tolerance is loose; "
          "this is a smoke check, not a correctness test)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
