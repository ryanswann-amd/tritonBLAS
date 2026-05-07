"""Bypass the kpack gate and measure raw kpack=2 latency at every K.

The companion to ``bench_kpack_cohort.py``.  The cohort benchmark goes
through the gate, so for K < ``_DEFAULT_KPACK2_K_MIN`` it never actually
exercises kpack=2 -- the gate clamps the request to 1 and the latency is
identical to the kpack=1 row.

This benchmark patches ``_kpack_for_selector`` to a constant ``2`` so we
get the raw kpack=2 number at every K, including K=512 and K=768 where
the gate normally clamps.  Combined with the gated cohort CSV this gives
the per-shape ``kpack1 -> kpack2`` delta at every K and
demonstrates that on the small-K cohort the gate prevents a real
regression rather than masking a zero-cost change.

Output: CSV identical schema to bench_kpack_cohort.py, but with
``kpack_effective`` always equal to the requested ``kpack_req`` (no
clamping).

Run:
  python3 benchmarks/bench_kpack_gate_off.py --csv branch_cohort_gate_off.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

import torch

import tritonblas
from tritonblas import OrigamiMatmulSelector
from tritonblas.matmul import persistent_matmul_lt


def _make_selector(M, N, K, dtype, kpack):
    sel = OrigamiMatmulSelector(
        M, N, K,
        a_dtype=dtype, b_dtype=dtype, out_dtype=dtype,
        device=torch.device("cuda:0"),
    )
    sel.kpack = int(kpack)
    return sel


def _bench_one(M, N, K, dtype, requested_kpack, warmup=10, iters=50):
    """Bench with the gate disabled so requested_kpack reaches the kernel verbatim.

    Timing methodology matches ``bench_kpack_cohort.py`` exactly (one event
    pair around the iters loop) so latencies from the two CSVs can be
    compared row-for-row -- the only difference between the two benches is
    whether the gate is active.
    """
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    c = torch.empty(M, N, device="cuda", dtype=dtype)
    sel = _make_selector(M, N, K, dtype, requested_kpack)

    # Patch the gate to identity for the duration of this measurement.  The
    # finally clause restores the original even if a kernel raises so the
    # matmul module is never left in a broken state.
    mm = sys.modules["tritonblas.matmul"]
    orig = mm._kpack_for_selector
    try:
        mm._kpack_for_selector = lambda sel, K: int(requested_kpack)
        for _ in range(warmup):
            persistent_matmul_lt(a, b, c, sel)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            persistent_matmul_lt(a, b, c, sel)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
    finally:
        mm._kpack_for_selector = orig

    ref = torch.matmul(a, b)
    ok = torch.allclose(c, ref, atol=2e-2, rtol=2e-2)
    return requested_kpack, ms, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/dev/stdout")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    mn_sweep = [512, 1024, 1536, 2048]
    k_sweep = [512, 768, 1024, 1536, 2048, 4096, 8192]
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

    if args.csv != "/dev/stdout":
        with open(args.csv, "w") as f:
            f.write("M,N,K,kpack_req,kpack_effective,latency_ms,gflops,correct\n")
            f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
