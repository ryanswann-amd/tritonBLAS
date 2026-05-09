"""K-1804 N-padding prototype harness (standalone — no tritonblas source change).

Tests the hypothesis from K-1795 that wave-misaligned N (∈{48,80,192}) can be
mitigated by padding N to the next wave-aligned multiple of {64,128,256} and
slicing the output back. Padding is applied **externally in this harness** — no
gate is added to tritonblas.matmul, addressing the K-1804-r1 reviewer concern
about dead env-gated dispatch logic in the hot path.

Cohort gating used here matches the PRD:
    M >= 2048  AND  K in [4096, 16384]  AND  dtype in {bf16, fp16}
    AND  (N % 64 != 0  OR  N % 128 != 0  OR  N % 256 != 0)
The gate now pads N to the next multiple of 64, 128, OR 256 (the smallest
wave-aligned multiple > N). This addresses the K-1804-r1 Skeptic concern that
N=192 is wave-misaligned per K-1795 yet 192%64==0 silently no-ops a 64-only
gate.

The harness:
  1. Asserts bit-exactness of (TB on padded B, then sliced) vs (TB on raw B)
     for every gated cell at the start of the run (correctness gate).
  2. Times the unpadded vs padded path with paired n=30 trials and shuffled
     order, plus a hipBLASLt baseline for the ratio.
  3. Writes a CSV row per cell and a JSON summary with cohort gmeans.

Usage:
    python3 scripts/bench_n_pad.py --out output/bench_n_pad_bf16.csv --n 30
"""
import argparse
import csv
import json
import math
import os
import statistics
import sys
import time

import torch

# K-1795 12-cell cohort (M=4096, mid-K)
COHORT_BF16 = [
    (4096,  64, 4096),  (4096,  64, 8192),  (4096,  64, 16384),
    (4096, 128, 4096),  (4096, 128, 8192),  (4096, 128, 16384),
    (4096,  48, 4096),  (4096,  48, 8192),
    (4096,  80, 4096),  (4096,  80, 8192),
    (4096, 192, 4096),  (4096, 192, 8192),
]

# Aligned-control set (per K-1795: N is wave-aligned power of 2)
ALIGNED_NS = {64, 128, 256, 512}


def _pad_n_to_next_wave(N):
    """Round N up to the smallest BLOCK_N tile size from {64, 128, 256} that
    fits N exactly, else round up to next 64 for N > 256.

    Semantics — turn a partial-tile shape into a full-tile shape:
        N <= 64  → 64    (e.g. N=48 → 64)
        N <= 128 → 128   (e.g. N=80 → 128)
        N <= 256 → 256   (e.g. N=192 → 256)
        N >  256 → round up to next multiple of 64 (no large padding)
    No-ops on already-tile-sized N: 64,128,256,320,384,...,N (when N%64==0).
    """
    for unit in (64, 128, 256):
        if N <= unit:
            return unit
    return ((N + 63) // 64) * 64


def _gate_triggered(M, N, K, dtype):
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    if M < 2048 or not (4096 <= K <= 16384):
        return False
    return _pad_n_to_next_wave(N) != N


def _is_misaligned(N):
    return N not in ALIGNED_NS


def _sync():
    torch.cuda.synchronize()


def _gmean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def tb_padded(a, b, M, N, K, out_buf):
    """Externally-padded TB matmul. Allocates padded B/out, calls TB, slices.

    This is the prototype dispatcher behavior, applied from the caller side so
    that the production tritonblas.matmul stays untouched.
    """
    import tritonblas
    n_pad = _pad_n_to_next_wave(N)
    if n_pad == N:
        tritonblas.matmul(a, b, out=out_buf)
        return out_buf
    b_pad = b.new_zeros(K, n_pad)
    b_pad[:, :N] = b
    out_pad = a.new_empty(M, n_pad)
    tritonblas.matmul(a, b_pad, out=out_pad)
    out_buf.copy_(out_pad[:, :N])
    return out_buf


def _paired_bench(engines, n, warmup, seed=0):
    import random as _random
    rng = _random.Random(seed)
    names = list(engines.keys())
    times = {nm: [] for nm in names}

    for _ in range(warmup):
        order = list(names)
        rng.shuffle(order)
        for nm in order:
            engines[nm]()
    _sync()

    for _ in range(n):
        order = list(names)
        rng.shuffle(order)
        starts = {nm: torch.cuda.Event(enable_timing=True) for nm in order}
        ends = {nm: torch.cuda.Event(enable_timing=True) for nm in order}
        for nm in order:
            starts[nm].record()
            engines[nm]()
            ends[nm].record()
        _sync()
        for nm in order:
            times[nm].append(starts[nm].elapsed_time(ends[nm]) / 1000.0)
    return times


def _bit_exact_check(a, b, M, N, K, dtype):
    """Assert TB-padded output is bit-identical to TB-unpadded for this cell."""
    import tritonblas
    out_raw = torch.empty(M, N, device=a.device, dtype=dtype)
    out_pad = torch.empty(M, N, device=a.device, dtype=dtype)
    tritonblas.matmul(a, b, out=out_raw)
    tb_padded(a, b, M, N, K, out_pad)
    _sync()
    if not torch.equal(out_raw, out_pad):
        diff = (out_raw.float() - out_pad.float()).abs()
        raise AssertionError(
            f"bit-exactness failed for M={M} N={N} K={K} {dtype}: "
            f"max |diff|={diff.max().item():.3e} mean={diff.mean().item():.3e}"
        )


def _bench_cell(M, N, K, dtype, device, n, warmup, seed):
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    out_raw = torch.empty(M, N, device=device, dtype=dtype)
    out_pad = torch.empty(M, N, device=device, dtype=dtype)
    out_hbl = torch.empty(M, N, device=device, dtype=dtype)

    import tritonblas

    # Pass A: unpadded TB vs hipBLASLt (paired, shuffled order)
    def tb_raw():
        tritonblas.matmul(a, b, out=out_raw)

    def hbl_a():
        torch.matmul(a, b, out=out_hbl)

    times_a = _paired_bench({"tb": tb_raw, "hbl": hbl_a}, n=n, warmup=warmup, seed=seed)
    tb_raw_med = statistics.median(times_a["tb"])
    hbl_a_med = statistics.median(times_a["hbl"])

    # Pass B: padded TB vs hipBLASLt
    def tb_pad_call():
        tb_padded(a, b, M, N, K, out_pad)

    def hbl_b():
        torch.matmul(a, b, out=out_hbl)

    times_b = _paired_bench({"tb": tb_pad_call, "hbl": hbl_b}, n=n, warmup=warmup, seed=seed + 1)
    tb_pad_med = statistics.median(times_b["tb"])
    hbl_b_med = statistics.median(times_b["hbl"])

    # Combined hbl baseline for ratio (use min of the two passes — closest to peak)
    hbl_med = min(hbl_a_med, hbl_b_med)

    # Final correctness re-check
    out_a = torch.empty(M, N, device=device, dtype=dtype)
    out_b = torch.empty(M, N, device=device, dtype=dtype)
    tritonblas.matmul(a, b, out=out_a)
    tb_padded(a, b, M, N, K, out_b)
    _sync()
    rel_err = (out_a.float() - out_b.float()).abs().max().item()

    del a, b, out_raw, out_pad, out_hbl, out_a, out_b
    torch.cuda.empty_cache()
    return tb_raw_med, tb_pad_med, hbl_med, hbl_a_med, hbl_b_med, rel_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = "cuda"
    torch.manual_seed(args.seed)

    cells = list(COHORT_BF16)

    # Bit-exactness pre-check for every gated cell
    print("[K-1804] running bit-exactness pre-check on gated cells ...", flush=True)
    for (M, N, K) in cells:
        if not _gate_triggered(M, N, K, dtype):
            continue
        a = torch.randn(M, K, device=device, dtype=dtype)
        b = torch.randn(K, N, device=device, dtype=dtype)
        _bit_exact_check(a, b, M, N, K, dtype)
        del a, b
        torch.cuda.empty_cache()
    print("[K-1804] bit-exactness OK on all gated cells.", flush=True)

    rows = []
    for (M, N, K) in cells:
        n_pad = _pad_n_to_next_wave(N)
        gated = int(_gate_triggered(M, N, K, dtype))
        misaligned = int(_is_misaligned(N))
        print(f"[K-1804] cell M={M} N={N} K={K} gated={gated} misaligned={misaligned} n_pad={n_pad}", flush=True)
        t0 = time.time()
        tb_raw_med, tb_pad_med, hbl_med, hbl_a, hbl_b, rel_err = _bench_cell(
            M, N, K, dtype, device, args.n, args.warmup, args.seed
        )
        ratio_base = hbl_med / tb_raw_med   # >1 means TB faster than HBL
        ratio_pad = hbl_med / tb_pad_med
        ratio_delta = ratio_pad - ratio_base
        speedup_pad_vs_base = tb_raw_med / tb_pad_med  # >1 = pad faster
        rows.append({
            "M": M, "N": N, "K": K,
            "dtype": args.dtype,
            "n": args.n,
            "n_pad": n_pad,
            "gate_triggered": gated,
            "misaligned": misaligned,
            "tb_base_med": tb_raw_med,
            "tb_pad_med": tb_pad_med,
            "hbl_med": hbl_med,
            "hbl_pass_a": hbl_a,
            "hbl_pass_b": hbl_b,
            "ratio_base": ratio_base,
            "ratio_pad": ratio_pad,
            "ratio_delta": ratio_delta,
            "speedup_pad_vs_base": speedup_pad_vs_base,
            "rel_err": rel_err,
        })
        print(f"   tb_base={tb_raw_med*1e6:.2f}us tb_pad={tb_pad_med*1e6:.2f}us "
              f"hbl={hbl_med*1e6:.2f}us speedup={speedup_pad_vs_base:.4f} "
              f"rel_err={rel_err:.3e} ({time.time()-t0:.1f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[K-1804] wrote CSV → {args.out}", flush=True)

    aligned_rows = [r for r in rows if not r["misaligned"]]
    misaligned_rows = [r for r in rows if r["misaligned"]]
    gated_rows = [r for r in rows if r["gate_triggered"]]
    not_gated_rows = [r for r in rows if not r["gate_triggered"]]

    def _block(rs):
        return {
            "n_cells": len(rs),
            "gmean_ratio_base": _gmean([r["ratio_base"] for r in rs]),
            "gmean_ratio_pad": _gmean([r["ratio_pad"] for r in rs]),
            "gmean_speedup_pad_vs_base": _gmean([r["speedup_pad_vs_base"] for r in rs]),
            "max_rel_err": max((r["rel_err"] for r in rs), default=0.0),
        }

    summary = {
        "cohort": args.dtype,
        "n": args.n,
        "aligned": _block(aligned_rows),
        "misaligned": _block(misaligned_rows),
        "gated": _block(gated_rows),
        "not_gated": _block(not_gated_rows),
        "all": _block(rows),
        "bit_exact_max_rel_err": max((r["rel_err"] for r in rows), default=0.0),
    }
    sjson = args.out.rsplit(".", 1)[0] + ".summary.json"
    with open(sjson, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[K-1804] wrote summary → {sjson}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    if summary["bit_exact_max_rel_err"] != 0.0:
        print(f"[K-1804] WARNING: bit-exactness violated, max rel_err={summary['bit_exact_max_rel_err']}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
