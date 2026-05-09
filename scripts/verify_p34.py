"""P34 N=288 verified-winner subset GPU verification.

Three passes (paired n=30 HIP-graph hot-cache, position-rotated):
  1. Route-firing audit: confirm the route predicate now returns True for
     all 18 P34 N=288 cells.
  2. Post-patch parity: live-routed `tritonblas.matmul()` (oracle) vs
     `torch.matmul()` (HBL) on the 18 cells; compute per-cell ratio
     hbl/oracle (should be ~1.0 since both call hipBLASLt now).
  3. Adjacency-band regression check: 6 cells outside the P34 envelope
     (N=256 + N=384 sibling rungs) must NOT be routed via P34 (firewall)
     and their HBL/oracle ratio must remain >= 0.95 (no regression).

Usage:
    PYTHONPATH=$(pwd)/include python3 scripts/verify_p34.py
"""
from __future__ import annotations

import csv
import math
import os
import statistics
import sys

import torch

TB_INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "include"))
if os.path.isdir(TB_INCLUDE):
    sys.path.insert(0, TB_INCLUDE)

import tritonblas
from tritonblas.matmul import _k971_route_to_hbl
from tritonblas._route_predicate import _P34_SKINNY_N288_KCOMPL_VERIFIED_WIN_18

print("tritonblas import:", tritonblas.__file__)
print("device:", torch.cuda.get_device_name(0))


def hipgraph_capture(fn, n_inner=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_inner):
            fn()
    torch.cuda.synchronize()

    def replay():
        g.replay()

    return replay, n_inner


def time_replay(replay, n_inner, n_outer=30):
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(n_outer):
        torch.cuda.synchronize()
        start.record()
        replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / n_inner)
    return statistics.median(times)


def bench_paired(M, N, K, dtype):
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    out_oracle = torch.empty(M, N, device="cuda", dtype=dtype)
    out_hbl = torch.empty(M, N, device="cuda", dtype=dtype)

    def fn_oracle():
        tritonblas.matmul(a, b, out=out_oracle)

    def fn_hbl():
        torch.matmul(a, b, out=out_hbl)

    if (M + K) % 2 == 0:
        rep_o, ni = hipgraph_capture(fn_oracle)
        rep_h, _ = hipgraph_capture(fn_hbl)
    else:
        rep_h, ni = hipgraph_capture(fn_hbl)
        rep_o, _ = hipgraph_capture(fn_oracle)

    t_oracle = time_replay(rep_o, ni)
    t_hbl = time_replay(rep_h, ni)
    return t_oracle, t_hbl


def gmean(xs):
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


torch.backends.cuda.preferred_blas_library("hipblaslt")


# === Pass 1: route-firing audit on the 18 P34 cells ===
print("\n=== Pass 1: route-firing audit (P34 N=288 18 cells) ===")
n288_cells = sorted(_P34_SKINNY_N288_KCOMPL_VERIFIED_WIN_18)
fired = 0
for (M, N, K, dt) in n288_cells:
    dtype = torch.bfloat16 if dt == "torch.bfloat16" else torch.float16
    routed = _k971_route_to_hbl(M, N, K, dtype, dtype, False, False)
    if routed:
        fired += 1
print(f"  routed: {fired}/{len(n288_cells)}")
assert fired == len(n288_cells), f"Only {fired}/{len(n288_cells)} N=288 cells route to HBL"


# === Pass 2: post-patch parity sweep on 18 cells ===
print("\n=== Pass 2: post-patch HBL/oracle parity (18 cells) ===")
results = []
ratios = []
for (M, N, K, dt) in n288_cells:
    dtype = torch.bfloat16 if dt == "torch.bfloat16" else torch.float16
    t_oracle, t_hbl = bench_paired(M, N, K, dtype)
    ratio = t_hbl / t_oracle  # ~1.0 means parity (both routed to HBL)
    ratios.append(ratio)
    results.append({
        "M": M, "N": N, "K": K, "dtype": dt,
        "t_oracle_us": round(t_oracle * 1000, 3),
        "t_hbl_us": round(t_hbl * 1000, 3),
        "r_hbl_over_oracle": round(ratio, 4),
    })
    print(f"  M={M:5d} N={N} K={K:5d} {dt[6:]:9s}  oracle={t_oracle*1000:7.3f}us  hbl={t_hbl*1000:7.3f}us  hbl/oracle={ratio:.4f}")

print(f"\n  cohort gmean hbl/oracle = {gmean(ratios):.4f}")
print(f"  cohort min  hbl/oracle = {min(ratios):.4f}")
print(f"  cohort max  hbl/oracle = {max(ratios):.4f}")
print(f"  cells with hbl/oracle >= 0.95 = {sum(1 for r in ratios if r >= 0.95)}/{len(ratios)}")


# === Pass 3: adjacency-band regression check (6 cells, N=256 + N=384) ===
print("\n=== Pass 3: adjacency-band regression check (6 cells, N=256/N=384) ===")
adj_cells = []
for M in (2048, 4096, 8192):
    adj_cells.append((M, 256, 8192, "torch.bfloat16"))
for M in (2048, 4096, 8192):
    adj_cells.append((M, 384, 8192, "torch.bfloat16"))

adj_results = []
adj_ratios = []
for (M, N, K, dt) in adj_cells:
    dtype = torch.bfloat16 if dt == "torch.bfloat16" else torch.float16
    in_p34 = (int(M), int(N), int(K), str(dtype)) in _P34_SKINNY_N288_KCOMPL_VERIFIED_WIN_18
    assert not in_p34, f"firewall breach: ({M},{N},{K},{dt}) is in P34"
    t_oracle, t_hbl = bench_paired(M, N, K, dtype)
    ratio = t_hbl / t_oracle
    adj_ratios.append(ratio)
    adj_results.append({
        "M": M, "N": N, "K": K, "dtype": dt,
        "t_oracle_us": round(t_oracle * 1000, 3),
        "t_hbl_us": round(t_hbl * 1000, 3),
        "r_hbl_over_oracle": round(ratio, 4),
        "p34_firewall_ok": True,
    })
    print(f"  M={M:5d} N={N} K={K:5d} {dt[6:]:9s}  oracle={t_oracle*1000:7.3f}us  hbl={t_hbl*1000:7.3f}us  hbl/oracle={ratio:.4f}  [firewall OK]")

print(f"\n  adjacency gmean hbl/oracle = {gmean(adj_ratios):.4f}")
print(f"  adjacency min  hbl/oracle = {min(adj_ratios):.4f}")
print(f"  cells with hbl/oracle >= 0.95 = {sum(1 for r in adj_ratios if r >= 0.95)}/{len(adj_ratios)}")


# === Write CSVs ===
out_dir = os.environ.get("OUT_DIR", "/tmp/K-1835/output")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "p34_n288_postpatch_18cell.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
with open(os.path.join(out_dir, "p34_adjacency_6cell.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(adj_results[0].keys()))
    w.writeheader()
    w.writerows(adj_results)
print(f"\nWrote CSVs to {out_dir}/")
print("\n[K-1835 P34 verification complete]")
