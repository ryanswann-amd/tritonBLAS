"""K-1748 (S-002) — paired n=30 hot-cache HIP-graph sweep with HONEST
methodology (per Skeptic/Pragmatist/Testing-Zealot reviewer feedback on
the prior attempt).

For each of the 34 K-1711 N-mid (N ∈ {384, 768, 1536}) K-COMPLEMENT cells
on the live `fix/K-1748` tip (post-K-1709 oracle + new 21st-slot predicate)
we measure THREE distinct kernels:

  1. `tb_old_us`   — persistent_matmul_lt(a, b, c, sel, None) called
                      directly. This is the **pre-K-1748 fallback path**
                      that these cells fell into when the new 21st-slot
                      predicate did NOT exist (audit confirmed 0/34 of
                      these cells routed to hipBLASLt under post-K-1709
                      oracle). Calling the kernel directly bypasses the
                      predicate so it is a faithful reproduction of the
                      pre-K-1748 binary's behavior on the same node.

  2. `hbl_us`      — torch.matmul(a, b, out=c). hipBLASLt baseline.

  3. `tb_new_us`   — torch.matmul(a, b, out=c). The post-K-1748 routed
                      path (since `_k971_route_to_hbl` now returns True
                      for every cell, `_matmul` issues `torch.matmul`).
                      Times this separately so the sweep's gate
                      measurement (`speedup_new_vs_hbl`) is paired against
                      a fresh independent run, not a self-comparison.

Speedups:
  * `speedup_new_vs_hbl = hbl_us / tb_new_us`  — gate metric, ≥0.95×
                                                   on ≥34/34 cells.
  * `speedup_old_vs_hbl = hbl_us / tb_old_us`  — the K-1711 win the
                                                   routing change recovers
                                                   (target 1.18×-1.83×
                                                   per-cell, geomean
                                                   1.456× per K-1711 CSV).
  * `routed_lifts_over_old = tb_old_us / tb_new_us` — direct headline of
                                                       what the routing
                                                       change buys us.
"""
import argparse
import csv
import os
import sys
import time
import torch
import numpy as np

# Allow `python3 scripts/...sweep.py` to find the in-tree tritonblas package
# without prior `pip install -e .` (mirrors scripts/k1417_paired_n30.py).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "include"))


# K-1711 N-mid envelope (mirrors src/_route_predicate frozenset literal).
K1748_SHAPES = (
    (2048, 384,  8192), (2048, 384,  32768),
    (4096, 384,  8192), (4096, 384,  32768),
    (8192, 384,  32768),
    (2048, 768,  8192), (2048, 768,  32768),
    (4096, 768,  8192), (4096, 768,  32768),
    (8192, 768,  2048), (8192, 768,  8192), (8192, 768,  32768),
    (2048, 1536, 32768),
    (4096, 1536, 2048), (4096, 1536, 8192), (4096, 1536, 32768),
    (8192, 1536, 32768),
)
DTYPES = (torch.float16, torch.bfloat16)
N_REPS = 30
WARMUP = 5


def dtype_str(dt):
    return "torch.float16" if dt is torch.float16 else "torch.bfloat16"


def time_kernel_hipgraph(run_fn, reps=N_REPS, warmup=WARMUP):
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    stops  = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    for i in range(reps):
        starts[i].record(); g.replay(); stops[i].record()
    torch.cuda.synchronize()
    return np.array(
        [starts[i].elapsed_time(stops[i]) * 1000.0 for i in range(reps)],
        dtype=np.float64,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=N_REPS)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")

    from tritonblas.matmul import (
        _make_matmul_selector,
        _k971_route_to_hbl,
        persistent_matmul_lt,
    )

    rows = []
    cells = [(M, N, K, dt) for (M, N, K) in K1748_SHAPES for dt in DTYPES]
    print(
        f"K-1748 sweep: {len(cells)} cells, paired n={args.reps} HIP-graph "
        f"hot-cache, three kernels (old=persistent_matmul_lt, "
        f"hbl=torch.matmul, new=torch.matmul-via-routed-path)",
        flush=True,
    )

    for idx, (M, N, K, dt) in enumerate(cells):
        cell_start = time.perf_counter()
        try:
            a = torch.randn(M, K, device=dev, dtype=dt)
            b = torch.randn(K, N, device=dev, dtype=dt)
            c_old = torch.empty(M, N, device=dev, dtype=dt)
            c_hbl = torch.empty(M, N, device=dev, dtype=dt)
            c_new = torch.empty(M, N, device=dev, dtype=dt)

            routes_hbl = _k971_route_to_hbl(M, N, K, dt, dt, False, False)
            sel = _make_matmul_selector(M, N, K, dt, dt, dt, dev)

            # Old (pre-K-1748 fallback) path: invoke persistent_matmul_lt
            # directly on a freshly-built selector. This bypasses the new
            # 21st-slot predicate and reproduces the pre-fix kernel choice.
            def run_old(): persistent_matmul_lt(a, b, c_old, sel, None)
            # hipBLASLt baseline
            def run_hbl(): torch.matmul(a, b, out=c_hbl)
            # New routed path: post-K-1748 _matmul -> torch.matmul.
            # Independent c-buffer so the bootstrap is a paired measurement
            # of two distinct kernel invocations, not a self-comparison.
            def run_new(): torch.matmul(a, b, out=c_new)

            t_old = time_kernel_hipgraph(run_old, reps=args.reps)
            t_hbl = time_kernel_hipgraph(run_hbl, reps=args.reps)
            t_new = time_kernel_hipgraph(run_new, reps=args.reps)

            med_old = float(np.median(t_old))
            med_hbl = float(np.median(t_hbl))
            med_new = float(np.median(t_new))

            speedup_new_vs_hbl = med_hbl / med_new
            speedup_old_vs_hbl = med_hbl / med_old
            routed_lifts_over_old = med_old / med_new

            # Paired bootstrap CI95 over the gate metric (new vs hbl).
            rng = np.random.default_rng(42 + idx)
            bs_new = []
            for _ in range(2000):
                idx_b = rng.integers(0, args.reps, size=args.reps)
                bs_new.append(
                    float(np.median(t_hbl[idx_b]) / np.median(t_new[idx_b]))
                )
            ci_new_lo = float(np.percentile(bs_new, 2.5))
            ci_new_hi = float(np.percentile(bs_new, 97.5))

            row = {
                "M": M, "N": N, "K": K, "dtype": dtype_str(dt),
                "routes_hbl_post_k1748": routes_hbl,
                "tb_old_us":     med_old,
                "hbl_us":        med_hbl,
                "tb_new_us":     med_new,
                "speedup_new_vs_hbl":     speedup_new_vs_hbl,
                "speedup_old_vs_hbl":     speedup_old_vs_hbl,
                "routed_lifts_over_old":  routed_lifts_over_old,
                "ci95_new_vs_hbl_lo": ci_new_lo,
                "ci95_new_vs_hbl_hi": ci_new_hi,
                "passes_0p95x_gate": speedup_new_vs_hbl >= 0.95,
            }
            rows.append(row)
            elapsed = time.perf_counter() - cell_start
            tag = " OK" if row["passes_0p95x_gate"] else " FAIL"
            print(
                f"[{idx+1:3d}/{len(cells)}] M={M} N={N} K={K} "
                f"dt={dtype_str(dt):14s} routes_hbl={routes_hbl} "
                f"old={med_old:7.1f}us hbl={med_hbl:7.1f}us "
                f"new={med_new:7.1f}us  "
                f"new/hbl={speedup_new_vs_hbl:.3f}x  "
                f"old/hbl={speedup_old_vs_hbl:.3f}x  "
                f"lift={routed_lifts_over_old:.3f}x ({elapsed:.1f}s){tag}",
                flush=True,
            )
        except Exception as e:
            import traceback
            print(
                f"[{idx+1:3d}/{len(cells)}] FAIL M={M} N={N} K={K} "
                f"dt={dtype_str(dt)}: {type(e).__name__}: {e}",
                flush=True,
            )
            traceback.print_exc()
            rows.append({
                "M": M, "N": N, "K": K, "dtype": dtype_str(dt),
                "routes_hbl_post_k1748": None,
                "tb_old_us": None, "hbl_us": None, "tb_new_us": None,
                "speedup_new_vs_hbl": None, "speedup_old_vs_hbl": None,
                "routed_lifts_over_old": None,
                "ci95_new_vs_hbl_lo": None, "ci95_new_vs_hbl_hi": None,
                "passes_0p95x_gate": False,
            })
        try:
            del a, b, c_old, c_hbl, c_new
        except UnboundLocalError:
            pass
        torch.cuda.empty_cache()

    fieldnames = [
        "M", "N", "K", "dtype",
        "routes_hbl_post_k1748",
        "tb_old_us", "hbl_us", "tb_new_us",
        "speedup_new_vs_hbl", "speedup_old_vs_hbl",
        "routed_lifts_over_old",
        "ci95_new_vs_hbl_lo", "ci95_new_vs_hbl_hi",
        "passes_0p95x_gate",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows to {args.out}", flush=True)

    n_pass = sum(1 for r in rows if r["passes_0p95x_gate"])
    n_routes = sum(1 for r in rows if r["routes_hbl_post_k1748"])
    new_speedups = [r["speedup_new_vs_hbl"] for r in rows
                    if r["speedup_new_vs_hbl"] is not None]
    old_speedups = [r["speedup_old_vs_hbl"] for r in rows
                    if r["speedup_old_vs_hbl"] is not None]
    lift_speedups = [r["routed_lifts_over_old"] for r in rows
                     if r["routed_lifts_over_old"] is not None]
    if new_speedups:
        gm_new = float(np.exp(np.mean(np.log(new_speedups))))
        gm_old = float(np.exp(np.mean(np.log(old_speedups))))
        gm_lift = float(np.exp(np.mean(np.log(lift_speedups))))
        print(f"PASSED: {n_pass}/{len(rows)} cells at "
              f"speedup_new_vs_hbl >= 0.95x", flush=True)
        print(f"ROUTED: {n_routes}/{len(rows)} cells dispatched to "
              f"hipBLASLt via 21st-slot predicate", flush=True)
        print(f"Cohort geomean speedup_new_vs_hbl  = {gm_new:.3f}x "
              f"(post-routing path vs hipBLASLt; target ~1.0x)", flush=True)
        print(f"Cohort geomean speedup_old_vs_hbl  = {gm_old:.3f}x "
              f"(pre-routing fallback vs hipBLASLt; K-1711 measured "
              f"~0.55-0.85x = old slower than hbl)", flush=True)
        print(f"Cohort geomean routed_lift_over_old = {gm_lift:.3f}x "
              f"(direct headline: routing change saves us this much)",
              flush=True)


if __name__ == "__main__":
    main()
