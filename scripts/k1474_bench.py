#!/usr/bin/env python3
"""K-1474 productionization bench.

Runs three measurement passes per arch invocation:

  1. K-1295 22-shape cohort PRE/POST: measures both tritonblas.matmul and
     torch.matmul per cell.  PRE = tb_us; POST analytically = hbl_us if
     cell in P19 frozenset else tb_us.  Reports cohort geomean shift.
  2. P19 envelope spot-revalidation: for each of the 29 productionised
     P19 cells, paired n=15 HIP-graph hot-cache measurement with
     B=2000 vectorised paired bootstrap CI95 on log-ratio.  Reports
     per-cell ratio_median + CI95 lo/hi.  This is the cross-arch
     transferability test.
  3. Routing-decision verification: confirms _k971_route_to_hbl returns
     True for all 29 P19 cells and False for the lone reject cell, on
     the patched tritonblas head.  Pure logic check, no GPU.

Usage:
  python3 k1474_bench.py --arch-tag mi300x \\
      --out-dir /work/output --frozenset-keys /work/scripts/p19_frozenset_keys.json
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from itertools import product
import numpy as np
import torch

# K-1295 22-shape cohort — same as K-1465 bench
COHORT = [
    (1024, 1024, 1024, "bf16"), (1024, 1024, 1024, "fp16"),
    (2048, 2048, 2048, "bf16"), (2048, 2048, 2048, "fp16"),
    (4096, 4096, 4096, "bf16"), (4096, 4096, 4096, "fp16"),
    (8192, 8192, 8192, "bf16"), (8192, 8192, 8192, "fp16"),
    (4096, 1024, 4096, "bf16"), (4096, 1024, 8192, "fp16"),
    (4096, 2048, 4096, "bf16"), (8192, 2048, 8192, "fp16"),
    (2048, 4096, 4096, "bf16"), (4096, 4096, 8192, "bf16"),
    (2048, 8192, 4096, "bf16"), (2048, 8192, 8192, "fp16"),
    (4096, 8192, 8192, "bf16"), (4096, 8192, 16384, "fp16"),
    (8192, 8192, 4096, "bf16"), (8192, 8192, 16384, "fp16"),
    (4096, 8192, 4096, "bf16"), (8192, 4096, 2048, "bf16"),
]

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

REPLAYS_FAST = 50
N_PAIRED_FAST = 20
REPLAYS_SLOW = 50
N_PAIRED_SLOW = 15
WARMUP = 4
BOOTSTRAP_B = 2000


def time_graph(g, replays):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(replays): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / replays


def capture(call):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): call()
    return g


def measure_paired(M, N, K, dt, n_paired=N_PAIRED_FAST, replays=REPLAYS_FAST):
    """Returns (tb_array, hb_array) of n_paired per-iteration medians (us)."""
    import tritonblas
    dtype = DTYPES[dt]
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    out_tb = torch.empty(M, N, device="cuda", dtype=dtype)
    out_hb = torch.empty(M, N, device="cuda", dtype=dtype)
    def tb_call(): out_tb.copy_(tritonblas.matmul(a, b))
    def hb_call(): torch.matmul(a, b, out=out_hb)
    for _ in range(WARMUP): tb_call(); hb_call()
    torch.cuda.synchronize()
    g_tb = capture(tb_call); g_hb = capture(hb_call)
    tb_us, hb_us = [], []
    for i in range(n_paired):
        if i % 2 == 0:
            t1 = time_graph(g_tb, replays); t2 = time_graph(g_hb, replays)
        else:
            t2 = time_graph(g_hb, replays); t1 = time_graph(g_tb, replays)
        tb_us.append(t1); hb_us.append(t2)
    return np.array(tb_us), np.array(hb_us)


def bootstrap_ratio_ci(tb_us, hb_us, B=BOOTSTRAP_B, lo=2.5, hi=97.5):
    """Vectorised paired-log-ratio bootstrap, returns (median, ci_lo, ci_hi)."""
    n = len(tb_us)
    log_r = np.log(tb_us / hb_us)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(B, n))
    boot_means = log_r[idx].mean(axis=1)
    median = float(np.exp(np.median(log_r)))
    return median, float(np.exp(np.percentile(boot_means, lo))), float(np.exp(np.percentile(boot_means, hi)))


def pass1_cohort(out_dir, productionised, arch_tag):
    """K-1295 22-shape cohort PRE/POST analytical."""
    rows = []
    for i, (M, N, K, dt) in enumerate(COHORT, 1):
        ts = time.time()
        tb_arr, hb_arr = measure_paired(M, N, K, dt)
        tb_us, hb_us = float(np.median(tb_arr)), float(np.median(hb_arr))
        in_p19 = (M, N, K, dt) in productionised
        pre = tb_us
        post = hb_us if in_p19 else tb_us
        rows.append({
            "M": M, "N": N, "K": K, "dtype": dt,
            "tb_us": tb_us, "hbl_us": hb_us,
            "in_p19_frozenset": in_p19,
            "pre_us": pre, "post_us": post,
            "speedup_pre_vs_hbl": hb_us / tb_us,
            "speedup_post_vs_hbl": hb_us / post,
            "post_to_pre_us_delta_pct": (post - pre) / pre * 100.0,
        })
        print(f"  [coh {i:2d}/{len(COHORT)}] {M}x{N}x{K} {dt} "
              f"tb={tb_us:.1f}us hbl={hb_us:.1f}us in_P19={in_p19} "
              f"pre/hbl={hb_us/tb_us:.3f} post/hbl={hb_us/post:.3f} "
              f"({time.time()-ts:.1f}s)", file=sys.stderr, flush=True)

    geo_pre = float(np.exp(np.mean(np.log([r["speedup_pre_vs_hbl"] for r in rows]))))
    geo_post = float(np.exp(np.mean(np.log([r["speedup_post_vs_hbl"] for r in rows]))))
    summary = {
        "arch_tag": arch_tag,
        "n_cells": len(rows),
        "n_in_p19": sum(1 for r in rows if r["in_p19_frozenset"]),
        "geomean_pre_routed_over_hbl": geo_pre,
        "geomean_post_routed_over_hbl": geo_post,
        "delta_geomean_pp": (geo_post - geo_pre) * 100.0,
        "delta_geomean_pct": (geo_post / geo_pre - 1.0) * 100.0,
    }
    with open(f"{out_dir}/k1474_{arch_tag}_k1295_cohort.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    fields = list(rows[0].keys())
    with open(f"{out_dir}/k1474_{arch_tag}_k1295_cohort.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[summary {arch_tag} cohort] PRE={geo_pre:.4f} POST={geo_post:.4f} "
          f"delta={summary['delta_geomean_pct']:+.2f}%", file=sys.stderr, flush=True)
    return summary


def pass2_envelope(out_dir, productionised, arch_tag):
    """29 P19 cells, paired n=15 + bootstrap CI."""
    rows = []
    cells = sorted(productionised)
    for i, (M, N, K, dt) in enumerate(cells, 1):
        ts = time.time()
        tb_arr, hb_arr = measure_paired(M, N, K, dt, n_paired=N_PAIRED_SLOW, replays=REPLAYS_SLOW)
        ratio_med, ci_lo, ci_hi = bootstrap_ratio_ci(tb_arr, hb_arr)
        # KEEP / MARGINAL / STRICT_FAIL classification per K-1430
        if ratio_med >= 1.0 and ci_lo >= 1.0:
            verdict = "KEEP"
        elif ratio_med < 0.95 and ci_lo < 1.0:
            verdict = "STRICT_FAIL"
        else:
            verdict = "MARGINAL"
        rows.append({
            "M": M, "N": N, "K": K, "dtype": dt,
            "tb_us_median": float(np.median(tb_arr)),
            "hbl_us_median": float(np.median(hb_arr)),
            "ratio_median_tb_over_hbl": ratio_med,
            "ci95_lo": ci_lo, "ci95_hi": ci_hi,
            "verdict": verdict,
        })
        print(f"  [env {i:2d}/{len(cells)}] {M}x{N}x{K} {dt} "
              f"r={ratio_med:.4f} ci95=[{ci_lo:.4f},{ci_hi:.4f}] {verdict} "
              f"({time.time()-ts:.1f}s)", file=sys.stderr, flush=True)

    geo = float(np.exp(np.mean(np.log([r["ratio_median_tb_over_hbl"] for r in rows]))))
    n_keep = sum(1 for r in rows if r["verdict"] == "KEEP")
    n_marg = sum(1 for r in rows if r["verdict"] == "MARGINAL")
    n_fail = sum(1 for r in rows if r["verdict"] == "STRICT_FAIL")
    summary = {
        "arch_tag": arch_tag,
        "n_cells": len(rows),
        "n_KEEP": n_keep, "n_MARGINAL": n_marg, "n_STRICT_FAIL": n_fail,
        "cohort_geomean_tb_over_hbl": geo,
        "min_ci95_lo": min(r["ci95_lo"] for r in rows),
        "max_ratio_median": max(r["ratio_median_tb_over_hbl"] for r in rows),
    }
    with open(f"{out_dir}/k1474_{arch_tag}_envelope.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    fields = list(rows[0].keys())
    with open(f"{out_dir}/k1474_{arch_tag}_envelope.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[summary {arch_tag} envelope] KEEP={n_keep}/{len(rows)} MARG={n_marg} FAIL={n_fail} "
          f"geomean={geo:.4f} min_ci95-lo={summary['min_ci95_lo']:.4f}",
          file=sys.stderr, flush=True)
    return summary


def pass3_routing_check(out_dir, productionised, arch_tag):
    """Pure logic check: every P19 cell routes to HBL via _k971_route_to_hbl."""
    from tritonblas._route_predicate import k971_route_decision
    DT_MAP = {"bf16": "torch.bfloat16", "fp16": "torch.float16"}
    rows = []
    for (M, N, K, dt) in sorted(productionised):
        a_dt_str = DT_MAP[dt]
        routed = k971_route_decision(M, N, K, a_dt_str, a_dt_str, False, False)
        rows.append({"M": M, "N": N, "K": K, "dtype": dt, "routed_to_hbl": routed})
    n_correct = sum(1 for r in rows if r["routed_to_hbl"])
    # Also check the reject cell falls through
    reject_M, reject_N, reject_K, reject_dt = (2048, 8192, 32768, "fp16")
    reject_routed = k971_route_decision(
        reject_M, reject_N, reject_K, DT_MAP[reject_dt], DT_MAP[reject_dt], False, False)
    summary = {
        "arch_tag": arch_tag,
        "n_p19_cells": len(rows),
        "n_routed_to_hbl": n_correct,
        "all_p19_routed": n_correct == len(rows),
        "reject_cell_routed_to_hbl": reject_routed,
    }
    with open(f"{out_dir}/k1474_{arch_tag}_routing_check.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"[summary {arch_tag} routing] {n_correct}/{len(rows)} P19 cells route to HBL; "
          f"reject cell routed_to_hbl={reject_routed} (must be False)",
          file=sys.stderr, flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch-tag", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--frozenset-keys", required=True)
    p.add_argument("--skip-cohort", action="store_true")
    p.add_argument("--skip-envelope", action="store_true")
    args = p.parse_args()

    productionised = {tuple(c) for c in json.load(open(args.frozenset_keys))}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"# device: {torch.cuda.get_device_name(0)}", file=sys.stderr)
    print(f"# gcnArch: {torch.cuda.get_device_properties(0).gcnArchName}", file=sys.stderr)
    print(f"# torch: {torch.__version__}", file=sys.stderr)
    print(f"# productionised P19 cells: {len(productionised)}", file=sys.stderr)

    # Pass 3 first (cheap logic check, fail fast if predicate not wired)
    r3 = pass3_routing_check(args.out_dir, productionised, args.arch_tag)
    if not r3["all_p19_routed"] or r3["reject_cell_routed_to_hbl"]:
        print("[FATAL] routing-decision check failed — abort before GPU work",
              file=sys.stderr)
        return 2

    if not args.skip_cohort:
        pass1_cohort(args.out_dir, productionised, args.arch_tag)
    if not args.skip_envelope:
        pass2_envelope(args.out_dir, productionised, args.arch_tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
