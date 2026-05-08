"""K-967 round-robin paired-CI benchmark across multiple cells.

For each cell, runs N_ROUNDS rounds, where each round is:
  subprocess(TB_K967_PROTO=0) -> 1 OFF p50
  subprocess(TB_K967_PROTO=1) -> 1 ON  p50
  subprocess(torch)            -> 1 REF p50

This avoids the K-888 in-process env-mutation gotcha (Triton selector
caching). Round-robin within rounds eliminates time-correlated drift
per K-901 M1.

Computes paired delta (ON/OFF - 1) with Student-t 95% CI (n=N_ROUNDS).
LAND if CI95-lower > 0.02 (2% lift). NO-LAND if upper < 0.0 (regression)
or geomean-of-rounds < 1.02.
"""
import csv
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time

CELLS = [
    # (M, N, K, dtype) — MFMA-issue-stall cohort from K-931 + a few K-832 neighbors
    (14208, 2048, 1024, "bf16"),  # S29 in-cohort
    (5972,  1792,  768, "bf16"),  # S18 MFMA-issue (different shape, OOC)
    (6016,  2048, 1024, "bf16"),  # S25 MFMA-issue (close neighbor, OOC)
    (4480,  3072,  768, "bf16"),  # S24 MFMA-issue (OOC)
    (12160, 2048, 1024, "bf16"),  # S28 LDS-BC (occupancy similar, leakage check)
    (16256, 2048, 1024, "bf16"),  # S30 occupancy-bound (closest spatial neighbor)
]

N_ROUNDS = int(os.environ.get("N_ROUNDS", "10"))
N_WARM = 5
N_CAP = 10
SCRIPT = "/home/ryaswann/mc2-workspaces/K-967/scripts/bench_one.py"

t_lookup = {5: 2.776, 10: 2.262, 15: 2.145, 20: 2.093, 30: 2.045}

def t_value(n):
    if n in t_lookup:
        return t_lookup[n]
    keys = sorted(t_lookup.keys())
    for k in keys:
        if n <= k:
            return t_lookup[k]
    return 1.96

def run_one(M, N, K, dt, backend, proto, csv_path):
    env = os.environ.copy()
    env["TB_K967_PROTO"] = str(proto)
    env["HIP_VISIBLE_DEVICES"] = "0"
    cmd = ["python3", SCRIPT, str(M), str(N), str(K), dt, backend,
           str(N_WARM), str(N_CAP), "1", csv_path]  # 1 round per subprocess
    out = subprocess.check_output(cmd, env=env, cwd="/home/ryaswann/mc2-workspaces/K-967/repos/tritonblas",
                                  stderr=subprocess.STDOUT, text=True)
    # Last line is "MxNxK ... p50=...us min=...us max=...us"
    for line in out.strip().splitlines()[::-1]:
        if "p50=" in line:
            p50us = float(line.split("p50=")[1].split("us")[0])
            return p50us / 1000.0  # to ms
    raise RuntimeError(f"no p50 in output: {out}")

def main():
    out_dir = "/home/ryaswann/mc2-workspaces/K-967/output"
    os.makedirs(out_dir, exist_ok=True)
    bench_csv = os.path.join(out_dir, "rr_raw.csv")
    summary_csv = os.path.join(out_dir, "rr_summary.csv")
    with open(bench_csv, "w") as f:
        f.write("M,N,K,dtype,backend,TB_K967_PROTO,iter,time_ms\n")

    summary_rows = []
    for (M, N, K, dt) in CELLS:
        off_ms = []; on_ms = []; ref_ms = []
        for r in range(N_ROUNDS):
            o = run_one(M, N, K, dt, "tb",    0, bench_csv)
            n = run_one(M, N, K, dt, "tb",    1, bench_csv)
            t = run_one(M, N, K, dt, "torch", 0, bench_csv)
            off_ms.append(o); on_ms.append(n); ref_ms.append(t)
            print(f"  round {r+1}/{N_ROUNDS}  OFF={o*1000:.2f}us ON={n*1000:.2f}us REF={t*1000:.2f}us speedup={o/n:.4f}")
        # paired delta = ON/OFF - 1 per round
        deltas = [(off / on) - 1.0 for off, on in zip(off_ms, on_ms)]
        mean_d = statistics.mean(deltas)
        sd_d = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        sem = sd_d / math.sqrt(len(deltas))
        tv = t_value(len(deltas))
        ci_lo = mean_d - tv * sem
        ci_hi = mean_d + tv * sem
        wins = sum(1 for d in deltas if d > 0)
        med_off = statistics.median(off_ms)
        med_on  = statistics.median(on_ms)
        med_ref = statistics.median(ref_ms)
        speedup = med_off / med_on
        tb_vs_ref = med_ref / med_off
        on_vs_ref = med_ref / med_on
        verdict = "LAND" if ci_lo > 0.02 else ("REGRESS" if ci_hi < 0 else "NULL")
        print(f"\n[{M}x{N}x{K} {dt}] OFF p50={med_off*1000:.2f}us ON p50={med_on*1000:.2f}us REF={med_ref*1000:.2f}us "
              f"speedup={speedup:.4f} CI95=[{ci_lo*100:+.3f}%, {ci_hi*100:+.3f}%] wins={wins}/{len(deltas)} VERDICT={verdict}")
        summary_rows.append({
            "M": M, "N": N, "K": K, "dtype": dt,
            "off_p50_us": med_off*1000, "on_p50_us": med_on*1000, "ref_p50_us": med_ref*1000,
            "speedup_off_over_on": speedup,
            "tb_vs_torch_off": tb_vs_ref, "tb_vs_torch_on": on_vs_ref,
            "delta_mean_pct": mean_d*100, "ci_lo_pct": ci_lo*100, "ci_hi_pct": ci_hi*100,
            "win_frac": wins / len(deltas), "verdict": verdict,
        })

    fieldnames = list(summary_rows[0].keys())
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"\n=== summary written to {summary_csv} ===")
    print(f"In-cohort cell (S29 14208x2048x1024 bf16): "
          f"speedup={summary_rows[0]['speedup_off_over_on']:.4f}, verdict={summary_rows[0]['verdict']}")

if __name__ == "__main__":
    main()
