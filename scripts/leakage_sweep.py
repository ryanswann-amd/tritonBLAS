"""K-967 leakage sweep — paired ON/OFF across 50 K-832 production neighbors.

For each neighbor cell, runs paired subprocess(OFF) + subprocess(ON), n_rounds=5.
Computes ON/OFF ratio and CI95. Flags REGRESS if mean_delta < 0 AND CI95-upper < 0.

If override is correctly gated, ALL 50 should be NULL (no fire) since gate is
strict-equality on (14208,2048,1024) bf16 only. Any leakage indicates harness bug.
"""
import csv
import math
import os
import statistics
import subprocess
import sys

# 50 neighbor shapes from K-832 top-100 (covers C2/C4/C6/C9 cohorts that are
# spatially near the in-cohort cell or share the MFMA-issue mechanism class).
NEIGHBORS = [
    # Strict-equality "near miss" — adjacent C9 sub-band (LDS-BC + occupancy bound)
    (10112, 2048, 1024, "bf16"), (12160, 2048, 1024, "bf16"), (16256, 2048, 1024, "bf16"),
    (18304, 2048, 1024, "bf16"), (20352, 2048, 1024, "bf16"), (22400, 2048, 1024, "bf16"),
    (24448, 2048, 1024, "bf16"), (26496, 2048, 1024, "bf16"),
    # Other MFMA-issue cells (would trigger if predicate were broadened)
    (5972, 1792,  768, "bf16"), (4480, 3072, 768, "bf16"), (6016, 2048, 1024, "bf16"),
    # K-832 top-traffic LDS-BC cells (defensive — wpeu=1 should NOT fire)
    (256,  2048,  256, "bf16"), (192, 2048, 512, "bf16"), (2304, 2048, 4800, "bf16"),
    (256,  1792, 2048, "bf16"), (736, 1792, 736, "bf16"), (2048, 1792, 256, "bf16"),
    (512,  2048, 192,  "bf16"), (512, 192, 2048, "bf16"), (156, 1792, 512, "bf16"),
    (160,  3072, 512,  "bf16"), (736, 1792, 3744, "bf16"), (256, 256, 2048, "bf16"),
    (512,  2048, 96,   "bf16"), (256, 409600, 384, "bf16"), (768, 3072, 4480, "bf16"),
    (1024, 2048, 6016, "bf16"), (1024, 2048, 8064, "bf16"),
    # K-832 occupancy-bound (vocab-N), large-rect — should NOT fire either
    (384, 409600, 384, "bf16"), (384, 409600, 256, "bf16"), (128, 409600, 384, "bf16"),
    (384, 409600, 128, "bf16"), (8064, 2048, 1024, "bf16"),
    # In-cohort + same-axis fp16 (off-dtype check — should NOT fire on fp16)
    (14208, 2048, 1024, "fp16"),
    # Mid-rect / mid-square cohorts (different mechanism — should NOT fire)
    (3072, 3072, 4096, "bf16"), (4096, 4096, 4096, "bf16"), (4096, 4096, 8192, "bf16"),
    (3072, 3072, 8192, "bf16"), (2048, 2048, 8192, "bf16"), (2048, 2048, 4096, "bf16"),
    (6144, 6144, 4096, "bf16"), (8192, 8192, 8192, "bf16"),
    # Skinny cohorts
    (768, 1792, 5972, "bf16"), (1024, 2048, 1240, "bf16"), (768, 1792, 1024, "bf16"),
    # Small / odd shapes
    (128, 128, 128, "bf16"), (256, 256, 256, "bf16"), (1024, 1024, 1024, "bf16"),
    (2048, 2048, 2048, "bf16"), (4096, 4096, 4096, "bf16"),
    # The IN-COHORT cell, included as the positive-control row
    (14208, 2048, 1024, "bf16"),
]

N_ROUNDS = int(os.environ.get("LEAKAGE_ROUNDS", "5"))
N_WARM, N_CAP = 5, 10
SCRIPT = "/home/ryaswann/mc2-workspaces/K-967/scripts/bench_one.py"

T = {3: 4.303, 5: 2.776, 10: 2.262, 30: 2.045}

def t_value(n):
    for k in sorted(T):
        if n <= k:
            return T[k]
    return 1.96

def run_one(M, N, K, dt, proto, tmp_csv):
    env = os.environ.copy()
    env["TB_K967_PROTO"] = str(proto); env["HIP_VISIBLE_DEVICES"] = "0"
    cmd = ["python3", SCRIPT, str(M), str(N), str(K), dt, "tb",
           str(N_WARM), str(N_CAP), "1", tmp_csv]
    out = subprocess.check_output(cmd, env=env, cwd="/home/ryaswann/mc2-workspaces/K-967/repos/tritonblas",
                                  stderr=subprocess.STDOUT, text=True)
    for line in out.strip().splitlines()[::-1]:
        if "p50=" in line:
            return float(line.split("p50=")[1].split("us")[0]) / 1000.0
    raise RuntimeError(out)

def main():
    out_dir = "/home/ryaswann/mc2-workspaces/K-967/output"
    bench_csv = os.path.join(out_dir, "leakage_raw.csv")
    summary_csv = os.path.join(out_dir, "leakage_summary.csv")
    with open(bench_csv, "w") as f:
        f.write("M,N,K,dtype,backend,TB_K967_PROTO,iter,time_ms\n")

    rows = []
    for i, (M, N, K, dt) in enumerate(NEIGHBORS):
        off_ms = []; on_ms = []
        for r in range(N_ROUNDS):
            try:
                o = run_one(M, N, K, dt, 0, bench_csv)
                n = run_one(M, N, K, dt, 1, bench_csv)
            except Exception as e:
                print(f"FAIL {M}x{N}x{K} {dt}: {e}")
                o = n = float("nan")
            off_ms.append(o); on_ms.append(n)
        med_off = statistics.median(off_ms); med_on = statistics.median(on_ms)
        deltas = [(o / n - 1.0) for o, n in zip(off_ms, on_ms) if not math.isnan(o) and not math.isnan(n)]
        if len(deltas) >= 2:
            mean_d = statistics.mean(deltas); sd_d = statistics.stdev(deltas)
            sem = sd_d / math.sqrt(len(deltas))
            tv = t_value(len(deltas))
            ci_lo, ci_hi = mean_d - tv*sem, mean_d + tv*sem
        else:
            mean_d = ci_lo = ci_hi = float("nan")
        if ci_hi < 0:
            verdict = "REGRESS"
        elif ci_lo > 0.02:
            verdict = "LIFT"
        else:
            verdict = "NULL"
        leak_pct = (med_on / med_off - 1.0) * 100  # negative = TB ON regresses
        print(f"{i+1:2d}/{len(NEIGHBORS)} {M:>6}x{N:>6}x{K:>5} {dt} "
              f"OFF={med_off*1000:7.2f}us ON={med_on*1000:7.2f}us "
              f"d={mean_d*100:+.3f}% CI=[{ci_lo*100:+.3f},{ci_hi*100:+.3f}]% leak={leak_pct:+.3f}% {verdict}")
        rows.append({"M":M,"N":N,"K":K,"dtype":dt,"off_p50_us":med_off*1000,
                     "on_p50_us":med_on*1000,"delta_pct":mean_d*100,
                     "ci_lo_pct":ci_lo*100,"ci_hi_pct":ci_hi*100,
                     "leak_pct":leak_pct,"verdict":verdict})

    fields = list(rows[0].keys())
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)

    n_regress = sum(1 for r in rows if r["verdict"] == "REGRESS")
    n_lift = sum(1 for r in rows if r["verdict"] == "LIFT")
    n_leak2 = sum(1 for r in rows if r["leak_pct"] < -2.0)
    print(f"\n=== summary: {len(rows)} cells, REGRESS={n_regress}, LIFT={n_lift}, leak<-2%={n_leak2}")
    print(f"verdict: {'LAND' if n_leak2 == 0 and any(r['verdict']=='LIFT' for r in rows if (r['M'],r['N'],r['K'],r['dtype'])==(14208,2048,1024,'bf16')) else 'NO-LAND'}")

if __name__ == "__main__":
    main()
