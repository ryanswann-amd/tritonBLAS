"""K_min boundary sweep: per-shape kpack=1 vs kpack=2 deltas.

Reads a cohort CSV (produced by ``benchmarks/bench_kpack_cohort.py``) and
answers two questions about the small-K kpack=2 boundary:

1) For each candidate K_min in {512, 768, 1024, 1536, 2048}, what is the
   per-shape latency when a caller always requests kpack=2 and the gate
   uses that K_min?  This is computed from the raw CSV by selecting:
       - kpack_req=2 row (effective=2 path) when K >= K_min
       - kpack_req=1 row when K < K_min

2) Per (M,N,K) shape, what is the kpack=1 -> kpack=2 percent change,
   computed as ``(t2 - t1) / t1 * 100``?  Positive = kpack=2 is slower
   (regression).  Negative = kpack=2 is faster (win).  The boundary
   K_min is the smallest K at which kpack=2 stops regressing on the
   small-K cohort.

Outputs:
  * per_shape_delta.csv -- (M,N,K,kpack1_ms,kpack2_ms,delta_pct)
  * kmin_boundary_summary.csv -- (K_min,ks_running_kpack2,
    cohort_geomean_ms,always_kpack1_geomean_ms,
    speedup_vs_always_kpack1_pct)

Run:
  python3 benchmarks/analyze_kmin_sweep.py --in cohort_gate_off.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict


def _geomean(xs):
    xs = [x for x in xs if x > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    # Load: keyed by (M,N,K,kpack_req)
    rows = {}
    with open(args.inp) as f:
        r = csv.DictReader(f)
        for row in r:
            key = (int(row["M"]), int(row["N"]), int(row["K"]),
                   int(row["kpack_req"]))
            rows[key] = row

    # 1) Per-shape kpack=1 vs kpack=2 delta
    shapes = sorted({(M, N, K) for (M, N, K, _) in rows})
    per_shape_rows = []
    for (M, N, K) in shapes:
        r1 = rows.get((M, N, K, 1))
        r2 = rows.get((M, N, K, 2))
        if r1 is None or r2 is None:
            continue
        t1 = float(r1["latency_ms"])
        # Reach into the *raw* kpack=2 latency:
        # When K < gate, kpack_req=2 is clamped to effective=1, so the
        # latency in the CSV equals kpack=1.  In that case we cannot
        # measure the unclamped kpack=2 cost from this CSV alone.  Mark it
        # with a sentinel so the analysis does not pretend the gap is 0%
        # at small K -- the reviewer specifically asked for raw kpack=2
        # deltas at every K.
        if int(r2["kpack_effective"]) == 2:
            t2 = float(r2["latency_ms"])
            measured_unclamped = True
        else:
            t2 = float("nan")
            measured_unclamped = False
        delta_pct = (t2 - t1) / t1 * 100.0 if measured_unclamped else float("nan")
        per_shape_rows.append({
            "M": M, "N": N, "K": K,
            "kpack1_ms": f"{t1:.4f}",
            "kpack2_ms": (f"{t2:.4f}" if measured_unclamped else "NA_clamped"),
            "delta_pct": (f"{delta_pct:+.2f}" if measured_unclamped else "NA_clamped"),
            "kpack2_measured_unclamped": int(measured_unclamped),
        })

    out_per = os.path.join(args.out_dir, "per_shape_delta.csv")
    with open(out_per, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_shape_rows[0].keys()))
        w.writeheader()
        w.writerows(per_shape_rows)
    print(f"wrote {out_per} ({len(per_shape_rows)} rows)")

    # 2) K_min boundary table.
    #    For each candidate K_min, we ask: if a caller always requests
    #    kpack=2 and we use this K_min, what fraction of shapes at each K
    #    regress vs kpack=1?
    candidates = [512, 768, 1024, 1536, 2048]
    summary_rows = []
    # First, the *unclamped* per-K characterization (does not depend on K_min).
    by_K = defaultdict(list)  # K -> list of (delta_pct, t1, t2)
    for s in per_shape_rows:
        if s["delta_pct"] == "NA_clamped":
            continue
        K = int(s["K"])
        d = float(s["delta_pct"])
        t1 = float(s["kpack1_ms"])
        t2 = float(s["kpack2_ms"])
        by_K[K].append((d, t1, t2))

    print("\n=== Unclamped kpack=2 vs kpack=1 per K (from CSV, gate K_min=1024) ===")
    print(f"{'K':>6} {'n':>4} {'n_reg':>6} {'n_neut':>7} {'n_win':>6} "
          f"{'min_pct':>8} {'med_pct':>8} {'max_pct':>8}")
    for K in sorted(by_K):
        ds = [d for (d, _, _) in by_K[K]]
        n = len(ds)
        n_reg = sum(1 for d in ds if d > 3.0)        # >3% slower
        n_neut = sum(1 for d in ds if -3.0 <= d <= 3.0)
        n_win = sum(1 for d in ds if d < -3.0)        # >3% faster
        ds_sorted = sorted(ds)
        med = ds_sorted[len(ds_sorted) // 2]
        print(f"{K:>6} {n:>4} {n_reg:>6} {n_neut:>7} {n_win:>6} "
              f"{ds_sorted[0]:+8.2f} {med:+8.2f} {ds_sorted[-1]:+8.2f}")

    # NOTE: at small K (where the gate would clamp), the CSV cannot give us
    # an unclamped kpack=2 number for that exact shape.  The K_min sweep
    # below therefore needs a *separate* run with the gate disabled to
    # measure kpack=2 latency at every K.  This script outputs the
    # K_min table assuming you have a "gate-off" CSV with effective==
    # requested for every shape, and falls back to "use kpack=1 latency"
    # whenever the small-K kpack=2 measurement is missing (which yields a
    # conservative lower bound on the regression).

    print("\n=== K_min boundary sweep (geomean across the small-K cohort) ===")
    print(f"{'K_min':>6} {'kpack=2-only_K':>20} {'cohort_geomean_ms':>20}")
    for kmin in candidates:
        cohort_lat = []
        for s in per_shape_rows:
            K = int(s["K"])
            t1 = float(s["kpack1_ms"])
            if K < kmin:
                lat = t1  # gate clamps to 1
            else:
                if s["kpack2_ms"] == "NA_clamped":
                    # Should not happen for K >= 1024, but guard anyway
                    lat = t1
                else:
                    lat = float(s["kpack2_ms"])
            cohort_lat.append(lat)
        gm = _geomean(cohort_lat)
        # Always-kpack=1 baseline geomean for comparison
        baseline = _geomean([float(s["kpack1_ms"]) for s in per_shape_rows])
        speedup_pct = (baseline - gm) / baseline * 100.0
        ks_kpack2 = [K for K in sorted({int(s["K"]) for s in per_shape_rows})
                     if K >= kmin]
        summary_rows.append({
            "K_min": kmin,
            "ks_running_kpack2": ",".join(str(k) for k in ks_kpack2),
            "cohort_geomean_ms": f"{gm:.4f}",
            "always_kpack1_geomean_ms": f"{baseline:.4f}",
            "speedup_vs_always_kpack1_pct": f"{speedup_pct:+.2f}",
        })
        print(f"{kmin:>6} {ks_kpack2!s:>20} "
              f"{gm:.4f}  speedup_vs_kpack1 {speedup_pct:+.2f}%")

    out_sum = os.path.join(args.out_dir, "kmin_boundary_summary.csv")
    with open(out_sum, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"wrote {out_sum}")


if __name__ == "__main__":
    main()
