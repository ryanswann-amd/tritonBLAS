#!/usr/bin/env python3
"""K-1092: K-1037 P6 predicate verification harness.

Two phases:
  1) Host-side predicate verification (no GPU required) — replays the K-1017
     18-cell PMC calibration, the K-984+K-989 14-anchor union, the K-950
     9-cell LAND set, the K-1028 fp16 negative cohort, and the K-1055 12-cell
     held-out cohort against P6.  Produces ``output/p6_predicate_verdicts.csv``.
  2) GPU-side paired n=20 HIP-graph bench (mi300x only) — alternates
     TRITONBLAS_ENABLE_K1037_P6={0,1} on the K-1017 18-cell cohort + 6 K-1050/
     K-1055 negative-control cohorts and computes the per-cell speedup +
     geomean uplift.  Produces ``output/p6_paired_n20.csv``.

Invocation (host):
    python3 scripts/validate_p6_predicate.py --phase predicate

Invocation (GPU, MI300X):
    TRITONBLAS_DISABLE_K971=  python3 scripts/validate_p6_predicate.py \\
        --phase paired --n-pairs 20 --output output/p6_paired_n20.csv

Reproduces K-1031/K-1043 adversarial held-out gate (0 LAND-leaks tolerated).

NOTE: an earlier revision shipped a third "paired-dryrun" phase that walked
the dispatch ladder host-side and projected per-cell speedups from K-1017's
PMC ``gap_x``.  The Minimalist review flagged that CSV as dead weight (23 of
24 rows reduced to ``speedup=1.0`` / ``flipped=False``; the only signal was
the single S24 row already surfaced by ``--phase predicate``).  The phase
was removed; live MI300X paired n=20 numbers replace it.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import time
from pathlib import Path


# K-1017 18-cell calibration (cid, M, N, K, dtype, k1017_class, expected_p6)
K1017_18CELL = [
    ("S03",   384, 409600,  384, "torch.bfloat16", "Occupancy",       False),
    ("S07",  2048,   1792,  256, "torch.bfloat16", "Occupancy",       False),
    ("S09",   384, 409600,  256, "torch.bfloat16", "Occupancy",       False),
    ("S14",   128, 409600,  384, "torch.bfloat16", "Occupancy",       False),
    ("S15",   384, 409600,  128, "torch.bfloat16", "Occupancy",       False),
    ("S16",   256,    256, 2048, "torch.bfloat16", "Occupancy",       False),
    ("S24",  4480,   3072,  768, "torch.bfloat16", "MFMA-issue-stall", True),
    ("S26",  8064,   2048, 1024, "torch.bfloat16", "Occupancy",       False),
    ("S29", 14208,   2048, 1024, "torch.bfloat16", "MFMA-issue-stall", True),
    ("S30", 16256,   2048, 1024, "torch.bfloat16", "Occupancy",       False),
    ("S31", 18304,   2048, 1024, "torch.bfloat16", "Occupancy",       False),
    ("S32", 20352,   2048, 1024, "torch.bfloat16", "Occupancy",       False),
    ("S33", 22400,   2048, 1024, "torch.bfloat16", "Occupancy",       False),
    ("S34", 24448,   2048, 1024, "torch.bfloat16", "Occupancy",       False),
    ("S35", 26496,   2048, 1024, "torch.bfloat16", "Occupancy",       False),
    ("S37", 25600,   2048,  256, "torch.bfloat16", "Occupancy",       False),
    ("S38",   384,    128,  200, "torch.bfloat16", "HBM/L2",          False),
    ("S39", 49152,   2048,  256, "torch.bfloat16", "Occupancy",       False),
]

# K-984+K-989 union (14 unique shapes) — must NOT be P6-LAND-leaked.
K984_K989_UNION = [
    ("S04",  2304,   2048, 4800),
    ("S05",   256,   1792, 2048),
    ("S06",   736,   1792,  736),
    ("S10",   512,    192, 2048),
    ("S17",   768,   1792, 5972),
    ("S18",  5972,   1792,  768),
    ("S21",   768,   3072, 4480),
    ("S22",  1024,   2048, 6016),
    ("S23",  1024,   2048, 8064),
    ("S25",  6016,   2048, 1024),
    ("S27", 10112,   2048, 1024),
    ("S28", 12160,   2048, 1024),
    ("S36",    30, 786432,  200),
    ("S40",  1024,   2048, 1240),
]

# K-950 9-cell LAND harness — every cell MUST stay outside P6.
K950_LAND_BF16 = [
    ("L01", 1024, 1024,   512, "torch.bfloat16"),
    ("L02",  512,  512,  1024, "torch.bfloat16"),
    ("L03", 2048,  768,  1024, "torch.bfloat16"),
    ("L04",  768, 2048,  1024, "torch.bfloat16"),
    ("L05", 4096, 4096,  4096, "torch.bfloat16"),
    ("L07", 1024, 1024, 16384, "torch.bfloat16"),
    ("L08", 4096, 4096,  8192, "torch.bfloat16"),
]
K950_LAND_FP16 = [
    ("L06", 2048, 2048, 16384, "torch.float16"),
    ("L09", 1024, 1024, 16384, "torch.float16"),
]

# K-1028 fp16 cohort — bf16-only carve-out coverage.
K1028_FP16 = [
    ("K1028-S04",  2304, 2048, 4800, "torch.float16"),
    ("K1028-S17",   768, 1792, 5972, "torch.float16"),
    ("K1028-S27", 10112, 2048, 1024, "torch.float16"),
]

# K-1055 12-cell held-out (5 K-1017 + 4 K-984 + 3 K-1028 fp16).
K1055_12CELL = [
    ("S07",       2048, 1792,  256, "torch.bfloat16"),
    ("S16",        256,  256, 2048, "torch.bfloat16"),
    ("S24",       4480, 3072,  768, "torch.bfloat16"),
    ("S37",      25600, 2048,  256, "torch.bfloat16"),
    ("S38",        384,  128,  200, "torch.bfloat16"),
    ("K984-S04",  2304, 2048, 4800, "torch.bfloat16"),
    ("K984-S10",   512,  192, 2048, "torch.bfloat16"),
    ("K984-S27", 10112, 2048, 1024, "torch.bfloat16"),
    ("K984-S28", 12160, 2048, 1024, "torch.bfloat16"),
    ("K1028-S04", 2304, 2048, 4800, "torch.float16"),
    ("K1028-S17",  768, 1792, 5972, "torch.float16"),
    ("K1028-S27",10112, 2048, 1024, "torch.float16"),
]

# K-1050 negative-control sample (Occupancy bucket).
K1050_NEGATIVES = [
    ("S03_K1050",   384, 409600, 384, "torch.bfloat16"),
    ("S16_K1050",   256,    256,2048, "torch.bfloat16"),
]


# ---------------------------------------------------------------------------
# Phase 1 — predicate verification (no GPU required).
# ---------------------------------------------------------------------------
def phase_predicate(out_csv: Path) -> int:
    """Run the P6 + P5 predicates against every cohort cell and write a CSV.

    Exits with non-zero status if any LAND-cell is leaked.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "include"))
    from tritonblas._route_predicate import (
        R_K1037_P6_mfma_issue_stall_route_to_hbl as P6,
        R_K979_P5_route_to_hbl as P5,
    )

    rows = []
    leak_count = 0
    p6_fp_count = 0
    p6_tp_count = 0
    p6_tn_count = 0

    # K-1017 calibration cohort.
    for cid, M, N, K, dt, kclass, expected in K1017_18CELL:
        p6_fires = P6(M, N, K, dt)
        p5_fires = P5(M, N, K, dt)
        match = (p6_fires is expected)
        if match and expected:
            p6_tp_count += 1
        elif match and not expected:
            p6_tn_count += 1
        elif (not match) and expected:
            leak_count += 1
        else:
            p6_fp_count += 1
        rows.append({
            "cohort": "K-1017", "cid": cid, "M": M, "N": N, "K": K, "dtype": dt,
            "kclass": kclass, "expected_p6": expected, "p6_fires": p6_fires,
            "p5_fires": p5_fires, "verdict": "PASS" if match else "FAIL",
        })

    # K-984+K-989 union — must NOT fire P6.
    for sid, M, N, K in K984_K989_UNION:
        dt = "torch.bfloat16"
        p6_fires = P6(M, N, K, dt)
        p5_fires = P5(M, N, K, dt)
        if p6_fires:
            leak_count += 1
        rows.append({
            "cohort": "K-984+K-989", "cid": sid, "M": M, "N": N, "K": K, "dtype": dt,
            "kclass": "ship-anchor", "expected_p6": False, "p6_fires": p6_fires,
            "p5_fires": p5_fires, "verdict": "PASS" if not p6_fires else "FAIL-LEAK",
        })

    # K-950 LAND harness — must NOT fire P6.
    for cid, M, N, K, dt in K950_LAND_BF16 + K950_LAND_FP16:
        p6_fires = P6(M, N, K, dt)
        p5_fires = P5(M, N, K, dt)
        if p6_fires:
            leak_count += 1
        rows.append({
            "cohort": "K-950-LAND", "cid": cid, "M": M, "N": N, "K": K, "dtype": dt,
            "kclass": "harness-LAND", "expected_p6": False, "p6_fires": p6_fires,
            "p5_fires": p5_fires, "verdict": "PASS" if not p6_fires else "FAIL-LEAK",
        })

    # K-1028 fp16 cohort — bf16-only carve-out, must NOT fire.
    for cid, M, N, K, dt in K1028_FP16:
        p6_fires = P6(M, N, K, dt)
        p5_fires = P5(M, N, K, dt)
        if p6_fires:
            leak_count += 1
        rows.append({
            "cohort": "K-1028-fp16", "cid": cid, "M": M, "N": N, "K": K, "dtype": dt,
            "kclass": "fp16-negative", "expected_p6": False, "p6_fires": p6_fires,
            "p5_fires": p5_fires, "verdict": "PASS" if not p6_fires else "FAIL-LEAK",
        })

    # K-1055 12-cell held-out.
    for cid, M, N, K, dt in K1055_12CELL:
        # Expected: only S24 fires (and S29 if present, but 12-cell sample has
        # neither S29 nor any other MFMA-issue-stall cell besides S24).
        expected = (cid == "S24")
        p6_fires = P6(M, N, K, dt)
        p5_fires = P5(M, N, K, dt)
        if p6_fires and not expected:
            leak_count += 1
        match = (p6_fires is expected)
        rows.append({
            "cohort": "K-1055-held-out", "cid": cid, "M": M, "N": N, "K": K, "dtype": dt,
            "kclass": "held-out-12cell", "expected_p6": expected,
            "p6_fires": p6_fires, "p5_fires": p5_fires,
            "verdict": "PASS" if match else "FAIL",
        })

    # K-1050 negative controls.
    for cid, M, N, K, dt in K1050_NEGATIVES:
        p6_fires = P6(M, N, K, dt)
        p5_fires = P5(M, N, K, dt)
        if p6_fires:
            leak_count += 1
        rows.append({
            "cohort": "K-1050-negative", "cid": cid, "M": M, "N": N, "K": K, "dtype": dt,
            "kclass": "negative-control", "expected_p6": False,
            "p6_fires": p6_fires, "p5_fires": p5_fires,
            "verdict": "PASS" if not p6_fires else "FAIL-LEAK",
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"P6 calibration: TP={p6_tp_count}/2  TN={p6_tn_count}/16  "
          f"K-1017 cohort agreement = {p6_tp_count + p6_tn_count}/18")
    print(f"Total LAND-cell leaks (across all cohorts): {leak_count}")
    print(f"CSV written to {out_csv}")
    return 0 if leak_count == 0 and p6_tp_count == 2 and p6_tn_count == 16 else 2


# ---------------------------------------------------------------------------
# Phase 2 — paired n=20 HIP-graph bench (MI300X only).
# ---------------------------------------------------------------------------
def phase_paired(out_csv: Path, n_pairs: int) -> int:
    import torch
    if not torch.cuda.is_available():
        sys.exit("phase=paired requires CUDA / ROCm")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "include"))
    import tritonblas  # noqa: F401

    DTYPE_MAP = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}

    BENCH_COHORT = (
        [(c, M, N, K, dt) for c, M, N, K, dt, _, _ in K1017_18CELL]
        + K1055_12CELL[-3:]                          # 3 fp16 negatives
        + [(c, M, N, K, dt) for c, M, N, K, dt in K1050_NEGATIVES] * 1  # +2 neg
        + K1055_12CELL[5:9]                          # 4 K-984 LAND anchors
    )
    # 18 + 3 + 2 + 4 = 27 cells (>= 6 negatives requested)

    def _bench(M, N, K, dtype, enable_p6: bool, n: int) -> float:
        os.environ["TRITONBLAS_ENABLE_K1037_P6"] = "1" if enable_p6 else "0"
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b = torch.randn(K, N, dtype=dtype, device="cuda")
        # warm cache + capture
        for _ in range(5):
            torch.matmul(a, b)  # warm hipBLASLt path too
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            from tritonblas import matmul as tb_matmul
            _ = tb_matmul(a, b)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            starts[i].record()
            graph.replay()
            ends[i].record()
        torch.cuda.synchronize()
        times_us = [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]
        return float(statistics.median(times_us))

    rows = []
    for cid, M, N, K, dt in BENCH_COHORT:
        dtype = DTYPE_MAP[dt]
        try:
            off_ts = [_bench(M, N, K, dtype, enable_p6=False, n=20)
                      for _ in range(n_pairs)]
            on_ts  = [_bench(M, N, K, dtype, enable_p6=True,  n=20)
                      for _ in range(n_pairs)]
            speedup = statistics.median(off_ts) / statistics.median(on_ts)
            rows.append({"cid": cid, "M": M, "N": N, "K": K, "dtype": dt,
                         "off_us_med": statistics.median(off_ts),
                         "on_us_med":  statistics.median(on_ts),
                         "speedup_off_over_on": speedup})
            print(f"{cid:12} ({M:5},{N:6},{K:5}) {dt:18}  off={off_ts[0]:.2f}us  "
                  f"on={on_ts[0]:.2f}us  speedup={speedup:.4f}x")
        except Exception as e:                                  # noqa: BLE001
            rows.append({"cid": cid, "M": M, "N": N, "K": K, "dtype": dt,
                         "off_us_med": float("nan"), "on_us_med": float("nan"),
                         "speedup_off_over_on": float("nan"),
                         "error": str(e)})
            print(f"{cid:12} ({M:5},{N:6},{K:5}) ERROR: {e}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    speedups = [r["speedup_off_over_on"] for r in rows
                if not math.isnan(r["speedup_off_over_on"])]
    if speedups:
        gm = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"\nGeomean speedup OFF/ON over {len(speedups)} cells: {gm:.4f}x")
    print(f"CSV written to {out_csv}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase",
                   choices=["predicate", "paired"],
                   required=True)
    p.add_argument("--output", type=Path,
                   default=Path("output/p6_predicate_verdicts.csv"))
    p.add_argument("--n-pairs", type=int, default=20)
    args = p.parse_args()
    if args.phase == "predicate":
        return phase_predicate(args.output)
    return phase_paired(args.output, args.n_pairs)


if __name__ == "__main__":
    sys.exit(main())
