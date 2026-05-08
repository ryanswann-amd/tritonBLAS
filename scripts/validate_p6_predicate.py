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
#
# IMPORTANT: each (cell, arm) pair runs in a fresh subprocess to avoid the
# K-888 / K-967 in-process Triton-cache gotcha — once Triton's autotune
# selects a kernel for a given shape on first call, the env-flip on the
# next call inside the same process does NOT re-trigger selection (the
# selector caches per-shape).  K-1037 round_robin_n30 used the same
# subprocess-per-arm protocol; this is the canonical paired methodology.
# ---------------------------------------------------------------------------
def _bench_one_inproc(M, N, K, dtype_name, enable_p6, n_inner) -> list[float]:
    """Single-process kernel-only HIP-graph timing for ONE (cell, arm).

    Called as a fresh subprocess by phase_paired so each arm starts with a
    clean Triton autotune cache + fresh tritonblas module state.

    Uses K-1037 round-robin protocol: preallocate output `c`, capture
    n_capture=20 calls per CUDAGraph, replay graph and divide elapsed
    time by n_capture to amortise launch overhead.
    """
    import time
    import torch
    DTYPE_MAP = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}
    dtype = DTYPE_MAP[dtype_name]
    os.environ["TRITONBLAS_ENABLE_K1037_P6"] = "1" if enable_p6 else "0"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "include"))
    import tritonblas  # noqa: F401
    from tritonblas import matmul as tb_matmul

    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(K, N, dtype=dtype, device="cuda")
    c = torch.empty(M, N, dtype=dtype, device="cuda")

    def fn():
        tb_matmul(a, b, out=c)

    # Warm both paths so JIT compile + autotune complete before timing.
    for _ in range(5):
        torch.matmul(a, b, out=c)
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    n_capture = 20
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(n_capture):
            fn()
    torch.cuda.synchronize()

    times_us = []
    for _ in range(n_inner):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6 / n_capture)
    return times_us


def phase_paired(out_csv: Path, n_pairs: int) -> int:
    import json
    import subprocess
    import torch
    if not torch.cuda.is_available():
        sys.exit("phase=paired requires CUDA / ROCm")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "include"))
    import tritonblas  # noqa: F401

    BENCH_COHORT = (
        [(c, M, N, K, dt) for c, M, N, K, dt, _, _ in K1017_18CELL]
        + K1055_12CELL[-3:]                          # 3 fp16 negatives
        + [(c, M, N, K, dt) for c, M, N, K, dt in K1050_NEGATIVES] * 1  # +2 neg
        + K1055_12CELL[5:9]                          # 4 K-984 LAND anchors
    )
    # 18 + 3 + 2 + 4 = 27 cells (>= 6 negatives requested)

    def _spawn(M, N, K, dt, enable_p6, n_inner) -> float:
        """Spawn a fresh subprocess for ONE (cell, arm) sample, return median us."""
        env = dict(os.environ)
        env["TRITONBLAS_ENABLE_K1037_P6"] = "1" if enable_p6 else "0"
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--bench-arm",
            "--M", str(M), "--N", str(N), "--K", str(K),
            "--dtype", dt,
            "--enable-p6", "1" if enable_p6 else "0",
            "--n-inner", str(n_inner),
        ]
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"subprocess failed: {result.stderr[-2000:]}")
        # Last non-empty stdout line is JSON {"median_us": ...}
        for line in reversed(result.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return float(json.loads(line)["median_us"])
        raise RuntimeError(f"no median_us in subprocess output: {result.stdout[-2000:]}")

    rows = []
    for cid, M, N, K, dt in BENCH_COHORT:
        try:
            off_ts = [_spawn(M, N, K, dt, enable_p6=False, n_inner=20)
                      for _ in range(n_pairs)]
            on_ts  = [_spawn(M, N, K, dt, enable_p6=True,  n_inner=20)
                      for _ in range(n_pairs)]
            off_med = statistics.median(off_ts)
            on_med  = statistics.median(on_ts)
            speedup = off_med / on_med
            rows.append({"cid": cid, "M": M, "N": N, "K": K, "dtype": dt,
                         "off_us_med": off_med, "on_us_med": on_med,
                         "speedup_off_over_on": speedup,
                         "n_pairs": n_pairs, "n_inner": 20})
            print(f"{cid:12} ({M:5},{N:6},{K:5}) {dt:18}  "
                  f"off_med={off_med:.2f}us  on_med={on_med:.2f}us  "
                  f"speedup={speedup:.4f}x", flush=True)
        except Exception as e:                                  # noqa: BLE001
            rows.append({"cid": cid, "M": M, "N": N, "K": K, "dtype": dt,
                         "off_us_med": float("nan"), "on_us_med": float("nan"),
                         "speedup_off_over_on": float("nan"),
                         "n_pairs": n_pairs, "n_inner": 20,
                         "error": str(e)})
            print(f"{cid:12} ({M:5},{N:6},{K:5}) ERROR: {e}", flush=True)

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
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--phase",
                   choices=["predicate", "paired"],
                   default=None)
    p.add_argument("--output", type=Path,
                   default=Path("output/p6_predicate_verdicts.csv"))
    p.add_argument("--n-pairs", type=int, default=20)
    # Subprocess-arm hooks (called by phase_paired's per-arm spawn).
    p.add_argument("--bench-arm", action="store_true",
                   help="Internal: run one (cell, arm) sample as fresh subprocess")
    p.add_argument("--M", type=int)
    p.add_argument("--N", type=int)
    p.add_argument("--K", type=int)
    p.add_argument("--dtype", type=str)
    p.add_argument("--enable-p6", type=int, default=0)
    p.add_argument("--n-inner", type=int, default=20)
    args = p.parse_args()
    if args.bench_arm:
        times = _bench_one_inproc(args.M, args.N, args.K, args.dtype,
                                  bool(args.enable_p6), args.n_inner)
        print(json.dumps({"median_us": statistics.median(times),
                          "n": len(times)}))
        return 0
    if args.phase == "predicate":
        return phase_predicate(args.output)
    if args.phase == "paired":
        return phase_paired(args.output, args.n_pairs)
    print("--phase {predicate|paired} required (or --bench-arm)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
