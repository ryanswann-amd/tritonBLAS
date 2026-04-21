#!/usr/bin/env python3
"""
CU Allocation Sweep — Data Collection for Comm/GEMM Overlap Heuristic

Runs overlap.py in 'standard' mode across a grid of:
  - GEMM shapes (M, N, K)
  - Communication tensor sizes
  - total_cus allocations (how many CUs the GEMM kernel gets)

All results are appended to a single CSV + per-config JSON files for
downstream heuristic training.

Usage (inside sbatch with 8 GPUs):
  python benchmarks/cu_allocation_sweep.py \
      --output-dir /shared/ryaswann/cu_sweep_results \
      --tier full         # or 'quick' for a fast sanity check
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime


GEMM_SHAPES = {
    "quick": [
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ],
    "full": [
        # Square shapes (common benchmarks)
        (1024, 1024, 4096),
        (2048, 2048, 4096),
        (4096, 4096, 4096),
        (4096, 4096, 8192),
        (8192, 8192, 4096),
        (8192, 8192, 8192),
        (16384, 16384, 8192),
        # Tall-skinny (activation-shaped, TP inference)
        (2048, 8192, 8192),
        (4096, 8192, 1024),
        (8192, 1024, 8192),
        (1024, 8192, 4096),
        # LLM-relevant (LLaMA-70B FFN dims with TP=8)
        (4096, 3584, 8192),
        (8192, 3584, 8192),
        # Medium non-power-of-2
        (3072, 3072, 4096),
        (6144, 6144, 4096),
    ],
}

COMM_SIZES = {
    "quick": [
        [8192, 4096],   # ~64 MB in bf16
    ],
    "full": [
        [4096, 1024],   # ~8 MB  — small comm
        [8192, 4096],   # ~64 MB — medium comm
        [8192, 8192],   # ~128 MB — large comm
    ],
}

# MI300X has 304 CUs (38 per XCD × 8 XCDs)
CU_SWEEP = {
    "quick": [304, 240, 176, 112],
    "full": [304, 288, 272, 256, 240, 224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64],
}

BACKEND = "ws"
COLLECTIVE = "all_reduce"
DTYPE = "bf16"
STEPS = 100
WARMUP = 10


def run_config(m, n, k, comm_size, total_cus, output_csv, output_json, nproc=8):
    """Run a single overlap.py standard benchmark and return success/failure."""
    comm_str = " ".join(str(s) for s in comm_size)
    json_path = os.path.join(
        os.path.dirname(output_json),
        f"json/m{m}_n{n}_k{k}_comm{'x'.join(str(s) for s in comm_size)}_cu{total_cus}.json"
    )
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    cmd = [
        "torchrun", f"--nproc_per_node={nproc}",
        "--", "benchmarks/overlap.py", "standard",
        "--backend", BACKEND,
        "--gemm-m", str(m), "--gemm-n", str(n), "--gemm-k", str(k),
        "--dtype", DTYPE,
        "--comm-size", *[str(s) for s in comm_size],
        "--collective", COLLECTIVE,
        "--steps", str(STEPS),
        "--warmup", str(WARMUP),
        "--total-cus", str(total_cus),
        "--output-csv", output_csv,
        "--output-json", json_path,
    ]

    print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
          f"M={m} N={n} K={k} comm={comm_size} CUs={total_cus} ... ",
          end="", flush=True)

    t0 = time.time()
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"OK ({elapsed:.1f}s)")
        return True
    else:
        print(f"FAIL (rc={result.returncode}, {elapsed:.1f}s)")
        stderr_tail = result.stderr.decode("utf-8", errors="replace")[-500:]
        print(f"    stderr: {stderr_tail}")
        return False


def main():
    parser = argparse.ArgumentParser(description="CU allocation sweep for overlap heuristic data")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for CSV + JSON results")
    parser.add_argument("--tier", type=str, default="full", choices=["quick", "full"],
                        help="Sweep tier: 'quick' (~16 configs) or 'full' (~720 configs)")
    parser.add_argument("--nproc", type=int, default=8,
                        help="Number of GPUs (torchrun --nproc_per_node)")
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Skip first N configs (for resuming after crash)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "json"), exist_ok=True)

    shapes = GEMM_SHAPES[args.tier]
    comms = COMM_SIZES[args.tier]
    cus = CU_SWEEP[args.tier]

    configs = list(itertools.product(shapes, comms, cus))
    total = len(configs)

    output_csv = os.path.join(args.output_dir, "cu_allocation_sweep.csv")

    print(f"=" * 72)
    print(f"CU Allocation Sweep — {args.tier} tier")
    print(f"  GEMM shapes:  {len(shapes)}")
    print(f"  Comm sizes:   {len(comms)}")
    print(f"  CU points:    {len(cus)}")
    print(f"  Total configs: {total}")
    print(f"  Output CSV:   {output_csv}")
    print(f"  Output JSON:  {args.output_dir}/json/")
    if args.resume_from > 0:
        print(f"  Resuming from config #{args.resume_from}")
    print(f"=" * 72)
    print()

    succeeded = 0
    failed = 0
    skipped = args.resume_from
    t_start = time.time()

    for i, ((m, n, k), comm, cu) in enumerate(configs):
        if i < args.resume_from:
            continue

        print(f"[{i+1}/{total}]", end="")
        try:
            ok = run_config(m, n, k, comm, cu, output_csv,
                            os.path.join(args.output_dir, "placeholder"), args.nproc)
            if ok:
                succeeded += 1
            else:
                failed += 1
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (300s limit)")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

        elapsed_total = time.time() - t_start
        done = i + 1 - args.resume_from
        rate = elapsed_total / done if done > 0 else 0
        remaining = (total - i - 1) * rate
        print(f"    Progress: {done}/{total - args.resume_from} done, "
              f"{succeeded} ok, {failed} fail, "
              f"~{remaining/60:.0f} min remaining")

    elapsed_total = time.time() - t_start
    print(f"\n{'=' * 72}")
    print(f"Sweep complete in {elapsed_total/60:.1f} minutes")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
    print(f"  Skipped:   {skipped}")
    print(f"  CSV:       {output_csv}")
    print(f"{'=' * 72}")

    summary = {
        "tier": args.tier,
        "total_configs": total,
        "succeeded": succeeded,
        "failed": failed,
        "elapsed_minutes": elapsed_total / 60,
        "timestamp": datetime.now().isoformat(),
        "shapes": [(m, n, k) for m, n, k in shapes],
        "comm_sizes": comms,
        "cu_points": cus,
    }
    with open(os.path.join(args.output_dir, "sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
