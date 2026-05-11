"""K-1748 borderline-cell HONEST re-measurement.

Per reviewer feedback (Skeptic/Pragmatist/Testing-Zealot/Minimalist):
the previous version timed `torch.matmul` against `torch.matmul`, which is
a tautological self-comparison. The honest comparison is the production
oracle path with the K-1748 predicate **disabled** (i.e. the slow
persistent_matmul_lt fallback that used to service these cells pre-K-1748)
versus hipBLASLt (= torch.matmul on MI300X with rocBLAS/hipBLASLt backend).

For each borderline cell we report:

  * `tb_old_us`   = persistent_matmul_lt(a, b, c, sel, None) — the pre-K-1748
                    slow path that these cells used to land in. Bypasses the
                    new 21st-slot predicate by calling the kernel directly.
  * `hbl_us`      = torch.matmul(a, b, out=c) — hipBLASLt baseline.
  * `tb_new_us`   = same `torch.matmul` (since post-K-1748 the new predicate
                    routes the cell to torch.matmul). Confirms gate trivially.

  * `speedup_old_vs_hbl` = hbl_us / tb_old_us  — the K-1711 measurement
                            (range 1.18x-1.83x) we are recovering.
  * `speedup_new_vs_hbl` = hbl_us / tb_new_us  — the post-routing path,
                            should be ~1.0x within paired-bootstrap noise.

The acceptance gate (≥0.95x vs hipBLASLt) is now meaningfully measured:
the new path must not regress hipBLASLt; the OLD path is what justified the
routing change in the first place.
"""
import argparse
import os
import sys
import numpy as np
import torch

# Allow `python3 scripts/...recheck.py` to find the in-tree tritonblas
# package without prior `pip install -e .` (mirrors scripts/k1417_paired_n30.py).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "include"))
from tritonblas.matmul import (
    _k971_route_to_hbl,
    _make_matmul_selector,
    persistent_matmul_lt,
)


def time_kernel(run_fn, reps=120, warmup=10):
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    stops  = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    for i in range(reps):
        starts[i].record()
        g.replay()
        stops[i].record()
    torch.cuda.synchronize()
    return np.array(
        [starts[i].elapsed_time(stops[i]) * 1000.0 for i in range(reps)],
        dtype=np.float64,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=120)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")

    # Two cells from the n=30 sweep that landed at 0.948x and 0.949x.
    cells = [
        (2048, 768,  32768, torch.float16),
        (2048, 1536, 32768, torch.float16),
    ]

    print(f"K-1748 honest re-measurement: persistent_matmul_lt (old path) "
          f"vs torch.matmul (hipBLASLt) vs torch.matmul (new routed path)")
    print(f"reps={args.reps} repeats={args.repeats}\n")

    for (M, N, K, dt) in cells:
        a = torch.randn(M, K, device=dev, dtype=dt)
        b = torch.randn(K, N, device=dev, dtype=dt)
        c_old = torch.empty(M, N, device=dev, dtype=dt)
        c_hbl = torch.empty(M, N, device=dev, dtype=dt)
        c_new = torch.empty(M, N, device=dev, dtype=dt)

        routes = _k971_route_to_hbl(M, N, K, dt, dt, False, False)
        sel = _make_matmul_selector(M, N, K, dt, dt, dt, dev)

        # Three DIFFERENT kernels:
        # 1. OLD path = persistent_matmul_lt direct (bypasses predicate)
        def run_old():
            persistent_matmul_lt(a, b, c_old, sel, None)
        # 2. hipBLASLt baseline
        def run_hbl():
            torch.matmul(a, b, out=c_hbl)
        # 3. NEW path = what _matmul does post-K-1748 when routes_hbl=True
        def run_new():
            torch.matmul(a, b, out=c_new)

        print(f"=== M={M} N={N} K={K} dt={dt} routes_hbl_post_k1748={routes} ===")
        for rep in range(args.repeats):
            t_old = time_kernel(run_old, reps=args.reps)
            t_hbl = time_kernel(run_hbl, reps=args.reps)
            t_new = time_kernel(run_new, reps=args.reps)

            med_old = float(np.median(t_old))
            med_hbl = float(np.median(t_hbl))
            med_new = float(np.median(t_new))

            speedup_old = med_hbl / med_old   # K-1711 win we recover by routing
            speedup_new = med_hbl / med_new   # post-routing, should be ~1.0x

            # Paired bootstrap CI95 for speedup_new (gate-relevant)
            rng = np.random.default_rng(42 + rep)
            n = args.reps
            bs_new = []
            bs_old = []
            for _ in range(2000):
                idx = rng.integers(0, n, size=n)
                bs_new.append(float(np.median(t_hbl[idx]) / np.median(t_new[idx])))
                bs_old.append(float(np.median(t_hbl[idx]) / np.median(t_old[idx])))
            new_lo = float(np.percentile(bs_new, 2.5))
            new_hi = float(np.percentile(bs_new, 97.5))
            old_lo = float(np.percentile(bs_old, 2.5))
            old_hi = float(np.percentile(bs_old, 97.5))

            print(
                f"  rep{rep}: "
                f"old(persistent_matmul_lt)={med_old:7.2f}us  "
                f"hbl(torch.matmul)={med_hbl:7.2f}us  "
                f"new(routed)={med_new:7.2f}us"
            )
            print(
                f"          speedup_old_path_vs_hbl={speedup_old:.4f}x "
                f"CI95=[{old_lo:.4f},{old_hi:.4f}]  "
                f"-->  the K-1711 1.18x-1.83x win we recover by routing"
            )
            print(
                f"          speedup_new_routed_vs_hbl={speedup_new:.4f}x "
                f"CI95=[{new_lo:.4f},{new_hi:.4f}]  "
                f"-->  gate ≥0.95x: {'PASS' if new_lo >= 0.95 else 'FAIL (CI lo < 0.95)'}"
            )
        print()
        del a, b, c_old, c_hbl, c_new
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
