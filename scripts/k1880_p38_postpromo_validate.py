#!/usr/bin/env python3
"""K-1880 P38 post-promotion paired n=30 HIP-graph hot-cache validation.

Per PRD step 5: re-run paired n=30 HIP-graph hot-cache benchmark on MI300X
to validate that the LIVE tritonblas dispatcher (post-P38 promotion) routes
the K-1873 N=416 K-COMPLEMENT verified-winner cells to hipBLASLt with
geomean TB-LIVE / HBL >= 1.0 (i.e., parity — confirms the dispatcher wiring
is correct and the routed cells inherit hipBLASLt's measured headroom from
K-1873 rather than dropping back to TB's slow path).

Methodology mirrors K-1417 / K-1873:
  * HIP-graph capture+replay (hot cache, low jitter) via cuda graphs
  * n=30 paired iterations (TB-LIVE then HBL, interleaved per replay)
  * Paired t-test (scipy.stats.ttest_rel) on per-replay times
  * Per-cell ratio = mean(TB-LIVE) / mean(HBL); cohort geomean over the
    18-cell envelope and the 17-cell admit set.

Per R-K1850 the post-promotion gmean ~ 1.0 ± noise on the dispatcher path
is the wiring signal (NOT headroom — headroom evidence comes from the K-1873
parent paired sweep).  The acceptance criterion is geomean >= 1.0 with no
cell regression below ~0.95x.

Cells: M ∈ {2048, 4096, 8192} × N=416 × K ∈ {4096, 8192, 16384}
       × {bf16, fp16}  =  18 cells

The 1 cell (2048, 416, 4096, bf16) routes via P5 (NOT P38) — already
covered upstream — so it is included to verify the parity it had at K-1873
is preserved.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import tritonblas


CELLS = [
    (M, 416, K, dtype)
    for M in (2048, 4096, 8192)
    for K in (4096, 8192, 16384)
    for dtype in (torch.bfloat16, torch.float16)
]


def _dtype_str(d: torch.dtype) -> str:
    return {
        torch.bfloat16: "torch.bfloat16",
        torch.float16: "torch.float16",
    }[d]


def make_graph(fn, a, b, n_replays):
    """Capture an n_replays-deep HIP graph for fn(a, b) and return a
    callable that replays it once."""
    # warm jit / autotune outside the graph
    for _ in range(3):
        _ = fn(a, b)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for _ in range(n_replays):
                _ = fn(a, b)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    return g


def time_graph(graph, n_iter):
    samples = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return np.asarray(samples)


def run_cell(M, N, K, dtype, n_iter, warmup, n_replays):
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)

    def tb_fn(x, y):
        return tritonblas.matmul(x, y)
    def hbl_fn(x, y):
        return torch.matmul(x, y)

    tb_graph = make_graph(tb_fn, a, b, n_replays)
    hbl_graph = make_graph(hbl_fn, a, b, n_replays)

    # Pre-warm both graphs
    for _ in range(warmup):
        tb_graph.replay(); hbl_graph.replay()
    torch.cuda.synchronize()

    # Interleave per-iter to control thermal drift
    tb_samples = np.empty(n_iter); hbl_samples = np.empty(n_iter)
    for i in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tb_graph.replay()
        torch.cuda.synchronize()
        tb_samples[i] = (time.perf_counter() - t0) / n_replays
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hbl_graph.replay()
        torch.cuda.synchronize()
        hbl_samples[i] = (time.perf_counter() - t0) / n_replays
    return tb_samples, hbl_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Per-cell CSV")
    p.add_argument("--summary", required=True, help="Cohort JSON summary")
    p.add_argument("--n_iter", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--n_replays", type=int, default=20)
    args = p.parse_args()

    print(f"K-1880 P38 post-promotion paired n=30 HIP-graph validation",
          flush=True)
    print(f"  n_iter={args.n_iter} warmup={args.warmup} n_replays={args.n_replays}",
          flush=True)
    print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    summary = {"cells": [], "params": vars(args)}
    out_rows = []
    for (M, N, K, dtype) in CELLS:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            tb, hbl = run_cell(M, N, K, dtype, args.n_iter, args.warmup,
                               args.n_replays)
        except Exception as e:
            print(f"  ({M},{N},{K},{_dtype_str(dtype):16s}) FAILED: {e}",
                  flush=True)
            continue
        ratio = float(tb.mean() / hbl.mean())
        ratio_med = float(np.median(tb) / np.median(hbl))
        # paired t-test on per-iter times: H0 = tb == hbl
        t_stat, p_val = stats.ttest_rel(tb, hbl)
        cell = {
            "M": M, "N": N, "K": K, "dtype": _dtype_str(dtype),
            "tb_mean_us": float(tb.mean() * 1e6),
            "hbl_mean_us": float(hbl.mean() * 1e6),
            "ratio_mean": ratio,
            "ratio_median": ratio_med,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "n_iter": args.n_iter,
            "n_replays": args.n_replays,
        }
        summary["cells"].append(cell)
        out_rows.append(",".join(str(x) for x in [
            M, N, K, _dtype_str(dtype),
            cell["tb_mean_us"], cell["hbl_mean_us"],
            ratio, ratio_med, t_stat, p_val,
        ]))
        print(f"  ({M},{N},{K:5d},{_dtype_str(dtype):16s}) "
              f"TB-LIVE={cell['tb_mean_us']:7.1f}us  HBL={cell['hbl_mean_us']:7.1f}us  "
              f"ratio={ratio:.4f} (med={ratio_med:.4f}) p={p_val:.2e}  "
              f"({time.perf_counter()-t0:.1f}s)",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("M,N,K,dtype,tb_mean_us,hbl_mean_us,ratio_mean,ratio_median,"
                "t_stat,p_value\n")
        f.write("\n".join(out_rows) + "\n")

    # Cohort geomean over all cells & over admit-only (17 cells; exclude the
    # P5-aliased (2048,416,4096,bf16))
    ratios = np.asarray([c["ratio_mean"] for c in summary["cells"]])
    summary["cohort_geomean_all_18"] = float(np.exp(np.mean(np.log(ratios))))
    admit = [c for c in summary["cells"]
             if not (c["M"] == 2048 and c["K"] == 4096
                     and c["dtype"] == "torch.bfloat16")]
    admit_ratios = np.asarray([c["ratio_mean"] for c in admit])
    summary["cohort_geomean_admit_17"] = float(
        np.exp(np.mean(np.log(admit_ratios))))
    summary["n_cells_ge_1p0"] = int(np.sum(ratios >= 1.0))
    summary["n_cells_ge_0p95"] = int(np.sum(ratios >= 0.95))
    summary["n_cells"] = len(summary["cells"])
    summary["min_ratio"] = float(ratios.min())
    summary["max_ratio"] = float(ratios.max())

    print(f"\nCOHORT geomean TB-LIVE/HBL (all 18 cells)   = "
          f"{summary['cohort_geomean_all_18']:.4f}", flush=True)
    print(f"COHORT geomean TB-LIVE/HBL (17 admit cells) = "
          f"{summary['cohort_geomean_admit_17']:.4f}", flush=True)
    print(f"  per-cell ratio range: [{summary['min_ratio']:.4f}, "
          f"{summary['max_ratio']:.4f}]", flush=True)
    print(f"  cells with ratio >= 1.00 : {summary['n_cells_ge_1p0']}/"
          f"{summary['n_cells']}", flush=True)
    print(f"  cells with ratio >= 0.95 : {summary['n_cells_ge_0p95']}/"
          f"{summary['n_cells']}", flush=True)

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
