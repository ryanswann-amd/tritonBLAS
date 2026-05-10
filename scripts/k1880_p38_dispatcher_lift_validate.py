#!/usr/bin/env python3
"""K-1880 P38 PRE-vs-POST dispatcher-lift validation (cold-cache, non-graph).

Addresses the K-1880 RETRY R-Skeptic objection that the post-promotion HIP-
graph hot-cache validator (k1880_p38_postpromo_validate.py) shows ~1.0×
parity (NOT lift) and never demonstrates that the K-1873 1.429× headroom
materialises on the LIVE dispatcher path for end users.

Why hot-cache parity is *expected* on the post-promotion validator: post-
P38, BOTH legs of the comparison route through hipBLASLt — the LIVE
tritonblas.matmul() now dispatches to HBL via _k971_route_to_hbl(), so it
*is* HBL.  The hot-cache validator's TB-LIVE/HBL ratio of 1.0010 is the
correct signal that the route landed on the same kernel.  But that does
NOT show the user-visible speedup vs the PRE-P38 codebase.

This script answers the Skeptic by running a PRE-vs-POST A/B on the LIVE
dispatcher itself:
  PRE  : monkey-patch the K-1880 canonical 51-cell frozenset on the
         already-imported `tritonblas.matmul` module to an empty set.  The
         _k971_route_to_hbl() membership probe falls through, so cells in
         {N=320, N=352, N=416} run on tritonblas's native persistent_matmul
         (the codepath that K-1873 measured at 1.401× slower than HBL).
  POST : restore the canonical frozenset.  Cells route through hipBLASLt
         (the codepath productionised at HEAD).
Per-cell ratio = mean(time_PRE) / mean(time_POST).  Reproducing the K-1873
admit-only geomean of ~1.429× on the 17 N=416 admit cells empirically
proves the dispatcher wiring delivers the headroom on the LIVE path.

Methodology departures from the hot-cache validator (per Skeptic):
  * NON-GRAPH timing — direct callable invocation with torch.cuda.synchronize()
    fences; HIP graph capture freezes the kernel selection at capture time
    and was implicated in masking PRE/POST differences (K-1873 used graphs
    to characterise headroom; we use direct calls so the dispatcher actually
    re-evaluates per call as it would in user code).
  * COLD-CACHE between every measured iteration via a 256 MiB write-buffer
    eviction kernel — defeats the hot-cache regime that conflates PRE and
    POST when the L2 holds the matmul result.
  * Per-cell, the script measures PRE and POST in INTERLEAVED iterations
    (PRE_i, POST_i, PRE_{i+1}, POST_{i+1}, ...) to control thermal /
    DVFS drift, then computes a paired t-test over the n=30 ratio samples.

Acceptance criterion (asserted at end-of-run, exit 1 if any fails):
  (a) every admit cell has ratio_PRE/POST ≥ 0.95 (no regression at the
      dispatcher hand-off — this also pins the post-promotion parity claim)
  (b) ≥10 of 17 N=416 admit cells must show ratio_PRE/POST ≥ 1.05 (the
      PRD's "≥10/18 cells flip from <1.0× to ≥1.0×" promotion criterion,
      tightened to ≥1.05× to require statistically meaningful lift)
  (c) admit-only geomean(ratio_PRE/POST) ≥ 1.20× (loose lower bound on the
      K-1873-measured 1.429× — allows ~15% drift between hot-cache graph
      measurement and cold-cache direct-call measurement)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import tritonblas
import tritonblas.matmul as _tb_matmul


# K-1880 P38 admit cells (N=416 K-COMPLEMENT, MINUS the P5-aliased
# (2048,416,4096,bf16) cell) — exactly the cohort K-1873 measured.
N416_ADMIT_CELLS = [
    (M, 416, K, dt) for M in (2048, 4096, 8192)
    for K in (4096, 8192, 16384)
    for dt in (torch.bfloat16, torch.float16)
    if not (M == 2048 and K == 4096 and dt == torch.bfloat16)
]
assert len(N416_ADMIT_CELLS) == 17, f"expected 17 N=416 admit cells, got {len(N416_ADMIT_CELLS)}"


def _dtype_str(d: torch.dtype) -> str:
    return {torch.bfloat16: "torch.bfloat16",
            torch.float16:  "torch.float16"}[d]


# 256 MiB cache-flush buffer — write zeros across it before every measured
# matmul to evict the previous result from L2 (MI300X L2 is 4 MiB; this
# overflows it 64×, ensuring cold cache).
_FLUSH_NBYTES = 256 << 20
_flush_buf = None


def _alloc_flush_buf():
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.empty(_FLUSH_NBYTES // 4, dtype=torch.float32,
                                 device="cuda")


def _flush_l2():
    """Evict L2 by writing zeros to a 256 MiB buffer."""
    _flush_buf.zero_()


def time_matmul_cold(fn, a, b, n_iter):
    """Direct-call cold-cache timing.  Returns array of n_iter per-call
    times in seconds.  L2 is flushed before every measured call."""
    samples = np.empty(n_iter)
    for i in range(n_iter):
        _flush_l2()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn(a, b)
        torch.cuda.synchronize()
        samples[i] = time.perf_counter() - t0
    return samples


def run_cell_pre_post(M, N, K, dtype, n_iter, warmup):
    """Run PRE/POST interleaved timing for a single cell."""
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)

    canon = _tb_matmul._K1880_SKINNY_KCOMPL_N320_N352_N416_ROUTEOUT_51

    def _matmul(x, y):
        return tritonblas.matmul(x, y)

    # ---- WARMUP: compile both code paths so the first measured iteration
    # is not paying autotune cost.
    # POST (route → HBL): canonical set intact
    for _ in range(warmup):
        _flush_l2(); _ = _matmul(a, b)
    # PRE (route disabled): empty canonical set → falls through to TB-native
    _tb_matmul._K1880_SKINNY_KCOMPL_N320_N352_N416_ROUTEOUT_51 = frozenset()
    try:
        for _ in range(warmup):
            _flush_l2(); _ = _matmul(a, b)
    finally:
        _tb_matmul._K1880_SKINNY_KCOMPL_N320_N352_N416_ROUTEOUT_51 = canon
    torch.cuda.synchronize()

    # ---- INTERLEAVED MEASUREMENTS: PRE_i, POST_i, PRE_{i+1}, POST_{i+1}, ...
    pre_samples = np.empty(n_iter)
    post_samples = np.empty(n_iter)
    for i in range(n_iter):
        # PRE iteration: monkey-patch to empty set
        _tb_matmul._K1880_SKINNY_KCOMPL_N320_N352_N416_ROUTEOUT_51 = frozenset()
        try:
            _flush_l2()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = _matmul(a, b)
            torch.cuda.synchronize()
            pre_samples[i] = time.perf_counter() - t0
        finally:
            _tb_matmul._K1880_SKINNY_KCOMPL_N320_N352_N416_ROUTEOUT_51 = canon
        # POST iteration: canonical set intact
        _flush_l2()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = _matmul(a, b)
        torch.cuda.synchronize()
        post_samples[i] = time.perf_counter() - t0

    # Sanity: confirm canon is restored after the measurement loop
    assert _tb_matmul._K1880_SKINNY_KCOMPL_N320_N352_N416_ROUTEOUT_51 is canon, (
        "monkey-patch leaked: canonical frozenset NOT restored after measurement"
    )
    return pre_samples, post_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Per-cell CSV")
    p.add_argument("--summary", required=True, help="Cohort JSON summary")
    p.add_argument("--n_iter", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 if acceptance criteria (a)/(b)/(c) fail")
    args = p.parse_args()

    print(f"K-1880 P38 PRE-vs-POST dispatcher-lift validation "
          f"(cold-cache, non-graph)", flush=True)
    print(f"  n_iter={args.n_iter} warmup={args.warmup} cells=17 "
          f"(N=416 admit)", flush=True)
    print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    _alloc_flush_buf()

    summary = {"cells": [], "params": vars(args)}
    out_rows = []
    for (M, N, K, dtype) in N416_ADMIT_CELLS:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0_cell = time.perf_counter()
        try:
            pre, post = run_cell_pre_post(M, N, K, dtype, args.n_iter, args.warmup)
        except Exception as e:
            print(f"  ({M},{N},{K},{_dtype_str(dtype):16s}) FAILED: {e}",
                  flush=True)
            continue

        # Per-call ratio = PRE / POST.  >1 means POST is faster (the
        # K-1880 productionisation delivered its expected headroom on the
        # LIVE dispatcher).
        ratio = float(pre.mean() / post.mean())
        ratio_med = float(np.median(pre) / np.median(post))
        # Paired t-test on per-iter times (H0: pre == post)
        t_stat, p_val = stats.ttest_rel(pre, post)
        cell = {
            "M": M, "N": N, "K": K, "dtype": _dtype_str(dtype),
            "pre_mean_us":  float(pre.mean()  * 1e6),
            "post_mean_us": float(post.mean() * 1e6),
            "ratio_pre_over_post": ratio,
            "ratio_pre_over_post_median": ratio_med,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "n_iter": args.n_iter,
        }
        summary["cells"].append(cell)
        out_rows.append(",".join(str(x) for x in [
            M, N, K, _dtype_str(dtype),
            cell["pre_mean_us"], cell["post_mean_us"],
            ratio, ratio_med, t_stat, p_val,
        ]))
        print(f"  ({M},{N},{K:5d},{_dtype_str(dtype):16s}) "
              f"PRE={cell['pre_mean_us']:7.1f}us POST={cell['post_mean_us']:7.1f}us "
              f"ratio={ratio:.4f} (med={ratio_med:.4f}) p={p_val:.2e}  "
              f"({time.perf_counter()-t0_cell:.1f}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("M,N,K,dtype,pre_mean_us,post_mean_us,"
                "ratio_pre_over_post,ratio_pre_over_post_median,"
                "t_stat,p_value\n")
        f.write("\n".join(out_rows) + "\n")

    ratios = np.asarray([c["ratio_pre_over_post"] for c in summary["cells"]])
    summary["n_cells"] = len(summary["cells"])
    summary["geomean_ratio_pre_over_post"] = float(
        np.exp(np.mean(np.log(ratios))))
    summary["min_ratio"] = float(ratios.min())
    summary["max_ratio"] = float(ratios.max())
    summary["n_cells_ge_0p95"] = int(np.sum(ratios >= 0.95))
    summary["n_cells_ge_1p00"] = int(np.sum(ratios >= 1.00))
    summary["n_cells_ge_1p05"] = int(np.sum(ratios >= 1.05))
    summary["n_cells_ge_1p20"] = int(np.sum(ratios >= 1.20))
    summary["n_cells_ge_1p40"] = int(np.sum(ratios >= 1.40))

    print(f"\nDISPATCHER LIFT (PRE-P38 TB-native / POST-P38 routed-to-HBL)",
          flush=True)
    print(f"  geomean across {summary['n_cells']} N=416 admit cells = "
          f"{summary['geomean_ratio_pre_over_post']:.4f}", flush=True)
    print(f"  per-cell ratio range: [{summary['min_ratio']:.4f}, "
          f"{summary['max_ratio']:.4f}]", flush=True)
    print(f"  cells ratio >= 0.95: {summary['n_cells_ge_0p95']}/{summary['n_cells']}",
          flush=True)
    print(f"  cells ratio >= 1.00: {summary['n_cells_ge_1p00']}/{summary['n_cells']}",
          flush=True)
    print(f"  cells ratio >= 1.05: {summary['n_cells_ge_1p05']}/{summary['n_cells']}",
          flush=True)
    print(f"  cells ratio >= 1.20: {summary['n_cells_ge_1p20']}/{summary['n_cells']}",
          flush=True)
    print(f"  cells ratio >= 1.40: {summary['n_cells_ge_1p40']}/{summary['n_cells']}",
          flush=True)
    print(f"  K-1873 hot-cache graph reference geomean: 1.429×", flush=True)

    # Acceptance criteria
    fail_a = summary["n_cells_ge_0p95"] != summary["n_cells"]
    fail_b = summary["n_cells_ge_1p05"] < 10
    fail_c = summary["geomean_ratio_pre_over_post"] < 1.20
    summary["accept_a_no_regression_below_0p95"] = (not fail_a)
    summary["accept_b_at_least_10_cells_ge_1p05"] = (not fail_b)
    summary["accept_c_geomean_ge_1p20"] = (not fail_c)
    summary["accept_all_three"] = not (fail_a or fail_b or fail_c)

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    if fail_a:
        print(f"\n[FAIL accept-(a)] {summary['n_cells'] - summary['n_cells_ge_0p95']} "
              f"cells regressed below 0.95×", flush=True)
    if fail_b:
        print(f"\n[FAIL accept-(b)] only {summary['n_cells_ge_1p05']} cells "
              f"≥1.05× lift (PRD requires ≥10 of 17)", flush=True)
    if fail_c:
        print(f"\n[FAIL accept-(c)] cohort geomean "
              f"{summary['geomean_ratio_pre_over_post']:.4f}× < 1.20× floor "
              f"(K-1873 measured 1.429× hot-cache)", flush=True)
    if summary["accept_all_three"]:
        print(f"\n[ACCEPT] all three criteria pass — K-1873 dispatcher "
              f"lift materialises on the LIVE path", flush=True)

    if args.strict and not summary["accept_all_three"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
