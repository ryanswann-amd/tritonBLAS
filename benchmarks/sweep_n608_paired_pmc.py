#!/usr/bin/env python3
"""
K-1938 / K-1926 — Paired n=30 HIP-graph bench + 3-pass rocprofv2 PMC sweep
for the N=608 wave-misaligned skinny-N K-COMPLEMENT cohort on MI300X
(gfx942).

Methodology mirrors K-1857 (N=384), K-1873 (N=416), K-1881 (N=448),
K-1888 (N=480), K-1910/K-1917/K-1922 (N=512/544/576) exactly.  The third
(streamK) arm is retained for completeness but the gate metric is the
two-arm tb-vs-hbl ratio.

Wave-misalignment signature for N=608:
  608 mod 64  = 32   (matches the validated P32-P40 wave-misalignment
                      cluster — same modular class as N=160/N=224/N=288/
                      N=352/N=416/N=480/N=544)
  608 mod 128 = 96   (same wave-misaligned modular class as N=96/N=224/
                      N=352/N=480 — partial tail-wave on BLOCK_N=128)

Cohort (18 paired cells, fixed N=608):
  N      = 608  (fixed; 608 mod 128 = 96 -> wave-misaligned)
  M      in {2048, 4096, 8192}        (3 values)
  K      in {4096, 8192, 16384}       (3 values)
  dtype  in {bf16, fp16}              (2 dtypes)
  Total  = 3 * 3 * 2 = 18 cells

Three engines per cell:
  tb  -- tritonblas.matmul   (LIVE oracle, persistent_matmul fallthrough
                              if no alias-stack hit at HEAD)
  hbl -- torch.matmul backed by hipBLASLt (preferred_blas_library)
  sk  -- tritonblas.matmul(..., enable_streamk=True, sk_grid=304)
         (304 = MI300X total CU count, 'all-CU' streamK regime)

Bench is paired n=30 HIP-graph hot-cache wall-time per (cell, engine).
PMC is 3-pass per engine x 18 cells x (3 warmup + 30 reps) = 594
dispatches per (engine, pass).  Total PMC = 9 passes (3 packs x 3 engines).

PMC packs (matches K-1843 / K-1888 protocol):
  pack 1 (LDS):       SQ_LDS_BANK_CONFLICT, SQ_WAIT_INST_LDS, SQ_INSTS_LDS
  pack 2 (VALU/MFMA): SQ_INSTS_MFMA, SQ_INSTS_VALU, SQ_INSTS_VALU_MFMA
  pack 3 (VMEM/L2):   SQ_INSTS_VMEM, TCC_HIT, TCC_MISS

Usage:
  # Bench arm (paired wall-time)
  python sweep_n608_paired_pmc.py bench --out /home/ryaswann/mc2-workspaces/K-1938/output/n608_bench.json

  # PMC arms (one rocprofv2 invocation per engine x pack)
  rocprofv2 -i pmc_lds.txt   -o n608_lds_tb.csv   -- python sweep_n608_paired_pmc.py kernel --engine tb
  rocprofv2 -i pmc_lds.txt   -o n608_lds_hbl.csv  -- python sweep_n608_paired_pmc.py kernel --engine hbl
  rocprofv2 -i pmc_valu.txt  -o n608_valu_tb.csv  -- python sweep_n608_paired_pmc.py kernel --engine tb
  ...etc 9 invocations total.
"""
import argparse, json, sys, statistics, os

# Source layout (adapt to whichever HEAD this is run from). Default expects
# the K-1938 workspace tritonblas clone at the standard path.
_DEFAULT_TB = "/home/ryaswann/mc2-workspaces/K-1938/repos/tritonblas/include"
_DEFAULT_ORI = "/home/ryaswann/mc2-workspaces/K-1938/origami_lib"
if os.path.isdir(_DEFAULT_ORI):
    sys.path.insert(0, _DEFAULT_ORI)
if os.path.isdir(_DEFAULT_TB):
    sys.path.insert(0, _DEFAULT_TB)
import torch
import tritonblas


N_FIX = 608
SK_GRID = 304  # MI300X N_CU
DT_NAME = {torch.bfloat16: "bf16", torch.float16: "fp16"}
ENGINES = ("tb", "hbl", "sk")

# 18 cells; N=608 is wave-misaligned (608 mod 128 = 96).  Iteration order
# matches K-1888 / K-1881 / K-1873 (dtype outer, M middle, K inner).
COHORT = []
for dt in (torch.bfloat16, torch.float16):
    for M in (2048, 4096, 8192):
        for K in (4096, 8192, 16384):
            COHORT.append((M, N_FIX, K, dt, False))
assert len(COHORT) == 18, f"got {len(COHORT)}"


def alloc(M, N, K, dt, seed=1938):
    g = torch.Generator(device="cuda").manual_seed(
        seed + M * 1009 + K * 31 + N * 7 + (1 if dt is torch.float16 else 0))
    a = torch.randn(M, K, device="cuda", dtype=dt, generator=g)
    b = torch.randn(K, N, device="cuda", dtype=dt, generator=g)
    out = torch.empty(M, N, device="cuda", dtype=dt)
    return a, b, out


def run_engine(engine, a, b, out):
    if engine == "tb":
        tritonblas.matmul(a, b, out=out)
    elif engine == "hbl":
        torch.matmul(a, b, out=out)
    elif engine == "sk":
        tritonblas.matmul(a, b, out=out, enable_streamk=True, sk_grid=SK_GRID)
    else:
        raise ValueError(engine)


def cmd_kernel(args):
    """All 18 cells under PMC wrapper.  Dispatch order = COHORT order x reps;
    warmup is outside the measured window so dispatch_idx -> cell mapping
    is deterministic.  PER_CELL = warmup + reps dispatches per cell."""
    if args.engine == "hbl":
        torch.backends.cuda.preferred_blas_library("hipblaslt")
    cells = []
    for (M, N, K, dt, _ali) in COHORT:
        a, b, out = alloc(M, N, K, dt)
        cells.append((M, N, K, dt, a, b, out))
    for (M, N, K, dt, a, b, out) in cells:
        for _ in range(args.warmup):
            run_engine(args.engine, a, b, out)
    torch.cuda.synchronize()
    for (M, N, K, dt, a, b, out) in cells:
        for _ in range(args.reps):
            run_engine(args.engine, a, b, out)
    torch.cuda.synchronize()


def cmd_bench(args):
    """Paired hot-cache wall-time TB vs HBL vs SK across 18 cells via
    HIP-graph replay.  For each cell: build a TB-only, HBL-only and SK-only
    graph each with REPS_INNER replays, then alternate timing them n=30
    times.  HIP-graph eliminates launch overhead variance.
    """
    out_rows = []
    torch.backends.cuda.preferred_blas_library("hipblaslt")
    REPS_INNER = args.inner_reps
    for (M, N, K, dt, ali) in COHORT:
        a, b, out = alloc(M, N, K, dt)
        for _ in range(args.warmup):
            tritonblas.matmul(a, b, out=out)
            torch.matmul(a, b, out=out)
            tritonblas.matmul(a, b, out=out, enable_streamk=True, sk_grid=SK_GRID)
        torch.cuda.synchronize()

        s_tb = torch.cuda.Stream(); s_tb.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_tb):
            for _ in range(2):
                tritonblas.matmul(a, b, out=out)
        torch.cuda.current_stream().wait_stream(s_tb); torch.cuda.synchronize()
        g_tb = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_tb, stream=s_tb):
            for _ in range(REPS_INNER):
                tritonblas.matmul(a, b, out=out)

        s_hb = torch.cuda.Stream(); s_hb.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_hb):
            for _ in range(2):
                torch.matmul(a, b, out=out)
        torch.cuda.current_stream().wait_stream(s_hb); torch.cuda.synchronize()
        g_hb = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_hb, stream=s_hb):
            for _ in range(REPS_INNER):
                torch.matmul(a, b, out=out)

        s_sk = torch.cuda.Stream(); s_sk.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_sk):
            for _ in range(2):
                tritonblas.matmul(a, b, out=out, enable_streamk=True, sk_grid=SK_GRID)
        torch.cuda.current_stream().wait_stream(s_sk); torch.cuda.synchronize()
        g_sk = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_sk, stream=s_sk):
            for _ in range(REPS_INNER):
                tritonblas.matmul(a, b, out=out, enable_streamk=True, sk_grid=SK_GRID)

        for _ in range(3):
            g_tb.replay(); g_hb.replay(); g_sk.replay()
        torch.cuda.synchronize()

        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        tb_us, hbl_us, sk_us = [], [], []
        for trial in range(args.reps):
            ev_s.record(); g_tb.replay(); ev_e.record(); torch.cuda.synchronize()
            tb_us.append(ev_s.elapsed_time(ev_e) * 1000.0 / REPS_INNER)
            ev_s.record(); g_hb.replay(); ev_e.record(); torch.cuda.synchronize()
            hbl_us.append(ev_s.elapsed_time(ev_e) * 1000.0 / REPS_INNER)
            ev_s.record(); g_sk.replay(); ev_e.record(); torch.cuda.synchronize()
            sk_us.append(ev_s.elapsed_time(ev_e) * 1000.0 / REPS_INNER)

        tb_med = statistics.median(tb_us)
        hbl_med = statistics.median(hbl_us)
        sk_med = statistics.median(sk_us)
        ratio_tb_hbl = tb_med / hbl_med
        ratio_sk_hbl = sk_med / hbl_med
        ratio_tb_sk = tb_med / sk_med
        out_rows.append(dict(
            M=M, N=N, K=K, dtype=DT_NAME[dt],
            wave_aligned=int(ali),
            tb_med_us=round(tb_med, 3),
            hbl_med_us=round(hbl_med, 3),
            sk_med_us=round(sk_med, 3),
            ratio_tb_over_hbl=round(ratio_tb_hbl, 4),
            ratio_sk_over_hbl=round(ratio_sk_hbl, 4),
            ratio_tb_over_sk=round(ratio_tb_sk, 4),
            speedup_hbl_over_tb=round(ratio_tb_hbl, 4),
            tb_us=[round(x, 3) for x in tb_us],
            hbl_us=[round(x, 3) for x in hbl_us],
            sk_us=[round(x, 3) for x in sk_us]))
        print(f"M={M:>4} N={N:>3} K={K:>5} {DT_NAME[dt]} ali={int(ali)}  "
              f"TB={tb_med:8.2f}us HBL={hbl_med:8.2f}us SK={sk_med:8.2f}us "
              f"TB/HBL={ratio_tb_hbl:6.3f} SK/HBL={ratio_sk_hbl:6.3f}",
              flush=True)

        del g_tb, g_hb, g_sk

    with open(args.out, "w") as f:
        json.dump(out_rows, f, indent=1)
    csv_path = args.out.rsplit(".", 1)[0] + ".csv"
    with open(csv_path, "w") as f:
        f.write("M,N,K,dtype,wave_aligned,tb_med_us,hbl_med_us,sk_med_us,"
                "ratio_tb_over_hbl,ratio_sk_over_hbl,ratio_tb_over_sk,"
                "speedup_hbl_over_tb\n")
        for r in out_rows:
            f.write(f"{r['M']},{r['N']},{r['K']},{r['dtype']},{r['wave_aligned']},"
                    f"{r['tb_med_us']},{r['hbl_med_us']},{r['sk_med_us']},"
                    f"{r['ratio_tb_over_hbl']},{r['ratio_sk_over_hbl']},"
                    f"{r['ratio_tb_over_sk']},{r['speedup_hbl_over_tb']}\n")
    print(f"wrote {args.out} and {csv_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p_b = sub.add_parser("bench")
    p_b.add_argument("--out", required=True)
    p_b.add_argument("--warmup", type=int, default=10)
    p_b.add_argument("--reps", type=int, default=30)
    p_b.add_argument("--inner-reps", type=int, default=10)
    p_b.set_defaults(fn=cmd_bench)
    p_k = sub.add_parser("kernel")
    p_k.add_argument("--engine", choices=ENGINES, required=True)
    p_k.add_argument("--warmup", type=int, default=3)
    p_k.add_argument("--reps", type=int, default=30)
    p_k.set_defaults(fn=cmd_kernel)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
