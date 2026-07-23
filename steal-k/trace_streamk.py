#!/usr/bin/env python3
"""Trace tritonBLAS Stream-K: separate compute vs reduction time via s_memrealtime.

Runs the instrumented `streamk_matmul_traced` in an all-Stream-K configuration
(every output tile split across the CU grid, so the reduction/fixup path is
exercised), then reports per-CU compute vs reduction vs lock-spin time and emits
a Chrome/perfetto trace.

Usage: python3 trace_streamk.py [M N K] [BM BN BK] [GRID]
Defaults: 256 256 8192, tile 128x128x64, grid=304 (MI300X full).
"""
import json
import os
import sys

import torch
import triton

from trace_helpers import REALTIME_HZ
from streamk_gemm_traced import (
    streamk_matmul_traced, TRACE_SLOTS,
    T_START, T_END, COMP_FULL, COMP_SK, RED, SPIN, PROD,
    N_FULL, N_SK, N_CONS, N_PROD, CU_ID, PID_SLOT,
)

M = int(sys.argv[1]) if len(sys.argv) > 1 else 256
N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
K = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
BM = int(sys.argv[4]) if len(sys.argv) > 4 else 128
BN = int(sys.argv[5]) if len(sys.argv) > 5 else 128
BK = int(sys.argv[6]) if len(sys.argv) > 6 else 64
GRID = int(sys.argv[7]) if len(sys.argv) > 7 else 304

TICKS_PER_US = REALTIME_HZ / 1e6  # 100 ticks per microsecond @100MHz


def us(ticks):
    return float(ticks) / TICKS_PER_US


def main():
    dev = "cuda"
    arch = torch.cuda.get_device_properties(0).gcnArchName
    torch.manual_seed(0)
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    B = torch.randn(K, N, device=dev, dtype=torch.float16)
    C = torch.zeros(M, N, device=dev, dtype=torch.float16)

    num_pid_m = triton.cdiv(M, BM)
    num_pid_n = triton.cdiv(N, BN)
    total_tiles = num_pid_m * num_pid_n
    STREAMK_TILES = total_tiles          # all tiles go through Stream-K
    NUM_SMS = GRID
    NUM_XCDS = 1                          # no chiplet remap -> pid == program_id
    CHUNK_SIZE = 1
    GROUP_SIZE_M = 8

    P = torch.zeros(NUM_SMS, BM * BN, device=dev, dtype=torch.float32)
    locks = torch.zeros(NUM_SMS, device=dev, dtype=torch.uint8)
    trace = torch.zeros(NUM_SMS, TRACE_SLOTS, device=dev, dtype=torch.int64)

    even_k = (K % BK == 0)
    grid = (NUM_SMS,)
    streamk_matmul_traced[grid](
        A, B, C, None, None, None, P, locks, trace,
        M, N, K,
        A.stride(0), B.stride(1), C.stride(0), C.stride(1), 0,
        A.stride(1), B.stride(0),
        BM, BN, BK, GROUP_SIZE_M, NUM_SMS, NUM_XCDS, CHUNK_SIZE, STREAMK_TILES,
        False, even_k, None, None, TRACE_SLOTS,
        False, torch.backends.cuda.matmul.allow_tf32,
    )
    torch.cuda.synchronize()

    # correctness
    ref = torch.matmul(A, B)
    err = (C.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    ok = torch.allclose(C.to(torch.float32), ref.to(torch.float32), atol=1.0, rtol=0.05)
    print(f"shape {M}x{N}x{K}  tile {BM}x{BN}x{BK}  grid={NUM_SMS}  "
          f"total_tiles={total_tiles}  arch={arch}")
    print(f"correctness: max_abs_err={err:.4f}  allclose={ok}")

    t = trace.cpu().numpy()
    # active CUs = those that recorded a kernel end
    active = t[:, T_END] > 0
    ta = t[active]
    g_start = ta[:, T_START].min()

    tot_comp = us(ta[:, COMP_FULL].sum() + ta[:, COMP_SK].sum())
    tot_red = us(ta[:, RED].sum())        # reduction/fixup, incl. lock-wait
    tot_all = tot_comp + tot_red
    wall = us(ta[:, T_END].max() - ta[:, T_START].min())

    print("\n=== aggregate (sum over CUs of in-kernel time) ===")
    print(f"  compute (full+sk) : {tot_comp:10.1f} us  ({100*tot_comp/tot_all:.1f}%)")
    print(f"  reduction (fixup) : {tot_red:10.1f} us  ({100*tot_red/tot_all:.1f}%)  "
          f"[includes lock-wait]")
    print(f"  kernel wall       : {wall:10.1f} us")
    print(f"  consumers={int(ta[:, N_CONS].sum())} producers={int(ta[:, N_PROD].sum())} "
          f"sk_segments={int(ta[:, N_SK].sum())} active_CUs={active.sum()}")

    # per-consumer view: the CUs that actually do reduction
    cons = ta[ta[:, N_CONS] > 0]
    if len(cons):
        red_us = [us(r[RED]) for r in cons]
        comp_us = [us(r[COMP_FULL] + r[COMP_SK]) for r in cons]
        print(f"\n=== consumer CUs (n={len(cons)}) — these own a tile & reduce partials ===")
        print(f"  reduction/CU: mean={sum(red_us)/len(red_us):.1f} us  "
              f"max={max(red_us):.1f} us   (compute/CU mean={sum(comp_us)/len(comp_us):.2f} us)")
        print(f"  => on the owner CUs, reduction is "
              f"{sum(red_us)/max(sum(comp_us),1e-9):.2f}x the compute time")

    os.makedirs("data", exist_ok=True)
    tag = f"{M}x{N}x{K}_{BM}x{BN}x{BK}_g{NUM_SMS}"

    # per-CU CSV
    import csv
    with open(f"data/trace_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "cu_id", "comp_full_us", "comp_sk_us", "red_us",
                    "spin_us", "prod_us", "active_us", "n_full", "n_sk",
                    "n_cons", "n_prod", "t_start_us", "t_end_us"])
        for r in ta:
            w.writerow([
                int(r[PID_SLOT]), int(r[CU_ID]),
                f"{us(r[COMP_FULL]):.3f}", f"{us(r[COMP_SK]):.3f}",
                f"{us(r[RED]):.3f}", f"{us(r[SPIN]):.3f}", f"{us(r[PROD]):.3f}",
                f"{us(r[T_END]-r[T_START]):.3f}",
                int(r[N_FULL]), int(r[N_SK]), int(r[N_CONS]), int(r[N_PROD]),
                f"{us(r[T_START]-g_start):.3f}", f"{us(r[T_END]-g_start):.3f}",
            ])

    # Chrome/perfetto trace: per CU row, approximate spans
    #   compute span  : [t_start, t_start+comp]
    #   reduction span: [t_end-red, t_end]  (spin shown nested at its start)
    events = []
    for r in ta:
        pid = int(r[PID_SLOT]); cu = int(r[CU_ID])
        ts0 = us(r[T_START] - g_start)
        comp = us(r[COMP_FULL] + r[COMP_SK])
        red = us(r[RED]); spin = us(r[SPIN])
        tend = us(r[T_END] - g_start)
        tid = f"CU{cu:03d}" if cu >= 0 else f"wg{pid:03d}"
        if comp > 0:
            events.append({"name": "compute", "ph": "X", "ts": ts0, "dur": comp,
                           "pid": 0, "tid": tid, "args": {"pid": pid}})
        if red > 0:
            events.append({"name": "reduction", "ph": "X", "ts": tend - red,
                           "dur": red, "pid": 0, "tid": tid,
                           "args": {"pid": pid, "spin_us": round(spin, 3)}})
        if spin > 0:
            events.append({"name": "lock_spin", "ph": "X", "ts": tend - red,
                           "dur": spin, "pid": 0, "tid": tid, "args": {"pid": pid}})
    with open(f"data/chrome_trace_{tag}.json", "w") as f:
        json.dump({"traceEvents": events, "displayTimeUnit": "us"}, f)

    print(f"\nwrote data/trace_{tag}.csv and data/chrome_trace_{tag}.json")


if __name__ == "__main__":
    main()
