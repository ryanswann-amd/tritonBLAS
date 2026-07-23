#!/usr/bin/env python3
"""Sweep #workgroups (NUM_SMS) x tile size for one shape on the WORK-STEALING
whole-tile path of steal_k_adaptive (STREAMK=False): each WG atomically steals
whole tiles and computes their full K in registers. Records correctness +
throughput and a torch.matmul (hipBLASLt) baseline. Writes data/sweep_ws_<shape>.csv.

Usage: python3 sweep_wgs_tiles.py [M N K]
"""
import csv
import itertools
import os
import sys

import torch
import triton
from steal_k import steal_k_adaptive, EVENT_FIELDS

M = int(sys.argv[1]) if len(sys.argv) > 1 else 32
N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
K = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
# Optional single-tile mode: sweep num_sms for one (BM,BN,BK) and append to CSV.
# A fault in one tile then only kills that subprocess (driver loops tiles).
ONE_TILE = None
if len(sys.argv) > 6:
    ONE_TILE = (int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
NSTAGES = int(sys.argv[7]) if len(sys.argv) > 7 else 2
DEV = "cuda"

BMS = [16, 32, 64, 128]
BNS = [32, 64, 128, 256]
BKS = [64, 128, 256]
NUM_SMS_LIST = [8, 16, 32, 64, 128, 256, 304]
LDS_CAP = 64 * 1024   # MI300X LDS/CU


def pick_warps(bm, bn):
    a = bm * bn
    if a >= 128 * 128:
        return 8
    if a >= 64 * 64:
        return 4
    return 1


def bench(fn, reset, n_warmup=10, n_repeat=30):
    flush = torch.empty(int(256 * 1024 * 1024 // 4), dtype=torch.int, device=DEV)
    reset(); fn(); torch.cuda.synchronize()
    for _ in range(n_warmup):
        reset(); flush.zero_(); fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    for i in range(n_repeat):
        reset(); flush.zero_(); torch.cuda.synchronize()
        st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    t = sorted(a.elapsed_time(b) for a, b in zip(st, en))
    return t[len(t) // 2]


def tflops(ms):
    return 2 * M * N * K / 1e12 / (ms * 1e-3)


def main():
    arch = torch.cuda.get_device_properties(0).gcnArchName
    torch.manual_seed(0)
    A = torch.randn(M, K, device=DEV, dtype=torch.float16)
    B = torch.randn(K, N, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B)
    dp_ms = bench(lambda: torch.matmul(A, B), lambda: None)
    print(f"shape {M}x{N}x{K}  arch={arch}")
    print(f"torch.matmul (hipBLASLt) baseline: {dp_ms:.4f} ms  {tflops(dp_ms):.1f} TF/s", flush=True)

    os.makedirs("data", exist_ok=True)
    out = f"data/sweep_ws_{M}x{N}x{K}.csv"
    fields = ["bm", "bn", "bk", "num_sms", "nt", "ipt", "lds_kb", "nwarps", "ok", "ms", "tflops", "vs_dp"]
    new_file = not os.path.exists(out) or ONE_TILE is None
    fcsv = open(out, "w" if new_file else "a", newline="")
    w = csv.DictWriter(fcsv, fieldnames=fields)
    if new_file:
        w.writeheader(); fcsv.flush()

    tile_list = [ONE_TILE] if ONE_TILE is not None else list(itertools.product(BMS, BNS, BKS))
    rows = []
    best = None
    for bm, bn, bk in tile_list:
        lds = (bm * bk + bk * bn) * 2  # ns=2 -> (ns-1)*(A_bytes+B_bytes), fp16
        if lds > LDS_CAP:
            continue
        nt = triton.cdiv(M, bm) * triton.cdiv(N, bn)
        ipt = triton.cdiv(K, bk)
        nwarps = pick_warps(bm, bn)
        for num_sms in NUM_SMS_LIST:
            # work-stealing: any #WGs works (idle if > nt); no P/locks reduction.
            grid = num_sms
            C = torch.zeros(M, N, device=DEV, dtype=torch.float16)
            P = torch.zeros(1, device=DEV, dtype=torch.float32)      # unused (register path)
            locks = torch.zeros(1, device=DEV, dtype=torch.int32)    # unused
            wc = torch.zeros(1, device=DEV, dtype=torch.int32)
            ev = torch.zeros(1, 1, EVENT_FIELDS, device=DEV, dtype=torch.int64)

            def run():
                steal_k_adaptive[(grid,)](
                    A, B, C, P, locks, wc, ev, M, N, K,
                    A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                    bm, bn, bk, nt, ipt, num_sms, 1, EVENT_FIELDS, False, False, False,
                    num_warps=nwarps, matrix_instr_nonkdim=16, kpack=1, num_stages=NSTAGES,
                )
            reset = lambda: wc.zero_()   # reset the work-steal counter
            try:
                reset(); run(); torch.cuda.synchronize()
                ok = torch.allclose(C.float(), ref.float(), atol=1.0, rtol=0.05)
                ms = bench(run, reset)
            except Exception as e:
                ok, ms = f"ERR:{repr(e)[:40]}", 0.0
            tf = tflops(ms) if ms > 0 else 0.0
            row = dict(bm=bm, bn=bn, bk=bk, num_sms=num_sms, nt=nt, ipt=ipt,
                       lds_kb=lds // 1024, nwarps=nwarps, ok=ok,
                       ms=round(ms, 5), tflops=round(tf, 2),
                       vs_dp=round(dp_ms / ms, 3) if ms > 0 else 0.0)
            rows.append(row)
            w.writerow(row); fcsv.flush()   # incremental: survive a later fault
            if ok is True and (best is None or tf > best["tflops"]):
                best = row
            print(f"  {bm}x{bn}x{bk} sms={num_sms:3d} nt={nt} ipt={ipt} lds={lds//1024}KB "
                  f"ok={ok} {tf:6.2f} TF/s ({dp_ms/ms if ms>0 else 0:.2f}x DP)", flush=True)

    fcsv.close()
    print(f"\nappended {len(rows)} rows to {out}")
    if best:
        print(f"BEST correct: {best['bm']}x{best['bn']}x{best['bk']} sms={best['num_sms']} "
              f"-> {best['tflops']} TF/s ({best['vs_dp']}x DP)")


if __name__ == "__main__":
    main()
