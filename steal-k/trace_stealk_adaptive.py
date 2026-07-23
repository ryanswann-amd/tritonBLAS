#!/usr/bin/env python3
"""Trace steal_k_adaptive (the fixed kernel) -> SimpleTrace / Chrome-trace JSON.

Emits per-workgroup tracks; compute slices colored by k-group (cat="kg{n}");
in split-K mode (SPLITK>1) also draws peer flow arrows connecting the k-group
partials of a tile to that tile's last-finishing partial (the atomic merge point).

Usage: python3 trace_stealk_adaptive.py [M N K] [BM BN BK] [SPLITK] [GRID]
"""
import json
import os
import sys

import torch
import triton

from trace_helpers import REALTIME_HZ
from steal_k import steal_k_adaptive, EVENT_FIELDS

M = int(sys.argv[1]) if len(sys.argv) > 1 else 512
N = int(sys.argv[2]) if len(sys.argv) > 2 else 512
K = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
BM = int(sys.argv[4]) if len(sys.argv) > 4 else 128
BN = int(sys.argv[5]) if len(sys.argv) > 5 else 128
BK = int(sys.argv[6]) if len(sys.argv) > 6 else 64
NUM_SMS = int(sys.argv[7]) if len(sys.argv) > 7 else 32
PER_ITER = bool(int(sys.argv[8])) if len(sys.argv) > 8 else False
TREE = bool(int(sys.argv[9])) if len(sys.argv) > 9 else False
MAX_SLOTS = 512
TICKS_PER_US = REALTIME_HZ / 1e6


def us(x):
    return float(x) / TICKS_PER_US


def main():
    dev = "cuda"
    arch = torch.cuda.get_device_properties(0).gcnArchName
    torch.manual_seed(0)
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    B = torch.randn(K, N, device=dev, dtype=torch.float16)
    C = torch.zeros(M, N, device=dev, dtype=torch.float16)
    ref = torch.matmul(A, B)

    npm = triton.cdiv(M, BM); npn = triton.cdiv(N, BN)
    nt = npm * npn
    ipt = triton.cdiv(K, BK)
    STREAMK = nt < NUM_SMS
    grid = NUM_SMS
    P = torch.zeros(NUM_SMS, BM * BN, device=dev, dtype=torch.float32) if STREAMK else torch.zeros(1, device=dev, dtype=torch.float32)
    locks = torch.zeros(NUM_SMS, device=dev, dtype=torch.int32) if STREAMK else torch.zeros(1, device=dev, dtype=torch.int32)
    wc = torch.zeros(1, device=dev, dtype=torch.int32)
    ev = torch.zeros(grid, MAX_SLOTS, EVENT_FIELDS, device=dev, dtype=torch.int64)

    steal_k_adaptive[(grid,)](
        A, B, C, P, locks, wc, ev, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BM, BN, BK, nt, ipt, NUM_SMS, MAX_SLOTS, EVENT_FIELDS, STREAMK, True, PER_ITER,
        EVEN_K=True, GROUP_M=8, TREE=TREE,
        num_warps=(8 if BM * BN >= 128 * 128 else (4 if BM * BN >= 64 * 64 else 1)),
        matrix_instr_nonkdim=16, kpack=1,
        num_stages=(1 if PER_ITER else 2),
    )
    torch.cuda.synchronize()

    err = (C.float() - ref.float()).abs().max().item()
    okc = torch.allclose(C.float(), ref.float(), atol=1.0, rtol=0.05)
    mode = "streamk" if STREAMK else "register-reduce"
    print(f"steal-k adaptive {M}x{N}x{K} tile {BM}x{BN}x{BK} nt={nt} NUM_SMS={NUM_SMS} grid={grid} "
          f"[{mode}]  max_err={err:.3f} allclose={okc}")

    e = ev.cpu().numpy().reshape(-1, EVENT_FIELDS)
    e = e[e[:, 5] > 0]  # valid: t_end>0
    g0 = e[:, 4].min()

    te = [{"name": "process_name", "ph": "M", "pid": 0, "args": {"name": f"steal-k {mode}"}}]
    wgs = sorted(set(int(r[0]) for r in e))
    for wg in wgs:
        te.append({"name": "thread_name", "ph": "M", "pid": 0, "tid": wg, "args": {"name": f"wg{wg}"}})
        te.append({"name": "thread_sort_index", "ph": "M", "pid": 0, "tid": wg, "args": {"sort_index": wg}})

    PHASE_NAME = {10: "prologue", 11: "compute", 12: "store", 13: "reduce"}
    owner_by_tile = {}     # tile -> (wg, reduce_start_ts)
    contrib_by_tile = {}   # tile -> [(wg, store_end_ts), ...]
    for r in e:
        wg, tile, kc, phase = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        ks_it, ke_it = int(r[6]), int(r[7])   # local k-iter range within tile
        ts, dur = us(r[4] - g0), us(r[5] - r[4])
        pid_m, pid_n = tile // npn, tile % npn
        m0, m1 = pid_m * BM, min((pid_m + 1) * BM, M)
        n0, n1 = pid_n * BN, min((pid_n + 1) * BN, N)
        k0, k1 = ks_it * BK, min(ke_it * BK, K)
        if phase == 10:
            cat = "prologue"; name = f"prologue tile{tile} pid{wg}"
        elif phase == 11:
            cat = f"kc{kc}"; name = f"compute tile{tile} pid{wg}  K[{k0}:{k1}]"
        elif phase == 12:
            cat = "store"; name = f"store partial tile{tile} pid{wg}  K[{k0}:{k1}]"
        else:  # 13
            cat = "reduce"; name = f"reduce(owner) C[{m0}:{m1},{n0}:{n1}]"
        te.append({
            "name": name, "cat": cat, "ph": "X", "pid": 0, "tid": wg, "ts": ts, "dur": dur,
            "args": {
                "phase": PHASE_NAME.get(phase, phase), "tile": tile, "pid": wg,
                "k_chunk_start": kc,
                "C_rows": f"[{m0}:{m1}]", "C_cols": f"[{n0}:{n1}]",
                "K_range": f"[{k0}:{k1}]", "K_len": k1 - k0,
                "A_region": f"A[{m0}:{m1}, {k0}:{k1}]",
                "B_region": f"B[{k0}:{k1}, {n0}:{n1}]",
            }})
        if phase == 12:
            contrib_by_tile.setdefault(tile, []).append((wg, us(r[5] - g0)))
        elif phase == 13:
            owner_by_tile[tile] = (wg, us(r[4] - g0))

    # peer flows: each contributor's partial store -> the tile owner's reduce
    flow_id = 0
    for tile, (owg, ots) in owner_by_tile.items():
        for (cwg, cend) in contrib_by_tile.get(tile, []):
            te.append({"name": "peer", "cat": "peer", "ph": "s", "id": flow_id,
                       "pid": 0, "tid": cwg, "ts": max(cend - 0.01, 0.0)})
            te.append({"name": "peer", "cat": "peer", "ph": "f", "bp": "e",
                       "id": flow_id, "pid": 0, "tid": owg, "ts": ots})
            flow_id += 1

    os.makedirs("data", exist_ok=True)
    tag = f"adaptive_{M}x{N}x{K}_{BM}x{BN}x{BK}_sms{NUM_SMS}_{mode}" + ("_tree" if TREE else "") + ("_periter" if PER_ITER else "")
    out = f"data/simpletrace_{tag}.json"
    with open(out, "w") as f:
        json.dump({"traceEvents": te, "displayTimeUnit": "us"}, f)

    per_wg = {}
    for r in e:
        per_wg[int(r[0])] = per_wg.get(int(r[0]), 0) + 1
    cnt = sorted(per_wg.values())
    print(f"  work-stealing: {len(e)} partials over {len(per_wg)} WGs; "
          f"per-WG min/med/max={cnt[0]}/{cnt[len(cnt)//2]}/{cnt[-1]}; peer_flows={flow_id}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
