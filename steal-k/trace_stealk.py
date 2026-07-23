#!/usr/bin/env python3
"""Run steal-k (work-stealing split-K) and emit a SimpleTrace-compatible trace.

Trace (Chrome Trace Event Format, read by SimpleTrace / Perfetto):
  - one track per workgroup (M process_name/thread_name/thread_sort_index)
  - X slices: compute slices colored by k-group (`cat = "kg{n}"`), reduction
    slices `cat = "reduce"`; args carry tile / k_group / k-range / wg
  - flow arrows (s -> f): each k-group partial -> the tile's reduction, so you
    see the peer (producer -> consumer) structure.

Usage: python3 trace_stealk.py [M N K] [BM BN BK] [SPLITK] [GRID]
"""
import json
import os
import sys

import torch
import triton

from trace_helpers import REALTIME_HZ
from steal_k import steal_k_matmul, EVENT_FIELDS

M = int(sys.argv[1]) if len(sys.argv) > 1 else 512
N = int(sys.argv[2]) if len(sys.argv) > 2 else 512
K = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
BM = int(sys.argv[4]) if len(sys.argv) > 4 else 128
BN = int(sys.argv[5]) if len(sys.argv) > 5 else 128
BK = int(sys.argv[6]) if len(sys.argv) > 6 else 64
SPLITK = int(sys.argv[7]) if len(sys.argv) > 7 else 8
GRID = int(sys.argv[8]) if len(sys.argv) > 8 else 32
MAX_SLOTS = 64

TICKS_PER_US = REALTIME_HZ / 1e6


def main():
    dev = "cuda"
    arch = torch.cuda.get_device_properties(0).gcnArchName
    torch.manual_seed(0)
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    B = torch.randn(K, N, device=dev, dtype=torch.float16)
    C = torch.zeros(M, N, device=dev, dtype=torch.float16)

    num_pid_m = triton.cdiv(M, BM)
    num_pid_n = triton.cdiv(N, BN)
    NUM_TILES = num_pid_m * num_pid_n
    ITERS_PER_TILE = triton.cdiv(K, BK)
    ITERS_PER_GROUP = triton.cdiv(ITERS_PER_TILE, SPLITK)
    total_items = NUM_TILES * SPLITK

    Ppartial = torch.zeros(total_items, BM * BN, device=dev, dtype=torch.float32)
    done = torch.zeros(total_items, device=dev, dtype=torch.int32)
    work_ctr = torch.zeros(1, device=dev, dtype=torch.int32)
    events = torch.zeros(GRID, MAX_SLOTS, EVENT_FIELDS, device=dev, dtype=torch.int64)

    steal_k_matmul[(GRID,)](
        A, B, C, Ppartial, done, work_ctr, events,
        M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BM, BN, BK, SPLITK, NUM_TILES, ITERS_PER_TILE, ITERS_PER_GROUP,
        GRID, MAX_SLOTS, EVENT_FIELDS, True,
        num_warps=8,
    )
    torch.cuda.synchronize()

    ref = torch.matmul(A, B)
    err = (C.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    ok = torch.allclose(C.to(torch.float32), ref.to(torch.float32), atol=1.0, rtol=0.05)
    print(f"steal-k {M}x{N}x{K}  tile {BM}x{BN}x{BK}  SPLITK={SPLITK}  grid={GRID}  "
          f"tiles={NUM_TILES}  items={total_items}  arch={arch}")
    print(f"correctness: max_abs_err={err:.4f}  allclose={ok}")

    ev = events.cpu().numpy().reshape(-1, EVENT_FIELDS)
    ev = ev[ev[:, 5] > 0]  # valid events have t_end>0
    if len(ev) == 0:
        print("no events recorded"); return
    g0 = ev[:, 4].min()

    def us(x):
        return float(x) / TICKS_PER_US

    # work-stealing distribution
    comp = ev[ev[:, 3] == 0]
    red = ev[ev[:, 3] == 1]
    steal = ev[ev[:, 3] == 2]
    per_wg = {}
    for r in comp:
        per_wg[int(r[0])] = per_wg.get(int(r[0]), 0) + 1
    counts = sorted(per_wg.values())
    print(f"\nwork-stealing: {len(comp)} partials stolen across {len(per_wg)} WGs; "
          f"per-WG min/median/max = {counts[0]}/{counts[len(counts)//2]}/{counts[-1]}")
    print(f"reductions: {len(red)} tiles reduced; atomic pulls: {len(steal)}")

    tot_comp = sum(us(r[5] - r[4]) for r in comp)
    tot_steal = sum(us(r[5] - r[4]) for r in steal)   # atomic queue-index pull
    tot_red = sum(us(r[5] - r[4]) for r in red)        # sk-reduction
    tot = tot_comp + tot_steal + tot_red
    print("\n=== aggregate (sum over WGs) — compute vs atomic-steal vs sk-reduction ===")
    print(f"  compute        : {tot_comp:9.1f} us  ({100*tot_comp/tot:.1f}%)")
    print(f"  atomic steal   : {tot_steal:9.1f} us  ({100*tot_steal/tot:.1f}%)  "
          f"[{len(steal)} pulls, {tot_steal/max(len(steal),1)*1000:.2f} ns/pull]")
    print(f"  sk-reduction   : {tot_red:9.1f} us  ({100*tot_red/tot:.1f}%)")
    # per-WG atomic-steal time
    steal_by_wg = {}
    for r in steal:
        steal_by_wg[int(r[0])] = steal_by_wg.get(int(r[0]), 0.0) + us(r[5] - r[4])
    sv = sorted(steal_by_wg.values())
    if sv:
        print(f"  per-WG steal time: min/median/max = "
              f"{sv[0]:.2f}/{sv[len(sv)//2]:.2f}/{sv[-1]:.2f} us")

    # ---- SimpleTrace / Chrome trace ----
    te = []
    wgs = sorted(set(int(r[0]) for r in ev))
    te.append({"name": "process_name", "ph": "M", "pid": 0, "args": {"name": "steal-k"}})
    for wg in wgs:
        te.append({"name": "thread_name", "ph": "M", "pid": 0, "tid": wg,
                   "args": {"name": f"wg{wg}"}})
        te.append({"name": "thread_sort_index", "ph": "M", "pid": 0, "tid": wg,
                   "args": {"sort_index": wg}})

    # compute + reduce slices; index compute events by tile for flows
    comp_by_tile = {}
    for r in comp:
        wg, tile, kg = int(r[0]), int(r[1]), int(r[2])
        ts, dur = us(r[4] - g0), us(r[5] - r[4])
        te.append({"name": f"t{tile} k{kg}", "cat": f"kg{kg}", "ph": "X",
                   "pid": 0, "tid": wg, "ts": ts, "dur": dur,
                   "args": {"tile": tile, "k_group": kg,
                            "k_start_iter": int(r[6]), "k_end_iter": int(r[7]), "wg": wg}})
        comp_by_tile.setdefault(tile, []).append((wg, us(r[5] - g0), kg))

    # atomic queue-index pull slices (separate from compute/reduction)
    for r in steal:
        wg = int(r[0])
        ts, dur = us(r[4] - g0), us(r[5] - r[4])
        te.append({"name": "steal", "cat": "atomic_steal", "ph": "X",
                   "pid": 0, "tid": wg, "ts": ts, "dur": dur, "args": {"wg": wg}})

    flow_id = 0
    for r in red:
        wg, tile = int(r[0]), int(r[1])
        ts, dur = us(r[4] - g0), us(r[5] - r[4])
        te.append({"name": f"reduce t{tile}", "cat": "reduce", "ph": "X",
                   "pid": 0, "tid": wg, "ts": ts, "dur": dur,
                   "args": {"tile": tile, "wg": wg, "splitk": SPLITK}})
        # peer flows: each producer partial -> this reduction
        for (pwg, pend, kg) in comp_by_tile.get(tile, []):
            te.append({"name": "peer", "cat": "peer", "ph": "s", "id": flow_id,
                       "pid": 0, "tid": pwg, "ts": max(pend - 0.01, us(0))})
            te.append({"name": "peer", "cat": "peer", "ph": "f", "bp": "e",
                       "id": flow_id, "pid": 0, "tid": wg, "ts": ts + dur * 0.5})
            flow_id += 1

    os.makedirs("data", exist_ok=True)
    tag = f"{M}x{N}x{K}_{BM}x{BN}x{BK}_sk{SPLITK}_g{GRID}"
    out = f"data/simpletrace_stealk_{tag}.json"
    with open(out, "w") as f:
        json.dump({"traceEvents": te, "displayTimeUnit": "us"}, f)
    print(f"\nwrote {out}  ({len(te)} events, {flow_id} peer flows)")


if __name__ == "__main__":
    main()
