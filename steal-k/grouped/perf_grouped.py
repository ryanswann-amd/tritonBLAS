#!/usr/bin/env python3
"""Perf of grouped-steal-k vs torch per-group loop on the VALIDATION config set
(broad, highly variant group sizes). Reports best-of-steal-k speedup, geomean,
win-rate, and which path/tile won.
"""
import math, torch, triton
from grouped_steal_k import make_problem, ref_loop, bench, grouped_matmul
from grouped_split_k import grouped_matmul_sk
from validate_grouped import gen_configs

DEV = "cuda"
CU = torch.cuda.get_device_properties(0).multi_processor_count
# small adaptive-ish candidate set (register for filled, split-K for underfilled)
CANDS = [("reg", grouped_matmul, 128, 256, 64),
         ("reg", grouped_matmul, 128, 128, 64),
         ("splitk", grouped_matmul_sk, 64, 64, 64),
         ("splitk", grouped_matmul_sk, 128, 128, 64)]


def run(Ms, N, K, label):
    A, B, off = make_problem(Ms, N, K, seed=1)
    ref = ref_loop(A, B, off, N).float()
    flops = 2 * sum(int(m) * N * K for m in Ms)
    lp_ms = bench(lambda: ref_loop(A, B, off, N), n_repeat=15)
    best_ms, best_v = None, "-"
    for (name, fn, bm, bn, bk) in CANDS:
        try:
            C = fn(A, B, off, Ms, N, K, bm, bn, bk)[0]; torch.cuda.synchronize()
            if (C.float() - ref).abs().max().item() >= 1.0:
                continue
            ms = bench(lambda: fn(A, B, off, Ms, N, K, bm, bn, bk), n_repeat=15)
            if best_ms is None or ms < best_ms:
                best_ms, best_v = ms, f"{name} {bm}x{bn}"
        except Exception:
            continue
    del A, B, ref
    if best_ms is None:
        return None
    tf = flops / 1e12 / (best_ms * 1e-3)
    sp = lp_ms / best_ms
    print(f"  {label[:38]:38} sumM={sum(Ms):6} | loop {lp_ms:7.3f}ms | steal-k {best_ms:7.3f}ms {tf:6.1f}TF/s [{best_v:11}] | {sp:5.2f}x", flush=True)
    return sp


def main():
    print(f"CUs={CU}  (speedup = torch-loop / best-steal-k)\n", flush=True)
    sps = []
    for (label, Ms, N, K) in gen_configs():
        if sum(Ms) == 0:
            continue
        s = run(Ms, N, K, label)
        if s:
            sps.append(s)
    sps.sort()
    geo = math.exp(sum(math.log(x) for x in sps) / len(sps))
    wins = sum(1 for x in sps if x >= 1.0)
    print(f"\n=== {len(sps)} configs | steal-k faster on {wins} ({100*wins/len(sps):.0f}%) | "
          f"speedup min={sps[0]:.2f} median={sps[len(sps)//2]:.2f} max={sps[-1]:.2f} geomean={geo:.2f} ===", flush=True)
    print("PERF_DONE", flush=True)


if __name__ == "__main__":
    main()
