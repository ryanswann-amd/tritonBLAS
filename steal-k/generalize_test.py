#!/usr/bin/env python3
"""Does Origami tile-selection generalize better for steal-k than a fixed tile?

Over a DIVERSE shape set (skinny + square + rectangular + large), run sk_tree
(stream-K + tree, grid=CU) under 4 tile-selection strategies and compare geomean
throughput vs torch:
  - fixed 64x64x64
  - Origami data-parallel selector's tile
  - Origami Stream-K selector's tile
  - tuned (best of a fixed tile set) -- the upper bound

CSV: CSV,m,n,k,torch_ms,fixed_ms,fixed_tile,odp_ms,odp_tile,osk_ms,osk_tile,tuned_ms,tuned_tile
"""
import sys, torch, triton
from steal_k import sk_tree
from tritonblas import OrigamiMatmulSelector

DEV = "cuda"
CU = torch.cuda.get_device_properties(0).multi_processor_count
P = torch.zeros(CU, 256 * 256, device=DEV, dtype=torch.float32)
LOCKS = torch.zeros(CU, device=DEV, dtype=torch.int32)
FLUSH = torch.empty(int(256 * 1024 * 1024 // 4), dtype=torch.int, device=DEV)
TUNE_TILES = [(64, 64, 64), (64, 128, 64), (128, 64, 64), (128, 128, 64),
              (128, 256, 64), (256, 128, 64), (256, 256, 64)]

SHAPES = [
    # squares
    (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096),
    (8192, 8192, 8192), (3072, 3072, 3072),
    # large rectangular
    (4096, 4096, 8192), (8192, 8192, 1024), (2048, 8192, 2048), (1024, 8192, 8192),
    (8192, 1024, 8192), (8192, 256, 8192), (256, 8192, 8192),
    # skinny big-K
    (64, 512, 16384), (128, 128, 32768), (256, 256, 16384), (512, 512, 16384),
    (32, 256, 16384), (96, 8192, 8192), (8192, 96, 8192),
    # medium
    (1536, 1536, 4096), (2048, 2048, 8192), (512, 8192, 4096), (6144, 6144, 6144),
]


def bench(fn, reset, n_warmup=4, n_repeat=10):
    reset(); fn(); torch.cuda.synchronize()
    for _ in range(n_warmup):
        reset(); fn()
    FLUSH.zero_(); torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    for i in range(n_repeat):
        reset(); st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    t = sorted(a.elapsed_time(b) for a, b in zip(st, en))
    return t[len(t) // 2]


def nwarps(bm, bn):
    return 8 if bm * bn >= 128 * 128 else (4 if bm * bn >= 64 * 64 else 1)


def try_tile(A, B, C, m, n, k, bm, bn, bk, ref):
    if bm * bn > 256 * 256:
        return None
    nt = triton.cdiv(m, bm) * triton.cdiv(n, bn); ipt = triton.cdiv(k, bk)
    even = (k % bk == 0)
    def f():
        sk_tree[(CU,)](A, B, C, P, LOCKS, m, n, k,
                       A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                       nt, ipt, CU, bm, bn, bk, 12, even,
                       num_warps=nwarps(bm, bn), matrix_instr_nonkdim=16, kpack=1, num_stages=2)
    reset = lambda: (LOCKS.zero_(), C.zero_())
    try:
        reset(); f(); torch.cuda.synchronize()
    except Exception:
        return None
    if not torch.allclose(C.float(), ref.float(), atol=1.0, rtol=0.05):
        return None
    return bench(f, reset)


def run_one(m, n, k):
    A = torch.randn(m, k, device=DEV, dtype=torch.float16)
    B = torch.randn(k, n, device=DEV, dtype=torch.float16)
    C = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B)
    pt = bench(lambda: torch.matmul(A, B), lambda: None)
    # fixed 64x64
    fx = try_tile(A, B, C, m, n, k, 64, 64, 64, ref)
    # origami dp tile
    sd = OrigamiMatmulSelector(m, n, k, torch.float16, torch.float16, torch.float16, A.device, streamk=False)
    odp_t = (sd.block_m, sd.block_n, sd.block_k)
    odp = try_tile(A, B, C, m, n, k, *odp_t, ref)
    # origami streamk tile
    ss = OrigamiMatmulSelector(m, n, k, torch.float16, torch.float16, torch.float16, A.device, streamk=True)
    osk_t = (ss.block_m, ss.block_n, ss.block_k)
    osk = try_tile(A, B, C, m, n, k, *osk_t, ref)
    # tuned best-of-set
    best = None; best_t = "-"
    for (bm, bn, bk) in TUNE_TILES:
        ms = try_tile(A, B, C, m, n, k, bm, bn, bk, ref)
        if ms and (best is None or ms < best):
            best, best_t = ms, f"{bm}x{bn}x{bk}"
    del A, B, ref
    g = lambda x: f"{x:.4f}" if x else "-"
    return (f"CSV,{m},{n},{k},{pt:.4f},{g(fx)},64x64x64,{g(odp)},{odp_t[0]}x{odp_t[1]}x{odp_t[2]},"
            f"{g(osk)},{osk_t[0]}x{osk_t[1]}x{osk_t[2]},{g(best)},{best_t}")


def main():
    print(f"# GEN CU={CU} shapes={len(SHAPES)}", flush=True)
    for (m, n, k) in SHAPES:
        print(run_one(m, n, k), flush=True)
    print("# GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
