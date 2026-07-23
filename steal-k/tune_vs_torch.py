#!/usr/bin/env python3
"""Per-shape TILE-tuned steal-k (stream-K + tree) vs torch.matmul.

For each shape, sweeps a set of tiles (sk_tree, grid=CU, tree reduction), picks
the best correct one, and compares to torch.matmul. Answers: does per-shape
tuning close the gap vs rocBLAS on skinny big-K GEMMs?

CSV: CSV,m,n,k,torch_ms,sk_ms,tile,torch_tf,sk_tf,speedup,ok
"""
import sys, torch, triton
from steal_k import sk_tree

DEV = "cuda"
CU = torch.cuda.get_device_properties(0).multi_processor_count
# (BM, BN, BK, num_warps) -- BK=64 so EVEN_K holds for K%64==0
TILES = [
    (64, 64, 64, 4),
    (32, 128, 64, 4), (128, 32, 64, 4),
    (64, 128, 64, 4), (128, 64, 64, 4),
    (128, 128, 64, 8),
    (128, 256, 64, 8), (256, 128, 64, 8),
    (256, 256, 64, 8),
]
MAXT = 256
P = torch.zeros(CU, MAXT * MAXT, device=DEV, dtype=torch.float32)
LOCKS = torch.zeros(CU, device=DEV, dtype=torch.int32)
FLUSH = torch.empty(int(256 * 1024 * 1024 // 4), dtype=torch.int, device=DEV)


def tflops(m, n, k, ms):
    return 2 * m * n * k / 1e12 / (ms * 1e-3)


def bench(fn, reset, n_warmup=3, n_repeat=8):
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


def sk_run(A, B, C, m, n, k, bm, bn, bk, nw):
    nt = triton.cdiv(m, bm) * triton.cdiv(n, bn); ipt = triton.cdiv(k, bk)
    sk_tree[(CU,)](A, B, C, P, LOCKS, m, n, k,
                   A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                   nt, ipt, CU, bm, bn, bk, 12, True,
                   num_warps=nw, matrix_instr_nonkdim=16, kpack=1, num_stages=2)


def run_one(m, n, k):
    A = torch.randn(m, k, device=DEV, dtype=torch.float16)
    B = torch.randn(k, n, device=DEV, dtype=torch.float16)
    C = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B)
    pt_ms = bench(lambda: torch.matmul(A, B), lambda: None)
    best = None
    for (bm, bn, bk, nw) in TILES:
        reset = lambda: (LOCKS.zero_(), C.zero_())
        try:
            reset(); sk_run(A, B, C, m, n, k, bm, bn, bk, nw); torch.cuda.synchronize()
        except Exception:
            continue
        if not torch.allclose(C.float(), ref.float(), atol=1.0, rtol=0.05):
            continue
        ms = bench(lambda: sk_run(A, B, C, m, n, k, bm, bn, bk, nw), reset)
        if best is None or ms < best[0]:
            best = (ms, f"{bm}x{bn}x{bk}")
    del A, B, C, ref
    if best is None:
        return f"CSV,{m},{n},{k},{pt_ms:.4f},-,-,-,-,-,nofit"
    sk_ms, tile = best
    return (f"CSV,{m},{n},{k},{pt_ms:.4f},{sk_ms:.4f},{tile},"
            f"{tflops(m,n,k,pt_ms):.1f},{tflops(m,n,k,sk_ms):.1f},{pt_ms/sk_ms:.3f},True")


def main():
    shard = sys.argv[1]
    shapes = [tuple(int(x) for x in l.split()) for l in open(shard) if l.split()]
    print(f"# TUNE shard={shard} shapes={len(shapes)} tiles={len(TILES)} CU={CU}", flush=True)
    for i, (m, n, k) in enumerate(shapes):
        print(run_one(m, n, k), flush=True)
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()
    print("# TUNE_DONE", flush=True)


if __name__ == "__main__":
    main()
