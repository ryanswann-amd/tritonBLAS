#!/usr/bin/env python3
"""Equal-opportunity tuning: steal-k vs library persistent(dp) vs library Stream-K,
all tuned over the SAME tile set per shape, vs torch.matmul.

For the library kernels we override Origami's pick by mutating the backing fields
(selector._result.config.mt.{m,n,k}, _num_stages, _grid) and sweeping the same
tiles as steal-k, then take the best correct config for each kernel.

CSV: CSV,m,n,k,torch_ms,sk_ms,sk_tile,dp_ms,dp_tile,streamk_ms,streamk_tile
"""
import sys, torch, triton
from steal_k import sk_tree
import tritonblas
from tritonblas import OrigamiMatmulSelector, matmul_preamble, matmul_lt

DEV = "cuda"
CU = torch.cuda.get_device_properties(0).multi_processor_count
TILES = [(64, 64, 64), (64, 128, 64), (128, 64, 64), (128, 128, 64), (128, 256, 64), (256, 256, 64)]
NS = 2
MAXT = 256
P = torch.zeros(CU, MAXT * MAXT, device=DEV, dtype=torch.float32)
LOCKS = torch.zeros(CU, device=DEV, dtype=torch.int32)
FLUSH = torch.empty(int(256 * 1024 * 1024 // 4), dtype=torch.int, device=DEV)


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


def okc(C, ref):
    return torch.allclose(C.float(), ref.float(), atol=1.0, rtol=0.05)


def nwarps(bm, bn):
    return 8 if bm * bn >= 128 * 128 else (4 if bm * bn >= 64 * 64 else 1)


def tune_stealk(A, B, m, n, k, ref):
    best = (None, "-")
    C = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    for (bm, bn, bk) in TILES:
        nt = triton.cdiv(m, bm) * triton.cdiv(n, bn); ipt = triton.cdiv(k, bk)
        def f():
            sk_tree[(CU,)](A, B, C, P, LOCKS, m, n, k,
                           A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                           nt, ipt, CU, bm, bn, bk, 12, True,
                           num_warps=nwarps(bm, bn), matrix_instr_nonkdim=16, kpack=1, num_stages=NS)
        reset = lambda: (LOCKS.zero_(), C.zero_())
        try:
            reset(); f(); torch.cuda.synchronize()
        except Exception:
            continue
        if not okc(C, ref):
            continue
        ms = bench(f, reset)
        if best[0] is None or ms < best[0]:
            best = (ms, f"{bm}x{bn}x{bk}")
    return best


def tune_lib(A, B, m, n, k, ref, streamk):
    best = (None, "-")
    C = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    # candidates: Origami's OWN default (None) + the same tiles steal-k swept.
    # Including the default means the library gets MAX opportunity (never worse
    # than its heuristic pick).
    for tile in [None] + TILES:
        try:
            sel = OrigamiMatmulSelector(m, n, k, torch.float16, torch.float16, torch.float16, A.device, streamk=streamk)
            if tile is not None:
                bm, bn, bk = tile
                sel._result.config.mt.m = bm
                sel._result.config.mt.n = bn
                sel._result.config.mt.k = bk
                sel._num_stages = NS
                if streamk:
                    sel._grid = CU
            tlabel = f"{sel.block_m}x{sel.block_n}x{sel.block_k}" + ("" if tile is not None else "*")
            cfg = matmul_preamble(sel)
            reset = (lambda: cfg.reset(streamk=True)) if streamk else (lambda: None)
            f = lambda: matmul_lt(A, B, C, sel, cfg, streamk)
            reset(); C.zero_(); f(); torch.cuda.synchronize()
        except Exception:
            continue
        if not okc(C, ref):
            continue
        ms = bench(f, reset)
        if best[0] is None or ms < best[0]:
            best = (ms, tlabel)
    return best


def run_one(m, n, k):
    A = torch.randn(m, k, device=DEV, dtype=torch.float16)
    B = torch.randn(k, n, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B)
    pt_ms = bench(lambda: torch.matmul(A, B), lambda: None)
    sk_ms, sk_t = tune_stealk(A, B, m, n, k, ref)
    dp_ms, dp_t = tune_lib(A, B, m, n, k, ref, False)
    st_ms, st_t = tune_lib(A, B, m, n, k, ref, True)
    del A, B, ref
    g = lambda x: f"{x:.4f}" if x else "-"
    return f"CSV,{m},{n},{k},{pt_ms:.4f},{g(sk_ms)},{sk_t},{g(dp_ms)},{dp_t},{g(st_ms)},{st_t}"


def main():
    shard = sys.argv[1]
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    off = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    shapes = [tuple(int(x) for x in l.split()) for l in open(shard) if l.split()][off::stride]
    print(f"# EQOPP shard={shard} stride={stride} off={off} n={len(shapes)} tiles={len(TILES)}", flush=True)
    for i, (m, n, k) in enumerate(shapes):
        print(run_one(m, n, k), flush=True)
        if (i + 1) % 25 == 0:
            torch.cuda.empty_cache()
    print("# EQOPP_DONE", flush=True)


if __name__ == "__main__":
    main()
