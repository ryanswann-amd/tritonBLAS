#!/usr/bin/env python3
"""steal-k using ORIGAMI's heuristic tile pick (no sweep) vs torch.matmul.

For each shape, asks the Origami Stream-K selector for its tile (block_m/n/k,
num_stages), runs sk_tree (stream-K + tree, grid=CU) with exactly that config,
and times vs torch. Answers: how much of steal-k's tuned perf does the Origami
heuristic recover, i.e. could steal-k ship with Origami as its selector?

CSV: CSV,m,n,k,torch_ms,sk_ms,orig_tile,torch_tf,sk_tf,speedup,ok
"""
import sys, torch, triton
from steal_k import sk_tree
from tritonblas import OrigamiMatmulSelector

DEV = "cuda"
CU = torch.cuda.get_device_properties(0).multi_processor_count
MAXEL = 256 * 256
P = torch.zeros(CU, MAXEL, device=DEV, dtype=torch.float32)
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


def tflops(m, n, k, ms):
    return 2 * m * n * k / 1e12 / (ms * 1e-3)


def nwarps(bm, bn):
    return 8 if bm * bn >= 128 * 128 else (4 if bm * bn >= 64 * 64 else 1)


def run_one(m, n, k):
    A = torch.randn(m, k, device=DEV, dtype=torch.float16)
    B = torch.randn(k, n, device=DEV, dtype=torch.float16)
    C = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B)
    pt_ms = bench(lambda: torch.matmul(A, B), lambda: None)
    # Origami's heuristic pick (Stream-K selector, matches steal-k's split-K nature)
    sel = OrigamiMatmulSelector(m, n, k, torch.float16, torch.float16, torch.float16, A.device, streamk=True)
    bm, bn, bk = sel.block_m, sel.block_n, sel.block_k
    ns = getattr(sel, "num_stages", 2)
    even = (k % bk == 0)
    if bm * bn > MAXEL:   # too big for P scratch -> clamp k so P fits (rare)
        return f"CSV,{m},{n},{k},{pt_ms:.4f},-,{bm}x{bn}x{bk},-,-,-,toobig"
    nt = triton.cdiv(m, bm) * triton.cdiv(n, bn); ipt = triton.cdiv(k, bk)
    def f():
        sk_tree[(CU,)](A, B, C, P, LOCKS, m, n, k,
                       A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                       nt, ipt, CU, bm, bn, bk, 12, even,
                       num_warps=nwarps(bm, bn), matrix_instr_nonkdim=16, kpack=1, num_stages=ns)
    reset = lambda: (LOCKS.zero_(), C.zero_())
    try:
        reset(); f(); torch.cuda.synchronize()
    except Exception as e:
        return f"CSV,{m},{n},{k},{pt_ms:.4f},-,{bm}x{bn}x{bk},-,-,-,fail:{type(e).__name__}"
    ok = torch.allclose(C.float(), ref.float(), atol=1.0, rtol=0.05)
    sk_ms = bench(f, reset)
    del A, B, ref
    return (f"CSV,{m},{n},{k},{pt_ms:.4f},{sk_ms:.4f},{bm}x{bn}x{bk},"
            f"{tflops(m,n,k,pt_ms):.1f},{tflops(m,n,k,sk_ms):.1f},{pt_ms/sk_ms:.3f},{ok}")


def main():
    shard = sys.argv[1]; stride = int(sys.argv[2]); off = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    shapes = [tuple(int(x) for x in l.split()) for l in open(shard) if l.split()][off::stride]
    print(f"# ORIGSK shard={shard} stride={stride} off={off} n={len(shapes)}", flush=True)
    for i, (m, n, k) in enumerate(shapes):
        print(run_one(m, n, k), flush=True)
        if (i + 1) % 25 == 0:
            torch.cuda.empty_cache()
    print("# ORIGSK_DONE", flush=True)


if __name__ == "__main__":
    main()
