#!/usr/bin/env python3
"""steal-k (64x64 stream-K + tree reduction) vs torch.matmul, big-K skinny sweep.

Reads shapes ("M N K" per line) from a shard file and, for each, benchmarks
torch.matmul (the PyTorch/rocBLAS baseline) and the single-compile sk_tree kernel,
checks correctness, and emits a CSV row. Designed for ~10k shapes: sk_tree compiles
ONCE (runtime tile counts), grid = CU.

CSV: CSV,m,n,k,torch_ms,sk_ms,torch_tf,sk_tf,speedup,ok
     speedup = torch_ms / sk_ms  (>1 => steal-k faster than pytorch)
"""
import sys, torch, triton
from steal_k import sk_tree

DEV = "cuda"
CU = torch.cuda.get_device_properties(0).multi_processor_count
BM = BN = 64
BK = 64
FLUSH = torch.empty(int(256 * 1024 * 1024 // 4), dtype=torch.int, device=DEV)
P = torch.zeros(CU, BM * BN, device=DEV, dtype=torch.float32)
LOCKS = torch.zeros(CU, device=DEV, dtype=torch.int32)


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
        reset()
        st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    t = sorted(a.elapsed_time(b) for a, b in zip(st, en))
    return t[len(t) // 2]


def sk_launch(A, B, C, m, n, k):
    nt = triton.cdiv(m, BM) * triton.cdiv(n, BN)
    ipt = triton.cdiv(k, BK)
    sk_tree[(CU,)](
        A, B, C, P, LOCKS, m, n, k,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        nt, ipt, CU, BM, BN, BK, 12, True,
        num_warps=4, matrix_instr_nonkdim=16, kpack=1, num_stages=2,
    )


def run_one(m, n, k):
    A = torch.randn(m, k, device=DEV, dtype=torch.float16)
    B = torch.randn(k, n, device=DEV, dtype=torch.float16)
    C = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B)
    reset = lambda: (LOCKS.zero_(), C.zero_())
    try:
        reset(); sk_launch(A, B, C, m, n, k); torch.cuda.synchronize()
    except Exception as e:
        return f"CSV,{m},{n},{k},-,-,-,-,-,launchfail"
    ok = torch.allclose(C.float(), ref.float(), atol=1.0, rtol=0.05)
    sk_ms = bench(lambda: sk_launch(A, B, C, m, n, k), reset)
    pt_ms = bench(lambda: torch.matmul(A, B), lambda: None)
    del A, B, C, ref
    return (f"CSV,{m},{n},{k},{pt_ms:.4f},{sk_ms:.4f},"
            f"{tflops(m,n,k,pt_ms):.1f},{tflops(m,n,k,sk_ms):.1f},{pt_ms/sk_ms:.3f},{ok}")


def main():
    shard = sys.argv[1]
    shapes = []
    for line in open(shard):
        line = line.split("#")[0].strip()
        if line:
            shapes.append(tuple(int(x) for x in line.split()))
    print(f"# shard={shard} shapes={len(shapes)} CU={CU}", flush=True)
    for i, (m, n, k) in enumerate(shapes):
        print(run_one(m, n, k), flush=True)
        if (i + 1) % 200 == 0:
            torch.cuda.empty_cache()
    print("# SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
