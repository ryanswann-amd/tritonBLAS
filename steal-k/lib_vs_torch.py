#!/usr/bin/env python3
"""Library Stream-K (sk) and data-parallel (dp) vs torch.matmul, skinny big-K.

Answers: how does the tritonBLAS *library* Stream-K fare against PyTorch on the
same regime where we benched steal-k? Samples shapes from a shard.

CSV: CSV,m,n,k,torch_ms,sk_ms,dp_ms,sk_vs_torch,dp_vs_torch,sk_ok,dp_ok
"""
import sys, torch, triton
import tritonblas
from tritonblas import OrigamiMatmulSelector, matmul_preamble, matmul_lt

DEV = "cuda"
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


def run_one(m, n, k):
    A = torch.randn(m, k, device=DEV, dtype=torch.float16)
    B = torch.randn(k, n, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B)
    pt_ms = bench(lambda: torch.matmul(A, B), lambda: None)
    try:
        selS = OrigamiMatmulSelector(m, n, k, torch.float16, torch.float16, torch.float16, A.device, streamk=True)
        cfgS = matmul_preamble(selS)
        Cs = torch.zeros(m, n, device=DEV, dtype=torch.float16)
        skf = lambda: matmul_lt(A, B, Cs, selS, cfgS, True)
        rS = lambda: cfgS.reset(streamk=True)
        rS(); skf(); torch.cuda.synchronize(); sk_ok = okc(Cs, ref)
        sk_ms = bench(skf, rS)
    except Exception:
        sk_ms, sk_ok = None, False
    try:
        selD = OrigamiMatmulSelector(m, n, k, torch.float16, torch.float16, torch.float16, A.device)
        cfgD = matmul_preamble(selD)
        Cd = torch.zeros(m, n, device=DEV, dtype=torch.float16)
        dpf = lambda: matmul_lt(A, B, Cd, selD, cfgD, False)
        dpf(); torch.cuda.synchronize(); dp_ok = okc(Cd, ref)
        dp_ms = bench(dpf, lambda: None)
    except Exception:
        dp_ms, dp_ok = None, False
    del A, B, ref
    sks = f"{pt_ms/sk_ms:.3f}" if sk_ms else "-"
    dps = f"{pt_ms/dp_ms:.3f}" if dp_ms else "-"
    return (f"CSV,{m},{n},{k},{pt_ms:.4f},{sk_ms if sk_ms else '-'},{dp_ms if dp_ms else '-'},"
            f"{sks},{dps},{sk_ok},{dp_ok}")


def main():
    shard = sys.argv[1]
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 5   # subsample every Nth
    shapes = [tuple(int(x) for x in l.split()) for l in open(shard) if l.split()]
    shapes = shapes[::stride]
    print(f"# LIBVT shard={shard} sampled={len(shapes)}", flush=True)
    for i, (m, n, k) in enumerate(shapes):
        print(run_one(m, n, k), flush=True)
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
    print("# LIBVT_DONE", flush=True)


if __name__ == "__main__":
    main()
