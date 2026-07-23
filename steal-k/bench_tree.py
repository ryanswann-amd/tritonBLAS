#!/usr/bin/env python3
"""Compare serial owner-fold vs TREE reduction over contributors (steal-k stream-K).

For each split-heavy shape: library dp/sk baselines, then steal-k stream-K with
TREE=False (serial) and TREE=True (log-depth tree), checking correctness + median
TFLOP/s. Focus: 64x64x64 tile, grid=304 (the regime where reduce dominates).

Usage: python3 bench_tree.py [M N K]
"""
import sys, torch, triton
import tritonblas
from tritonblas import OrigamiMatmulSelector, matmul_preamble, matmul_lt
from steal_k import steal_k_adaptive, EVENT_FIELDS

DEV = "cuda"
CU = torch.cuda.get_device_properties(0).multi_processor_count


def bench(fn, reset, n_warmup=12, n_repeat=30):
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


def tflops(m, n, k, ms):
    return 2 * m * n * k / 1e12 / (ms * 1e-3)


def okc(C, ref):
    return torch.allclose(C.to(torch.float32), ref.to(torch.float32), atol=1.0, rtol=0.05)


def steal(A, B, m, n, k, bm, bn, bk, grid, tree, ref):
    nt = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    ipt = triton.cdiv(k, bk)
    Ck = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    P = torch.zeros(grid, bm * bn, device=DEV, dtype=torch.float32)
    locks = torch.zeros(grid, device=DEV, dtype=torch.int32)
    wc = torch.zeros(1, device=DEV, dtype=torch.int32)
    ev = torch.zeros(1, 1, EVENT_FIELDS, device=DEV, dtype=torch.int64)
    nw = 8 if bm * bn >= 128 * 128 else (4 if bm * bn >= 64 * 64 else 1)

    def f():
        steal_k_adaptive[(grid,)](
            A, B, Ck, P, locks, wc, ev, m, n, k,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), Ck.stride(0), Ck.stride(1),
            bm, bn, bk, nt, ipt, grid, 1, EVENT_FIELDS, True, False, False,
            True, 8, tree, 10,
            num_warps=nw, matrix_instr_nonkdim=16, kpack=1, num_stages=2,
        )

    reset = lambda: (wc.zero_(), locks.zero_())
    try:
        reset(); f(); torch.cuda.synchronize()
    except Exception as e:
        return None, False, str(e)[:80]
    ok = okc(Ck, ref)
    ms = bench(f, reset)
    return ms, ok, ""


def run(m, n, k):
    torch.manual_seed(0)
    A = torch.randn(m, k, device=DEV, dtype=torch.float16)
    B = torch.randn(k, n, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B)
    print(f"\n### {m}x{n}x{k} (CU={CU})", flush=True)

    selS = OrigamiMatmulSelector(m, n, k, torch.float16, torch.float16, torch.float16, A.device, streamk=True)
    cfgS = matmul_preamble(selS)
    Cs = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    sk = lambda: matmul_lt(A, B, Cs, selS, cfgS, True)
    rS = lambda: cfgS.reset(streamk=True)
    rS(); sk(); torch.cuda.synchronize()
    sk_ms = bench(sk, rS)
    print(f"  library streamk : {sk_ms:.4f} ms  {tflops(m,n,k,sk_ms):6.1f} TF/s  ok={okc(Cs,ref)}", flush=True)

    for (bm, bn, bk, grid) in [(64, 64, 64, 304)]:
        for tree in (False, True):
            ms, ok, err = steal(A, B, m, n, k, bm, bn, bk, grid, tree, ref)
            tag = "TREE  " if tree else "serial"
            if ms is None:
                print(f"  steal {tag} {bm}x{bn}x{bk} g{grid}: FAIL {err}", flush=True)
            else:
                print(f"  steal {tag} {bm}x{bn}x{bk} g{grid}: {ms:.4f} ms  {tflops(m,n,k,ms):6.1f} TF/s  "
                      f"ok={ok}  vsSK={sk_ms/ms:.2f}", flush=True)


def main():
    if len(sys.argv) >= 4:
        run(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    else:
        for shp in [(64, 512, 16384), (512, 512, 16384), (32, 256, 16384),
                    (128, 128, 32768), (256, 256, 16384), (128, 512, 16384)]:
            run(*shp)


if __name__ == "__main__":
    main()
