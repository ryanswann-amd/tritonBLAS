#!/usr/bin/env python3
"""Big steal-k tuning sweep.

For each GEMM shape, benchmarks the library data-parallel + Stream-K baselines,
then sweeps steal-k over {tile, path (register/stream-K), grid=#workgroups,
num_stages}, checking correctness and median TFLOP/s for every config. Emits one
CSV row per (shape, variant/config) to stdout so results can be aggregated.

CSV columns:
  shape,m,n,k,variant,bm,bn,bk,path,grid,ns,nwarps,ms,tflops,ok,vs_dp,vs_sk
    - variant: 'dp' | 'sk' | 'steal'
    - path:    'dp' | 'sk' | 'register' | 'streamk'  (kernel code path)
    - grid:    number of workgroups launched
    - vs_dp/vs_sk: speedup of this row's time vs the library dp/sk baseline

Usage: python3 sweep_tune.py M N K [tag]
"""
import sys
import torch
import triton

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


def pick_warps(bm, bn):
    area = bm * bn
    if area >= 128 * 128:
        return 8
    if area >= 64 * 64:
        return 4
    return 1


def emit(row):
    print("CSV," + ",".join(str(x) for x in row), flush=True)


def steal_cfg(A, B, m, n, k, bm, bn, bk, streamk, grid, ns, nwarps, ref):
    """Launch one steal-k config; return (ms, ok) or (None, False) on failure."""
    npm = triton.cdiv(m, bm); npn = triton.cdiv(n, bn)
    nt = npm * npn
    ipt = triton.cdiv(k, bk)
    Ck = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    P = torch.zeros(grid, bm * bn, device=DEV, dtype=torch.float32) if streamk else torch.zeros(1, device=DEV, dtype=torch.float32)
    locks = torch.zeros(grid, device=DEV, dtype=torch.int32) if streamk else torch.zeros(1, device=DEV, dtype=torch.int32)
    wc = torch.zeros(1, device=DEV, dtype=torch.int32)
    ev = torch.zeros(1, 1, EVENT_FIELDS, device=DEV, dtype=torch.int64)

    def f():
        steal_k_adaptive[(grid,)](
            A, B, Ck, P, locks, wc, ev, m, n, k,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), Ck.stride(0), Ck.stride(1),
            bm, bn, bk, nt, ipt, grid, 1, EVENT_FIELDS, streamk, False, False,
            num_warps=nwarps, matrix_instr_nonkdim=16, kpack=1, num_stages=ns,
        )

    reset = (lambda: (wc.zero_(), locks.zero_())) if streamk else (lambda: wc.zero_())
    try:
        reset(); f(); torch.cuda.synchronize()
    except Exception:
        return None, False
    ok = okc(Ck, ref)
    ms = bench(f, reset)
    return ms, ok


def candidates(m, n, k):
    """Config list: (bm,bn,bk, streamk, grid, ns). Shape-aware, ~10-16 per shape."""
    cfgs = []
    tiles = [(128, 128, 64), (256, 256, 64), (128, 256, 64), (256, 128, 64), (64, 64, 64)]
    for (bm, bn, bk) in tiles:
        nt = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        ipt = triton.cdiv(k, bk)
        # skip tiles that produce absurd tiling for this shape
        if nt > 24 * CU:            # tile too small -> way too many tiles
            continue
        if bm > max(m, 64) or bn > max(n, 64):   # tile bigger than the whole dim
            continue
        # ---- register / work-stealing path ----
        grids_reg = sorted(set([min(nt, CU), CU]))
        for g in grids_reg:
            cfgs.append((bm, bn, bk, False, g, 2))
        if nt <= CU:               # underfilled: ns=1 register sometimes wins
            cfgs.append((bm, bn, bk, False, min(nt, CU), 1))
        # ---- persistent stream-K path (only when it can actually split K) ----
        if nt < CU and nt * ipt >= CU:
            cfgs.append((bm, bn, bk, True, CU, 2))
            cfgs.append((bm, bn, bk, True, CU, 1))
    return cfgs


def run_shape(m, n, k, tag=""):
    torch.manual_seed(0)
    A = torch.randn(m, k, device=DEV, dtype=torch.float16)
    B = torch.randn(k, n, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B)
    shape = f"{m}x{n}x{k}"
    print(f"### shape {shape} (CU={CU}) {tag}", flush=True)

    # library data-parallel
    selD = OrigamiMatmulSelector(m, n, k, torch.float16, torch.float16, torch.float16, A.device)
    cfgD = matmul_preamble(selD)
    Cd = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    dp = lambda: matmul_lt(A, B, Cd, selD, cfgD, False)
    dp(); torch.cuda.synchronize()
    dp_ms = bench(dp, lambda: None)
    dp_ok = okc(Cd, ref)
    emit([shape, m, n, k, "dp", selD.block_m, selD.block_n, selD.block_k, "dp", "-", "-", "-",
          f"{dp_ms:.4f}", f"{tflops(m,n,k,dp_ms):.1f}", dp_ok, "1.00", "-"])

    # library stream-K
    selS = OrigamiMatmulSelector(m, n, k, torch.float16, torch.float16, torch.float16, A.device, streamk=True)
    cfgS = matmul_preamble(selS)
    Cs = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    sk = lambda: matmul_lt(A, B, Cs, selS, cfgS, True)
    resetS = lambda: cfgS.reset(streamk=True)
    resetS(); sk(); torch.cuda.synchronize()
    sk_ms = bench(sk, resetS)
    sk_ok = okc(Cs, ref)
    emit([shape, m, n, k, "sk", selS.block_m, selS.block_n, selS.block_k, "sk", "-", "-", "-",
          f"{sk_ms:.4f}", f"{tflops(m,n,k,sk_ms):.1f}", sk_ok, f"{dp_ms/sk_ms:.2f}", "1.00"])

    # steal-k sweep
    for (bm, bn, bk, streamk, grid, ns) in candidates(m, n, k):
        nw = pick_warps(bm, bn)
        ms, ok = steal_cfg(A, B, m, n, k, bm, bn, bk, streamk, grid, ns, nw, ref)
        path = "streamk" if streamk else "register"
        if ms is None:
            emit([shape, m, n, k, "steal", bm, bn, bk, path, grid, ns, nw, "FAIL", "-", "launchfail", "-", "-"])
            continue
        emit([shape, m, n, k, "steal", bm, bn, bk, path, grid, ns, nw,
              f"{ms:.4f}", f"{tflops(m,n,k,ms):.1f}", ok, f"{dp_ms/ms:.2f}", f"{sk_ms/ms:.2f}"])
    print(f"### done {shape}", flush=True)


def main():
    if len(sys.argv) >= 4:
        tag = sys.argv[4] if len(sys.argv) > 4 else ""
        run_shape(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), tag)
    else:
        print("usage: sweep_tune.py M N K [tag]")


if __name__ == "__main__":
    main()
