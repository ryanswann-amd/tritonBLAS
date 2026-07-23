#!/usr/bin/env python3
"""Head-to-head: data-parallel vs Stream-K (library, Origami-tuned) vs steal-k.

For each shape, measures median kernel time (L2 flushed) and TFLOP/s for:
  - data-parallel : tritonblas.matmul_lt (persistent, Origami config)
  - streamk       : tritonblas.matmul_lt(enable_streamk=True) (Origami)
  - steal-k       : steal_k_matmul (this repo), TRACE off, Origami tile + split-K
                    sized to target full CU occupancy.
All checked for correctness vs torch.matmul.

Requires the real `origami` module built + a GPU.
"""
import torch
import triton

import tritonblas
from tritonblas import OrigamiMatmulSelector, matmul_preamble, matmul_lt
from steal_k import steal_k_adaptive, EVENT_FIELDS

DEV = "cuda"
CU = torch.cuda.get_device_properties(0).multi_processor_count


def bench(fn, reset, n_warmup=15, n_repeat=40):
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


def ok(C, ref):
    return torch.allclose(C.to(torch.float32), ref.to(torch.float32), atol=1.0, rtol=0.05)


def pick_warps(bm, bn):
    """num_warps valid for the tile (8 warps is wrong for tiny tiles)."""
    area = bm * bn
    if area >= 128 * 128:
        return 8
    if area >= 64 * 64:
        return 4
    return 1


def run_shape(m, n, k):
    A = torch.randn(m, k, device=DEV, dtype=torch.float16)
    B = torch.randn(k, n, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B)

    print(f"[shape {m}x{n}x{k}] starting", flush=True)
    # ---- data-parallel (library) ----
    print("  [dp] ...", flush=True)
    selD = OrigamiMatmulSelector(m, n, k, torch.float16, torch.float16, torch.float16, A.device)
    cfgD = matmul_preamble(selD)
    Cd = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    dp = lambda: matmul_lt(A, B, Cd, selD, cfgD, False)
    dp(); torch.cuda.synchronize()
    dp_ok = ok(Cd, ref)
    dp_ms = bench(dp, lambda: None)

    # ---- Stream-K (library) ----
    print("  [sk] ...", flush=True)
    selS = OrigamiMatmulSelector(m, n, k, torch.float16, torch.float16, torch.float16, A.device, streamk=True)
    cfgS = matmul_preamble(selS)
    Cs = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    sk = lambda: matmul_lt(A, B, Cs, selS, cfgS, True)
    resetS = lambda: cfgS.reset(streamk=True)
    resetS(); sk(); torch.cuda.synchronize()
    sk_ok = ok(Cs, ref)
    sk_ms = bench(sk, resetS)

    # ---- steal-k (this repo) ----
    print("  [steal-k] ...", flush=True)
    # steal-k picks its own tile. Large (register) regime -> big 256x256 tile
    # (more compute/tile, fewer atomic steals); skinny (stream-k) -> 128x128.
    # Large (register) regime -> 256x256x64 (library's preferred tile; ns=2 fits
    # 64KB LDS since Triton pipelines with (ns-1)*(A+B) buffers). Skinny/streamk
    # -> 128x128x64. (256x256 needs TRACE off so the s_memrealtime timers don't
    # spill the 256x256 accumulator -> bench launches with TRACE=False.)
    if triton.cdiv(m, 128) * triton.cdiv(n, 128) >= CU:
        bm, bn, bk = 256, 256, 64
    else:
        bm, bn, bk = 128, 128, 64
    npm = triton.cdiv(m, bm); npn = triton.cdiv(n, bn)
    nt = npm * npn
    ipt = triton.cdiv(k, bk)
    # Persistent Stream-K when underfilled (nt < CU); else work-stealing tiles.
    # grid == CU so all WGs are resident (occupancy 1/CU) -> fixup deadlock-free.
    STREAMK = (nt < CU) and (nt * ipt >= CU)   # need enough total K-iters to split
    SPLITK = triton.cdiv(CU, nt) if STREAMK else 1   # descriptive only (avg split)
    grid = CU
    Ck = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    P = torch.zeros(CU, bm * bn, device=DEV, dtype=torch.float32) if STREAMK else torch.zeros(1, device=DEV, dtype=torch.float32)
    locks = torch.zeros(CU, device=DEV, dtype=torch.int32) if STREAMK else torch.zeros(1, device=DEV, dtype=torch.int32)
    wc = torch.zeros(1, device=DEV, dtype=torch.int32)
    ev = torch.zeros(1, 1, EVENT_FIELDS, device=DEV, dtype=torch.int64)

    nwarps = pick_warps(bm, bn)

    def stealf():
        steal_k_adaptive[(grid,)](
            A, B, Ck, P, locks, wc, ev, m, n, k,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), Ck.stride(0), Ck.stride(1),
            bm, bn, bk, nt, ipt, CU, 1, EVENT_FIELDS, STREAMK, False, False,
            num_warps=nwarps, matrix_instr_nonkdim=16, kpack=1, num_stages=2,
        )

    if STREAMK:
        resetK = lambda: (wc.zero_(), locks.zero_())
    else:
        resetK = lambda: wc.zero_()
    resetK(); stealf(); torch.cuda.synchronize()
    k_ok = ok(Ck, ref)
    k_ms = bench(stealf, resetK)

    print(f"{m}x{n}x{k}  tile {bm}x{bn}x{bk}  tiles={nt}  splitK={SPLITK}  grid={grid}")
    mode = f"streamk ~x{SPLITK}" if STREAMK else "register-reduce"
    print(f"  data-parallel   : {dp_ms:8.4f} ms  {tflops(m,n,k,dp_ms):7.1f} TF/s  ok={dp_ok}")
    print(f"  stream-k        : {sk_ms:8.4f} ms  {tflops(m,n,k,sk_ms):7.1f} TF/s  ok={sk_ok}")
    print(f"  steal-k adaptive: {k_ms:8.4f} ms  {tflops(m,n,k,k_ms):7.1f} TF/s  ok={k_ok}"
          f"  [{mode}]  (vs DP {dp_ms/k_ms:.2f}x, vs SK {sk_ms/k_ms:.2f}x)")


def main():
    import sys
    if len(sys.argv) >= 4:
        run_shape(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        return
    shapes = [
        (4096, 4096, 4096),
        (2048, 2048, 2048),
        (1024, 1024, 1024),
        (512, 512, 16384),
        (256, 256, 16384),
        (128, 128, 32768),
    ]
    print(f"arch={torch.cuda.get_device_properties(0).gcnArchName}  CUs={CU}\n")
    for (m, n, k) in shapes:
        run_shape(m, n, k)
        print()


if __name__ == "__main__":
    main()
