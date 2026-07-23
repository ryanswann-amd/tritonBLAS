#!/usr/bin/env python3
"""Option 1: aiter batched GEMM vs grouped-steal-k on EQUAL-M groups (bf16).

Uniform per-group M -> grouped == batched, so aiter.batched_gemm_bf16_CK
(CK/asm, XQ[B,M,K] @ WQ[B,N,K]^T -> [B,M,N]) is a fair, direct baseline.
Compares aiter vs grouped-steal-k (best of register / split-K) vs torch bmm.
"""
import torch, triton
import aiter
from grouped_steal_k import grouped_matmul, bench
from grouped_split_k import grouped_matmul_sk

DEV = "cuda"
CU = torch.cuda.get_device_properties(0).multi_processor_count
DT = torch.bfloat16


def aiter_bgemm(XQ, WQ):
    return aiter.batched_gemm_bf16_CK(XQ, WQ)


def steal_best(A2d, B, off, G, M, N, K):
    Ms = [M] * G
    best = None; bv = "-"
    cands = [("reg128x256", grouped_matmul, 128, 256, 64),
             ("reg128x128", grouped_matmul, 128, 128, 64),
             ("splitk64", grouped_matmul_sk, 64, 64, 64),
             ("splitk128", grouped_matmul_sk, 128, 128, 64)]
    for (nm, fn, bm, bn, bk) in cands:
        try:
            fn(A2d, B, off, Ms, N, K, bm, bn, bk)[0]
            torch.cuda.synchronize()
            ms = bench(lambda: fn(A2d, B, off, Ms, N, K, bm, bn, bk), n_repeat=20)
            if best is None or ms < best:
                best, bv = ms, nm
        except Exception:
            continue
    return best, bv


def run(G, M, N, K):
    torch.manual_seed(0)
    A = torch.randn(G, M, K, device=DEV, dtype=DT) * 0.1
    B = torch.randn(G, K, N, device=DEV, dtype=DT) * 0.1          # per-expert [K,N]
    ref = torch.bmm(A.float(), B.float())                         # [G,M,N]
    WQ = B.transpose(1, 2).contiguous()                           # [G,N,K] for aiter X@W^T

    # correctness: aiter
    try:
        Ca = aiter_bgemm(A, WQ)
        torch.cuda.synchronize()
        ok_a = torch.allclose(Ca.float(), ref, atol=1.0, rtol=0.05)
    except Exception as e:
        print(f"[G={G} M={M} N={N} K={K}] aiter FAIL: {type(e).__name__} {str(e)[:100]}", flush=True)
        ok_a = None; Ca = None

    # steal-k (stacked layout)
    A2d = A.reshape(G * M, K).contiguous()
    off = torch.tensor([i * M for i in range(G + 1)], device=DEV, dtype=torch.int32)
    Ck = grouped_matmul(A2d, B, off, [M] * G, N, K, 128, 256, 64)[0]
    torch.cuda.synchronize()
    ok_k = torch.allclose(Ck.reshape(G, M, N).float(), ref, atol=1.0, rtol=0.05)

    flops = 2 * G * M * N * K
    tf = lambda ms: flops / 1e12 / (ms * 1e-3)
    a_ms = bench(lambda: aiter_bgemm(A, WQ), n_repeat=20) if Ca is not None else None
    k_ms, kv = steal_best(A2d, B, off, G, M, N, K)
    bmm_ms = bench(lambda: torch.bmm(A, B), n_repeat=20)

    print(f"[G={G} M={M} N={N} K={K}]  (flops {flops/1e9:.0f} GF)", flush=True)
    if a_ms:
        print(f"  aiter batched_gemm_bf16 : {a_ms:8.4f} ms  {tf(a_ms):7.1f} TF/s  ok={ok_a}", flush=True)
    print(f"  grouped-steal-k (best)  : {k_ms:8.4f} ms  {tf(k_ms):7.1f} TF/s  ok={ok_k} [{kv}]", flush=True)
    print(f"  torch.bmm               : {bmm_ms:8.4f} ms  {tf(bmm_ms):7.1f} TF/s", flush=True)
    if a_ms:
        print(f"  --> steal-k vs aiter = {a_ms/k_ms:.2f}x   (aiter vs bmm {bmm_ms/a_ms:.2f}x)", flush=True)
    print(flush=True)


def main():
    print(f"CUs={CU} dtype={DT}\n", flush=True)
    cfgs = [
        (8, 2048, 4096, 4096),
        (8, 4096, 4096, 4096),
        (16, 512, 4096, 4096),
        (32, 256, 2048, 4096),
        (128, 128, 2048, 4096),
        (8, 64, 512, 16384),     # skinny experts
        (16, 128, 1024, 8192),
    ]
    for c in cfgs:
        run(*c)
    print("AIVG_DONE", flush=True)


if __name__ == "__main__":
    main()
