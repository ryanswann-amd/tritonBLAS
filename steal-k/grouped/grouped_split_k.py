#!/usr/bin/env python3
"""grouped-steal-k with PER-GROUP SPLIT-K (persistent Stream-K + tree reduction).

Extends grouped_steal_k to the underfilled regime (skinny experts, big K): the
flat (group, tile, k-iter) work space is split contiguously across grid=CU
persistent workgroups. Each chunk resolves its (group, local tile) via the
prefix-sum, computes a partial K-range, and each global tile is reduced with the
hardened log-depth tree (system-scope release/acquire). Fills all CUs even when
total tiles << CU.

Reuses the single-GEMM sk_tree structure; only adds the group resolution for the
A/B/C addressing.
"""
import sys, torch, triton
import triton.language as tl
from grouped_steal_k import make_problem, ref_loop, bench, grouped_matmul, _pick_warps

DEV = "cuda"
CU = torch.cuda.get_device_properties(0).multi_processor_count


@triton.jit
def grouped_sk_tree(
    A, B, C, group_off, tile_prefix, locks, P,
    N, K,
    stride_am, stride_ak, stride_bg, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    G: tl.constexpr, NUM_TILES, NUM_SMS, TREE_ROUNDS: tl.constexpr = 12, EVEN_K: tl.constexpr = True,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    ITERS = tl.cdiv(K, BLOCK_K)                 # k-iters per tile (shared K)
    total = NUM_TILES * ITERS
    q = total // NUM_SMS
    rem = total % NUM_SMS
    start_iter = pid * q + tl.minimum(pid, rem)
    last_iter = (pid + 1) * q + tl.minimum(pid + 1, rem)

    while start_iter < last_iter:
        remainder = start_iter % ITERS
        end_iter = tl.minimum(start_iter + (ITERS - remainder), last_iter)
        gtile = start_iter // ITERS
        tile_iter = gtile * ITERS
        # resolve group from global tile id
        g = 0
        for gg in range(0, G):
            g += tl.where(tl.load(tile_prefix + gg + 1) <= gtile, 1, 0)
        local = gtile - tl.load(tile_prefix + g)
        row0 = tl.load(group_off + g)
        Mg = tl.load(group_off + g + 1) - row0
        pid_m = local // num_pid_n
        pid_n = local % num_pid_n
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = rm < Mg
        mask_n = rn < N
        rm_w = row0 + (rm % Mg)
        rn_w = rn % N
        A_BASE = A + rm_w[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_K * stride_ak * remainder
        B_BASE = B + g * stride_bg + rk[:, None] * stride_bk + rn_w[None, :] * stride_bn + BLOCK_K * stride_bk * remainder
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for it in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(A_BASE)
                b = tl.load(B_BASE)
            else:
                gko = (it - tile_iter) * BLOCK_K
                kmask = gko + rk < K
                a = tl.load(A_BASE, mask=kmask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=kmask[:, None], other=0.0)
            acc += tl.dot(a, b, allow_tf32=False)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk

        # ---- tree reduction over the tile's contributors (flat-iter space) ----
        thr = rem * (q + 1)
        ti = tile_iter
        tj = tile_iter + ITERS - 1
        base = tl.where(ti < thr, ti // (q + 1), rem + (ti - thr) // tl.maximum(q, 1))
        last = tl.where(tj < thr, tj // (q + 1), rem + (tj - thr) // tl.maximum(q, 1))
        s = last - base + 1
        lp = pid - base
        done_tree = lp < 0
        for r in range(0, TREE_ROUNDS):
            stride = 1 << r
            if ((stride < s) & (~done_tree)):
                if (lp % (stride * 2)) == 0:
                    partner = lp + stride
                    if partner < s:
                        ppid = base + partner
                        while tl.atomic_add(locks + ppid, 0, sem="acquire", scope="sys") < (r + 1):
                            pass
                        Pf = P + ppid * BLOCK_M * BLOCK_N + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
                        acc += tl.load(Pf, cache_modifier=".cv")
                else:
                    Ps = P + pid * BLOCK_M * BLOCK_N + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
                    tl.store(Ps, acc, cache_modifier=".wt")
                    tl.debug_barrier()
                    tl.atomic_xchg(locks + pid, r + 1, sem="release", scope="sys")
                    done_tree = lp >= 0
        if lp == 0:
            C_ = C + rm_w[:, None] * stride_cm + rn_w[None, :] * stride_cn
            tl.store(C_, acc.to(C.type.element_ty), mask=mask_m[:, None] & mask_n[None, :])
        start_iter = end_iter


def grouped_matmul_sk(A, B, group_off, Ms, N, K, bm=64, bn=64, bk=64):
    G = B.shape[0]
    npn = triton.cdiv(N, bn)
    tiles = [triton.cdiv(int(m), bm) * npn for m in Ms]
    tile_prefix = torch.tensor([0] + list(torch.tensor(tiles).cumsum(0).tolist()), device=DEV, dtype=torch.int32)
    num_tiles = int(tile_prefix[-1].item())
    C = torch.zeros(A.shape[0], N, device=DEV, dtype=A.dtype)
    P = torch.zeros(CU, bm * bn, device=DEV, dtype=torch.float32)
    locks = torch.zeros(CU, device=DEV, dtype=torch.int32)
    grouped_sk_tree[(CU,)](
        A, B, C, group_off, tile_prefix, locks, P, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), B.stride(2), C.stride(0), C.stride(1),
        bm, bn, bk, G, num_tiles, CU, 12, (K % bk == 0),
        num_warps=_pick_warps(bm, bn), matrix_instr_nonkdim=16, kpack=1, num_stages=2,
    )
    return C, num_tiles, locks


def run(Ms, N, K, bm=64, bn=64, bk=64, tag=""):
    A, B, off = make_problem(Ms, N, K)
    ref = ref_loop(A, B, off, N).float()
    flops = 2 * sum(int(m) * N * K for m in Ms)
    tf = lambda ms: flops / 1e12 / (ms * 1e-3)
    reg_tiles = sum(triton.cdiv(int(m), bm) * triton.cdiv(N, bn) for m in Ms)

    def sk_reset_call():
        C, _, _ = grouped_matmul_sk(A, B, off, Ms, N, K, bm, bn, bk)
        return C
    C_sk, nt, _ = grouped_matmul_sk(A, B, off, Ms, N, K, bm, bn, bk); torch.cuda.synchronize()
    ok_sk = torch.allclose(C_sk.float(), ref, atol=1.0, rtol=0.05)
    C_rg, _ = grouped_matmul(A, B, off, Ms, N, K, bm, bn, bk); torch.cuda.synchronize()
    ok_rg = torch.allclose(C_rg.float(), ref, atol=1.0, rtol=0.05)

    sk_ms = bench(lambda: grouped_matmul_sk(A, B, off, Ms, N, K, bm, bn, bk))
    rg_ms = bench(lambda: grouped_matmul(A, B, off, Ms, N, K, bm, bn, bk))
    lp_ms = bench(lambda: ref_loop(A, B, off, N))
    print(f"[G={len(Ms)} N={N} K={K} tile {bm}x{bn}x{bk} tiles={reg_tiles}(<CU={CU}?) sumM={sum(Ms)}] {tag}", flush=True)
    print(f"  grouped SPLIT-K (streamk+tree): {sk_ms:8.4f} ms  {tf(sk_ms):7.1f} TF/s  ok={ok_sk}", flush=True)
    print(f"  grouped register (no split)   : {rg_ms:8.4f} ms  {tf(rg_ms):7.1f} TF/s  ok={ok_rg}  (split-K vs reg {rg_ms/sk_ms:.2f}x)", flush=True)
    print(f"  torch per-group loop          : {lp_ms:8.4f} ms  {tf(lp_ms):7.1f} TF/s  (split-K vs loop {lp_ms/sk_ms:.2f}x)", flush=True)


def main():
    print(f"arch={torch.cuda.get_device_properties(0).gcnArchName} CUs={CU}\n", flush=True)
    # underfilled: skinny experts, big K -> few tiles, split-K should fill CUs
    cfgs = [
        ("8 skinny experts M=64 N=512 K=16384", [64] * 8, 512, 16384),
        ("8 experts M=128 N=256 K=16384", [128] * 8, 256, 16384),
        ("4 experts M=64 N=512 K=32768", [64] * 4, 512, 32768),
        ("16 experts M=32 N=256 K=16384", [32] * 16, 256, 16384),
        ("8 experts M=256 N=512 K=8192", [256] * 8, 512, 8192),
    ]
    for tag, Ms, N, K in cfgs:
        run(Ms, N, K, 64, 64, 64, tag=tag)
        print()


if __name__ == "__main__":
    main()
