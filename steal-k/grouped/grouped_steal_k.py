#!/usr/bin/env python3
"""grouped-steal-k: a work-stealing GROUPED (MoE-style) GEMM.

Many independent GEMMs (groups/experts) that share N and K but have varying row
counts M_g are computed in ONE launch. All output tiles across all groups are
flattened into a single work list; each persistent workgroup atomically steals
the next global tile index, maps it back to (group, local tile), and computes it
-- so imbalanced groups (the MoE pathology) are load-balanced automatically.

Layout (MoE FFN):
  A : [total_M, K]        tokens of all experts stacked (row-partitioned by group)
  B : [G, K, N]           per-expert weights
  C : [total_M, N]
  group_off  : [G+1] int32  row offsets: group g owns rows [off[g], off[g+1])
  tile_prefix: [G+1] int32  prefix-sum of per-group tile counts; total = prefix[G]

Correctness pattern reuses steal-k's tricks: unmasked EVEN_K loads with rm%Mg /
rn%N address wrapping + a masked store, so the software pipeliner (num_stages>=2)
works. No global reduction (each tile is fully owned -> register accumulate).
"""
import torch
import triton
import triton.language as tl

DEV = "cuda"


@triton.jit
def grouped_steal_k(
    A, B, C, group_off, tile_prefix, work_ctr,
    N, K,
    stride_am, stride_ak, stride_bg, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    G: tl.constexpr, NUM_TILES, EVEN_K: tl.constexpr = True,
):
    num_pid_n = tl.cdiv(N, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    iters = tl.cdiv(K, BLOCK_K)
    wid = tl.atomic_add(work_ctr, 1, scope="gpu")
    while wid < NUM_TILES:
        # --- map global tile id -> (group g, local tile) ---
        # g = number of groups whose prefix-end is <= wid (skips empty groups)
        g = 0
        for gg in range(0, G):
            g += tl.where(tl.load(tile_prefix + gg + 1) <= wid, 1, 0)
        base_tile = tl.load(tile_prefix + g)
        local = wid - base_tile
        row0 = tl.load(group_off + g)
        Mg = tl.load(group_off + g + 1) - row0

        pid_m = local // num_pid_n
        pid_n = local % num_pid_n
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = rm < Mg
        mask_n = rn < N
        rm_w = row0 + (rm % Mg)     # absolute rows, wrapped -> always valid addr
        rn_w = rn % N

        A_BASE = A + rm_w[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + g * stride_bg + rk[:, None] * stride_bk + rn_w[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for it in range(0, iters):
            if EVEN_K:
                a = tl.load(A_BASE)
                b = tl.load(B_BASE)
            else:
                kmask = (it * BLOCK_K + rk) < K
                a = tl.load(A_BASE, mask=kmask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=kmask[:, None], other=0.0)
            acc += tl.dot(a, b, allow_tf32=False)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk

        C_ = C + rm_w[:, None] * stride_cm + rn_w[None, :] * stride_cn
        tl.store(C_, acc.to(C.type.element_ty), mask=mask_m[:, None] & mask_n[None, :])
        wid = tl.atomic_add(work_ctr, 1, scope="gpu")


def _pick_warps(bm, bn):
    a = bm * bn
    return 8 if a >= 128 * 128 else (4 if a >= 64 * 64 else 1)


def grouped_matmul(A, B, group_off, Ms, N, K, bm=128, bn=128, bk=64, grid=None):
    """Launch grouped_steal_k. A:[sumM,K], B:[G,K,N], returns C:[sumM,N]."""
    G = B.shape[0]
    CU = torch.cuda.get_device_properties(0).multi_processor_count
    npn = triton.cdiv(N, bn)
    tiles = [triton.cdiv(int(m), bm) * npn for m in Ms]
    tile_prefix = torch.tensor([0] + list(torch.tensor(tiles).cumsum(0).tolist()), device=DEV, dtype=torch.int32)
    num_tiles = int(tile_prefix[-1].item())
    C = torch.zeros(A.shape[0], N, device=DEV, dtype=A.dtype)
    wc = torch.zeros(1, device=DEV, dtype=torch.int32)
    grid = grid or min(num_tiles, CU) or 1
    grouped_steal_k[(grid,)](
        A, B, C, group_off, tile_prefix, wc, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), B.stride(2), C.stride(0), C.stride(1),
        bm, bn, bk, G, num_tiles, (K % bk == 0),
        num_warps=_pick_warps(bm, bn), matrix_instr_nonkdim=16, kpack=1, num_stages=2,
    )
    return C, num_tiles


def make_problem(Ms, N, K, seed=0):
    torch.manual_seed(seed)
    G = len(Ms)
    total = sum(Ms)
    A = torch.randn(total, K, device=DEV, dtype=torch.float16)
    B = torch.randn(G, K, N, device=DEV, dtype=torch.float16)
    off = torch.tensor([0] + list(torch.tensor(Ms).cumsum(0).tolist()), device=DEV, dtype=torch.int32)
    return A, B, off


def ref_loop(A, B, off, N):
    C = torch.empty(A.shape[0], N, device=DEV, dtype=A.dtype)
    for g in range(B.shape[0]):
        s, e = int(off[g]), int(off[g + 1])
        if e > s:
            C[s:e] = A[s:e] @ B[g]
    return C


def bench(fn, n_warmup=10, n_repeat=30):
    flush = torch.empty(int(256 * 1024 * 1024 // 4), dtype=torch.int, device=DEV)
    for _ in range(n_warmup):
        flush.zero_(); fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    for i in range(n_repeat):
        flush.zero_(); torch.cuda.synchronize()
        st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    t = sorted(a.elapsed_time(b) for a, b in zip(st, en))
    return t[len(t) // 2]


def run(Ms, N, K, bm=128, bn=128, bk=64, tag=""):
    A, B, off = make_problem(Ms, N, K)
    ref = ref_loop(A, B, off, N).float()
    C, nt = grouped_matmul(A, B, off, Ms, N, K, bm, bn, bk)
    torch.cuda.synchronize()
    ok = torch.allclose(C.float(), ref, atol=1.0, rtol=0.05)
    err = (C.float() - ref).abs().max().item()
    flops = 2 * sum(int(m) * N * K for m in Ms)
    gk_ms = bench(lambda: grouped_matmul(A, B, off, Ms, N, K, bm, bn, bk))
    lp_ms = bench(lambda: ref_loop(A, B, off, N))
    tf = lambda ms: flops / 1e12 / (ms * 1e-3)
    print(f"[G={len(Ms)} N={N} K={K} tile {bm}x{bn}x{bk} tiles={nt} sumM={sum(Ms)}] {tag}", flush=True)
    print(f"  grouped-steal-k: {gk_ms:8.4f} ms  {tf(gk_ms):7.1f} TF/s  ok={ok} max_err={err:.3f}", flush=True)
    print(f"  torch per-group loop: {lp_ms:8.4f} ms  {tf(lp_ms):7.1f} TF/s  (speedup {lp_ms/gk_ms:.2f}x)", flush=True)


def main():
    import random
    print(f"arch={torch.cuda.get_device_properties(0).gcnArchName} CUs={torch.cuda.get_device_properties(0).multi_processor_count}\n", flush=True)
    # MoE-like: many experts, shared N,K, varying tokens per expert (imbalanced)
    random.seed(0)
    cfgs = [
        ("balanced 8x2048", [2048] * 8, 4096, 4096),
        ("imbalanced 8", [4096, 2048, 1024, 512, 256, 128, 64, 4096], 4096, 4096),
        ("skewed 16 (one hot expert)", [8192] + [128] * 15, 4096, 4096),
        ("many small 64 experts", [random.choice([64, 128, 256, 512]) for _ in range(64)], 2048, 4096),
        ("moe 32 experts x~1k", [random.randint(256, 2048) for _ in range(32)], 5120, 4096),
    ]
    for tag, Ms, N, K in cfgs:
        for tile in [(128, 128, 64), (64, 64, 64), (128, 256, 64)]:
            run(Ms, N, K, *tile, tag=tag)
        print()


if __name__ == "__main__":
    main()
