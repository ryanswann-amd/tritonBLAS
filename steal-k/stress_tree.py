"""Stress the TREE reduction correctness over many repeated launches (TRACE off).

Detects latent cross-WG races that a single ok-check would miss.
"""
import sys, torch, triton
from steal_k import steal_k_adaptive, EVENT_FIELDS
DEV = "cuda"


def run(m, n, k, bm, bn, bk, grid, tree, iters=100):
    torch.manual_seed(0)
    A = torch.randn(m, k, device=DEV, dtype=torch.float16)
    B = torch.randn(k, n, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B).float()
    nt = triton.cdiv(m, bm) * triton.cdiv(n, bn); ipt = triton.cdiv(k, bk)
    Ck = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    P = torch.zeros(grid, bm * bn, device=DEV, dtype=torch.float32)
    locks = torch.zeros(grid, device=DEV, dtype=torch.int32)
    wc = torch.zeros(1, device=DEV, dtype=torch.int32)
    ev = torch.zeros(1, 1, EVENT_FIELDS, device=DEV, dtype=torch.int64)
    nw = 8 if bm * bn >= 128 * 128 else (4 if bm * bn >= 64 * 64 else 1)
    bad = 0; worst = 0.0
    for i in range(iters):
        Ck.zero_(); wc.zero_(); locks.zero_()
        steal_k_adaptive[(grid,)](
            A, B, Ck, P, locks, wc, ev, m, n, k,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), Ck.stride(0), Ck.stride(1),
            bm, bn, bk, nt, ipt, grid, 1, EVENT_FIELDS, True, False, False,
            True, 8, tree, 10, num_warps=nw, matrix_instr_nonkdim=16, kpack=1, num_stages=2)
        torch.cuda.synchronize()
        e = (Ck.float() - ref).abs().max().item(); worst = max(worst, e)
        if e > 1.0: bad += 1
    print(f"{m}x{n}x{k} tile{bm}x{bn}x{bk} g{grid} tree={tree}: {bad}/{iters} bad, worst_err={worst:.3f}", flush=True)


if __name__ == "__main__":
    shp = [(64, 512, 16384), (512, 512, 16384), (128, 512, 16384), (256, 256, 16384)]
    for (m, n, k) in shp:
        run(m, n, k, 64, 64, 64, 304, True)
