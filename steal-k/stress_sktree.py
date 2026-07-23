"""Stress sk_tree correctness on previously-failing shapes (detect race)."""
import torch, triton
from steal_k import sk_tree
DEV = "cuda"; CU = torch.cuda.get_device_properties(0).multi_processor_count
BM = BN = BK = 64
P = torch.zeros(CU, BM * BN, device=DEV, dtype=torch.float32)
LOCKS = torch.zeros(CU, device=DEV, dtype=torch.int32)

SHAPES = [(192, 256, 16384), (48, 512, 16384), (320, 320, 12288), (192, 160, 49152),
          (224, 256, 8192), (80, 384, 2560)]
NITER = 300


def once(A, B, C, m, n, k):
    nt = triton.cdiv(m, BM) * triton.cdiv(n, BN); ipt = triton.cdiv(k, BK)
    sk_tree[(CU,)](A, B, C, P, LOCKS, m, n, k,
                   A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                   nt, ipt, CU, BM, BN, BK, 12, True,
                   num_warps=4, matrix_instr_nonkdim=16, kpack=1, num_stages=2)


for (m, n, k) in SHAPES:
    torch.manual_seed(0)
    A = torch.randn(m, k, device=DEV, dtype=torch.float16)
    B = torch.randn(k, n, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B).float()
    C = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    bad = 0; worst = 0.0
    for _ in range(NITER):
        C.zero_(); LOCKS.zero_()
        once(A, B, C, m, n, k); torch.cuda.synchronize()
        e = (C.float() - ref).abs().max().item(); worst = max(worst, e)
        if e > 1.0:
            bad += 1
    print(f"{m}x{n}x{k}: {bad}/{NITER} bad, worst={worst:.2f}", flush=True)
