#!/usr/bin/env python3
"""Verify steal_k_adaptive (atomic split-K) correctness for a given config."""
import sys
import torch
import triton
from steal_k import steal_k_adaptive, EVENT_FIELDS

m = int(sys.argv[1]) if len(sys.argv) > 1 else 512
n = int(sys.argv[2]) if len(sys.argv) > 2 else 512
k = int(sys.argv[3]) if len(sys.argv) > 3 else 512
bm = int(sys.argv[4]) if len(sys.argv) > 4 else 128
bn = int(sys.argv[5]) if len(sys.argv) > 5 else 128
bk = int(sys.argv[6]) if len(sys.argv) > 6 else 64
NUM_SMS = int(sys.argv[7]) if len(sys.argv) > 7 else 304
STREAMK = bool(int(sys.argv[8])) if len(sys.argv) > 8 else True

dev = "cuda"
torch.manual_seed(0)
A = torch.randn(m, k, device=dev, dtype=torch.float16)
B = torch.randn(k, n, device=dev, dtype=torch.float16)
C = torch.zeros(m, n, device=dev, dtype=torch.float16)
ref = torch.matmul(A, B)

npm = triton.cdiv(m, bm); npn = triton.cdiv(n, bn)
nt = npm * npn
ipt = triton.cdiv(k, bk)
grid = NUM_SMS
P = torch.zeros(NUM_SMS, bm * bn, device=dev, dtype=torch.float32)
locks = torch.zeros(NUM_SMS, device=dev, dtype=torch.int32)
wc = torch.zeros(1, device=dev, dtype=torch.int32)
ev = torch.zeros(1, 1, EVENT_FIELDS, device=dev, dtype=torch.int64)

print(f"{m}x{n}x{k} tile {bm}x{bn}x{bk} nt={nt} ipt={ipt} NUM_SMS={NUM_SMS} grid={grid} STREAMK={STREAMK}")
steal_k_adaptive[(grid,)](
    A, B, C, P, locks, wc, ev, m, n, k,
    A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
    bm, bn, bk, nt, ipt, NUM_SMS, 1, EVENT_FIELDS, STREAMK, False, False,
    num_warps=8, matrix_instr_nonkdim=16, kpack=1, num_stages=2,
)
torch.cuda.synchronize()

err = (C.float() - ref.float()).abs().max().item()
nwrong = ((C.float() - ref.float()).abs() > 1.0).sum().item()
print(f"C: max_err={err:.3f}  wrong_elems={nwrong}/{m*n}  "
      f"allclose={torch.allclose(C.float(), ref.float(), atol=1.0, rtol=0.05)}")
