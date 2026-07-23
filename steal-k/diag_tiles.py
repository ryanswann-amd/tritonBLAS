"""Reproduce sk_tree failures on arbitrary tiles -> classify launch-fail vs wrong."""
import torch, triton
from steal_k import sk_tree
DEV = "cuda"; CU = torch.cuda.get_device_properties(0).multi_processor_count
P = torch.zeros(CU, 256 * 256, device=DEV, dtype=torch.float32)
LOCKS = torch.zeros(CU, device=DEV, dtype=torch.int32)


def nwarps(bm, bn):
    return 8 if bm * bn >= 128 * 128 else (4 if bm * bn >= 64 * 64 else 1)


def test(m, n, k, bm, bn, bk):
    torch.manual_seed(0)
    A = torch.randn(m, k, device=DEV, dtype=torch.float16)
    B = torch.randn(k, n, device=DEV, dtype=torch.float16)
    C = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B).float()
    nt = triton.cdiv(m, bm) * triton.cdiv(n, bn); ipt = triton.cdiv(k, bk)
    even = (k % bk == 0)
    try:
        LOCKS.zero_()
        sk_tree[(CU,)](A, B, C, P, LOCKS, m, n, k,
                       A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                       nt, ipt, CU, bm, bn, bk, 12, even,
                       num_warps=nwarps(bm, bn), matrix_instr_nonkdim=16, kpack=1, num_stages=2)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  {m}x{n}x{k} tile {bm}x{bn}x{bk} evenK={even}: LAUNCH_FAIL {type(e).__name__}: {str(e)[:90]}", flush=True)
        return
    err = (C.float() - ref).abs().max().item()
    tag = "OK" if err < 1.0 else "WRONG"
    print(f"  {m}x{n}x{k} tile {bm}x{bn}x{bk} evenK={even}: {tag} max_err={err:.2f} (nt={nt} ipt={ipt})", flush=True)


def repeat(m, n, k, bm, bn, bk, iters=40):
    torch.manual_seed(0)
    A = torch.randn(m, k, device=DEV, dtype=torch.float16)
    B = torch.randn(k, n, device=DEV, dtype=torch.float16)
    C = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B).float()
    nt = triton.cdiv(m, bm) * triton.cdiv(n, bn); ipt = triton.cdiv(k, bk)
    bad = 0; errs = []
    for _ in range(iters):
        C.zero_(); LOCKS.zero_()
        sk_tree[(CU,)](A, B, C, P, LOCKS, m, n, k,
                       A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                       nt, ipt, CU, bm, bn, bk, 12, True,
                       num_warps=nwarps(bm, bn), matrix_instr_nonkdim=16, kpack=1, num_stages=2)
        torch.cuda.synchronize()
        e = (C.float() - ref).abs().max().item()
        errs.append(e)
        if e > 1.0: bad += 1
    print(f"  REPEAT {m}x{n}x{k} {bm}x{bn}x{bk}: {bad}/{iters} bad, err range [{min(errs):.2f},{max(errs):.2f}]", flush=True)


for c in [(2048, 2048, 2048, 128, 128, 64), (2048, 2048, 2048, 128, 128, 128),
          (2048, 2048, 8192, 128, 128, 128), (2048, 2048, 2048, 256, 256, 64),
          (2048, 2048, 2048, 128, 256, 64), (4096, 4096, 4096, 256, 256, 64),
          (3072, 3072, 3072, 128, 128, 128)]:
    repeat(*c, iters=100)
print("DIAG_TILES_DONE", flush=True)
