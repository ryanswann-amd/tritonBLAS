"""Minimal trace-free work-stealing GEMM to isolate the register-path 256x256 bug.

If this is correct at 256x256 but steal_k_adaptive's register path is not, the
bug is the tracing/timer overhead (register pressure); if this also fails, it's
the work-stealing structure or the store.
"""
import sys, torch, triton
import triton.language as tl

DEV = "cuda"
CU = torch.cuda.get_device_properties(0).multi_processor_count


@triton.jit
def ws_min(A, B, C, work_ctr, M, N, K,
           stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
           NUM_TILES: tl.constexpr, ITERS_PER_TILE: tl.constexpr, EVEN_K: tl.constexpr = True):
    num_pid_n = tl.cdiv(N, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    wid = tl.atomic_add(work_ctr, 1, scope="gpu")
    while wid < NUM_TILES:
        tile = wid
        pid_m = tile // num_pid_n
        pid_n = tile % num_pid_n
        rm_raw = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn_raw = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = rm_raw < M
        mask_n = rn_raw < N
        rm = rm_raw % M
        rn = rn_raw % N
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for it in range(0, ITERS_PER_TILE):
            if EVEN_K:
                a = tl.load(A_BASE)
                b = tl.load(B_BASE)
            else:
                kmask = (it * BLOCK_K + rk) < K
                a = tl.load(A_BASE, mask=mask_m[:, None] & kmask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=kmask[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a, b, allow_tf32=False)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk
        mask = mask_m[:, None] & mask_n[None, :]
        C_ = C + rm_raw[:, None] * stride_cm + rn_raw[None, :] * stride_cn
        tl.store(C_, acc.to(C.type.element_ty), mask=mask)
        wid = tl.atomic_add(work_ctr, 1, scope="gpu")


def run(m, n, k, bm, bn, bk, ns, nw, grid):
    torch.manual_seed(0)
    A = torch.randn(m, k, device=DEV, dtype=torch.float16)
    B = torch.randn(k, n, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B).float()
    nt = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    ipt = triton.cdiv(k, bk)
    C = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    wc = torch.zeros(1, device=DEV, dtype=torch.int32)
    try:
        ws_min[(grid,)](A, B, C, wc, m, n, k,
                        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                        bm, bn, bk, nt, ipt,
                        num_warps=nw, num_stages=ns, matrix_instr_nonkdim=16, kpack=1)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  MIN tile {bm}x{bn}x{bk} ns={ns} nw={nw} grid={grid}: LAUNCH FAIL {e}", flush=True)
        return
    Cf = C.float()
    err = (Cf - ref).abs()
    print(f"  MIN tile {bm}x{bn}x{bk} ns={ns} nw={nw} grid={grid} nt={nt}: "
          f"allclose={torch.allclose(Cf, ref, atol=1.0, rtol=0.05)} max_err={err.max().item():.3f} "
          f"frac0={(Cf==0).float().mean().item():.3f}", flush=True)


def main():
    m = n = k = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    print(f"CUs={CU} shape {m}x{n}x{k}", flush=True)
    for grid in (int(sys.argv[2]) if len(sys.argv) > 2 else 304,):
        run(m, n, k, 128, 128, 64, 2, 8, grid)
        run(m, n, k, 256, 256, 64, 2, 8, grid)   # library-exact config
        run(m, n, k, 256, 256, 64, 1, 8, grid)
    print("MIN_DONE", flush=True)


if __name__ == "__main__":
    main()
