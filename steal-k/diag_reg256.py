"""Localize the register-path 256x256 correctness bug.

Forces the work-stealing (register) path (STREAMK=False) for a small shape and
reports WHERE C mismatches vs torch: per-(pid_m,pid_n) 256-tile error, so we can
tell 'all tiles slightly off' (dot/layout) from 'some tiles zero/garbage'
(tile-mapping / not-computed).
"""
import sys, torch, triton
from steal_k import steal_k_adaptive, EVENT_FIELDS

DEV = "cuda"
CU = torch.cuda.get_device_properties(0).multi_processor_count


def run(m, n, k, bm, bn, bk, nstages, nwarps, grid):
    torch.manual_seed(0)
    A = torch.randn(m, k, device=DEV, dtype=torch.float16)
    B = torch.randn(k, n, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B).float()
    nt = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    ipt = triton.cdiv(k, bk)
    STREAMK = False  # force register/work-stealing whole-tile path
    Ck = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    P = torch.zeros(1, device=DEV, dtype=torch.float32)
    locks = torch.zeros(1, device=DEV, dtype=torch.int32)
    wc = torch.zeros(1, device=DEV, dtype=torch.int32)
    ev = torch.zeros(1, 1, EVENT_FIELDS, device=DEV, dtype=torch.int64)
    try:
        steal_k_adaptive[(grid,)](
            A, B, Ck, P, locks, wc, ev, m, n, k,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), Ck.stride(0), Ck.stride(1),
            bm, bn, bk, nt, ipt, grid, 1, EVENT_FIELDS, STREAMK, False, False,
            num_warps=nwarps, matrix_instr_nonkdim=16, kpack=1, num_stages=nstages,
        )
        torch.cuda.synchronize()
    except Exception as e:
        print(f"tile {bm}x{bn}x{bk} ns={nstages} nw={nwarps} grid={grid}: LAUNCH FAIL {e}", flush=True)
        return
    C = Ck.float()
    err = (C - ref).abs()
    good = torch.allclose(C, ref, atol=1.0, rtol=0.05)
    print(f"\ntile {bm}x{bn}x{bk} ns={nstages} nw={nwarps} grid={grid} nt={nt}: "
          f"allclose={good} max_err={err.max().item():.3f} n_bad={int((err>1.0).sum())}/{m*n}", flush=True)
    # per 256-tile error grid
    npm, npn = triton.cdiv(m, bm), triton.cdiv(n, bn)
    print("  per-tile max_err (rows=pid_m, cols=pid_n):", flush=True)
    for pm in range(npm):
        row = []
        for pn in range(npn):
            blk = err[pm*bm:(pm+1)*bm, pn*bn:(pn+1)*bn]
            row.append(f"{blk.max().item():7.2f}")
        print("   ", " ".join(row), flush=True)
    # is the whole C maybe shifted/zeroed?
    print(f"  C[0,0]={C[0,0].item():.2f} ref={ref[0,0].item():.2f} | "
          f"C mean={C.mean().item():.3f} ref mean={ref.mean().item():.3f} | "
          f"frac C==0: {(C==0).float().mean().item():.3f}", flush=True)


def main():
    # small shape, force register path with a modest grid so it's fast + inspectable
    m = n = k = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    grid = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    print(f"arch={torch.cuda.get_device_properties(0).gcnArchName} CUs={CU} shape {m}x{n}x{k} (register forced)", flush=True)
    run(m, n, k, 128, 128, 64, 2, 8, grid)   # known-good reference
    run(m, n, k, 256, 256, 64, 1, 8, grid)   # the suspect
    run(m, n, k, 256, 256, 32, 1, 8, grid)
    print("DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
