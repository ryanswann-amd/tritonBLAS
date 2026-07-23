"""Probe large-shape (register-path) steal-k configs for correctness + perf.

Isolates why 256x256 tiles go ok=False while 128x128 is correct: sweeps
tile shape x num_stages x num_warps for a fixed large square GEMM.
"""
import sys, torch, triton
from steal_k import steal_k_adaptive

DEV = "cuda"
EVENT_FIELDS = 8
CU = torch.cuda.get_device_properties(0).multi_processor_count


def tflops(m, n, k, ms):
    return 2 * m * n * k / (ms * 1e-3) / 1e12


def okc(C, ref):
    return torch.allclose(C.to(torch.float32), ref.to(torch.float32), atol=1.0, rtol=0.05)


def bench(fn, reset, n_warmup=15, n_repeat=40):
    # L2-flushed, CUDA-event timed, RESET every iteration (the work counter must
    # be re-zeroed or repeated launches do no work -> bogus fast times).
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


def run(m, n, k, bm, bn, bk, nstages, nwarps):
    A = torch.randn(m, k, device=DEV, dtype=torch.float16)
    B = torch.randn(k, n, device=DEV, dtype=torch.float16)
    ref = torch.matmul(A, B)
    nt = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    ipt = triton.cdiv(k, bk)
    STREAMK = (nt < CU) and (nt * ipt >= CU)
    grid = CU
    Ck = torch.zeros(m, n, device=DEV, dtype=torch.float16)
    P = torch.zeros(CU, bm * bn, device=DEV, dtype=torch.float32) if STREAMK else torch.zeros(1, device=DEV, dtype=torch.float32)
    locks = torch.zeros(CU, device=DEV, dtype=torch.int32) if STREAMK else torch.zeros(1, device=DEV, dtype=torch.int32)
    wc = torch.zeros(1, device=DEV, dtype=torch.int32)
    ev = torch.zeros(1, 1, EVENT_FIELDS, device=DEV, dtype=torch.int64)

    def f():
        steal_k_adaptive[(grid,)](
            A, B, Ck, P, locks, wc, ev, m, n, k,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), Ck.stride(0), Ck.stride(1),
            bm, bn, bk, nt, ipt, CU, 1, EVENT_FIELDS, STREAMK, False, False,
            num_warps=nwarps, matrix_instr_nonkdim=16, kpack=1, num_stages=nstages,
        )

    reset = (lambda: (wc.zero_(), locks.zero_())) if STREAMK else (lambda: wc.zero_())
    try:
        reset(); f(); torch.cuda.synchronize()
    except Exception as e:
        print(f"  tile {bm}x{bn}x{bk} ns={nstages} nw={nwarps}: LAUNCH FAIL {type(e).__name__}", flush=True)
        return
    good = okc(Ck, ref)
    ms = bench(f, reset)
    tag = "streamk" if STREAMK else "register"
    print(f"  tile {bm}x{bn}x{bk} ns={nstages} nw={nwarps}: {ms:7.3f} ms  {tflops(m,n,k,ms):6.1f} TF/s  ok={good}  [{tag}]", flush=True)


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
    print(f"arch={torch.cuda.get_device_properties(0).gcnArchName} CUs={CU}  shape {m}x{n}x{k}\n", flush=True)
    configs = [
        (128, 128, 64, 2, 8),   # current shipping baseline (correct)
        (256, 256, 64, 1, 8),   # big tile, ns=1 -> LDS 64KB, fits
        (256, 256, 64, 1, 4),
        (256, 256, 32, 1, 8),
        (256, 256, 32, 2, 4),   # ns=2 but half warps
        (256, 128, 64, 1, 8),
        (128, 256, 64, 1, 8),
        (256, 256, 64, 2, 8),   # expected LDS-overflow / wrong (for contrast)
    ]
    for (bm, bn, bk, ns, nw) in configs:
        run(m, n, k, bm, bn, bk, ns, nw)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
