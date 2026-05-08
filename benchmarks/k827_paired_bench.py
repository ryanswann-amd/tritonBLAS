# SPDX-License-Identifier: MIT
# K-827 paired ON/OFF benchmark for the M=N=2048 large-K cohort + 12-cell guard.
#
# ON arm:  fix/K-827 includes split-K=2 routed override → fires on 6 cohort cells,
#          falls through on guard cells.
# OFF arm: simulated by monkey-patching maybe_dispatch_splitk2 to a no-op so
#          the same process measures both arms with shared allocator state.

import argparse
import csv
import math
import statistics
import sys

import torch


def parse_shapes(name):
    in_cohort = []
    for K in (4096, 8192, 16384):
        for dt in (torch.float16, torch.bfloat16):
            in_cohort.append((2048, 2048, K, dt))

    guard = []
    for M in (4096, 1024):
        for K in (4096, 8192, 16384):
            for dt in (torch.float16, torch.bfloat16):
                guard.append((M, M, K, dt))

    if name == "in_cohort":
        return in_cohort
    if name == "guard":
        return guard
    if name == "all":
        return in_cohort + guard
    raise ValueError(name)


def make_inputs(M, N, K, dtype, device="cuda"):
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    out = torch.empty(M, N, device=device, dtype=dtype)
    return a, b, out


def time_graph(fn, n_capture=10, n_warm=5, n_rounds=50):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_capture):
            fn()
    for _ in range(n_warm):
        g.replay()
    torch.cuda.synchronize()
    rounds = []
    for _ in range(n_rounds):
        s_evt = torch.cuda.Event(enable_timing=True)
        e_evt = torch.cuda.Event(enable_timing=True)
        s_evt.record()
        g.replay()
        e_evt.record()
        e_evt.synchronize()
        rounds.append(s_evt.elapsed_time(e_evt) / n_capture)
    return rounds


def correctness(M, N, K, dt, atol):
    """Compare ON-arm output to torch.matmul reference."""
    import tritonblas
    a = torch.randn(M, K, device="cuda", dtype=dt)
    b = torch.randn(K, N, device="cuda", dtype=dt)
    out = torch.empty(M, N, device="cuda", dtype=dt)
    tritonblas.matmul(a, b, out=out)
    ref = torch.matmul(a, b)
    diff = (out.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-6)
    return diff.mean().item(), diff.max().item(), rel.mean().item()


_ORIG_DISPATCH = None


def run_arm(arm, shapes, n_capture, n_warm, n_rounds):
    """Return list of (M,N,K,dtype_str,backend_str,median_ms)."""
    import importlib
    import tritonblas
    # `tritonblas.matmul` is the public function; the source module is .matmul (file).
    _tbm = importlib.import_module("tritonblas.matmul")

    global _ORIG_DISPATCH
    if _ORIG_DISPATCH is None:
        _ORIG_DISPATCH = _tbm.maybe_dispatch_splitk2

    if arm == "off":
        _tbm.maybe_dispatch_splitk2 = lambda a, b, out: False
    elif arm == "on":
        _tbm.maybe_dispatch_splitk2 = _ORIG_DISPATCH

    if arm == "off":
        assert _tbm.maybe_dispatch_splitk2.__name__ == "<lambda>", "OFF patch failed"
    else:
        assert _tbm.maybe_dispatch_splitk2 is _ORIG_DISPATCH, "ON restore failed"

    out_rows = []
    for (M, N, K, dt) in shapes:
        a, b, out = make_inputs(M, N, K, dt)
        def fn():
            tritonblas.matmul(a, b, out=out)
        rounds = time_graph(fn, n_capture, n_warm, n_rounds)
        med = statistics.median(rounds)
        # Also measure hipBLASLt for reference.
        a2, b2, out2 = make_inputs(M, N, K, dt)
        def fn_hbl():
            torch.matmul(a2, b2, out=out2)
        hbl_rounds = time_graph(fn_hbl, n_capture, n_warm, n_rounds)
        hbl_med = statistics.median(hbl_rounds)
        in_cohort = (M == 2048 and N == 2048 and
                     K in (4096, 8192, 16384) and
                     dt in (torch.float16, torch.bfloat16))
        out_rows.append((M, N, K, str(dt).split(".")[-1],
                         arm, med, hbl_med, int(in_cohort)))
        print(f"  {arm:3s} {M:5d}×{N:5d}×{K:5d} {str(dt).split('.')[-1]:8s}  "
              f"tb={med:.4f}ms  hbl={hbl_med:.4f}ms  hbl/tb={hbl_med/med:.3f}  "
              f"in_cohort={int(in_cohort)}", flush=True)
    return out_rows


def geomean(xs):
    xs = [x for x in xs if x > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default="all", choices=["in_cohort", "guard", "all"])
    ap.add_argument("--n-capture", type=int, default=10)
    ap.add_argument("--n-warm", type=int, default=5)
    ap.add_argument("--n-rounds", type=int, default=50)
    ap.add_argument("--check", action="store_true",
                    help="Also verify ON-arm correctness vs torch.matmul on cohort.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("cuda unavailable", file=sys.stderr); sys.exit(2)

    shapes = parse_shapes(args.shapes)

    if args.check:
        print("=== correctness check (ON arm vs torch.matmul) ===", flush=True)
        from tritonblas.splitk2_kernel import lookup_splitk2_override
        for K in (4096, 8192, 16384):
            for dt in (torch.float16, torch.bfloat16):
                M = N = 2048
                spec = lookup_splitk2_override(M, N, K, dt, dt, dt)
                assert spec is not None, f"override missing for K={K} {dt}"
                mean_abs, max_abs, mean_rel = correctness(M, N, K, dt, atol=None)
                print(f"  2048²×{K:5d} {str(dt).split('.')[-1]:8s}  "
                      f"mean_abs={mean_abs:.3e} max_abs={max_abs:.3e} "
                      f"mean_rel={mean_rel:.3e}", flush=True)

    # OFF arm first then ON, paired in a single process for shared allocator state.
    print("=== OFF arm (composite baseline, override patched out) ===", flush=True)
    off_rows = run_arm("off", shapes, args.n_capture, args.n_warm, args.n_rounds)
    print("=== ON arm (split-K=2 routed override active) ===", flush=True)
    on_rows = run_arm("on", shapes, args.n_capture, args.n_warm, args.n_rounds)

    # Combine
    by_key = {}
    for r in off_rows + on_rows:
        key = (r[0], r[1], r[2], r[3])
        by_key.setdefault(key, {})[r[4]] = (r[5], r[6], r[7])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "N", "K", "dtype",
                    "off_ms", "on_ms", "hbl_ms_off", "hbl_ms_on",
                    "in_cohort", "on_over_off", "on_over_hbl", "off_over_hbl"])
        cohort_ratios = []
        guard_ratios = []
        for key, arms in by_key.items():
            off_ms, hbl_off, ic = arms["off"]
            on_ms, hbl_on, _   = arms["on"]
            on_off = on_ms / off_ms if off_ms > 0 else float("nan")
            on_hbl = on_ms / hbl_on if hbl_on > 0 else float("nan")
            off_hbl = off_ms / hbl_off if hbl_off > 0 else float("nan")
            w.writerow([*key, f"{off_ms:.6f}", f"{on_ms:.6f}",
                        f"{hbl_off:.6f}", f"{hbl_on:.6f}",
                        ic, f"{on_off:.4f}", f"{on_hbl:.4f}", f"{off_hbl:.4f}"])
            (cohort_ratios if ic else guard_ratios).append(on_off)
        print(flush=True)
        print(f"=== cohort (n={len(cohort_ratios)}) on/off geomean: "
              f"{geomean(cohort_ratios):.4f}  (>1.0 = ON slower)", flush=True)
        print(f"=== guard  (n={len(guard_ratios)}) on/off geomean: "
              f"{geomean(guard_ratios):.4f}  (~1.0 expected: gate must not fire)", flush=True)


if __name__ == "__main__":
    main()
