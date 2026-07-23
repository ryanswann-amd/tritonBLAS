#!/usr/bin/env python3
"""Broad correctness validation for grouped-steal-k (register + split-K paths).

Stresses highly variant group sizes and every masking edge case:
  - empty experts (M_g = 0)                 -> prefix-map must skip them
  - single-row experts (M_g = 1)            -> rm % Mg wrapping
  - odd M_g not aligned to BLOCK_M (7,63,65,129)
  - N not a multiple of BLOCK_N             -> rn % N + store mask
  - K not a multiple of BLOCK_K             -> EVEN_K=False masked K path
  - huge + tiny mixed in one batch (1 .. 8192)
  - G from 1 to 128
Both paths checked vs a torch per-group loop, across several tiles.
"""
import random, torch, triton
from grouped_steal_k import make_problem, ref_loop, grouped_matmul
from grouped_split_k import grouped_matmul_sk

DEV = "cuda"
TILES = [(64, 64, 64), (128, 128, 64), (128, 256, 64)]


def check(Ms, N, K, seed, label):
    # skip degenerate all-empty
    if sum(Ms) == 0:
        return []
    A, B, off = make_problem(Ms, N, K, seed=seed)
    ref = ref_loop(A, B, off, N).float()
    out = []
    for (bm, bn, bk) in TILES:
        for path, fn in [("reg", grouped_matmul), ("splitk", grouped_matmul_sk)]:
            try:
                res = fn(A, B, off, Ms, N, K, bm, bn, bk)
                C = res[0]
                torch.cuda.synchronize()
                err = (C.float() - ref).abs().max().item()
                ok = err < 1.0
            except Exception as e:
                err, ok = float("nan"), False
                out.append((label, f"{bm}x{bn}x{bk}", path, "EXC:" + type(e).__name__, False))
                continue
            out.append((label, f"{bm}x{bn}x{bk}", path, f"{err:.3f}", ok))
    return out


def gen_configs():
    random.seed(0)
    configs = []
    # ---- explicit edge cases ----
    configs += [
        ("single group G1", [4096], 4096, 4096),
        ("single tiny row", [1], 512, 2048),
        ("empty experts mixed", [0, 512, 0, 1024, 0, 256], 512, 4096),
        ("all-but-one empty", [0, 0, 2048, 0], 512, 4096),
        ("odd M_g", [7, 63, 65, 129, 200, 1000], 512, 4096),
        ("tiny experts", [1, 2, 3, 5, 8, 13, 16, 32], 256, 2048),
        ("huge+tiny mix", [8192, 1, 4096, 2, 1, 6144, 3], 512, 4096),
        ("N not mult of BN", [512, 300, 700], 300, 4096),      # N=300
        ("N=500", [256, 512, 1024], 500, 2048),
        ("K not mult of BK", [512, 256, 128], 512, 5000),      # K=5000 -> EVEN_K=False
        ("K=4000", [1024, 64, 256], 384, 4000),
        ("K and N both odd", [333, 77, 900], 260, 4100),
        ("G=1 skinny bigK", [64], 512, 32768),
        ("G=128 tiny", [random.randint(1, 64) for _ in range(128)], 256, 2048),
        ("G=64 powerlaw", [max(1, int(4096 / (i + 1))) for i in range(64)], 512, 4096),
        ("extreme variance", [1, 8192, 2, 4096, 1, 1, 7000, 3, 16], 512, 4096),
    ]
    # ---- random variant batches ----
    for i in range(40):
        G = random.choice([2, 3, 5, 8, 16, 32])
        style = random.choice(["uniform", "skewed", "withzeros", "odd", "powerlaw"])
        if style == "uniform":
            Ms = [random.randint(1, 4096) for _ in range(G)]
        elif style == "skewed":
            Ms = [random.randint(1, 64) for _ in range(G)]; Ms[random.randrange(G)] = random.randint(4096, 8192)
        elif style == "withzeros":
            Ms = [random.choice([0, 0, random.randint(1, 2048)]) for _ in range(G)]
        elif style == "odd":
            Ms = [random.choice([1, 7, 33, 63, 65, 127, 129, 257, 500, 999]) for _ in range(G)]
        else:
            Ms = [max(1, int(6000 / (j + 1))) for j in range(G)]
        N = random.choice([256, 384, 500, 512, 768, 1024])
        K = random.choice([2048, 4000, 4096, 5000, 8192, 16384])
        configs.append((f"rand{i} {style} G{G} N{N} K{K}", Ms, N, K))
    return configs


def main():
    configs = gen_configs()
    total, passed, fails = 0, 0, []
    for idx, (label, Ms, N, K) in enumerate(configs):
        rows = check(Ms, N, K, seed=idx + 1, label=label)
        for (lab, tile, path, err, ok) in rows:
            total += 1
            if ok:
                passed += 1
            else:
                fails.append((lab, tile, path, err))
    print(f"\n=== VALIDATION: {passed}/{total} checks passed ({len(fails)} FAIL) ===", flush=True)
    for (lab, tile, path, err) in fails[:60]:
        print(f"  FAIL {lab} | {tile} {path} | err={err}", flush=True)
    print("VALIDATE_DONE", flush=True)


if __name__ == "__main__":
    main()
