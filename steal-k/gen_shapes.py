#!/usr/bin/env python3
"""Generate ~10000 big-K, small-M and/or small-N GEMM shapes and shard them.

Regime where steal-k's 64x64 full-split stream-K should beat rocBLAS/PyTorch:
at least one of M,N is small; K is large. K constrained to multiples of 64 so
the EVEN_K unmasked path is exact. Writes shards to data/shapes/shard_XX.txt.
"""
import os, random

random.seed(0)
SMALL = [16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512]
BIG = SMALL + [640, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
KS = [1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 10240,
      12288, 14336, 16384, 20480, 24576, 28672, 32768, 40960, 49152, 57344, 65536]  # all %64==0
NUM = 10000
NSHARDS = 12


def main():
    # full valid set: at least one of M,N small, K large
    full = set()
    for k in KS:
        for m in SMALL:
            for n in BIG:
                full.add((m, n, k))
        for m in BIG:
            for n in SMALL:
                full.add((m, n, k))
    full = list(full)
    random.shuffle(full)
    shapes = full[:NUM] if len(full) >= NUM else full
    shapes = sorted(shapes, key=lambda s: (s[0] * s[1] * s[2]))  # cheap->costly
    os.makedirs("data/shapes", exist_ok=True)
    # round-robin into shards so each shard has a similar cost mix
    shards = [[] for _ in range(NSHARDS)]
    for i, s in enumerate(shapes):
        shards[i % NSHARDS].append(s)
    for j, sh in enumerate(shards):
        with open(f"data/shapes/shard_{j:02d}.txt", "w") as f:
            for (m, n, k) in sh:
                f.write(f"{m} {n} {k}\n")
    print(f"wrote {len(shapes)} shapes into {NSHARDS} shards "
          f"(~{len(shapes)//NSHARDS}/shard) under data/shapes/")


if __name__ == "__main__":
    main()
