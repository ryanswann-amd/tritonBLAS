# Experiment: grouped-steal-k — work-stealing grouped (MoE) GEMM

**Date:** 2026-07-23
**Runner:** ryaswann (agent-assisted)
**Question:** Does steal-k's work-stealing pattern extend to **grouped GEMM**
(many independent per-expert GEMMs with varying row counts — the MoE FFN case),
and does it beat the naive per-group loop?

## Idea

Many groups/experts share N and K but have varying M_g. Flatten **all output
tiles across all groups** into one work list; each persistent workgroup
atomically steals the next global tile index, maps it back to `(group, local
tile)` via a prefix-sum table, and computes it. Imbalanced groups (a hot expert,
many tiny experts) are load-balanced automatically — no static assignment, one
launch instead of G launches.

Reuses steal-k's correctness tricks: **unmasked EVEN_K loads with `rm % Mg` /
`rn % N` address wrapping + a masked store**, so the software pipeliner
(`num_stages=2`) works. Each tile is fully owned → register accumulate, no global
reduction (a per-group Stream-K + tree split is the natural next extension for
underfilled groups).

- Kernel + harness: `grouped_steal_k.py` (`grouped_steal_k` jit fn,
  `grouped_matmul` launcher).
- Layout: `A[total_M, K]`, `B[G, K, N]`, `C[total_M, N]`, `group_off[G+1]`
  (row offsets), `tile_prefix[G+1]` (prefix-sum of per-group tile counts).

## Environment

- MI300X (gfx942), fp16, container `lmsysorg/sglang:v0.5.9-rocm720-mi30x`,
  Triton 3.6. Repo commit: see `git log`. Only needs Triton + torch (no origami).
- Baseline: **torch per-group loop** (`for g: C[g] = A[g] @ B[g]`) — the natural
  baseline since neither PyTorch nor tritonBLAS has grouped GEMM. (A stronger
  baseline would be a Triton grouped-gemm reference or hipBLASLt grouped; the
  loop is launch-overhead-bound for many-small-experts, which is realistic.)

## Results (best tile 128x256x64; all correct, allclose atol=1.0 rtol=0.05)

| scenario | grouped-steal-k TF/s | torch loop TF/s | speedup |
|---|---|---|---|
| balanced 8x2048 (N=K=4096) | 417.0 | 319.0 | 1.31x |
| imbalanced 8 | 353.4 | 275.9 | 1.28x |
| skewed 16 (one hot expert) | 321.2 | 180.1 | **1.78x** |
| many small 64 experts (N=2048) | 234.9 | 51.0 | **4.61x** |
| MoE 32 experts x~1k (N=5120) | 383.2 | 276.7 | 1.38x |

- **Biggest win: many small experts (4.6x)** — the torch loop is launch-overhead
  bound (64 tiny GEMMs at ~51 TF/s); the single work-stealing launch stays busy.
- **Load imbalance (skewed / one-hot expert): 1.8x** — work-stealing hides the
  imbalance that would stall a static split.
- Tile matters (same lesson as single-GEMM steal-k): **128x256x64 best**,
  128x128x64 decent, 64x64x64 poor for these moderate shapes (too many tiles).
- Correctness: `rm % Mg` wrapping handles ragged group boundaries; empty experts
  (M_g=0) are skipped by the prefix-sum map (no tile ever lands on them).

Raw: `data/raw/grouped_results.txt` (all tiles x scenarios).

## Reproduce

```
sbatch sbatch_grouped.sh    # writes grouped.log
# or: python3 grouped_steal_k.py
```

## Next steps

- Per-group Stream-K + tree reduction for **underfilled** groups (skinny experts,
  big K) — steal-k's skinny-regime win.
- Per-XCD L2-local work counters (as in tritonBLAS's single-GEMM work-stealing
  kernel) for locality.
- Compare against a real grouped baseline (hipBLASLt grouped / Triton group-gemm).
- Fuse the routing/gather (token permutation) for a true MoE path.

## Per-group split-K (grouped_split_k.py) — 2026-07-23

Extends grouped-steal-k to the **underfilled** regime (skinny experts, big K):
the flat `(group, tile, k-iter)` space is split across grid=CU persistent WGs
(Stream-K), the group is resolved per chunk from the prefix-sum, and each global
tile is reduced with the **hardened tree** (system-scope release/acquire). Fills
all CUs even when total tiles << CU. Kernel `grouped_sk_tree`.

Results (MI300X, 64x64x64, all ok=True):

| scenario | split-K TF/s | vs register (no split) | vs torch loop |
|---|---|---|---|
| 4 experts M=64 K=32768 | 36.4 | **2.68x** | 1.50x |
| 8 experts M=128 N=256 K=16384 | 37.4 | 1.68x | 2.56x |
| 8 skinny experts M=64 K=16384 | 35.0 | 1.67x | 2.53x |
| 16 experts M=32 K=16384 | 16.9 | 1.55x | 4.44x |
| 8 experts M=256 (256 tiles, near-filled) | 63.1 | 1.00x | 2.27x |

- Split-K wins **1.5-2.7x over the register (no-split) grouped path** when experts
  are underfilled; the gain grows with K / fewer tiles (register leaves CUs idle).
- **Neutral (1.0x) when already filled** (M=256) -> the adaptive rule is: use
  split-K when total tiles < CU, register otherwise (mirrors single-GEMM steal-k).
- Correct across ragged groups; the tree + group-resolution compose cleanly.

Raw: `data/raw/grouped_splitk_results.txt`.

## Broad correctness validation (validate_grouped.py) — 2026-07-23

Stress matrix over both paths (register + split-K) x 3 tiles (64x64, 128x128,
128x256) x 56 configs = **336 checks, all PASS (0 fail)** vs torch per-group loop.
Covers highly variant group sizes and every masking edge case:
- empty experts (M_g=0, incl. all-but-one-empty) — skipped by the prefix map
- single-row / tiny experts (1,2,3,7,13...) — `rm % Mg` wrapping
- odd M_g not aligned to BLOCK_M (63,65,129,257,500,999)
- N not a multiple of BLOCK_N (300,500) — `rn % N` + store mask
- K not a multiple of BLOCK_K (4000,5000,4100) — EVEN_K=False masked-K path
- huge+tiny mixed in one batch (1..8192), power-law / skewed distributions
- G from 1 to 128, plus 40 randomized variant batches

Confirms the ragged-group wrapping/masking and the tree reduction compose
correctly across arbitrary group-size distributions.

## Perf vs torch loop on the validation configs (perf_grouped.py) — 2026-07-23

Best-of-steal-k (adaptive over {register 128x256/128x128, split-K 64x64/128x128})
vs torch per-group loop, across all 56 broad/variant-size validation configs:

**steal-k faster on 49/56 (88%); speedup geomean 1.85x, median 1.69x, max 18.07x,
min 0.53x.**

- Biggest wins scale with group count / imbalance (launch-overhead-bound loop):
  G=128 tiny 18.1x, G=64 power-law 7.6x, 32-group odd/skewed 3.5-5.5x,
  huge+tiny mix 1.95x, empty-experts 1.69x.
- The 7 losses (0.53-0.98x) are all tiny-G / tiny-total (single group 0.70x,
  single row 0.54x, G=1 skinny 0.53x) -> effectively single-GEMM, where rocBLAS
  (the loop's per-call kernel) wins, consistent with the single-GEMM findings.

Takeaway: grouped steal-k's advantage grows with #groups and size variance (the
MoE regime); it only fades toward the single-GEMM limit. Raw:
`data/raw/grouped_perf_vs_loop.txt`.

## vs aiter batched GEMM (option 1: equal-M / batched, bf16) — 2026-07-23

aiter is installed (CK/asm MoE + GEMM lib). It has no bare variable-M grouped
GEMM (that lives in its fused-MoE stage kernels); its directly-comparable op is
`batched_gemm_bf16_CK` (uniform M). On equal-M groups (bf16, MI300X), all correct:

| shape | aiter TF/s | steal-k best TF/s | torch.bmm | steal-k vs aiter |
|---|---|---|---|---|
| G8 M2048 N4096 K4096 | 413 | 430 | 627 | 1.04x |
| G8 M4096 | 421 | 448 | 652 | 1.07x |
| G16 M512 | 405 | 314 | 506 | 0.77x |
| G32 M256 N2048 | 319 | 234 | 466 | 0.73x |
| G128 M128 N2048 | 288 | 213 | 350 | 0.74x |
| G8 M64 N512 K16384 (skinny) | 28.7 | 36.4 | 50.2 | 1.27x |
| G16 M128 N1024 K8192 | 131 | 105 | 226 | 0.80x |

- **steal-k is on par with aiter's batched GEMM (0.73-1.27x)** — wins large-M and
  skinny/big-K (split-K), loses small-M/many-batch.
- **Both trail torch.bmm (rocBLAS batched, ~1.3-1.5x)** — rocBLAS is the champion
  for uniform GEMM, as with single-GEMM.
- **Caveat: aiter ran UNTUNED** ("not found tuned config in CKGEMM, will use
  default") -- `batched_gemm_bf16_tune` would raise its numbers.
- **Scope: uniform-M only.** aiter's batched kernel can't express variable-M
  grouped GEMM (steal-k's actual strength); that head-to-head needs aiter's MoE
  stage kernels (`ck_moe_stage1` + `moe_sorting`) = option 2.

Raw: `data/raw/aiter_vs_grouped.txt`.
