# Split-K Routing Analysis: M=N=2048 Large-K FP16/BF16 Cohort

This note documents why a user-space Triton split-K=2 override is **not** a
viable speedup for the M=N=2048 / K∈{4096, 8192, 16384} / {fp16, bf16}
cohort on MI300X (gfx942), even though the vendor BLAS library
(`hipBLASLt`) dispatches a Global-Split-U=2 (GSU=2) kernel for the
K=16384 cells in the same cohort. This analysis is recorded so future
contributors do not re-attempt the same routed-override pattern.

## Decision

**No routing override added for this cohort.**

A 6-entry shape-and-dtype-guarded routed override that fires
`matmul_split_k(split_k=2)` only on the residual cohort cells was
prototyped, paired-benchmarked against the existing
`tritonblas.matmul` baseline (`off`) and `hipBLASLt` (`hbl`) on
c42/MI300X with a hot HIP-graph harness (n=50 timed iterations,
n_warmup=10), and rejected.

Measured inside the project's reference `rocm/pytorch` container image
(see `Dockerfile`) on Ubuntu 24.04 / Python 3.12 / PyTorch 2.10.0
release / Triton 3.6.0:

| M    | N    | K     | dtype | off (ms, med) | on (ms, med) | on/off | router action  |
|------|------|-------|-------|--------------:|-------------:|-------:|----------------|
| 2048 | 2048 |  4096 | fp16  | 0.1050        | 0.1080       | 0.972  | fall-through   |
| 2048 | 2048 |  4096 | bf16  | 0.1075        | 0.1028       | 1.046  | fall-through   |
| 2048 | 2048 |  8192 | fp16  | 0.1920        | 0.1924       | 0.998  | fall-through   |
| 2048 | 2048 |  8192 | bf16  | 0.1803        | 0.1890       | 0.954  | fall-through   |
| 2048 | 2048 | 16384 | fp16  | 0.3567        | 0.5755       | **0.620** | **split-K=2 fires** |
| 2048 | 2048 | 16384 | bf16  | 0.3456        | 0.5631       | **0.614** | **split-K=2 fires** |

The two cells where the new code path actually executes regress by
38–39 % vs the existing baseline. The four "fall-through" cells show
deltas in the run-to-run noise band (±5 %) for identical kernels, as
expected.

## Mechanism

The `hipBLASLt` advantage on this cohort is structurally different
between the K=4096/8192 cells and the K=16384 cells, and neither
mechanism can be reproduced from a user-space Triton override.

### K = 4096 and K = 8192 (4 of 6 cells)

`hipBLASLt` dispatches `GSU=1` on these cells. `rocprofv2` atomic
counters return zero atomic events. The remaining off-vs-`hbl` gap
(roughly 0.87×–0.90×) is per-shard codegen efficiency:

* The Tensile-generated kernel uses an `MIWT4_7` macro pattern that
  emits ≈ 28 MFMA instructions per LDS load.
* The corresponding Triton kernel emits 1 MFMA per LDS load, paying a
  per-load LDS round-trip on every MFMA.
* Tensile additionally uses an `LDSB1` row-padded LDS swizzle that
  eliminates LDS bank conflicts on K-stride reads. The Triton lowering
  pays a 4–8 % bank-conflict tax on the same access pattern.

Splitting K does not address either constraint — it adds an extra
reduction step in a regime where `hipBLASLt` itself never split K.
Routing here is, by construction, regression. The router correctly
does **not** fire on these four cells.

### K = 16384 (2 of 6 cells)

`hipBLASLt` dispatches `GSU=2` with the GSUAMBSK algorithm (Atomic
Multi-Buffer Single Kernel; ≈ 4256 atomic events per matmul, ≈ 4
atomics per output tile). The Triton split-K=2 prototype loses on
these cells because two costs compound:

1. **Per-shard codegen.** The `split_k=1` baseline of the same Triton
   kernel already runs at ~0.93× of the existing `tritonblas.matmul`
   path on this shape. The `persistent_matmul` codegen used by the
   default path beats the split-K kernel's per-shard codegen *before
   atomics enter*. Splitting K halves the per-shard arithmetic
   intensity but does not change this gap.
2. **Reduction overhead.** `tl.atomic_add` on the FP32 partial-sum
   buffer adds ~25–35 % on top of (1) on this shape. Tensile's
   GSUAMBSK fuses the atomic accumulation into the tail of the MFMA
   loop and reduces inside a single launch; a user-space Triton
   split-K must launch a separate reduction (or contend on the FP32
   buffer across the two shards on every output tile).

Composing the two effects predicts ~0.93 × ~0.72 ≈ **0.67×**, within
±5 pp of the measured **0.62×** / **0.61×**.

The two binding constraints — per-shard MFMA density and
atomic-reduction fusion — live in the Triton-AMD compiler backend and
the Tensile kernel-codegen pipeline (which mints GSUAMBSK), not in
the user-facing autotune surface that `tritonblas.matmul` exposes.
A user-space override on top of `tritonblas.matmul` therefore cannot,
by construction, recover the gap.

## Recommended Next Steps (priority order)

1. **Triton-AMD backend codegen.** Emit chained-MFMA macros that
   match Tensile's `MIWT4_7` density (≥ 16 MFMAs per LDS load), and
   apply an `LDSB1`-equivalent row-padded LDS swizzle in the Triton
   lowering. This closes the K=4096 and K=8192 cells without a
   split-K kernel ever being involved. Owner: Triton-AMD backend.
2. **Persistent Stream-K with explicit work-decomposition tuning.**
   Stream-K decomposes the K-axis into a fixed pool of persistent
   CTAs that pull tiles dynamically; the reduction stays inside one
   CTA's accumulator until the output tile is complete, avoiding the
   GSU=2 atomic-reduction overhead. Worth a one-week prototype on the
   K=16384 cells, but only after item 1 lands — the per-shard codegen
   gap dominates regardless of how K is decomposed.
3. **Atomic-FP16 / BF16 reduction kernel.** Halves atomic-buffer
   bandwidth versus the FP32 partial-sum buffer and may halve the
   ~25–35 % reduction tax. Numerical correctness vs cuBLAS reference
   on K=16384 is the gate; estimated two-week investigation, low
   confidence.
4. **Last resort: route the K=16384 cells to `torch.matmul`** (which
   dispatches to `hipBLASLt` directly) via a narrow shape-guarded
   override. Defer until items 1–3 are exhausted.

## Reproduction

The paired benchmark used a hot HIP-graph harness (n=50 timed,
n_warmup=10) on MI300X (gfx942) inside the project's reference
`rocm/pytorch` container image, with `tritonBLAS` checked out at the
merge base of this branch (`95e2c47`). The decision is robust at any tail-vs-median
choice within the measured n=50 distribution: the K=16384 regressions
exceed the ±5 % run-to-run noise band by an order of magnitude.
