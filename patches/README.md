# `amdgpu-max-vmcnt-before-mfma` — upstream-fixable lever (REFERENCE ONLY, do not enable)

This directory contains two cross-repo patches that, *if applied to LLVM
and the Triton AMD backend*, would expose a per-function attribute
`amdgpu-max-vmcnt-before-mfma=<N>` allowing kernel authors to relax the
conservative `s_waitcnt vmcnt(0)` floor that LLVM's `SIInsertWaitcnts`
pass emits immediately before MFMA / MAI instructions to
`s_waitcnt vmcnt(N)`.

The patches are checked in as the documented **upstream-fixable** form
of this lever — the clean alternative to the post-compilation ISA
rewriter scaffold investigated and rejected in K-6138 / K-6244.

> **The matching tritonblas Python wrapper plumbing was deliberately
> NOT shipped.** Empirical results from the prior tickets in this line
> (K-6244, K-6635, K-6663) show no surviving vmcnt site in current
> Triton + current ROCm codegen where `N > 0` is both correct *and* faster
> on the K-5156 / K-4967 worst-gap shape cohort. Until a benchmark
> identifies such a site, exposing a configurable Python surface in
> tritonblas would be unverified scaffolding.

## Why this lever was investigated

On the K-5156 / K-4967 worst-gap shape cohort (BF16 GEMM, MI300X /
gfx942), tritonblas's persistent matmul issues each MFMA preceded by
`s_waitcnt vmcnt(0)` (drain ALL outstanding loads), while hipBLASLt
issues MFMA with `vmcnt(N)` where 1 ≤ N ≤ 14. K-5101 / K-5694 / K-5647
/ K-6555 quantified this as the dominant single lever in the +30%
TFLOPS gap.

## Why no Python wrapper is shipped — the empirical record

Every prior in-tree attempt to lower the floor was tried and refused:

| Ticket  | Lever                                                                | Outcome                                                                          |
|---------|----------------------------------------------------------------------|----------------------------------------------------------------------------------|
| K-6028  | `cl::opt -amdgpu-waitcnt-load-forcezero=N`                           | hard-coded 0/inf; AMDGCN byte-identical                                          |
| K-6138  | post-compilation ISA rewriter `vmcnt(0)` → `vmcnt(N)`                | 14/18 noop, strict variants fault, lenient produces NaN                          |
| K-6244  | full-suite rewriter on 27 shapes × 5 modes                           | 0/27 sites rewritten — current Triton+ROCm no longer emits the K-loop pattern    |
| K-6244† | the remaining 3 vmcnt(0) sites that do exist                         | all immediately precede `ds_write_b128` (VMEM→LDS handoff barriers, MUST stay 0) |
| K-6635  | analogous `lgkmcnt(N)` ISA rewriter on the 7 candidate sites         | all 7 are operand-true RAW edges (correctness collapse on N=1, NaN on N≥2)       |
| K-6663  | joint `(vmcnt(N), lgkmcnt(M))` 5×5 grid                              | joint best matches single-axis, all N≥1 cells produce NaN                        |

K-6244 explicitly recommends: **CLOSE the entire vmcnt(N) line of
investigation.** The patches in this directory are checked in as the
clean upstream-fixable expression of the lever for completeness —
**they are not wired into tritonblas's matmul API** until a follow-up
ticket identifies a vmcnt site in current codegen where `N > 0` is
provably both correct (no NaN) and faster on a real shape.

## Files

| File                                                                | Lands in        | Purpose                                                                          |
|---------------------------------------------------------------------|-----------------|----------------------------------------------------------------------------------|
| `0001-llvm-amdgpu-max-vmcnt-before-mfma-attr.patch`                 | `llvm-project`  | Adds `amdgpu-max-vmcnt-before-mfma` function attribute to `SIInsertWaitcnts.cpp` |
| `0002-triton-amd-backend-max-vmcnt-before-mfma-option.patch`        | `triton`        | Adds `HIPOptions.max_vmcnt_before_mfma` and stamps the LLVM attribute            |

`include/tritonblas/matmul.py` is intentionally left untouched — see
the boxed note above.

## How to apply (for follow-up investigation only)

```bash
# 1. LLVM (against the commit Triton's third_party/amd pins to)
cd ${LLVM_PROJECT_DIR}
git apply --3way --reject ${TRITONBLAS_DIR}/patches/0001-llvm-amdgpu-max-vmcnt-before-mfma-attr.patch
# (line numbers may need adjustment for newer LLVM; the surrounding context
#  in SIInsertWaitcnts.cpp is stable.)

# 2. Triton AMD backend
cd ${TRITON_DIR}
git apply --3way --reject ${TRITONBLAS_DIR}/patches/0002-triton-amd-backend-max-vmcnt-before-mfma-option.patch

# 3. Build LLVM, then Triton against it (see triton/python/setup.py
#    for the $LLVM_INCLUDE_DIRS / $LLVM_LIBRARY_DIR knobs).
```

After the patches are built in, the attribute can be stamped on a
per-kernel basis from the Triton backend by passing
`max_vmcnt_before_mfma=N` through the kernel launch. **Do not surface
this as a tritonblas-level kwarg** until a follow-up benchmark
satisfies the gating below.

## Gating: what a follow-up ticket must demonstrate before any wrapper lands

A future ticket may revive a tritonblas Python wrapper iff *all* of
the following hold on at least one shape from the K-5156 / K-4967
worst-gap cohort, on MI300X with the patched LLVM + patched Triton AMD
backend:

1. **AMDGCN diff**: at least one `s_waitcnt vmcnt(0)` immediately
   preceding a `v_mfma_*` instruction in the K-loop body is rewritten
   to `s_waitcnt vmcnt(N)` with `N > 0`.
   `TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1` and
   `grep -nE 's_waitcnt vmcnt\([1-9][0-9]*\)'` on the dumped
   `*.amdgcn` is the witness.
2. **Numerical correctness**: output is bit-equivalent or within 1e-2
   (BF16) of the reference `torch.matmul` on the same inputs. **Past
   attempts produced NaN** (K-6635 / K-6663); this is the gate that
   has so far not been passed.
3. **TFLOPS delta**: measured TFLOPS exceeds the same shape's measured
   TFLOPS with the attribute absent, on at least 3 independent runs,
   and is reported alongside hipBLASLt's TFLOPS on the same shape.

Without all three, do not ship a Python wrapper. The patches stay in
this directory as documentation of the lever.

## Correctness rationale

`max_vmcnt_before_mfma > 0` is a **caller promise** that the MFMA does
not consume a VGPR whose backing load is among the relaxed N
outstanding loads. K-5684 documents two failure modes from binary
patches that violated this promise:

* **NaN output** when the relaxed load was the MFMA's data operand.
* **GPU memory-access fault** when the relaxed load fed an SGPR/scalar
  index used for address computation.

K-6244 / K-6635 / K-6663 then established empirically that in current
Triton + current ROCm codegen for tritonblas's persistent matmul, every
surviving `vmcnt(0)` site has the operand-true RAW dependency that
binds — i.e., the caller cannot honestly make the promise. That is why
no Python wrapper is shipped here.
