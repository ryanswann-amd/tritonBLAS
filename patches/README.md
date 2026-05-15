# `amdgpu-max-vmcnt-before-mfma` — opt-in vmcnt(0)→vmcnt(N) relaxation

This directory contains two cross-repo patches that, together with the
`max_vmcnt_before_mfma` kwarg / `TRITONBLAS_MAX_VMCNT_BEFORE_MFMA` env-var
plumbed through `include/tritonblas/matmul.py`, give kernel authors a
per-function knob to relax the conservative `s_waitcnt vmcnt(0)` floor that
LLVM's `SIInsertWaitcnts` pass emits immediately before MFMA / MAI
instructions to `s_waitcnt vmcnt(N)`.

## Why this exists

On the K-5156 / K-4967 worst-gap shape cohort (BF16 GEMM, MI300X /
gfx942), tritonblas's persistent matmul issues each MFMA preceded by
`s_waitcnt vmcnt(0)` (drain ALL outstanding loads), while hipBLASLt
issues MFMA with `vmcnt(N)` where 1 ≤ N ≤ 14. K-5101, K-5694, K-5647,
and K-6555 quantified this as the dominant lever in the +30% TFLOPS gap.

Prior in-tree levers were tried and refused:

| Ticket  | Lever                                                                | Outcome                                                                          |
|---------|----------------------------------------------------------------------|----------------------------------------------------------------------------------|
| K-6028  | `cl::opt -amdgpu-waitcnt-load-forcezero=N`                           | hard-coded 0/inf; AMDGCN byte-identical                                          |
| K-6138  | post-compilation ISA rewriter `vmcnt(0)` → `vmcnt(N)`                | 14/18 noop, strict variants fault, lenient produces NaN                          |
| K-6244  | full-suite rewriter on 27 shapes × 5 modes                           | 0/27 sites rewritten — current Triton+ROCm no longer emits the K-loop pattern    |
| K-6244† | remaining 3 vmcnt(0) sites                                           | all immediately precede `ds_write_b128` (VMEM→LDS handoff barriers, MUST stay 0) |
| K-6635  | analogous `lgkmcnt(N)` ISA rewriter on the 7 candidate sites         | all 7 are operand-true RAW edges (correctness collapse on N=1, NaN on N≥2)       |
| K-6663  | joint `(vmcnt(N), lgkmcnt(M))` 5×5 grid                              | joint best matches single-axis, all N≥1 cells produce NaN                        |

K-6244 explicitly recommends: **CLOSE the entire vmcnt(N) line of
investigation.** This patch is the clean upstream-fixable expression of
the lever, intended for kernels where the author *can* prove the
operand-true RAW dependency does not bind — e.g., persistent matmuls
with `num_stages` ≥ 2 where the K iteration whose loads are being
relaxed is not the same iteration whose loads the MFMA reads.

The default (`max_vmcnt_before_mfma=0`, attribute absent) is a no-op
and preserves today's correctness-first LLVM behavior.

## Files

| File                                                                | Lands in        | Purpose                                                                          |
|---------------------------------------------------------------------|-----------------|----------------------------------------------------------------------------------|
| `0001-llvm-amdgpu-max-vmcnt-before-mfma-attr.patch`                 | `llvm-project`  | Adds `amdgpu-max-vmcnt-before-mfma` function attribute to `SIInsertWaitcnts.cpp` |
| `0002-triton-amd-backend-max-vmcnt-before-mfma-option.patch`        | `triton`        | Adds `HIPOptions.max_vmcnt_before_mfma` and stamps the LLVM attribute            |
| `../include/tritonblas/matmul.py` (already in this repo)            | `tritonblas`    | Accepts `max_vmcnt_before_mfma=N` kwarg and `TRITONBLAS_MAX_VMCNT_BEFORE_MFMA` env-var |

## How to apply

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
#    for the $LLVM_INCLUDE_DIRS / $LLVM_LIBRARY_DIR knobs), then
#    pip install -e ${TRITONBLAS_DIR}.
```

## How to use

After the patches are built in:

```python
from tritonblas import matmul                # public API
out = matmul(a, b, max_vmcnt_before_mfma=3)  # OPT-IN (when wired through)
```

Or via env-var (no code changes):

```bash
TRITONBLAS_MAX_VMCNT_BEFORE_MFMA=3 python my_benchmark.py
```

The kwarg is **only forwarded into the Triton kernel launch when N > 0**
so unpatched Triton installs are unaffected (`HIPOptions` does not yet
have the field; passing it would `KeyError`).

## Verifying the attribute took effect

Dump the AMDGCN of a compiled kernel and grep:

```bash
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 \
TRITONBLAS_MAX_VMCNT_BEFORE_MFMA=3 \
python -c "import torch, tritonblas as tb; \
           a=torch.randn(256,16384,device='cuda',dtype=torch.bfloat16); \
           b=torch.randn(16384,256,device='cuda',dtype=torch.bfloat16); \
           tb.matmul(a, b)"

grep -nE 's_waitcnt vmcnt\([1-9][0-9]*\)' ~/.triton/dump/*/persistent_matmul.amdgcn | \
  awk -F: '/s_waitcnt vmcnt\([1-9]/{print $0}' | head -5
```

You should see `s_waitcnt vmcnt(3)` immediately preceding `v_mfma_*`
where the upstream compiler would have emitted `vmcnt(0)`.

## Correctness disclaimer

`max_vmcnt_before_mfma > 0` is a **caller promise** that the MFMA does
not consume a VGPR whose backing load is among the relaxed N
outstanding loads. K-5684 documents two failure modes from binary
patches that violated this promise:

* **NaN output** when the relaxed load was the MFMA's data operand.
* **GPU memory-access fault** when the relaxed load fed an SGPR/scalar
  index used for address computation.

The Triton frontend (or the kernel author) is responsible for proving
the dependency does not bind. For tritonblas's `persistent_matmul`,
this is true iff:

1. `num_stages >= 2` (so the prefetched K iteration's loads are at
   least one MFMA-burst behind the consuming MFMAs), AND
2. The chosen N is < the number of outstanding loads from the
   prefetched-but-not-yet-consumed K iteration.

If unsure, leave the default (0).
