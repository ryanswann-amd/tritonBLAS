# K-6830 — Triton AMD scheduler experiment (NEGATIVE FINDING)

This branch tracks an experimental Triton patch that did NOT close the
gfx942 BF16 GEMM gap to hipBLASLt on MI300X. Kept here so the next
investigator does not redo the work.

## What was tried

Two `.cpp` cleanups in upstream Triton's AMD backend:

- `third_party/amd/lib/TritonAMDGPUTransforms/LowerLoops.cpp` — restored
  the dead `localStoreCluster = 2` branch in `SingleDotSchedule::initSchedule`
  (statement-order bug; the `pairedGlobalLoadLocalStore` predicate was
  computed before `stages[SCHED_LOCAL_STORE] += maxDist`).
- `third_party/amd/lib/TritonAMDGPUTransforms/BlockPingpong.cpp` —
  parameterised cluster-barrier emission so `transformFourPPClusters`
  passes `schedMask = 0x108` (`SCHED_BARRIER_MASK_MFMA |
  SCHED_BARRIER_MASK_DS_READ`) on mem→dot transitions, matching
  hipBLASLt's mainloop semantics.

## What we observed

On a clean A/B (upstream `3c71c5f` vs patched `f7feddad`, both built from
source in the same container, c42 / mi300x / `gfx942:sramecc+:xnack-`):

- 64×64×4096 BF16 NS=4 SingleDot probe: bit-identical AMDGCN, 0.45 TFLOPS
  vs 0.45 TFLOPS (within ±2% noise).
- `tritonblas.matmul` sweep over (64,64,K) and (M,M,4096) for
  M ∈ {64, 128, 256, 1024}: bit-identical `persistent_matmul.amdgcn`,
  TFLOPS within noise.

`scripts/test_K6830_hypothesis.py` is a machine-checkable encoding of the
pre-registered PRD hypothesis; it exits **1** on the data in `output/`.
Reviewers / CI can use that exit code to gate a "this is a fix PR"
classification.

## Where the patch lives

The actual Triton change is **not** vendored in tritonBLAS — it would
silently rot on rebase, and tritonBLAS's build does not consume it.
To reproduce, build Triton from:

- Repo:    https://github.com/ryanswann-amd/triton
- Branch:  `s-002/inject-ds-read-mfma-interleaving-gfx94`
- Commit:  `f7feddad0ac6d3964412c3340400c4bb25c44fc4`
- Touches: `third_party/amd/lib/TritonAMDGPUTransforms/{LowerLoops,BlockPingpong}.cpp`

## Follow-up tickets

- The `LowerLoops` patch fires only at `maxDist >= 1`; the autotuner /
  selector for the tested shapes appears to pick `maxDist == 0`. Audit
  StreamPipelinerV2 stage assignment.
- The `BlockPingpong` patch only fires inside `transformFourPPClusters`;
  the cached `persistent_matmul` may not be exercising that path. Add a
  diagnostic that prints when the four-cluster transform fires.
- K-6808 measured that LLVM's AMDGPU `MachineScheduler` re-coalesces
  `ds_read*` regardless of source order; an MLIR-only fix is necessary
  but not sufficient.

Origin tracking ticket: K-6830.
