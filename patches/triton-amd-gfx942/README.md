# Triton AMD gfx942 scheduler patches

These patches modify Triton's AMD backend to close the `ds_read` / `v_mfma`
interleaving gap vs. hipBLASLt on MI300X (gfx942). They are *upstream* Triton
changes — tritonBLAS itself does not need modification, but the kernels in
this repo will only see the perf uplift when built against a Triton that
carries these patches.

## Files

- `0001-K-6830-interleave-ds-read-v-mfma-gfx942.patch` — two-file source
  patch for the Triton AMD MLIR pipeline (`LowerLoops.cpp` +
  `BlockPingpong.cpp`). Captures the change pushed to
  `ryanswann-amd/triton` branch
  `s-002/inject-ds-read-mfma-interleaving-gfx94`
  (commit `f7feddad0ac6d3964412c3340400c4bb25c44fc4`).

## What the patch does

1. **`LowerLoops.cpp`** — fixes a statement-order bug in
   `SingleDotSchedule::initSchedule` that made
   `pairedGlobalLoadLocalStore` unconditionally `true`, killing the
   `localStoreCluster = 2` branch. With the fix, `num_stages >= 3`
   schedules interleave `ds_write` with `v_mfma` instead of trailing
   them.

2. **`BlockPingpong.cpp`** — adds an optional `schedMask` parameter to
   `genClusterBarrier` / `appendClusterBarrier` /
   `prependClusterBarrier`. The four mem→dot transitions in
   `transformFourPPClusters` now pass `schedMask = 0x108`
   (`MFMA | DS_READ`), letting the LLVM AMDGPU MachineScheduler
   interleave next-iter `ds_read` groups with the current cluster's
   `v_mfma`. Dot→mem boundaries keep `mask = 0` to preserve the
   Membar correctness fence.

Both targets are gfx942-only by virtue of where they live in the AMD
backend; gfx950/gfx1100/gfx12 paths are untouched.

## Applying

```bash
# In a checkout of triton-lang/triton (or a fork at the same SHA range
# as ryanswann-amd/triton main):
cd /path/to/triton
git am < /path/to/tritonblas/patches/triton-amd-gfx942/0001-K-6830-interleave-ds-read-v-mfma-gfx942.patch
# or, for a non-mailbox-formatted apply:
git apply /path/to/tritonblas/patches/triton-amd-gfx942/0001-K-6830-interleave-ds-read-v-mfma-gfx942.patch
```

## Provenance

- Origin task: K-6830 (S-002 close tritonBLAS vs. hipBLASLt gap on MI300X)
- Upstream branch: `ryanswann-amd/triton:s-002/inject-ds-read-mfma-interleaving-gfx94`
- Upstream commit: `f7feddad0ac6d3964412c3340400c4bb25c44fc4`
- Hardware: MI300X (gfx942) c42
- Container: a recent `rocm/pytorch` image (ROCm + PyTorch nightly) on Ubuntu 24.04
