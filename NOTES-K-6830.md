# K-6830 — Triton AMD scheduler dependency

The performance fix for the BF16 GEMM `ds_read` / `v_mfma` clustering on
MI300X (gfx942) lives in upstream Triton, not in tritonBLAS. To pick up
the speed-up, build Triton from this branch / commit:

- Repo:    https://github.com/ryanswann-amd/triton
- Branch:  `s-002/inject-ds-read-mfma-interleaving-gfx94`
- Commit:  `f7feddad0ac6d3964412c3340400c4bb25c44fc4`
- Touches: `third_party/amd/lib/TritonAMDGPUTransforms/{LowerLoops,BlockPingpong}.cpp`
- Origin tracking ticket: K-6830

No tritonBLAS source change is required — kernels in this repo automatically
benefit when compiled against the patched Triton.
