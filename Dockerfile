FROM rocm/pytorch:latest-release

RUN apt-get update && \
    apt-get install -y git openssh-client && \
    rm -rf /var/lib/apt/lists/*

# IMPORTANT: Do NOT install upstream triton here.
# The rocm/pytorch base image ships with pytorch-triton-rocm which includes
# AMD MLIR patches required for FP8 fnuz types (float8_e4m3fnuz, float8_e5m2fnuz)
# on gfx942/gfx950. Installing upstream triton replaces these patches and causes
# MLIR assertion failures: "DenseElementsAttr::get(): floatAttr.getType() == eltType"
# on any tl.load(..., other=0.0) with FP8 fnuz-typed tensors.
