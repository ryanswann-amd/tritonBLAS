FROM rocm/pytorch:latest-release

RUN apt-get update && \
    apt-get install -y git openssh-client && \
    rm -rf /var/lib/apt/lists/*

# ── Triton (ROCm-patched from base image) ────────────────────
# The rocm/pytorch base image ships pytorch-triton-rocm which includes
# AMD-specific MLIR patches for FP8 fnuz types (float8_e4m3fnuz,
# float8_e5m2fnuz) on gfx942. DO NOT replace with upstream PyPI triton —
# upstream lacks these patches and causes MLIR assertion failures:
#   DenseElementsAttr::get(): floatAttr.getType() == eltType
# when compiling FP8 kernels with masked loads (other=0.0).
#
# Projects needing a specific Triton commit can override via PYTHONPATH
# in their bind-mounted repo.
RUN python3 -c "import triton; print(f'Triton: {triton.__version__}')" 2>/dev/null || \
    (echo "WARNING: No triton in base image -- installing from PyPI as fallback" && \
     pip install --quiet triton)
