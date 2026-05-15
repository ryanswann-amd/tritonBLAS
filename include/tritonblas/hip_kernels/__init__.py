"""Hand-written HIP kernel fallbacks for shapes where Triton's emitted
inner-loop schedule clusters LDS reads (K-6860 worst-case shapes).

The kernel source comes verbatim from K-6808's `mfma_interleaved` variant,
which produces numerically-equivalent BF16 output to torch.matmul (max_abs_err
within the BF16 MFMA tolerance of 1.0) and runs ~3-4x faster than
tritonblas's persistent_matmul on small shapes (K-6808 results.csv).

The kernel is gfx942-only (MI300X CDNA3 v_mfma_f32_16x16x16_bf16_1k).
On any other arch, `hip_interleaved_matmul` raises and the dispatch layer
falls through to the Triton path.
"""
from .dispatch import (
    HIP_FALLBACK_SHAPES,
    hip_interleaved_matmul,
    hip_kernel_available,
    should_use_hip_fallback,
)

__all__ = [
    "HIP_FALLBACK_SHAPES",
    "hip_interleaved_matmul",
    "hip_kernel_available",
    "should_use_hip_fallback",
]
