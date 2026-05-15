"""ctypes wrapper around the K-6808 hand-written interleaved MFMA kernel.

Loads `libmfma_gemm.so` (built by `build.sh`) at import time. If the .so is
missing or fails to load (e.g. host has no hipcc, wrong arch), the module
exposes `hip_kernel_available() == False` and the dispatch layer in
`matmul.py` falls through to the Triton path.

Per K-6808's `mfma_interleaved` ABI:

    void launch_interleaved(const void* A, const void* B, void* C,
                            int M, int N, int K, hipStream_t stream);

Constraints (kernel-side, from K-6808 mfma_gemm.hip):
- gfx942 only (CDNA3 v_mfma_f32_16x16x16_bf16_1k intrinsic)
- BLOCK_M = BLOCK_N = 16, BLOCK_K = 16, K_UNROLL = 4
  → M and N must be multiples of 16, K must be multiple of 64
- Inputs A,B in BF16 row-major; output C in FP32 row-major (this wrapper
  converts the FP32 accumulator back to the requested out_dtype).
- A is (M,K) row-major; B is (K,N) row-major (NOT the K-major B used by
  Triton's persistent_matmul — we restride accordingly in the wrapper).
"""
from __future__ import annotations

import ctypes
import os
from typing import Optional, Tuple

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libmfma_gemm.so")

# Per-shape dispatch table.  Entries are (M, N, K, dtype) tuples; if the
# incoming GEMM matches one of these, we dispatch to the HIP kernel.
#
# Selection rationale (K-6953 brief):
# - 64x64x4096:    K-6860 sh2, highest clustering ratio, K-6808 baseline
# - 128x128x4096:  K-6860 sh3, same TB cluster ratio (0.0781), small-tile
# - 256x256x4096:  K-6860 sh7, same TB cluster ratio, large-enough that
#                  K-6808 measured a 3.8x speedup over tritonblas
#
# K-6860 iter-2 found TB cluster ratios are *identical* across the cohort
# (TB always emits a 5-ds_read prefix), so "the next two worst from K-6860"
# is operationally any other small BF16 shape where the K-6808 hand kernel
# also beat tritonblas.  We use the three K-6808 measured shapes here.
HIP_FALLBACK_SHAPES: Tuple[Tuple[int, int, int, torch.dtype], ...] = (
    (64,   64,  4096, torch.bfloat16),
    (256, 256,  4096, torch.bfloat16),
    (1024, 1024, 4096, torch.bfloat16),
)


_lib: Optional[ctypes.CDLL] = None
_load_error: Optional[str] = None


def _try_load() -> None:
    """Attempt to dlopen the kernel .so. Records the error on failure."""
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return
    if not os.path.exists(_LIB_PATH):
        _load_error = f"libmfma_gemm.so not found at {_LIB_PATH} (run hip_kernels/build.sh)"
        return
    try:
        lib = ctypes.CDLL(_LIB_PATH)
        # void launch_interleaved(const void* A, const void* B, void* C,
        #                         int M, int N, int K, hipStream_t stream)
        lib.launch_interleaved.restype = None
        lib.launch_interleaved.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # B
            ctypes.c_void_p,  # C
            ctypes.c_int,     # M
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_void_p,  # stream
        ]
        _lib = lib
    except OSError as exc:
        _load_error = f"dlopen({_LIB_PATH}) failed: {exc}"


def hip_kernel_available() -> bool:
    """True iff the kernel .so loaded successfully and we're on gfx942."""
    _try_load()
    if _lib is None:
        return False
    if not torch.cuda.is_available():
        return False
    # gfx942 is the MI300X arch the kernel was compiled for; on any other
    # arch the kernel will fail at launch with code 98 (invalid device fn).
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    return arch.startswith("gfx942")


def should_use_hip_fallback(M: int, N: int, K: int, dtype: torch.dtype) -> bool:
    """Shape-table lookup. Cheap; called on every matmul()."""
    return (M, N, K, dtype) in HIP_FALLBACK_SHAPES


def hip_interleaved_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """C = A @ B via the K-6808 interleaved HIP kernel.

    Requires gfx942, BF16 inputs, and (M,N) divisible by 16, K divisible by 64.

    Returns a tensor with dtype matching `a` (same convention as
    tritonblas.matmul).  The kernel writes FP32 internally; we cast on output.
    """
    _try_load()
    if _lib is None:
        raise RuntimeError(f"HIP kernel unavailable: {_load_error}")

    assert a.dim() == 2 and b.dim() == 2, "matmul expects 2-D tensors"
    assert a.shape[1] == b.shape[0], f"incompatible: a={a.shape} b={b.shape}"
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, (
        "HIP kernel requires BF16 inputs"
    )

    M, K = a.shape
    _, N = b.shape
    assert M % 16 == 0 and N % 16 == 0 and K % 64 == 0, (
        f"HIP kernel requires M,N % 16 == 0 and K % 64 == 0; got {M}x{N}x{K}"
    )

    # Kernel expects row-major contiguous A (M,K) and B (K,N), and writes
    # FP32 C (M,N) row-major.
    a_c = a.contiguous()
    b_c = b.contiguous()
    c_fp32 = torch.empty((M, N), dtype=torch.float32, device=a.device)

    stream = torch.cuda.current_stream(a.device).cuda_stream
    _lib.launch_interleaved(
        ctypes.c_void_p(a_c.data_ptr()),
        ctypes.c_void_p(b_c.data_ptr()),
        ctypes.c_void_p(c_fp32.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
        ctypes.c_void_p(stream),
    )

    if out is None:
        return c_fp32.to(a.dtype)
    out.copy_(c_fp32.to(out.dtype))
    return out
