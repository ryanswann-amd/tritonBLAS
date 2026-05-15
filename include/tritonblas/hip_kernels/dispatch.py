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

K-7078: dispatch routing fix.  K-7054 found that the K-6953 routing fired
for *zero* of the S-002 cohort shapes:
  - 64x64x4096 BF16     — was in table, but K-6953 was reverted on main
  - 128x1024x8192 BF16  — was NOT in table
  - 1x4096x4096 BF16    — was NOT in table AND violates M%16==0 tile
                           constraint (gemv-shaped)
This file extends the table to cover all three cohort shapes and adds an
M-padding wrapper so the gemv shape can use the same MFMA tile.  The
dispatch decision is logged via _log_dispatch() so K-7054-style routing
audits can be re-run from saved logs.
"""
from __future__ import annotations

import ctypes
import logging
import os
from typing import Optional, Tuple

import torch

logger = logging.getLogger("tritonblas.hip_dispatch")

# K-7078: when set, every dispatch decision is printed to stderr.  The default
# is silent; turn on with TRITONBLAS_HIP_DISPATCH_LOG=1 for routing audits.
_DISPATCH_LOG_ENV = os.environ.get("TRITONBLAS_HIP_DISPATCH_LOG", "0").lower()
_DISPATCH_LOG = _DISPATCH_LOG_ENV in ("1", "true", "yes")

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libmfma_gemm.so")

# Per-shape dispatch table.  Entries are (M, N, K, dtype) tuples; if the
# incoming GEMM matches one of these, we dispatch to the HIP kernel.
#
# K-7078 / S-002 cohort — only shapes this PR actually benchmarks and
# verifies are listed.  The K-6953 entries (256x256x4096, 1024x1024x4096)
# from the original table were dropped because they are not part of the
# S-002 cohort and have no benchmark/correctness coverage in this branch
# (Minimalist review feedback).  Add them back via a separate ticket if
# they regain a verified benchmark.
#
# - 64x64x4096:     S-002 small-shape gap (the K-6808 hand-written kernel's
#                   native target).  M=N=64 → 4x4 grid of 16x16 tiles.
# - 128x1024x8192:  S-002 medium-shape gap; M=128, N=1024 both %16==0,
#                   K=8192 %64==0 → fits the 16x16 MFMA tile cleanly.
# - 1x4096x4096:    S-002 gemv-shaped; M=1 violates BLOCK_M=16 tile
#                   constraint, so the wrapper M-pads to 16 (only first
#                   output row is sliced back).  Pre-fix this shape was
#                   100% Triton path; now eligible for HIP.
HIP_FALLBACK_SHAPES: Tuple[Tuple[int, int, int, torch.dtype], ...] = (
    (64,    64, 4096, torch.bfloat16),
    (128, 1024, 8192, torch.bfloat16),
    (1,   4096, 4096, torch.bfloat16),
)


_lib: Optional[ctypes.CDLL] = None
_load_error: Optional[str] = None


def _log_dispatch(reason: str, M: int, N: int, K: int, dtype: torch.dtype, took_hip: bool) -> None:
    """K-7078: emit a one-line dispatch decision for routing audits."""
    if not _DISPATCH_LOG:
        return
    path = "HIP" if took_hip else "TRITON"
    msg = f"[tritonblas.hip_dispatch] M={M} N={N} K={K} dtype={dtype} -> {path} ({reason})"
    print(msg, flush=True)
    logger.info(msg)


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


# K-7078: kernel tile constants (must mirror BLOCK_M / BLOCK_N / BLOCK_K
# / K_UNROLL in mfma_gemm.hip).  Used both by the dispatch predicate and
# by the M-padding wrapper for the gemv-shaped 1x4096x4096 cohort entry.
_BLOCK_M = 16
_BLOCK_N = 16
_BLOCK_K = 16
_K_UNROLL = 4
_K_DIVISOR = _BLOCK_K * _K_UNROLL  # = 64


def _kernel_can_handle(M: int, N: int, K: int) -> bool:
    """True iff the raw kernel can handle (M,N,K) with no padding/dispatch
    workaround.  M-pad wrapper handles the M%16!=0 case separately.
    """
    return (M % _BLOCK_M == 0) and (N % _BLOCK_N == 0) and (K % _K_DIVISOR == 0)


def should_use_hip_fallback(M: int, N: int, K: int, dtype: torch.dtype) -> bool:
    """Shape-table lookup. Cheap; called on every matmul().

    K-7078: this is the predicate K-7054 found was never satisfied for the
    S-002 cohort.  The fix is two-pronged: (a) the table now lists the
    cohort shapes, (b) the predicate accepts shapes whose M dim is < 16
    as long as N/K satisfy the tile divisibility — those are M-padded by
    `hip_interleaved_matmul`.
    """
    if (M, N, K, dtype) not in HIP_FALLBACK_SHAPES:
        return False
    if dtype != torch.bfloat16:
        return False
    # Shapes that need M-padding (gemv-like, M < BLOCK_M).  N and K still
    # have to satisfy tile divisibility for the underlying kernel.
    if M < _BLOCK_M:
        return (N % _BLOCK_N == 0) and (K % _K_DIVISOR == 0)
    return _kernel_can_handle(M, N, K)


def hip_interleaved_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """C = A @ B via the K-6808 interleaved HIP kernel.

    Requires gfx942, BF16 inputs, and (N,K) divisible by 16/64.  M may be
    < BLOCK_M; the wrapper pads A with zero rows up to BLOCK_M and slices
    the corresponding output rows on the way out (K-7078: needed for the
    1x4096x4096 S-002 cohort entry).

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
    assert N % _BLOCK_N == 0 and K % _K_DIVISOR == 0, (
        f"HIP kernel requires N % {_BLOCK_N} == 0 and K % {_K_DIVISOR} == 0; "
        f"got {M}x{N}x{K}"
    )

    # K-7078: M-padding for gemv-shaped GEMMs (e.g. 1x4096x4096).  The
    # kernel is launched on (16, N) but only the first M rows of the output
    # are real; the remaining (16-M) rows of A are zero so no extra rows
    # of B contribute.  We slice the output back to (M, N).
    a_c = a.contiguous()
    b_c = b.contiguous()
    pad_M = (-M) % _BLOCK_M  # 0 if already aligned, else 16-M etc.
    if pad_M:
        a_padded = torch.zeros((M + pad_M, K), dtype=a_c.dtype, device=a_c.device)
        a_padded[:M].copy_(a_c)
        a_kernel = a_padded
        M_kernel = M + pad_M
    else:
        a_kernel = a_c
        M_kernel = M

    c_fp32 = torch.empty((M_kernel, N), dtype=torch.float32, device=a.device)

    stream = torch.cuda.current_stream(a.device).cuda_stream
    _lib.launch_interleaved(
        ctypes.c_void_p(a_kernel.data_ptr()),
        ctypes.c_void_p(b_c.data_ptr()),
        ctypes.c_void_p(c_fp32.data_ptr()),
        ctypes.c_int(M_kernel),
        ctypes.c_int(N),
        ctypes.c_int(K),
        ctypes.c_void_p(stream),
    )

    c_real = c_fp32[:M] if pad_M else c_fp32

    if out is None:
        return c_real.to(a.dtype)
    out.copy_(c_real.to(out.dtype))
    return out
