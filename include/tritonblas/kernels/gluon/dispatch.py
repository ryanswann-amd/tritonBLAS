"""
Gluon kernel dispatch for tritonblas.

Routes to the Gluon v9 256x256x64 GEMM kernel on gfx950 (MI350/MI355)
when ALL of the following conditions are met:

  1. TRITONBLAS_ENABLE_GLUON=1 environment variable is set
  2. triton.experimental.gluon is importable (requires Triton built from
     the gfx950-tutorial branch of triton-lang/triton)
  3. Origami selects a 256x256 tile (block_m >= 256 and block_n >= 256)
  4. M >= 2048 and N >= 2048 (smaller shapes crash the LLIR scheduler)
  5. B matrix is K-contiguous (stride(0) == 1), i.e. created as
     torch.randn(N, K).T — row-major B (stride(0) == N) triggers an
     LLVM lowering bug on gfx950

If any condition fails, returns None and the caller falls back to the
standard Triton persistent_matmul kernel.

Performance (BF16, MI355X, K-contiguous B, LLIR_SCHED only):
  - 4096x4096x8192:  1441 TFLOPS (0.97x hipBLASLt)
  - 8192x8192x8192:  1472 TFLOPS (0.97x hipBLASLt)
  - 5120x5120x5120:  1266 TFLOPS (1.11x hipBLASLt)

With AMDGCN_AS added (set externally, crashes on K < 4096):
  - 4096x4096x8192:  ~1500 TFLOPS (~1.02x hipBLASLt)
"""

import os
import torch


_COMPILE_OK = None


def gluon_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Run the Gluon v9 kernel. Returns c on success, None on failure."""
    global _COMPILE_OK

    if _COMPILE_OK is False:
        return None

    # B must be K-contiguous (created as randn(N,K).T)
    if b.stride(0) != 1:
        return None

    # Set LLIR scheduler for Gluon compilation only — the standard
    # tritonblas persistent_matmul kernel crashes with this flag.
    os.environ["TRITON_ENABLE_LLIR_SCHED"] = "1"

    try:
        from .fp16_gfx950 import matmul as _gluon_matmul
        _gluon_matmul(a, b, c)
        _COMPILE_OK = True
        return c
    except Exception:
        _COMPILE_OK = False
        return None
    finally:
        os.environ.pop("TRITON_ENABLE_LLIR_SCHED", None)
