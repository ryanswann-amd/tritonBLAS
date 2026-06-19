"""
Gluon kernel dispatch for tritonblas.

Routes to the Gluon v9 256x256x64 kernel when:
  - TRITONBLAS_ENABLE_GLUON=1
  - Origami selects a 256x256 tile
  - M, N >= 2048
  - B is K-contiguous (stride(0) == 1)

Falls back to standard Triton otherwise.

For peak Gluon performance, set TRITON_ENABLE_LLIR_SCHED=1 and
TRITON_ENABLE_AMDGCN_AS=1 externally. These are NOT auto-set because
they can cause crashes on some shapes.
"""

import torch


_COMPILE_OK = None


def gluon_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    global _COMPILE_OK

    if _COMPILE_OK is False:
        return None

    if b.stride(0) != 1:
        return None

    try:
        from .fp16_gfx950 import matmul as _gluon_matmul
        _gluon_matmul(a, b, c)
        _COMPILE_OK = True
        return c
    except Exception:
        _COMPILE_OK = False
        return None
