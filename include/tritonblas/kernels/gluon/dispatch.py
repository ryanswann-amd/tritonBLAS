"""
Gluon kernel dispatch for tritonblas.

Routes to the Gluon v9 256x256x64 kernel when conditions are met.
Sets LLIR_SCHED and AMDGCN_AS only for the Gluon kernel compilation,
not globally (the standard tritonblas kernel crashes with these flags).
"""

import os
import torch


_COMPILE_OK = None


def gluon_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    global _COMPILE_OK

    if _COMPILE_OK is False:
        return None

    if b.stride(0) != 1:
        return None

    # Enable scheduler flags ONLY for the Gluon kernel compilation.
    # These must NOT be set during standard tritonblas kernel compilation.
    os.environ["TRITON_ENABLE_LLIR_SCHED"] = "1"

    try:
        from .fp16_gfx950 import matmul as _gluon_matmul
        _gluon_matmul(a, b, c)
        _COMPILE_OK = True
        return c
    except Exception:
        _COMPILE_OK = False
        return None
