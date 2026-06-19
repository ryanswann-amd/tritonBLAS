"""
Gluon kernel dispatch for tritonblas.

Routes to the Gluon v9 256x256x64 kernel when Origami selects a 256x256 tile.
Falls back to standard Triton for smaller tiles or if compilation fails.
"""

import os
import torch


_COMPILE_OK = None


def _ensure_env():
    os.environ.setdefault("TRITON_ENABLE_LLIR_SCHED", "1")
    os.environ.setdefault("TRITON_ENABLE_AMDGCN_AS", "1")


def gluon_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Run the Gluon v9 256x256x64 GEMM kernel.

    Returns c on success, or None if the kernel fails to compile
    (signaling the caller to fall back to the standard path).
    """
    global _COMPILE_OK

    if _COMPILE_OK is False:
        return None

    _ensure_env()

    try:
        from .fp16_gfx950 import matmul as _gluon_matmul
        _gluon_matmul(a, b, c)
        _COMPILE_OK = True
        return c
    except Exception:
        _COMPILE_OK = False
        return None
