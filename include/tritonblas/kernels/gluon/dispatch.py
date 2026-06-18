"""
Gluon kernel dispatch for tritonblas.

Routes FP16/BF16 GEMM to the best Gluon tile variant for the given shape.
Falls back to the standard Triton kernel path if compilation fails.

Tile variants:
  128x128x64  — small M and N (< 512)
  128x256x64  — small M, large N
  256x128x64  — large M, small N
  256x256x64  — large M and N (default, fastest on big shapes)
"""

import os
import torch


_COMPILE_OK = {}


def _ensure_env():
    os.environ.setdefault("TRITON_ENABLE_LLIR_SCHED", "1")
    os.environ.setdefault("TRITON_ENABLE_AMDGCN_AS", "1")


def _select_tile(M, N):
    if M >= 512 and N >= 512:
        return "256x256"
    if M >= 512:
        return "256x128"
    if N >= 512:
        return "128x256"
    return "128x128"


def _get_kernel(tile_key):
    if tile_key == "256x256":
        from .fp16_gfx950 import matmul
    elif tile_key == "128x128":
        from .fp16_128x128_gfx950 import matmul
    elif tile_key == "128x256":
        from .fp16_128x256_gfx950 import matmul
    elif tile_key == "256x128":
        from .fp16_256x128_gfx950 import matmul
    else:
        return None
    return matmul


def gluon_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Run the best Gluon tile variant for this shape.

    Returns c on success, or None if the kernel fails to compile
    (signaling the caller to fall back to the standard path).
    """
    M, K = a.shape
    _, N = b.shape
    tile_key = _select_tile(M, N)

    if _COMPILE_OK.get(tile_key) is False:
        return None

    _ensure_env()

    try:
        kernel = _get_kernel(tile_key)
        if kernel is None:
            return None
        kernel(a, b, c)
        _COMPILE_OK[tile_key] = True
        return c
    except Exception:
        _COMPILE_OK[tile_key] = False
        return None
