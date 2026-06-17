"""
Gluon kernel dispatch for tritonblas.

Bridges the tritonblas matmul interface to the Gluon gfx950 kernels.
Each kernel module exports a `matmul(a, b, c=None)` function; this
module re-exposes them with the tile parameters driven by Origami.
"""

import torch
import triton

from . import ensure_scheduler_env


def gluon_matmul_lt(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    selector,
    bias=None,
    a_scale=None,
    b_scale=None,
    quantized=False,
):
    """Launch the Gluon FP16/BF16 GEMM kernel on gfx950.

    The v9 kernel has layouts hardcoded for 256x256x64 tiles. We use the
    kernel's own matmul() wrapper which handles tile config and grid setup.
    Origami tile selection will be integrated once the kernel is parameterized
    for multiple tile sizes.
    """
    ensure_scheduler_env()

    from .fp16_gfx950 import matmul as _fp16_matmul

    _fp16_matmul(a, b, c)
    return c


def gluon_matmul_fp8_lt(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    selector,
    a_scale=None,
    b_scale=None,
):
    """Launch the Gluon FP8 GEMM kernel on gfx950."""
    ensure_scheduler_env()

    from .fp8_gfx950 import matmul as _fp8_matmul

    _fp8_matmul(a, b, c)
    return c


def gluon_matmul_fp4_lt(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    selector,
):
    """Launch the Gluon MXFP4 GEMM kernel on gfx950."""
    ensure_scheduler_env()

    from .fp4_gfx950 import matmul as _fp4_matmul

    _fp4_matmul(a, b, c)
    return c
