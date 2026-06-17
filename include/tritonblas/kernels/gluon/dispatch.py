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
    """Launch the Gluon FP16/BF16 GEMM kernel on gfx950."""
    ensure_scheduler_env()

    from .fp16_gfx950 import v9_beyond_hotloop

    M, K = a.shape
    _, N = b.shape

    BLK_M = selector.block_m
    BLK_N = selector.block_n
    BLK_K = selector.block_k
    GROUP_SIZE_M = selector.group_m
    NUM_XCDS = selector.num_sms

    GRID_MN = triton.cdiv(M, BLK_M) * triton.cdiv(N, BLK_N)
    grid = (GRID_MN, 1)

    v9_beyond_hotloop[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLK_M,
        BLOCK_N=BLK_N,
        BLOCK_K=BLK_K,
        GRID_MN=GRID_MN,
        NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=4,
    )

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
