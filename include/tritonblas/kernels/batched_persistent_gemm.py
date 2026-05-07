# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Batched persistent GEMM kernel.

Fuses the batch dimension into the launch grid so a single kernel
launch processes all B independent (M, K) x (K, N) matmuls in one
dispatch instead of B sequential single-shape launches.

Design
------
Grid layout: ``(num_tiles_per_batch, B)``.

* ``program_id(0)`` = tile index inside the (M, N) tile grid for one
  batch (identical groupings/chiplet routing as the single-shape
  ``persistent_matmul`` kernel).
* ``program_id(1)`` = batch index ``b in [0, B)``.

Each program handles exactly one (tile, batch) pair. The base pointers
for A, B, and C are advanced by ``b * stride_a_batch`` etc. before the
inner GEMM K-loop runs. The K-loop, layout, scheduling, and epilogue
logic are reused unchanged from the single-shape persistent kernel via
the same ``GemmContext``/``ScheduleContext``/view aggregates, so the
inner-loop ISA is bit-identical to the single-shape path.

This collapses the per-shape launch overhead (~30-100 us on MI300X
for small shapes) from ``B`` launches to 1, which is the structural
cause of low ratios on launch-overhead-bound batched residuals
(e.g. 2048^3 fp16 B=4 ratio 0.30; 1024^3 bf16 B=8 ratio 0.07 with
the prior Python-host-loop path).
"""

import triton
import triton.language as tl

from tritonblas.kernels.stages import (
    GemmContext,
    make_schedule_context,
    make_input_view,
    make_output_view,
    make_scale_view,
    make_bias_view,
)


@triton.jit()
def batched_persistent_matmul(
    A,
    B,
    C,
    A_scale_ptr,
    B_scale_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_a_batch,
    stride_am,
    stride_ak,
    stride_b_batch,
    stride_bk,
    stride_bn,
    stride_c_batch,
    stride_cm,
    stride_cn,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    QUANTIZED: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = True,
):
    """Batched persistent GEMM, one program per (tile, batch).

    Grid is ``(num_tiles_per_batch, B)``.

    Strides for the batch dimension are passed explicitly because
    contiguous-batched and stacked-batched tensors have different
    leading strides; carrying ``stride_*_batch`` keeps the entrypoint
    layout-agnostic.
    """
    # ------------------------------------------------------------------
    # Per-batch pointer base
    # ------------------------------------------------------------------
    # int64 cast on batch_id is mandatory: B * stride_*_batch can exceed
    # 2^31 for moderately large GEMMs (e.g. B=8, M=N=K=2048, fp16 →
    # 8 * 4_194_304 * 2 B = 64 MiB which fits in int32 but the int32
    # multiplication can overflow with intermediate products if Triton
    # picks int32 for ``pid_b`` (it does by default for ``program_id``).
    # The promotion is also necessary for any future quantised batched
    # path with much larger per-batch strides.
    pid_b = tl.program_id(1).to(tl.int64)
    A_b = A + pid_b * stride_a_batch
    B_b = B + pid_b * stride_b_batch
    C_b = C + pid_b * stride_c_batch
    # Scale pointers and bias are not currently used by the batched
    # entrypoint (matmul_a8w8 / addmm have not yet been batched), but
    # we keep the parameter slots so the kernel signature lines up
    # with the single-shape persistent_matmul and any future
    # batched-quantised path can be added without re-plumbing the
    # signature. Guard with QUANTIZED/BIAS so unused tensors are not
    # re-bound.
    if QUANTIZED:
        # Per-batch scale pointers: assume scales are also batched
        # and contiguous along the batch dim (M scales per batch for A,
        # N scales per batch for B). If scales are shared across the
        # batch the caller should pass stride_scale_batch=0, which we
        # currently do not expose; use single-shape matmul_a8w8 for
        # quantised batched until that plumbing is added.
        A_scale_b = A_scale_ptr + pid_b * M
        B_scale_b = B_scale_ptr + pid_b * N
    else:
        A_scale_b = A_scale_ptr
        B_scale_b = B_scale_ptr
    if BIAS:
        # Bias is per-row of C; assume per-batch contiguous (length N
        # per batch). Same caveat as scales above.
        bias_b = bias_ptr + pid_b * N
    else:
        bias_b = bias_ptr

    # Determine accumulator dtype based on output type
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    # ------------------------------------------------------------------
    # Build views (single-shape, base pointers already batch-shifted)
    # ------------------------------------------------------------------
    tensorA = make_input_view(A_b, M, K, stride_am, stride_ak)
    tensorB = make_input_view(B_b, K, N, stride_bk, stride_bn)
    tensorC = make_output_view(C_b, M, N, stride_cm, stride_cn)

    scale_view = make_scale_view(A_scale_b, B_scale_b, M, N) if A_scale_ptr is not None else None
    bias_view = make_bias_view(bias_b, N, stride_bias) if BIAS else None

    ctx = GemmContext(
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        NUM_SMS, NUM_XCDS,
        GROUP_SIZE_M, CHUNK_SIZE,
        CACHE_MODIFIER_A, CACHE_MODIFIER_B,
        acc_dtype, ALLOW_TF32, EVEN_K, QUANTIZED,
    )

    sched = make_schedule_context(M, N, K, ctx)

    # ------------------------------------------------------------------
    # Single-tile-per-program walk over the M*N grid for this batch
    # ------------------------------------------------------------------
    # Unlike the single-shape persistent kernel which uses
    # NUM_SMS = total_tiles to make the schedule degenerate to one tile
    # per program, we still rely on the schedule's cleanup semantics
    # but only iterate once: the (tile, batch) grid is sized so each
    # program owns exactly one tile.
    start_tile, total_tiles, stride = sched.persistent_tile_range()
    for tile_id in range(start_tile, total_tiles, stride):
        out_tile = sched.get_tile_from_idx(tile_id)
        acc = ctx.reduce_axis(tensorA, tensorB, out_tile)
        tensorC.store(acc, out_tile, scale=scale_view, bias=bias_view)
