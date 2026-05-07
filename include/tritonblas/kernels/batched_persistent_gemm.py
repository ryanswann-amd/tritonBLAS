# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Batched persistent GEMM kernel using tritonblas aggregates.

This kernel implements a true 3D-grid batched matmul: the launch grid is
(total_tiles_per_batch, batch_size), and each program processes one (tile,
batch) work unit. Compared to a Python-level loop over the batch dimension,
this fuses the batch into a single grid and eliminates per-batch kernel-launch
overhead.

The tile/scheduling logic is identical to the rank-2 persistent kernel; the
only addition is a per-batch base-pointer offset computed from
``tl.program_id(1)`` and the user-supplied batch strides.
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
    A_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    B_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    bias_ptr,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_bias,
    stride_a_scale_b,
    stride_b_scale_b,
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
    BROADCAST_BIAS: tl.constexpr = True,
    QUANTIZED: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = True,
):
    """
    Batched persistent GEMM kernel — fused 3D launch grid.

    **Why this kernel exists.** A Python-level loop over the batch dim pays one
    kernel-launch latency per batch element. For shapes where each per-batch
    matmul completes in ~hundreds of microseconds, launch overhead dominates and
    starves the GPU (e.g. ``B=4, M=N=K=2048`` fp16 and ``B=8, M=N=K=1024``
    bf16, two of the most expensive shapes in the batched-residual sweep).
    This kernel folds every (batch, tile) work unit into a single grid so the
    GPU sees one launch and B× the parallelism.

    **Launch grid contract.** The host launches with
    ``grid = (total_tiles_per_batch, batch_size)``:

      * ``tl.program_id(0)`` — index of the (M, N)-tile *within one batch
        element*. The persistent scheduler in :mod:`stages` consumes this exactly
        as it does in the rank-2 kernel; no batch-aware logic lives in the
        scheduler itself.
      * ``tl.program_id(1)`` — index of the batch element this program owns.
        Each program adds ``program_id(1) * stride_*b`` to its A/B/C base
        pointers (and to the optional A/B-scale pointers) so it operates on its
        own per-batch slice.

    **Stride-0 broadcast convention.** Stride parameters mirror the rank-2
    kernel with an extra *batch stride* per tensor (``stride_ab``, ``stride_bb``,
    ``stride_cb``, ``stride_a_scale_b``, ``stride_b_scale_b``). Setting any of
    these to ``0`` is the documented broadcast signal: every batch program then
    reads from the same base address, which is exactly what is needed to reuse
    a rank-2 operand across a rank-3 batch (e.g. shared weights × per-batch
    activations). The host (:func:`tritonblas.matmul.batched_persistent_matmul_lt`)
    sets ``stride_*b = 0`` when the corresponding operand is rank-2.

    **What is NOT batched.** Per-tensor row/column strides
    (``stride_am``, ``stride_ak``, ``stride_bk``, ``stride_bn``,
    ``stride_cm``, ``stride_cn``) are read from the underlying tensor's
    ``stride()`` and so handle arbitrary contiguous-or-non-contiguous layouts
    inside one batch element. The kernel makes no assumption that any input is
    contiguous; only that the row/column strides are sensible.

    **Edge case ``batch_size == 1``.** The grid degenerates to
    ``(total_tiles, 1)``. Only ``program_id(1) == 0`` ever runs, so the batch
    stride is multiplied by 0 and is a no-op — the kernel produces the same
    output as the rank-2 path.
    """

    # Determine accumulator dtype based on output type
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    # ════════════════════════════════════════════════════════════════════════
    # BATCH OFFSET — single fused 3D launch grid
    # ════════════════════════════════════════════════════════════════════════
    bid = tl.program_id(1)
    A_b = A + bid * stride_ab
    B_b = B + bid * stride_bb
    C_b = C + bid * stride_cb

    # Optional epilogue tensor offsets. ``stride_*_b`` may be 0 to broadcast.
    if QUANTIZED:
        A_scale_b = A_scale_ptr + bid * stride_a_scale_b
        B_scale_b = B_scale_ptr + bid * stride_b_scale_b
    else:
        A_scale_b = A_scale_ptr
        B_scale_b = B_scale_ptr

    if BIAS:
        if BROADCAST_BIAS:
            bias_b = bias_ptr
        else:
            bias_b = bias_ptr + bid * N
    else:
        bias_b = bias_ptr

    # ════════════════════════════════════════════════════════════════════════
    # CREATE MATRIX VIEWS
    # ════════════════════════════════════════════════════════════════════════
    tensorA = make_input_view(A_b, M, K, stride_am, stride_ak)
    tensorB = make_input_view(B_b, K, N, stride_bk, stride_bn)
    tensorC = make_output_view(C_b, M, N, stride_cm, stride_cn)

    # ════════════════════════════════════════════════════════════════════════
    # CREATE EPILOGUE VIEWS (optional scale and bias)
    # ════════════════════════════════════════════════════════════════════════
    scale_view = make_scale_view(A_scale_b, B_scale_b, M, N) if QUANTIZED else None
    bias_view = make_bias_view(bias_b, N, stride_bias) if BIAS else None

    # ════════════════════════════════════════════════════════════════════════
    # CONSTRUCT GEMM CONTEXT
    # ════════════════════════════════════════════════════════════════════════
    ctx = GemmContext(
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        NUM_SMS, NUM_XCDS,
        GROUP_SIZE_M, CHUNK_SIZE,
        CACHE_MODIFIER_A, CACHE_MODIFIER_B,
        acc_dtype, ALLOW_TF32, EVEN_K, QUANTIZED,
    )

    # ════════════════════════════════════════════════════════════════════════
    # SCHEDULE — uses program_id(0) for tile selection inside one batch
    # ════════════════════════════════════════════════════════════════════════
    sched = make_schedule_context(M, N, K, ctx)

    start_tile, total_tiles, stride = sched.persistent_tile_range()
    for tile_id in range(start_tile, total_tiles, stride):
        out_tile = sched.get_tile_from_idx(tile_id)
        acc = ctx.reduce_axis(tensorA, tensorB, out_tile)
        tensorC.store(acc, out_tile, scale=scale_view, bias=bias_view)
