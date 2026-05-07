# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Batched persistent GEMM kernel.

Single-launch batched GEMM: computes C[b] = A[b] @ B[b] for b in [0, BATCH).
A 2-D launch grid is used: axis 0 is the persistent tile id within a batch,
axis 1 is the batch index. Each program biases its A/B/C base pointers by
the per-batch strides (stride_az / stride_bz / stride_cz) before calling the
standard persistent GEMM body.

The kernel reuses the production stages framework (`GemmContext`,
`ScheduleContext`, `make_input_view`, `make_output_view`) so the inner loop is
identical to the rank-2 `persistent_matmul` kernel — meaning the autotuned /
Origami-selected blocking factors and per-tile compute throughput transfer
directly. The only addition over the rank-2 kernel is the per-batch pointer
biasing at entry, plus accepting `stride_az / stride_bz / stride_cz` (which
may be 0 to broadcast that operand across the batch dim).
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
    stride_az,
    stride_am,
    stride_ak,
    stride_bz,
    stride_bk,
    stride_bn,
    stride_cz,
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
    """Batched persistent GEMM with grid = (NUM_SMS, BATCH).

    The kernel is identical to the rank-2 ``persistent_matmul`` body except
    that it accepts an additional ``stride_az/_bz/_cz`` triple for the batch
    dimension and biases the A/B/C base pointers by ``batch_id * stride_*z``
    before constructing the per-batch matrix views. Each (program_id_0,
    program_id_1) pair therefore computes one tile of one batch slice; the
    inner GEMM loop and tile-scheduler logic are shared verbatim with the
    production rank-2 kernel through ``GemmContext`` / ``ScheduleContext``.

    Stride parameters:
      stride_az / stride_bz / stride_cz : per-batch (Z) strides in elements.
        Pass 0 to broadcast that operand across the batch dimension (e.g.
        a shared rank-2 ``B`` matrix multiplied against a rank-3 ``A``).
        The kernel performs the multiplication in 64-bit so that batches
        whose slice byte-size exceeds 2 GiB do not overflow.
      stride_am, stride_ak : A's M and K row/col strides per batch slice.
      stride_bk, stride_bn : B's K and N row/col strides per batch slice.
      stride_cm, stride_cn : C's M and N row/col strides per batch slice.

    Grid sizing (chosen by the Python-side dispatcher):
      ``NUM_SMS`` controls the persistent loop length per batch. The
      dispatcher picks one of: ``total_tiles`` (one program per tile),
      ``NUM_CUS`` (one persistent wave per batch), ``NUM_CUS//2``
      (lighter persistence to amortize tile scheduling), or
      ``total_tiles*2`` (oversubscribe to hide tail latency on small grids).
      See ``_BatchedKernelConfig.nsm_mode`` and the autotune sweep results
      in the K-678 workspace for the rationale per shape.
    """
    # Determine accumulator dtype based on output type.
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    # ════════════════════════════════════════════════════════════════════════
    # PER-BATCH POINTER BIASING
    # ════════════════════════════════════════════════════════════════════════
    # Multiply tl.int32 batch_id by an int64 stride to keep the offset 64-bit;
    # batches whose slice byte-size exceeds 2 GiB would otherwise overflow.
    batch_id = tl.program_id(1)
    A_b = A + batch_id.to(tl.int64) * stride_az
    B_b = B + batch_id.to(tl.int64) * stride_bz
    C_b = C + batch_id.to(tl.int64) * stride_cz

    # ════════════════════════════════════════════════════════════════════════
    # CREATE MATRIX VIEWS (using batch-biased pointers)
    # ════════════════════════════════════════════════════════════════════════
    tensorA = make_input_view(A_b, M, K, stride_am, stride_ak)
    tensorB = make_input_view(B_b, K, N, stride_bk, stride_bn)
    tensorC = make_output_view(C_b, M, N, stride_cm, stride_cn)

    # ════════════════════════════════════════════════════════════════════════
    # CREATE EPILOGUE VIEWS (optional scale and bias)
    # ════════════════════════════════════════════════════════════════════════
    scale_view = make_scale_view(A_scale_ptr, B_scale_ptr, M, N) if A_scale_ptr is not None else None
    bias_view = make_bias_view(bias_ptr, N, stride_bias) if BIAS else None

    # ════════════════════════════════════════════════════════════════════════
    # GEMM CONTEXT
    # ════════════════════════════════════════════════════════════════════════
    ctx = GemmContext(
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        NUM_SMS, NUM_XCDS,
        GROUP_SIZE_M, CHUNK_SIZE,
        CACHE_MODIFIER_A, CACHE_MODIFIER_B,
        acc_dtype, ALLOW_TF32, EVEN_K, QUANTIZED,
    )

    # ════════════════════════════════════════════════════════════════════════
    # SCHEDULE CONTEXT (per-batch tile schedule, shared across all batches)
    # ════════════════════════════════════════════════════════════════════════
    sched = make_schedule_context(M, N, K, ctx)

    # ════════════════════════════════════════════════════════════════════════
    # PERSISTENT LOOP: process this batch's tiles
    # ════════════════════════════════════════════════════════════════════════
    start_tile, total_tiles, stride = sched.persistent_tile_range()
    for tile_id in range(start_tile, total_tiles, stride):
        out_tile = sched.get_tile_from_idx(tile_id)
        acc = ctx.reduce_axis(tensorA, tensorB, out_tile)
        tensorC.store(acc, out_tile, scale=scale_view, bias=bias_view)
