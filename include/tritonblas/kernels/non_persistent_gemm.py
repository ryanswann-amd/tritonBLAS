# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Non-persistent GEMM kernel — direct grid-stride dispatch.

Used for K-6399 A/B test against persistent_gemm.py for shapes where
n_tiles ≤ NUM_SMS (e.g. 64x64x4096 → 1 tile). Eliminates:

* persistent loop (`for tile_id in range(start, total, stride)`)
* chiplet_transform_chunked() pid remap
* tile-id → group → (pid_m,pid_n) decomposition

A single tile per workgroup; (pid_m, pid_n) come straight from
`tl.program_id(0)` / `tl.program_id(1)` on a 2D launch grid.

Same accumulator + epilogue path as persistent_matmul, so any TFLOPS
delta is attributable to scheduler-related instructions only.
"""

import triton
import triton.language as tl

from tritonblas.kernels.stages import (
    Tile,
    GemmContext,
    make_input_view,
    make_output_view,
    make_scale_view,
    make_bias_view,
)


@triton.jit()
def non_persistent_matmul(
    A,
    B,
    C,
    A_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    B_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    stride_ak,
    stride_bk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,        # kept for API symmetry; unused
    NUM_XCDS: tl.constexpr,       # kept for API symmetry; unused
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    QUANTIZED: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = True,
):
    """
    Non-persistent GEMM kernel — one workgroup per output tile.

    Launch grid: ``(cdiv(M, BLOCK_SIZE_M), cdiv(N, BLOCK_SIZE_N))``

    Compared to ``persistent_matmul`` this kernel removes:

    * outer persistent loop and its stride
    * chiplet (XCD) pid remap
    * GROUP_SIZE_M re-ordering (linear pid_m, pid_n)

    Everything inside the K-loop and the epilogue is identical to the
    persistent kernel — uses the same ``GemmContext.reduce_axis`` and
    ``OutputView.store`` so any K-loop differences are not the variable
    under study.
    """
    # Determine accumulator dtype based on output type
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    # 2D dispatch — pid_m and pid_n come straight from program_id.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Matrix views (same as persistent path)
    tensorA = make_input_view(A, M, K, stride_am, stride_ak)
    tensorB = make_input_view(B, K, N, stride_bk, stride_bn)
    tensorC = make_output_view(C, M, N, stride_cm, stride_cn)

    scale_view = make_scale_view(A_scale_ptr, B_scale_ptr, M, N) if A_scale_ptr is not None else None
    bias_view = make_bias_view(bias_ptr, N, stride_bias) if BIAS else None

    # GemmContext encapsulates the K-loop. We pass NUM_SMS=1, NUM_XCDS=1
    # because this kernel does no scheduling — those fields are not used
    # by reduce_axis / reduce_tile, only by the persistent ScheduleContext.
    ctx = GemmContext(
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        1, 1,                      # num_sms, num_xcds (unused here)
        1, 1,                      # group_size_m, chunk_size (unused)
        CACHE_MODIFIER_A, CACHE_MODIFIER_B,
        acc_dtype, ALLOW_TF32, EVEN_K, QUANTIZED,
    )

    # Build the output tile and run the same K-loop as the persistent kernel.
    out_tile = Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
    acc = ctx.reduce_axis(tensorA, tensorB, out_tile)
    tensorC.store(acc, out_tile, scale=scale_view, bias=bias_view)
