# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Skinny-GEMV (undertile-aware) GEMM kernel.

When min(M, N) <= 32 — e.g. matvec / decode-shape GEMMs like 16x4096x4096 —
the general-purpose persistent kernel wastes most of its 128x128 tile because
either the M or N dimension is padded out to BLOCK with mask-zeros, so >87%
of the MFMA fragment is unused.  hipBLASLt picks a GEMV-specialized kernel for
these shapes.  This module provides the matching path for tritonblas:

  * Tile is 16 along the small dim (matches MFMA M=16) and 128 along the wide
    dim (matches a single wave of bf16 lanes).
  * Plain reduction-only K-loop, no persistent / chunked / chiplet rewriting.
  * num_warps=2, kpack=1 — the small tile cannot keep 8 warps fed.
"""

import triton
import triton.language as tl


@triton.jit()
def skinny_gemv_matmul(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Reduction-only K-loop GEMM tuned for one skinny dimension.

    Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N)).  No persistent loop, no
    chiplet remapping — for skinny shapes the total tile count is so small
    that those overheads dominate.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    mask_m = rm < M
    mask_n = rn < N

    A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    loop_k = tl.cdiv(K, BLOCK_SIZE_K)
    if not EVEN_K:
        loop_k -= 1
    for _ in range(0, loop_k):
        if stride_ak == 1:
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)), mask=mask_m[:, None], other=0.0)
        else:
            a = tl.load(tl.multiple_of(A_BASE, (16, 1)), mask=mask_m[:, None], other=0.0)
        if stride_bk == 1:
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)), mask=mask_n[None, :], other=0.0)
        else:
            b = tl.load(tl.multiple_of(B_BASE, (1, 16)), mask=mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A_BASE += BLOCK_SIZE_K * stride_ak
        B_BASE += BLOCK_SIZE_K * stride_bk

    if not EVEN_K:
        k_iter = loop_k
        rk_tail = k_iter * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = rk_tail < K
        A_TAIL = A + rm[:, None] * stride_am + rk_tail[None, :] * stride_ak
        B_TAIL = B + rk_tail[:, None] * stride_bk + rn[None, :] * stride_bn
        a = tl.load(A_TAIL, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(B_TAIL, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    c = acc.to(C.type.element_ty)
    c_mask = mask_m[:, None] & mask_n[None, :]
    C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(C_, c, mask=c_mask)
