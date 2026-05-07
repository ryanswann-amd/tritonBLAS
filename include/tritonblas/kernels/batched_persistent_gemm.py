# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Batched persistent GEMM kernel.

Fuses the batch dimension into the kernel grid so a rank-3 BMM
(C[b] = A[b] @ B[b]) executes as a single Triton launch with
grid = (BATCH * cdiv(M, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N),).
Per-batch base-pointer arithmetic is performed inside the kernel
via stride_az / stride_bz / stride_cz.

This eliminates the Tcold launch overhead paid by a Python-level
loop over single-GEMM kernel launches (root cause of the K-654
batched-residual gap surfaced in K-659).
"""

import triton
import triton.language as tl

from .stages.indexing.pid_transforms import chiplet_transform_chunked


@triton.jit()
def batched_persistent_matmul(
    A,
    B,
    C,
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
    BATCH: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    ALLOW_TF32: tl.constexpr = True,
):
    """
    Batched persistent GEMM kernel — one launch handles BATCH * MN tiles.

    The grid is data-parallel: each program handles exactly one (batch, M-tile, N-tile)
    triple. The batch index is decoded from the program id by integer division
    against the per-batch tile count.

    A: (BATCH, M, K)
    B: (BATCH, K, N)
    C: (BATCH, M, N)
    bias_ptr: optional (M,) per-row bias broadcast over batches
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tiles_per_batch = num_pid_m * num_pid_n

    # ─── decode (batch_id, tile_in_batch) from flat program id ─────────────────
    batch_id = pid // tiles_per_batch
    tile_id = pid - batch_id * tiles_per_batch

    # Skip out-of-range programs (the host launches BATCH * tiles_per_batch
    # programs, so this branch should be statically dead in the common path).
    if batch_id >= BATCH:
        return

    # Apply chiplet remap on the within-batch tile id only — keeps batches
    # placed independently across XCDs without crossing batch boundaries.
    if NUM_XCDS != 1:
        tile_id = chiplet_transform_chunked(tile_id, tiles_per_batch, NUM_XCDS, CHUNK_SIZE)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    # ─── grouped tile mapping (the standard Triton swizzle) ───────────────────
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # ─── per-batch base offsets ───────────────────────────────────────────────
    A_batch = A + batch_id.to(tl.int64) * stride_az
    B_batch = B + batch_id.to(tl.int64) * stride_bz
    C_batch = C + batch_id.to(tl.int64) * stride_cz

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rk = tl.arange(0, BLOCK_SIZE_K)
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    A_BASE = A_batch + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_BASE = B_batch + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    if BIAS:
        bias_ = bias_ptr + rm * stride_bias
        bias = tl.load(bias_, mask=rm < M, other=0.0)

    loop_k = tl.cdiv(K, BLOCK_SIZE_K)
    if not EVEN_K:
        loop_k -= 1

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in range(0, loop_k):
        if stride_ak == 1:
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_A)
        else:
            a = tl.load(tl.multiple_of(A_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_A)

        if stride_bk == 1:
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_B)
        else:
            b = tl.load(tl.multiple_of(B_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_B)

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A_BASE += BLOCK_SIZE_K * stride_ak
        B_BASE += BLOCK_SIZE_K * stride_bk

    if not EVEN_K:
        k = loop_k
        rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        A_BASE = A_batch + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B_batch + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        if stride_ak == 1:
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
        else:
            A_BASE = tl.multiple_of(A_BASE, (16, 1))

        if stride_bk == 1:
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
        else:
            B_BASE = tl.multiple_of(B_BASE, (1, 16))
        a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
        b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    if BIAS:
        c = acc.to(C.type.element_ty)
        c += bias[:, None]
    else:
        c = acc.to(C.type.element_ty)

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    C_ = C_batch + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(C_, c, c_mask)
