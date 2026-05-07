# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Batched persistent GEMM kernel.

This kernel is the batched counterpart of ``persistent_gemm_monolithic``: it adds
a batch dimension as an outer ``program_id`` so a single kernel launch processes
``BATCH`` independent (M, K) x (K, N) GEMMs.  Per-batch base pointers are
computed from the supplied ``stride_ab/stride_bb/stride_cb`` strides, allowing
both contiguous (B, M, K) layouts and broadcasted rank-2 inputs (stride==0).

The intra-batch tile schedule mirrors the non-batched persistent kernel so the
existing Origami tile-selection heuristic can be reused without modification.

Stride-zero broadcasting (no separate code path required)
---------------------------------------------------------
The batched dispatcher in ``matmul.py:_normalize_bmm_strides`` returns
``stride_b == 0`` for rank-2 operands that should be broadcast across the batch
dimension (e.g. (M, K) x (B, K, N) — A is shared across all batches).  Inside
this kernel the per-batch base pointer is computed as

    A_BATCH = A + pid_b.to(tl.int64) * stride_ab

When ``stride_ab == 0`` every batch reads from the same base pointer, which is
exactly the broadcast semantics torch.matmul/torch.bmm define.  No explicit
``if BROADCAST_A`` constexpr branch is needed: the multiplication just yields
zero and the existing per-tile load addresses are correct.  This keeps the
kernel monomorphic across the broadcast/non-broadcast cases — a single compiled
kernel handles both, avoiding compile-time fan-out by 2^N where N is the number
of broadcast-capable operands.

The ``tl.assume(stride_ab >= 0)`` (vs ``> 0`` for the row/col strides) is the
only acknowledgement of the stride-zero case; relaxing the assumption matters
only for the address-generation simplifier, not for the schedule.
"""

import triton
import triton.language as tl

from .stages.indexing.pid_transforms import chiplet_transform_chunked


@triton.jit()
def batched_persistent_matmul(
    A,
    B,
    C,
    A_scale_ptr,        # Optional: per-batch scales for QUANTIZED path; None otherwise.
    B_scale_ptr,
    bias_ptr,           # Optional: per-batch bias vector (length M).
    M,
    N,
    K,
    stride_ab,          # bytes (in elements) between consecutive batches of A.
    stride_am,
    stride_ak,
    stride_bb,          # bytes (in elements) between consecutive batches of B.
    stride_bk,
    stride_bn,
    stride_cb,          # bytes (in elements) between consecutive batches of C.
    stride_cm,
    stride_cn,
    stride_bias_b,      # bytes (in elements) between consecutive batches of bias.
    stride_bias,
    stride_a_scale_b,   # bytes (in elements) between batches of A_scale.
    stride_b_scale_b,   # bytes (in elements) between batches of B_scale.
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
    QUANTIZED: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = True,
):
    """Single-launch batched GEMM.

    Grid is (NUM_SMS, BATCH).  ``program_id(0)`` is the persistent SM/wg slot,
    ``program_id(1)`` is the batch index.  Per-batch pointer offsets are added to
    the base pointers; the inner persistent tile loop is identical to the
    non-batched kernel so the same Origami tile heuristic remains valid.
    """
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ab >= 0)
    tl.assume(stride_bb >= 0)
    tl.assume(stride_cb > 0)

    # Per-batch base pointer offsets.
    A_BATCH = A + pid_b.to(tl.int64) * stride_ab
    B_BATCH = B + pid_b.to(tl.int64) * stride_bb
    C_BATCH = C + pid_b.to(tl.int64) * stride_cb

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A_BATCH + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B_BATCH + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_off = bias_ptr + pid_b.to(tl.int64) * stride_bias_b + rm * stride_bias
            bias = tl.load(bias_off, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1
        tl.assume(loop_k > 0)

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

            if QUANTIZED:
                acc += tl.dot(a, b, input_precision="ieee")
            else:
                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A_BATCH + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B_BATCH + rk[:, None] * stride_bk + rn[None, :] * stride_bn
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

            if QUANTIZED:
                acc += tl.dot(a, b, input_precision="ieee")
            else:
                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

        if QUANTIZED:
            rm_A_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
            rn_B_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
            A_scale_b = A_scale_ptr + pid_b.to(tl.int64) * stride_a_scale_b
            B_scale_b = B_scale_ptr + pid_b.to(tl.int64) * stride_b_scale_b
            A_scale = tl.load(A_scale_b + rm_A_scale)
            B_scale = tl.load(B_scale_b + rn_B_scale)
            acc *= A_scale[:, None] * B_scale[None, :]

        if BIAS:
            if QUANTIZED:
                bias_float = bias.to(tl.float32)
                c = acc + bias_float[:, None]
                c = c.to(C.type.element_ty)
            else:
                c = acc.to(C.type.element_ty)
                c += bias[:, None]
        else:
            c = acc.to(C.type.element_ty)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C_BATCH + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)
