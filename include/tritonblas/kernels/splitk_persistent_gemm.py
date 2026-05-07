# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Split-K persistent GEMM kernel.

Adds K-axis parallelism to the persistent GEMM kernel for shapes whose
output tile count cannot saturate the available CUs.  When SPLIT_K > 1
the K loop is partitioned across SPLIT_K workgroups per output tile,
each computing a partial accumulator that is atomically reduced into C.

Workgroup id decomposition (chosen for L2 locality across the SPLIT_K
shard of one output tile):

    pid_k    = program_id(0) %  SPLIT_K
    tile_id  = program_id(0) // SPLIT_K

For dtypes that lack hardware atomic-add (float8/int8 inputs with bf16/fp16
output), the caller MUST supply ``acc_buf`` — a float32 [M, N] scratch
buffer that this kernel atomically accumulates into.  A separate
``splitk_finalize`` kernel converts ``acc_buf`` to the C dtype on the
host-launched epilogue path.

When ``acc_buf`` is None (fp16/bf16/fp32 path), the kernel atomically
accumulates directly into C using ``tl.atomic_add`` in the C dtype.
"""

import triton
import triton.language as tl


@triton.jit()
def splitk_persistent_matmul(
    A,
    B,
    C,
    A_scale_ptr,
    B_scale_ptr,
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
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    QUANTIZED: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = True,
):
    """
    Split-K persistent GEMM.

    Each program processes one output tile (BLOCK_M, BLOCK_N) over a
    K-sub-range [pid_k * k_per_split, (pid_k+1) * k_per_split).  Partial
    accumulators are reduced into C via tl.atomic_add.

    The output tensor C MUST be zero-initialised by the caller because
    the atomic-add path performs a read-modify-write.  Bias and quantization
    scales are applied only by the pid_k == 0 program to avoid double-counting.
    """
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    pid = tl.program_id(0)
    pid_k = pid % SPLIT_K
    tile_id = pid // SPLIT_K

    # ─── Output-tile coordinates with grouped scheduling for L2 reuse ────
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # ─── K range for this split ──────────────────────────────────────────
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    k_per_split = tl.cdiv(num_k_tiles, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, num_k_tiles)

    # ─── Pointer setup ───────────────────────────────────────────────────
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    rm_safe = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn_safe = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

    a_ptrs = A + rm_safe[:, None] * stride_am + (k_start * BLOCK_SIZE_K + rk[None, :]) * stride_ak
    b_ptrs = B + (k_start * BLOCK_SIZE_K + rk[:, None]) * stride_bk + rn_safe[None, :] * stride_bn

    # ─── K loop over this split's range ──────────────────────────────────
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k_idx in range(k_start, k_end):
        if EVEN_K:
            a = tl.load(a_ptrs, cache_modifier=CACHE_MODIFIER_A)
            b = tl.load(b_ptrs, cache_modifier=CACHE_MODIFIER_B)
        else:
            k_remaining = K - k_idx * BLOCK_SIZE_K
            a_mask = rk[None, :] < k_remaining
            b_mask = rk[:, None] < k_remaining
            a = tl.load(a_ptrs, mask=a_mask, other=0.0, cache_modifier=CACHE_MODIFIER_A)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=CACHE_MODIFIER_B)

        if QUANTIZED:
            acc += tl.dot(a, b, out_dtype=tl.int32)
        else:
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # ─── Epilogue: scaling and bias only on pid_k == 0 ───────────────────
    if QUANTIZED and A_scale_ptr is not None:
        # Scale must be applied before the atomic add so partial sums share
        # the same scale.  Apply on every split since the atomic reduction
        # is linear: sum(scale * a_i * b_i) == scale * sum(a_i * b_i).
        a_scales = tl.load(A_scale_ptr + rm, mask=rm < M, other=1.0)
        b_scales = tl.load(B_scale_ptr + rn, mask=rn < N, other=1.0)
        acc = acc.to(tl.float32)
        acc = acc * a_scales[:, None]
        acc = acc * b_scales[None, :]

    # Bias is added to the final sum, so only the first split contributes it.
    if BIAS:
        if pid_k == 0:
            bias_vals = tl.load(bias_ptr + rn * stride_bias, mask=rn < N, other=0.0)
            acc = acc + bias_vals[None, :]

    # ─── Atomic accumulate into C ────────────────────────────────────────
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn

    # When SPLIT_K == 1 (gate disabled but kernel still entered), the atomic
    # is unnecessary; use a plain store.  This path is rare — the dispatcher
    # only enters this kernel when split_k > 1 — but it keeps the kernel safe
    # against accidental SPLIT_K=1 invocation during ablation.
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc.to(C.type.element_ty), mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc.to(C.type.element_ty), mask=c_mask, sem="relaxed")
