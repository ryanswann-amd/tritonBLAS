# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Persistent GEMM variant tuned for small-M GEMV-like shapes (M <= 32).

The default persistent kernel sizes its grid at one program per output
tile.  When M is tiny the M-dimension contributes a single tile, so
``grid == cdiv(N, BLOCK_N)`` — typically 4-32 programs — which leaves
~270 of MI300X's 304 CUs idle and starves the device.

This variant maps one program per CU and splits each output tile along K
so every CU has work.  Programs accumulate partial sums into ``C`` via
``tl.atomic_add``; the host zero-fills ``C`` first when ``NUM_K_SPLITS > 1``.

Design choices:
  * ``grid = NUM_SMS = N_CU``   one program per CU
  * Each ``(pid_n, split_id)`` pair is a work unit; the persistent loop
    strides over ``num_pid_n * NUM_K_SPLITS`` units in chunks of ``NUM_SMS``
  * BLOCK_M is small (16 or 32) — masking the unused rows is cheaper
    than the wasted MFMAs that the generic kernel performs
  * NUM_K_SPLITS is chosen by the host so that ``num_pid_n * NUM_K_SPLITS``
    is at least ``NUM_SMS`` and ``iters_per_split >= 2`` (avoid degenerate splits)
  * fp16/bf16 atomic_add is used directly — gfx942 has hardware PK_ADD
    for both, so the precision penalty over a few partial sums is bounded
"""

import triton
import triton.language as tl


@triton.jit()
def persistent_matmul_small_m(
    A,
    B,
    C,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_K_SPLITS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    ALLOW_TF32: tl.constexpr = True,
):
    """Persistent small-M GEMM with K-splitting and atomic accumulation."""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_work = num_pid_n * NUM_K_SPLITS

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    iters_per_split = tl.cdiv(iters_per_tile, NUM_K_SPLITS)

    rm = tl.arange(0, BLOCK_SIZE_M)
    rk = tl.arange(0, BLOCK_SIZE_K)
    m_mask = rm < M

    # Persistent loop: each program processes a contiguous slice of work units
    for wid in range(pid, total_work, NUM_SMS):
        pid_n = wid // NUM_K_SPLITS
        split_id = wid % NUM_K_SPLITS

        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn_safe = rn % N
        rn_safe = tl.max_contiguous(tl.multiple_of(rn_safe, BLOCK_SIZE_N), BLOCK_SIZE_N)

        k_start = split_id * iters_per_split
        k_end = tl.minimum(k_start + iters_per_split, iters_per_tile)

        A_BASE = A + rm[:, None] * stride_am + (k_start * BLOCK_SIZE_K + rk[None, :]) * stride_ak
        B_BASE = B + (k_start * BLOCK_SIZE_K + rk[:, None]) * stride_bk + rn_safe[None, :] * stride_bn

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k_iter in range(k_start, k_end):
            if EVEN_K:
                if stride_ak == 1:
                    a = tl.load(
                        tl.multiple_of(A_BASE, (1, 16)),
                        mask=m_mask[:, None],
                        other=0.0,
                        cache_modifier=CACHE_MODIFIER_A,
                    )
                else:
                    a = tl.load(
                        tl.multiple_of(A_BASE, (16, 1)),
                        mask=m_mask[:, None],
                        other=0.0,
                        cache_modifier=CACHE_MODIFIER_A,
                    )
                if stride_bk == 1:
                    b = tl.load(
                        tl.multiple_of(B_BASE, (16, 1)),
                        cache_modifier=CACHE_MODIFIER_B,
                    )
                else:
                    b = tl.load(
                        tl.multiple_of(B_BASE, (1, 16)),
                        cache_modifier=CACHE_MODIFIER_B,
                    )
            else:
                global_k = k_iter * BLOCK_SIZE_K
                k_mask = (global_k + rk) < K
                a = tl.load(
                    A_BASE,
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0,
                    cache_modifier=CACHE_MODIFIER_A,
                )
                b = tl.load(
                    B_BASE,
                    mask=k_mask[:, None],
                    other=0.0,
                    cache_modifier=CACHE_MODIFIER_B,
                )

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        # Bias is applied only by the first split to avoid double-add
        if BIAS and split_id == 0:
            bias_ = bias_ptr + rn_safe * stride_bias
            bias = tl.load(bias_, mask=rn < N, other=0.0).to(tl.float32)
            acc += bias[None, :]

        c_val = acc.to(C.type.element_ty)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn

        if NUM_K_SPLITS == 1:
            tl.store(C_, c_val, mask=c_mask)
        else:
            tl.atomic_add(C_, c_val, mask=c_mask, sem="relaxed")
