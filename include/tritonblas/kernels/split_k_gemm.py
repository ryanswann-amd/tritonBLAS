"""Split-K GEMM kernel.

Standard data-parallel GEMM ranges programs over the (M, N) output tile grid.
For shapes where the output tile count (cdiv(M, BM) * cdiv(N, BN)) is much
smaller than the device CU count, the kernel underutilises hardware: most CUs
sit idle while a handful chew through the entire K dimension on their own.
The classic remedy is to split the K-loop across SPLIT_K parallel programs
that share the same (pid_m, pid_n) tile and atomically add their partial
sums into C.

This file provides ``split_k_matmul`` — a monolithic Triton kernel modelled
on persistent_gemm_monolithic.py but launched on a 3D grid (M-tiles, N-tiles,
SPLIT_K) where each program does cdiv(K, SPLIT_K) of work.  The host is
responsible for zero-initialising C before launch (the persistent_matmul_lt
wrapper handles this automatically when SPLIT_K > 1).
"""

import triton
import triton.language as tl


@triton.jit()
def split_k_matmul(
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
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    QUANTIZED: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = True,
):
    """Split-K GEMM kernel.

    Grid layout:  (cdiv(M, BM) * cdiv(N, BN), SPLIT_K)
        program_id(0)  → (pid_m, pid_n) tile (with optional grouped ordering)
        program_id(1)  → which K-slice this program owns

    Each program loads its slice of A[:, k_start:k_end] and B[k_start:k_end, :],
    accumulates into a register tile, then atomically adds (or directly stores
    when SPLIT_K == 1) into C[pid_m, pid_n].
    """
    pid = tl.program_id(0)
    pid_sk = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Grouped ordering for L2 reuse (same as persistent monolithic kernel)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # Each program walks SPLIT_K-strided chunks of the K-loop.
    # k tile index = pid_sk + i * SPLIT_K, total chunks = cdiv(K, BK).
    total_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    rk = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    # Number of full chunks this program is responsible for.
    # When EVEN_K and K % (BLOCK_SIZE_K * SPLIT_K) == 0 each program owns
    # exactly K / (BLOCK_SIZE_K * SPLIT_K) chunks; otherwise the last few
    # programs may own one fewer (handled by the runtime mask below).
    for k_idx in range(pid_sk, total_k_tiles, SPLIT_K):
        if EVEN_K:
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)),
                            cache_modifier=CACHE_MODIFIER_A)
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)),
                            cache_modifier=CACHE_MODIFIER_A)
            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)),
                            cache_modifier=CACHE_MODIFIER_B)
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)),
                            cache_modifier=CACHE_MODIFIER_B)
        else:
            k_offset = k_idx * BLOCK_SIZE_K
            mask_k = (k_offset + tl.arange(0, BLOCK_SIZE_K)) < K
            a = tl.load(A_BASE, mask=mask_k[None, :], other=0.0,
                        cache_modifier=CACHE_MODIFIER_A)
            b = tl.load(B_BASE, mask=mask_k[:, None], other=0.0,
                        cache_modifier=CACHE_MODIFIER_B)

        if QUANTIZED:
            acc += tl.dot(a, b, input_precision="ieee")
        else:
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

        # Stride forward by SPLIT_K * BLOCK_SIZE_K to skip slices owned by peers.
        A_BASE += SPLIT_K * BLOCK_SIZE_K * stride_ak
        B_BASE += SPLIT_K * BLOCK_SIZE_K * stride_bk

    if QUANTIZED:
        # Apply quant scales once per tile (not per K-chunk).
        rm_A_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
        rn_B_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
        A_scale = tl.load(A_scale_ptr + rm_A_scale)
        B_scale = tl.load(B_scale_ptr + rn_B_scale)
        acc *= A_scale[:, None] * B_scale[None, :]

    # Bias is added only by program pid_sk == 0 to avoid SPLIT_K-fold double-add.
    if BIAS:
        if pid_sk == 0:
            bias = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
            if QUANTIZED:
                bias_float = bias.to(tl.float32)
                acc = acc + bias_float[:, None]
            else:
                acc = acc + bias[:, None].to(acc_dtype)

    # Cast to output dtype.
    c = acc.to(C.type.element_ty)

    rm_out = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn_out = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rm_out = tl.max_contiguous(tl.multiple_of(rm_out, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn_out = tl.max_contiguous(tl.multiple_of(rn_out, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_mask = (rm_out[:, None] < M) & (rn_out[None, :] < N)
    C_ = C + rm_out[:, None] * stride_cm + rn_out[None, :] * stride_cn

    if SPLIT_K == 1:
        tl.store(C_, c, c_mask)
    else:
        tl.atomic_add(C_, c, mask=c_mask, sem="relaxed")
