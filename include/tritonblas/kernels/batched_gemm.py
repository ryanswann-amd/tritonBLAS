# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
True batched GEMM kernel.

Computes ``C[b] = A[b] @ B[b]`` for ``b`` in 0..BATCH-1 in a single grid
launch.  Replaces the host-side Python loop
``[matmul(a[i], b[i]) for i in range(batch)]`` which previously serialised
launches and prevented the GPU from overlapping batches.

Design:
- Persistent grid of ``NUM_SMS`` workgroups (sized to the hardware CU
  count).  Each workgroup iterates the global tile space
  ``[0, BATCH * tiles_per_batch)`` in stride-NUM_SMS chunks.
- Tile-id decoder splits the global id into ``(pid_b, pid_m, pid_n)``;
  the (pid_m, pid_n) ordering uses the same group-M reorder as the
  single-shot kernel for L2 locality.
- ``chiplet_transform_chunked`` applies the MI300X XCD-aware chunked
  remap (matches what ``persistent_matmul`` uses) so consecutive XCDs
  stay close in tile-id space.
- Per-batch base pointers advance via ``stride_ab`` / ``stride_bb`` /
  ``stride_cb``.  The kernel does not assume contiguous batches.
"""

import triton
import triton.language as tl

from .stages.indexing.pid_transforms import chiplet_transform_chunked


@triton.jit()
def batched_matmul(
    A,
    B,
    C,
    M,
    N,
    K,
    BATCH,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr = True,
):
    """
    Persistent batched GEMM kernel.

    Grid = (NUM_SMS,).  Each workgroup iterates the global tile space
    [0, BATCH * cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)) in stride-NUM_SMS
    chunks.
    """
    pid = tl.program_id(0)
    if NUM_XCDS > 1:
        pid = chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tiles_per_batch = num_pid_m * num_pid_n
    total_tiles = BATCH * tiles_per_batch

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ab > 0)
    tl.assume(stride_bb > 0)
    tl.assume(stride_cb > 0)

    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        pid_b = tile_id // tiles_per_batch
        pid_in_batch = tile_id % tiles_per_batch

        # L2-friendly group-M reorder of (pid_m, pid_n) within each batch slice.
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid_in_batch // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid_in_batch % num_pid_in_group) % group_size_m)
        pid_n = (pid_in_batch % num_pid_in_group) // group_size_m

        tl.assume(pid_b >= 0)
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        A_b = A + pid_b * stride_ab
        B_b = B + pid_b * stride_bb
        C_b = C + pid_b * stride_cb

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        A_BASE = A_b + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B_b + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1
        tl.assume(loop_k > 0)

        for k in range(0, loop_k):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)))
            if stride_bk == 1:
                b_tile = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                b_tile = tl.load(tl.multiple_of(B_BASE, (1, 16)))

            acc += tl.dot(a, b_tile, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk_tail = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_TAIL = A_b + rm[:, None] * stride_am + rk_tail[None, :] * stride_ak
            B_TAIL = B_b + rk_tail[:, None] * stride_bk + rn[None, :] * stride_bn
            if stride_ak == 1:
                A_TAIL = tl.multiple_of(A_TAIL, (1, 16))
            else:
                A_TAIL = tl.multiple_of(A_TAIL, (16, 1))
            if stride_bk == 1:
                B_TAIL = tl.multiple_of(B_TAIL, (16, 1))
            else:
                B_TAIL = tl.multiple_of(B_TAIL, (1, 16))
            a = tl.load(A_TAIL, mask=rk_tail[None, :] < K, other=0.0)
            b_tile = tl.load(B_TAIL, mask=rk_tail[:, None] < K, other=0.0)
            acc += tl.dot(a, b_tile, allow_tf32=ALLOW_TF32)

        c = acc.to(C.type.element_ty)

        rm_o = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn_o = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm_o = tl.max_contiguous(tl.multiple_of(rm_o, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn_o = tl.max_contiguous(tl.multiple_of(rn_o, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_o[:, None] < M) & (rn_o[None, :] < N)
        C_OUT = C_b + rm_o[:, None] * stride_cm + rn_o[None, :] * stride_cn
        tl.store(C_OUT, c, c_mask)
