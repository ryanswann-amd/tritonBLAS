# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
K-1698 prototype: persistent split-K GEMM combining the three remediation axes
that K-1655 falsified in isolation on the M=N=4096 K-COMPLEMENT cohort:

  * BLOCK_SIZE_K = 128  (vs the productionised 64)
  * num_stages   = 3    (vs the productionised 2)
  * persistent grid sized to NUM_CU with K-axis split

K-1681's PMC verdict was that the dominant bottleneck on these cells is
SQ_WAIT_INST_LDS / SQ_ACTIVE_INST_LDS  (A4 SCHEDULER_LDS at 10x-24x HBL),
i.e. LDS-issue back-pressure caused by shallow (~1.2-instruction) prefetch
staging in persistent_matmul.  K-1681 explicitly recommended NO single-axis
kernel patch.  This prototype tests whether the *combination* of the three
falsified axes can deepen the LDS prefetch window enough to drain WAIT/ACTIVE
back to HBL's 0.08-0.16 range -- because in isolation:

  * BK=64->128  doubles per-iter LDS bytes but halves loop_k -> in-flight
                LDS ops should rise from ~1.2 to ~2.4 if num_stages permits.
  * NS=2->3     adds a third pipeline buffer -> deeper double-buffering
                (one in-flight load can overlap with two MFMA tiles).
  * split-K     splits K-loop across multiple persistent workers so each
                worker has fewer iterations but the same effective LDS
                buffering depth -- crucially, keeps the persistent-grid
                worker count saturated at NUM_CU even when M*N is small.

This is a partial-K / atomic-reduce variant: each work-item computes a
(BLOCK_M, BLOCK_N) tile across a K-chunk of width K/SPLIT_K and uses
tl.atomic_add into a fp32 workspace.  A separate writeback kernel converts
the workspace to the output dtype.  This avoids the streamk locks/P scheme
to keep the prototype minimal and to expose the LDS-pipeline behaviour
without inter-worker synchronisation noise.
"""

import triton
import triton.language as tl
import torch

from .stages.indexing.pid_transforms import chiplet_transform_chunked


@triton.jit()
def persistent_split_k_matmul(
    A,
    B,
    C_partial,        # fp32 workspace, shape (M, N) -- atomic-add target
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
    Persistent split-K GEMM.  Grid is (NUM_SMS,).  Each program iterates over
    work items (m_tile, n_tile, split_k_idx) in stride-NUM_SMS persistent
    fashion.  Partial accumulators are atomic-added into C_partial (fp32).
    A separate writeback kernel converts C_partial -> output dtype + bias.
    """
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    total_work = total_tiles * SPLIT_K

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    iters_per_full_tile = tl.cdiv(K, BLOCK_SIZE_K)
    iters_per_split = iters_per_full_tile // SPLIT_K  # constexpr-ish; assume EVEN_K & SPLIT_K | iters

    for work_id in range(pid, total_work, NUM_SMS):
        tile_id = work_id // SPLIT_K
        split_id = work_id % SPLIT_K

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

        # K offset for this split: each split processes iters_per_split
        # contiguous BLOCK_SIZE_K iterations.
        k_offset_iters = split_id * iters_per_split
        k_offset = k_offset_iters * BLOCK_SIZE_K

        A_BASE = A + rm[:, None] * stride_am + (rk[None, :] + k_offset) * stride_ak
        B_BASE = B + (rk[:, None] + k_offset) * stride_bk + rn[None, :] * stride_bn

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, iters_per_split):
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
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        # atomic_add into the fp32 workspace at the (rm, rn) block.
        rm_full = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
        rn_full = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        c_mask = (rm_full[:, None] < M) & (rn_full[None, :] < N)
        Cp = (C_partial
              + rm_full[:, None] * stride_cm
              + rn_full[None, :] * stride_cn)
        tl.atomic_add(Cp, acc, mask=c_mask, sem="relaxed")


@triton.jit()
def split_k_writeback(
    C_partial,
    C_out,
    bias_ptr,
    M,
    N,
    stride_pm,
    stride_pn,
    stride_cm,
    stride_cn,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    P = (C_partial + rm[:, None] * stride_pm + rn[None, :] * stride_pn)
    val = tl.load(P, mask=mask, other=0.0)
    if BIAS:
        rn_b = rn % N
        b = tl.load(bias_ptr + rn_b * stride_bias, mask=rn < N, other=0.0)
        val = val + b[None, :].to(tl.float32)
    Cout = (C_out + rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(Cout, val.to(C_out.type.element_ty), mask=mask)
