##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

"""
Gluon v9-style FP16 GEMM kernel: 128x256x64 tiles on gfx950 (MI350/MI355).

Tile shape rationale
--------------------
The standard v9 kernel uses 256x256x64 tiles. This variant uses 128x256x64
(tall-N, short-M) for workloads where M is small but N is large, giving better
occupancy and reducing wasted work on the M dimension.

Layout derivation
-----------------
BLOCK_M=128, BLOCK_N=256, BLOCK_K=64.

The "sliceMN" technique splits the tile into 4 quadrants:
  - Half-M = BLOCK_M // 2 = 64
  - Half-N = BLOCK_N // 2 = 128
  - Each accumulator quadrant is (64, 128)

MFMA config: 16x16x32 (version=4), warps_per_cta=[2,2], 4 warps total.
Each warp covers (64/2)x(128/2) = 32x64 of output in MFMA tiles.

Global load layouts use portable BlockedLayout instead of hand-tuned
DistributedLinearLayout bases:

  A layout: [64, 64] tile
    sizePerThread=[2, 8]  -> each thread loads 2 M-elements x 8 K-elements = 16 elements
    threadsPerWarp=[8, 8]  -> 64 threads per warp cover 16x64
    warpsPerCTA=[4, 1]     -> 4 warps stack along M: 4*16 = 64 rows
    order=[1, 0]           -> K-contiguous (coalesced along K for row-major A)

  B layout: [64, 128] tile
    sizePerThread=[8, 4]  -> each thread loads 8 K-elements x 4 N-elements = 32 elements
    threadsPerWarp=[8, 8]  -> warp tile = [8*8, 8*4] = [64, 32]
    warpsPerCTA=[1, 4]     -> 4 warps stack along N: CTA tile = [64, 4*32] = [64, 128]
    order=[0, 1]           -> K-contiguous (coalesced along K for column-major B)
    Note: 32 elements/thread (64 bytes in FP16) fully utilizes a cacheline per thread.

Shared memory uses SwizzledSharedLayout(1,1,1) which is portable across
gfx950 configurations and avoids bank conflicts for the MFMA access pattern.

Store layout: BlockedLayout for the (64, 128) quadrants, adjusted from the
256x256 variant's [1,8],[4,16],[4,1] pattern.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def get_pids(
    M,
    N,
    BM: gl.constexpr,
    BN: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    """XCD-aware PID remapping + GROUP_SIZE_M workgroup swizzling for L2 locality."""
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BM)
    num_pid_n = gl.cdiv(N, BN)

    if NUM_XCDS != 1:
        ## pid remapping on xcds
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


@gluon.jit
def v9_128x256_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,  #
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,  #
):
    """
    Gluon v9-style GEMM kernel with 128x256x64 tiles.

    Adapts the v9 "beyond the hot loop" design for an asymmetric tile:
      - 4 quadrant accumulators of shape (64, 128) each
      - Double-buffered async copy pipeline
      - XCD-aware PID remapping + workgroup swizzling
      - Interleaved epilogue stores

    Half-tile dimensions:
      Half-M = BLOCK_M // 2 = 64
      Half-N = BLOCK_N // 2 = 128
    """

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # =========================================================================
    # Layout definitions
    # =========================================================================

    # Global load layout for A: loads a [Half-M, BLOCK_K] = [64, 64] tile.
    #
    # BlockedLayout parameters:
    #   sizePerThread  = [2, 8]  -> each thread handles 2 rows x 8 cols = 16 elems
    #   threadsPerWarp = [8, 8]  -> warp tile = [8*2, 8*8] = [16, 64]
    #   warpsPerCTA    = [4, 1]  -> CTA tile = [4*16, 1*64] = [64, 64]
    #   order          = [1, 0]  -> K-major (contiguous along K dimension)
    #
    # This makes global loads coalesced when A is row-major (K-contiguous).
    gLoadLayoutA: gl.constexpr = gl.BlockedLayout([2, 8], [8, 8], [4, 1], [1, 0])

    # Global load layout for B: loads a [BLOCK_K, Half-N] = [64, 128] tile.
    #
    # BlockedLayout parameters:
    #   sizePerThread  = [8, 4]  -> each thread handles 8 rows x 4 cols = 32 elems
    #   threadsPerWarp = [8, 8]  -> warp tile = [8*8, 8*4] = [64, 32]
    #   warpsPerCTA    = [1, 4]  -> CTA tile = [1*64, 4*32] = [64, 128]
    #   order          = [0, 1]  -> K-major (contiguous along K dimension)
    #
    # For column-major B (K-contiguous), this yields coalesced global loads.
    # Each thread loads 32 elements (64 bytes in FP16), saturating a single
    # 64B cacheline fetch per thread.
    gLoadLayoutB: gl.constexpr = gl.BlockedLayout([8, 4], [8, 8], [1, 4], [0, 1])

    # Shared memory layouts: SwizzledSharedLayout(vec=1, perPhase=1, maxPhase=1)
    # is the portable baseline that avoids bank conflicts for MFMA-sized accesses.
    # The (1,1,1) parameters mean minimal swizzling -- sufficient for the 16x16x32
    # MFMA access pattern where each warp reads a contiguous 16-row slab.
    sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])

    # =========================================================================
    # Shared memory allocation: double-buffered (nBuffers=2)
    # =========================================================================
    # Each half-tile gets its own double-buffered SMEM allocation.
    # A: top [64, 64] and bottom [64, 64] halves of the [128, 64] A tile
    # B: left [64, 128] and right [64, 128] halves of the [64, 256] B tile
    nBuffers: gl.constexpr = 2
    smemA_top = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], sharedLayoutA
    )
    smemA_bot = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], sharedLayoutA
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )

    # =========================================================================
    # Offset computation for global loads
    # =========================================================================
    # A offsets: range over [Half-M, BLOCK_K] = [64, 64]
    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    # B offsets: range over [BLOCK_K, Half-N] = [64, 128]
    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    # Two sets of offsets for double-buffered pipeline:
    # base offsets (even K-steps) and _next offsets (odd K-steps, shifted by BLOCK_K)
    a_top_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_bot_offsets = a_top_offsets + (BLOCK_M // 2) * stride_am
    b_left_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b_right_offsets = b_left_offsets + (BLOCK_N // 2) * stride_bn

    a_top_offsets_next = a_top_offsets + BLOCK_K * stride_ak
    a_bot_offsets_next = a_bot_offsets + BLOCK_K * stride_ak
    b_left_offsets_next = b_left_offsets + BLOCK_K * stride_bk
    b_right_offsets_next = b_right_offsets + BLOCK_K * stride_bk

    # =========================================================================
    # MFMA and dot-operand layouts
    # =========================================================================
    # MFMA 16x16x32, version=4 (CDNA4/gfx950), transposed=True
    # warps_per_cta=[2,2]: 4 warps in a 2x2 grid
    #   - Each warp handles (Half-M/2) x (Half-N/2) = 32 x 64 output elements
    #   - The 2x2 warp grid covers the full 64 x 128 quadrant
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    # =========================================================================
    # Accumulator initialization: 4 quadrants, each (Half-M, Half-N) = (64, 128)
    # =========================================================================
    #   tl = top-left:     A_top * B_left   -> C[0:64,   0:128]
    #   tr = top-right:    A_top * B_right  -> C[0:64, 128:256]
    #   bl = bottom-left:  A_bot * B_left   -> C[64:128,  0:128]
    #   br = bottom-right: A_bot * B_right  -> C[64:128,128:256]
    acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)

    # =========================================================================
    # Prologue: prefetch first two K-steps into both double-buffer slots
    # =========================================================================
    # Buffer 0: K-step 0
    g_idx = 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_base, b_left_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(g_idx), a_base, a_top_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(g_idx), a_base, a_bot_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_base, b_right_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    # Buffer 1: K-step 1
    g_idx = 1
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(g_idx), b_base, b_left_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(g_idx), a_base, a_top_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(g_idx), a_base, a_bot_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(g_idx), b_base, b_right_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    # Advance base pointers past the two prefetched K-steps
    a_base += BLOCK_K * stride_ak * 2
    b_base += BLOCK_K * stride_bk * 2

    # Wait for B_left and A_top from buffer 0 (groups 0 and 1 of 8 total)
    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)

    gl.assume(iterMax > 3)

    # =========================================================================
    # Main loop: process K in steps of 2*BLOCK_K (unrolled by 2)
    # =========================================================================
    # Each iteration processes two K-steps:
    #   Sub-iter 0: consume buffer 0, prefetch next even K-step into buffer 0
    #   Sub-iter 1: consume buffer 1, prefetch next odd K-step into buffer 1
    for k in range(0, iterMax - 2, 2):

        ## =============================================================
        ## Sub-iteration 0: consume buffer 0, prefetch into buffer 0
        ## =============================================================

        ########################################
        ## Region 0: C_tl += A_top * B_left
        ########################################
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl)

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(0).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_left_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 1: C_bl += A_bot * B_left
        ########################################
        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl)

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(0).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_top_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 2: C_tr += A_top * B_right
        ########################################
        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr)

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(1).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base, a_bot_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 3: C_br += A_bot * B_right
        ########################################
        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br)

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(1).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_base, b_right_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ## =============================================================
        ## Sub-iteration 1: consume buffer 1, prefetch into buffer 1
        ## Uses _next offsets (odd K-step within the double-buffer pair).
        ## =============================================================

        ########################################
        ## Region 0: C_tl += A_top * B_left
        ########################################
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl)

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(1).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(1), b_base, b_left_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 1: C_bl += A_bot * B_left
        ########################################
        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl)

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(1).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_top.index(1), a_base, a_top_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 2: C_tr += A_top * B_right
        ########################################
        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr)

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(0).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_bot.index(1), a_base, a_bot_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 3: C_br += A_bot * B_right
        ########################################
        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br)

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(0).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(1), b_base, b_right_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2

    # =========================================================================
    # Epilogue: drain pipeline + interleaved stores
    # =========================================================================
    # Store layout for (Half-M, Half-N) = (64, 128) quadrants.
    #
    # Reuses the v9 store layout: BlockedLayout([1,8], [4,16], [4,1], [1,0]).
    # Per-warp tile = [4*1, 16*8] = [4, 128] -> 4 warps along M -> [16, 128].
    # The layout covers 16 M-rows per store pass; the remaining M-rows (64 total)
    # are distributed across registers by the MFMA layout. convert_layout handles
    # the MFMA-to-blocked remapping, and buffer_store emits all the stores.
    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])

    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_tl_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_tr_offsets = c_tl_offsets + (BLOCK_N // 2) * stride_cn
    c_bl_offsets = c_tl_offsets + (BLOCK_M // 2) * stride_cm
    c_br_offsets = c_bl_offsets + (BLOCK_N // 2) * stride_cn

    ## Iteration iterMax - 2: same 4-region MFMA pattern, no new async copies
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl)
    gl.amd.cdna4.async_copy.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = smemA_bot.index(l_idx).load(dotOpLayoutA)

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl)
    gl.amd.cdna4.async_copy.wait_group(4)
    b_right = smemB_right.index(l_idx).load(dotOpLayoutB)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr)
    gl.amd.cdna4.async_copy.wait_group(3)
    g_idx = 1 - l_idx
    b_left = smemB_left.index(g_idx).load(dotOpLayoutB)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br)
    gl.amd.cdna4.async_copy.wait_group(2)
    a_top = smemA_top.index(g_idx).load(dotOpLayoutA)

    ## Iteration iterMax - 1: final MFMAs + interleaved stores
    ## Natural-pipeline epilogue: each store follows its MFMA with one
    ## MFMA cycle of gap, yielding uniform MFMA-store interleaving.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl)
    gl.amd.cdna4.async_copy.wait_group(1)
    a_bot = smemA_bot.index(g_idx).load(dotOpLayoutA)

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl)
    gl.amd.cdna4.async_copy.wait_group(0)
    b_right = smemB_right.index(g_idx).load(dotOpLayoutB)

    # Store top-left quadrant while top-right MFMA is in flight
    c_tl = acc_tl.to(a_ptr.dtype.element_ty)
    c_tl = gl.convert_layout(c_tl, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_tl_offsets, stored_value=c_tl)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr)

    # Store bottom-left quadrant while bottom-right MFMA is in flight
    c_bl = acc_bl.to(a_ptr.dtype.element_ty)
    c_bl = gl.convert_layout(c_bl, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_bl_offsets, stored_value=c_bl)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br)

    # Store top-right quadrant
    c_tr = acc_tr.to(a_ptr.dtype.element_ty)
    c_tr = gl.convert_layout(c_tr, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_tr_offsets, stored_value=c_tr)

    # Store bottom-right quadrant
    c_br = acc_br.to(a_ptr.dtype.element_ty)
    c_br = gl.convert_layout(c_br, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_br_offsets, stored_value=c_br)


def matmul(a, b, c=None):
    """
    FP16/BF16 GEMM using the 128x256x64 Gluon v9-style kernel on gfx950.

    Args:
        a: [M, K] input tensor (row-major, contiguous)
        b: [K, N] input tensor
        c: optional [M, N] output tensor (allocated if None)

    Returns:
        c: [M, N] result tensor
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
    num_warps = 4

    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    NUM_XCDS = 8
    GROUP_SIZE_M = 4

    v9_128x256_kernel[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GRID_MN=GRID_MN,
        NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
    )
    return c
