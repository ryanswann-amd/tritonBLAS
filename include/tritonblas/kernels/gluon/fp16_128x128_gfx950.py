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
Gluon v9-style FP16 GEMM kernel for gfx950 (MI350/MI355) with 128x128x64 tiles.

Adapted from the 256x256x64 v9 kernel (fp16_gfx950.py) for better occupancy
on small M/N shapes (< 1024). Key differences:

  - BLOCK_M=128, BLOCK_N=128, BLOCK_K=64 (was 256x256x64)
  - Half-tile is 64x64 (was 128x128)
  - Uses BlockedLayout for global loads (portable across Triton variants)
  - Uses SwizzledSharedLayout for shared memory (portable, no manual padding)
  - Same 4-quadrant sliceMN accumulator structure
  - Same double-buffered async copy pipeline
  - Same XCD-aware PID remapping and GROUP_SIZE_M swizzling

Layout math for 128x128x64:
  - 4 warps, wave64 (64 lanes/warp), 256 threads total
  - Half-M = 64, Half-N = 64
  - MFMA v4 [16,16,32] with warps_per_cta=[2,2]:
      Each warp covers 64/2=32 rows x 64/2=32 cols of the half-tile via
      multiple MFMA instructions. With 2x2 warp layout, the 4 warps tile
      the full 64x64 half-tile.
  - For the 4-quadrant strategy: each accumulator is 64x64 in mfmaLayout,
    and 4 of them tile the full 128x128 output.
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
def v9_128x128_kernel(
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
    128x128x64 v9-style GEMM kernel for gfx950.

    Uses the same sliceMN strategy as the 256x256 v9 kernel:
    - Loads half-tiles: A_top [64, 64], A_bot [64, 64], B_left [64, 64], B_right [64, 64]
    - Computes 4 quadrant accumulators: TL, BL, TR, BR (each 64x64)
    - Double-buffered async copy pipeline
    - Interleaved epilogue stores

    Global load layouts use BlockedLayout (portable).
    Shared memory uses SwizzledSharedLayout (portable, no manual padding calc).
    """

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # =========================================================================
    # Global load layout for A half-tile [BLOCK_M//2, BLOCK_K] = [64, 64]
    # =========================================================================
    # BlockedLayout(sizePerThread, threadsPerWarp, warpsPerCTA, order)
    #
    # For A [64, 64] with 4 warps (256 threads total), 128-bit loads (8 FP16):
    #   sizePerThread = [2, 8] -> each thread loads 2 rows x 8 cols = 16 elements
    #   threadsPerWarp = [8, 8] -> 64 threads arranged 8 in M x 8 in K
    #   warpsPerCTA = [2, 1]   -> 2 warps in M, 1 in K (only 2 warps load A)
    #   Coverage: M = 2 * 8 * 2 = 32, K = 8 * 8 * 1 = 64
    #   But we need M = 64, so with 4 warps: warpsPerCTA = [4, 1]
    #   Coverage: M = 2 * 8 * 4 = 64, K = 8 * 8 * 1 = 64. Correct!
    #   order = [1, 0] -> K is the fast-moving dimension (coalesced loads)
    # =========================================================================
    gLoadLayoutA: gl.constexpr = gl.BlockedLayout([2, 8], [8, 8], [4, 1], [1, 0])

    # =========================================================================
    # Global load layout for B half-tile [BLOCK_K, BLOCK_N//2] = [64, 64]
    # =========================================================================
    # For B [64, 64] with 4 warps, 128-bit loads (8 FP16):
    #   sizePerThread = [8, 2] -> each thread loads 8 rows x 2 cols = 16 elements
    #   threadsPerWarp = [8, 8] -> 64 threads arranged 8 in K x 8 in N
    #   warpsPerCTA = [1, 4]   -> 1 warp in K, 4 warps in N
    #   Coverage: K = 8 * 8 * 1 = 64, N = 2 * 8 * 4 = 64. Correct!
    #   order = [1, 0] -> N is the fast-moving dimension (coalesced loads)
    # =========================================================================
    gLoadLayoutB: gl.constexpr = gl.BlockedLayout([8, 2], [8, 8], [1, 4], [1, 0])

    # =========================================================================
    # Shared memory layouts: SwizzledSharedLayout is portable
    # =========================================================================
    # SwizzledSharedLayout(vec, perPhase, maxPhase, order)
    # vec=1: minimum vectorization
    # perPhase=1, maxPhase=1: minimal swizzle pattern (equivalent to row-major)
    # For better bank-conflict avoidance on actual hardware, these can be tuned.
    # A: [64, 64], K is inner -> order=[1, 0] (K contiguous in shared mem)
    # B: [64, 64], N is inner -> order=[1, 0] (N contiguous in shared mem)
    # =========================================================================
    sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    # =========================================================================
    # Allocate double-buffered shared memory for half-tiles
    # =========================================================================
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
    # Compute global load offsets for half-tiles
    # =========================================================================
    # A: offs_am indexes rows [0, BLOCK_M//2), offs_ak indexes cols [0, BLOCK_K)
    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    # B: offs_bn indexes cols [0, BLOCK_N//2), offs_bk indexes rows [0, BLOCK_K)
    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    # Two sets of offsets: base (even K-steps) and _next (odd K-steps).
    # This avoids recomputing offsets inside the loop (saves VALU).
    a_top_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_bot_offsets = a_top_offsets + BLOCK_M * stride_am // 2
    b_left_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b_right_offsets = b_left_offsets + BLOCK_N * stride_bn // 2

    a_top_offsets_next = a_top_offsets + BLOCK_K * stride_ak
    a_bot_offsets_next = a_bot_offsets + BLOCK_K * stride_ak
    b_left_offsets_next = b_left_offsets + BLOCK_K * stride_bk
    b_right_offsets_next = b_right_offsets + BLOCK_K * stride_bk

    # =========================================================================
    # MFMA layout and dot operand layouts
    # =========================================================================
    # AMDMFMALayout v4 with 16x16x32 instructions, warps_per_cta=[2,2]
    # This is identical to the 256x256 kernel. With 2x2 warps and 64x64 acc:
    #   Each warp covers a 32x32 region of the accumulator via multiple MFMAs.
    #   4 warps tile the full 64x64 half-tile accumulator.
    # =========================================================================
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    # =========================================================================
    # 4-quadrant accumulators, each 64x64 (half-M x half-N)
    # Together they tile the full 128x128 output tile:
    #   TL = [0:64,   0:64  ]    TR = [0:64,   64:128]
    #   BL = [64:128, 0:64  ]    BR = [64:128, 64:128]
    # =========================================================================
    acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)

    # =========================================================================
    # Prologue: prefetch first two K-iterations into double buffers
    # =========================================================================
    # Buffer 0: K-step 0 (even)
    g_idx = 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_base, b_left_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(g_idx), a_base, a_top_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(g_idx), a_base, a_bot_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_base, b_right_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    # Buffer 1: K-step 1 (odd) -- uses _next offsets
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

    # Wait for buffer 0's B_left and A_top to be ready (6 groups pending -> wait for 6)
    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)

    gl.assume(iterMax > 3)

    # =========================================================================
    # Main loop: process K in steps of 2 (unrolled by 2 for double buffering)
    # Each sub-iteration computes 4 quadrant MFMAs and issues 4 async copies.
    # =========================================================================
    for k in range(0, iterMax - 2, 2):

        ## =============================================================
        ## Sub-iteration 0: consume buffer 0, prefetch into buffer 0
        ## =============================================================

        ########################################
        ## Region 0: C_tl = DOT(a_top, b_left)
        ########################################
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl)

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(0).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_left_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 1: C_bl = DOT(a_bot, b_left)
        ########################################
        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl)

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(0).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_top_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 2: C_tr = DOT(a_top, b_right)
        ########################################
        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr)

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(1).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base, a_bot_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 3: C_br = DOT(a_bot, b_right)
        ########################################
        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br)

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(1).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_base, b_right_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ## =============================================================
        ## Sub-iteration 1: consume buffer 1, prefetch into buffer 1
        ## Uses _next offsets (odd K-step).
        ## =============================================================

        ########################################
        ## Region 0: C_tl = DOT(a_top, b_left)
        ########################################
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl)

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(1).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(1), b_base, b_left_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 1: C_bl = DOT(a_bot, b_left)
        ########################################
        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl)

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(1).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_top.index(1), a_base, a_top_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 2: C_tr = DOT(a_top, b_right)
        ########################################
        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr)

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(0).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_bot.index(1), a_base, a_bot_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 3: C_br = DOT(a_bot, b_right)
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
    # Epilogue: drain the last two K-iterations and store results
    # =========================================================================
    # Uses BlockedLayout for the global stores as well (portable).
    #
    # For C half-tile [64, 64] with 4 warps, 128-bit stores (8 FP16):
    #   sizePerThread = [1, 8] -> each thread stores 1 row x 8 cols
    #   threadsPerWarp = [4, 16] -> 4 rows x 16 cols per warp
    #   warpsPerCTA = [4, 1]    -> 4 warps in M, 1 in N
    #   Coverage: M = 1 * 4 * 4 = 16, N = 8 * 16 * 1 = 128
    #   That gives 16x128 -- too wide. We need 64x64.
    #
    #   Alternative: sizePerThread = [2, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1]
    #   Coverage: M = 2 * 8 * 4 = 64, N = 8 * 8 * 1 = 64. Correct!
    # =========================================================================
    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([2, 8], [8, 8], [4, 1], [1, 0])

    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_tl_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_tr_offsets = c_tl_offsets + BLOCK_N * stride_cn // 2
    c_bl_offsets = c_tl_offsets + BLOCK_M * stride_cm // 2
    c_br_offsets = c_bl_offsets + BLOCK_N * stride_cn // 2

    ## Iter iterMax - 2: same 4-region pattern as main loop, no new async copies
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

    ## Iter iterMax - 1: last K-step with interleaved epilogue stores
    ## Natural-pipeline: each store follows its MFMA with one MFMA cycle gap,
    ## yielding uniform MFMA-store interleaving to spread write traffic.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl)
    gl.amd.cdna4.async_copy.wait_group(1)
    a_bot = smemA_bot.index(g_idx).load(dotOpLayoutA)

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl)
    gl.amd.cdna4.async_copy.wait_group(0)
    b_right = smemB_right.index(g_idx).load(dotOpLayoutB)

    # Store TL quadrant while TR MFMA executes
    c_tl = acc_tl.to(a_ptr.dtype.element_ty)
    c_tl = gl.convert_layout(c_tl, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_tl_offsets, stored_value=c_tl)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr)

    # Store BL quadrant while BR MFMA executes
    c_bl = acc_bl.to(a_ptr.dtype.element_ty)
    c_bl = gl.convert_layout(c_bl, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_bl_offsets, stored_value=c_bl)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br)

    # Store TR quadrant
    c_tr = acc_tr.to(a_ptr.dtype.element_ty)
    c_tr = gl.convert_layout(c_tr, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_tr_offsets, stored_value=c_tr)

    # Store BR quadrant
    c_br = acc_br.to(a_ptr.dtype.element_ty)
    c_br = gl.convert_layout(c_br, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_br_offsets, stored_value=c_br)


def matmul(a, b, c=None):
    """
    FP16 GEMM using 128x128x64 tiles on gfx950.

    Use this variant for small M/N (< 1024) where the 256x256 v9 kernel
    generates too few tiles for good occupancy. For large shapes, prefer
    the 256x256 variant (fp16_gfx950.py) which amortizes overhead better.
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    num_warps = 4
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    # MI350/MI355 have 8 XCDs; GROUP_SIZE_M=4 for L2 locality
    NUM_XCDS = 8
    GROUP_SIZE_M = 4
    v9_128x128_kernel[grid](
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
