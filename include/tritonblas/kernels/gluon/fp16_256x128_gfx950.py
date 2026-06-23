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
Gluon v9-style GEMM kernel: BLOCK_M=256, BLOCK_N=128, BLOCK_K=64 (gfx950).

Variant of the 256x256x64 v9 kernel optimized for tall-M, short-N shapes
(e.g. M>>N). Uses 4-quadrant accumulation with half-tiles of (128, 64).

Layout derivation (256x128 tile, 4 warps, 256 threads total):
  - A tile: two halves of [128, 64] (top/bot along M)
  - B tile: two halves of [64, 64] (left/right along N)
  - Accumulators: 4 quadrants of (128, 64) in MFMA layout

Global load layouts use BlockedLayout (portable across Triton versions):
  - gLoadLayoutA: [128, 64] = sizePerThread=[1,8], [8,8],
    warpsPerCTA=[4,1], [1,0]
    Each thread loads 8 contiguous elements along K (vectorized).
    128 / 1 = 128 threads along M, but 128/8=16 warps along M -- we have
    only 4 warps, so threadsPerWarp covers 8 positions along M.
    Total coverage: 4 warps * 8 threads/warp along M * 1 elem/thread = 32...
    Actually: sizePerThread[0]*threadsPerWarp[0]*warpsPerCTA[0] = 1*8*4 = 32,
    sizePerThread[1]*threadsPerWarp[1]*warpsPerCTA[1] = 8*8*1 = 64.
    So each "pass" covers [32, 64]; to fill [128, 64] we need 128/32 = 4
    iterations of the blocked layout, which Gluon handles internally.

    Correction per user spec:
    threadsPerWarp = [512//64, 64//8] = [8, 8]
    This is consistent: 8 threads along M * 8 threads along K = 64 per warp.

  - gLoadLayoutB: [64, 64] = sizePerThread=[8,1], [8,8],
    warpsPerCTA=[1,4], [0,1]
    Each thread loads 8 contiguous elements along K (vectorized).
    Coverage per pass: sizePerThread[0]*threadsPerWarp[0]*warpsPerCTA[0]
    = 8*8*1 = 64 along K, sizePerThread[1]*threadsPerWarp[1]*warpsPerCTA[1]
    = 1*8*4 = 32 along N. [64, 32] per pass, 2 passes for [64, 64].

    Correction per user spec:
    threadsPerWarp = [64//8, 512//64] = [8, 8]

Shared memory layouts use SwizzledSharedLayout (simpler, avoids bank
conflicts via hardware swizzling):
  - sharedLayoutA: SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    Row-major for A (M-contiguous in outer dim, K-contiguous in inner dim)
  - sharedLayoutB: SwizzledSharedLayout(1, 1, 1, order=[0, 1])
    Column-major for B (K-contiguous in outer dim, N-contiguous in inner dim)

MFMA configuration:
  - version=4 (CDNA4 / gfx950), instr_shape=[16,16,32], transposed=True
  - warps_per_cta=[2,2]: 2x2 warp grid for 4-quadrant accumulation
  - DotOperandLayout with k_width=8 (FP16: 8 elements = 16 bytes = 128 bits)
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
    """XCD-aware PID remapping + workgroup swizzling for L2 locality.

    Identical to v9 256x256. The remapping distributes consecutive PIDs
    across XCDs to avoid L2 hotspots, then applies GROUP_SIZE_M swizzling
    so that nearby M-tiles share L2 lines for the B operand.
    """
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
def v9_256x128_kernel(
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
    v9-style GEMM for 256x128x64 tiles on gfx950 (MI350/MI355).

    Tile decomposition:
        BLOCK_M=256, BLOCK_N=128, BLOCK_K=64
        Half-M = 128 = BLOCK_M // 2
        Half-N = 64  = BLOCK_N // 2

    The tile is split into 4 quadrants:
        acc_tl (128, 64): rows [0..128),    cols [0..64)
        acc_tr (128, 64): rows [0..128),    cols [64..128)
        acc_bl (128, 64): rows [128..256),  cols [0..64)
        acc_br (128, 64): rows [128..256),  cols [64..128)

    Pipeline structure (from v9):
        - Double-buffered async copy (nBuffers=2)
        - Main loop unrolled by 2 (even/odd K-steps use different offset sets)
        - Interleaved epilogue: stores interleaved with final MFMAs

    Key difference from 256x256:
        - B half-tiles are [64, 64] instead of [64, 128]
        - Accumulators are (128, 64) instead of (128, 128)
        - Uses BlockedLayout for global loads (portable)
        - Uses SwizzledSharedLayout for shared memory (simpler bank-conflict avoidance)
    """

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # =========================================================================
    # Global load layouts (BlockedLayout -- portable across Triton versions)
    # =========================================================================
    #
    # A layout: loads half-M tile [128, 64]
    #   sizePerThread = [1, 8]: each thread owns 1 row, 8 cols (vectorized K load)
    #   threadsPerWarp = [512//64, 64//8] = [8, 8]: 64 threads arranged 8x8
    #   warpsPerCTA = [4, 1]: 4 warps stacked along M
    #   order = [1, 0]: K dimension is contiguous (column-major load pattern)
    #   Per-pass coverage: 1*8*4 = 32 along M, 8*8*1 = 64 along K -> [32, 64]
    #   Needs 128/32 = 4 internal iterations to fill [128, 64]
    gLoadLayoutA: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])

    # B layout: loads half-N tile [64, 64]
    #   sizePerThread = [8, 1]: each thread owns 8 rows, 1 col (vectorized K load)
    #   threadsPerWarp = [64//8, 512//64] = [8, 8]: 64 threads arranged 8x8
    #   warpsPerCTA = [1, 4]: 4 warps spread along N
    #   order = [0, 1]: K dimension is contiguous (row-major load pattern)
    #   Per-pass coverage: 8*8*1 = 64 along K, 1*8*4 = 32 along N -> [64, 32]
    #   Needs 64/32 = 2 internal iterations to fill [64, 64]
    gLoadLayoutB: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, 4], [0, 1])

    # =========================================================================
    # Shared memory layouts (SwizzledSharedLayout -- hardware bank swizzling)
    # =========================================================================
    #
    # SwizzledSharedLayout(vec, perPhase, maxPhase, order):
    #   vec=1, perPhase=1, maxPhase=1: minimal swizzle configuration
    #   The hardware applies XOR-based swizzling to avoid bank conflicts.
    #   order=[1,0] for A (K inner), order=[0,1] for B (K inner, transposed)
    sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])

    # =========================================================================
    # Shared memory allocation -- double-buffered, split into half-tiles
    # =========================================================================
    #
    # A: two half-M buffers of [128, 64] (top and bottom halves of 256-row tile)
    # B: two half-N buffers of [64, 64] (left and right halves of 128-col tile)
    # Each has nBuffers=2 for double-buffering
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
    #
    # SliceLayout extracts a 1D index range from the 2D blocked layout,
    # used to create the row/col offset vectors.
    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    # Base pointers for this workgroup's tile
    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    # Two sets of offsets for double-buffered pipeline:
    #   base offsets = even K-steps (k=0, k=2*BLOCK_K, ...)
    #   _next offsets = odd K-steps (k=BLOCK_K, k=3*BLOCK_K, ...)
    a_top_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_bot_offsets = a_top_offsets + BLOCK_M * stride_am // 2
    b_left_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b_right_offsets = b_left_offsets + BLOCK_N * stride_bn // 2

    a_top_offsets_next = a_top_offsets + BLOCK_K * stride_ak
    a_bot_offsets_next = a_bot_offsets + BLOCK_K * stride_ak
    b_left_offsets_next = b_left_offsets + BLOCK_K * stride_bk
    b_right_offsets_next = b_right_offsets + BLOCK_K * stride_bk

    # =========================================================================
    # MFMA and dot-operand layouts
    # =========================================================================
    #
    # MFMA v4 (CDNA4/gfx950): 16x16x32 instruction, transposed=True
    # warps_per_cta=[2,2]: 2 warps along M, 2 along N
    # This means each warp handles a (HalfM/2, HalfN/2) = (64, 32) sub-tile,
    # and the 4 warps together cover (128, 64).
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    # =========================================================================
    # Accumulators -- 4 quadrants of (128, 64) = (Half-M, Half-N)
    # =========================================================================
    #
    # For 256x256: each quadrant was (128, 128)
    # For 256x128: each quadrant is (128, 64) -- halved along N
    acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)

    # =========================================================================
    # Prologue: prefetch first 2 K-iterations into both double-buffer slots
    # =========================================================================
    #
    # Buffer 0: load at K-offset 0
    g_idx = 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_base, b_left_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(g_idx), a_base, a_top_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(g_idx), a_base, a_bot_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_base, b_right_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    # Buffer 1: load at K-offset BLOCK_K (uses _next offsets)
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

    # Advance base pointers past the 2 prefetched K-steps
    a_base += BLOCK_K * stride_ak * 2
    b_base += BLOCK_K * stride_bk * 2

    # Wait for buffer 0's B_left and A_top to be ready (6 groups still pending)
    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)

    gl.assume(iterMax > 3)

    # =========================================================================
    # Main loop -- unrolled by 2 (even/odd K-steps alternate buffer slots)
    # =========================================================================
    #
    # Each full iteration processes 2 K-steps. Within each sub-iteration:
    #   4 MFMA regions (tl, bl, tr, br) interleaved with async loads for the
    #   next K-step. Shared memory loads for the current step are overlapped
    #   with MFMAs from the previous region.
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
        ## Async copies use _next offsets (odd K-step within the pair)
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
    #
    # Store layout for (128, 64) quadrants.
    # BlockedLayout: sizePerThread=[1,8] -> 8 contiguous elements per store
    # threadsPerWarp=[4,16] -> 64 threads per warp in 4x16 arrangement
    # warpsPerCTA=[4,1] -> 4 warps stacked along M
    # order=[1,0] -> N dimension is contiguous for coalesced writes
    #
    # Per-pass coverage: 1*4*4 = 16 along M, 8*16*1 = 128 along N...
    # but Half-N = 64, so we need threadsPerWarp to match.
    #
    # For (128, 64) output:
    #   sizePerThread=[1, 4]: each thread stores 4 contiguous N elements
    #   threadsPerWarp=[16, 4]: 64 threads in 16x4 arrangement
    #   warpsPerCTA=[4, 1]: 4 warps along M
    #   Coverage: 1*16*4 = 64 along M, 4*4*1 = 16 along N -> [64, 16]
    #   Needs 128/64 * 64/16 = 2*4 = 8 iterations for [128, 64]
    #
    # Alternative (keeping v9's pattern adapted):
    #   sizePerThread=[1, 8], [4, 16], [4, 1]
    #   Coverage: 1*4*4=16 M, 8*16*1=128 N -- but Half-N=64, not 128.
    #   So we use [1, 4] to fit within 64 columns, or adjust threadPerWarp.
    #
    # Simplest correct choice matching the 64-wide N dimension:
    #   [1, 8] * [8, 8] * [4, 1] = [32, 64] per pass -> 4 passes for [128, 64]
    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])

    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_tl_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_tr_offsets = c_tl_offsets + BLOCK_N * stride_cn // 2
    c_bl_offsets = c_tl_offsets + BLOCK_M * stride_cm // 2
    c_br_offsets = c_bl_offsets + BLOCK_N * stride_cn // 2

    # ----- Drain iteration iterMax - 2 (no new async copies) -----
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

    # ----- Final iteration iterMax - 1 with interleaved stores -----
    # Natural-pipeline epilogue: each store follows its MFMA with one
    # MFMA cycle of gap, yielding uniform MFMA-store interleaving.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl)
    gl.amd.cdna4.async_copy.wait_group(1)
    a_bot = smemA_bot.index(g_idx).load(dotOpLayoutA)

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl)
    gl.amd.cdna4.async_copy.wait_group(0)
    b_right = smemB_right.index(g_idx).load(dotOpLayoutB)

    # Store top-left quadrant (interleaved with remaining MFMAs)
    c_tl = acc_tl.to(a_ptr.dtype.element_ty)
    c_tl = gl.convert_layout(c_tl, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_tl_offsets, stored_value=c_tl)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr)

    # Store bottom-left quadrant
    c_bl = acc_bl.to(a_ptr.dtype.element_ty)
    c_bl = gl.convert_layout(c_bl, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_bl_offsets, stored_value=c_bl)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br)

    # Store top-right quadrant
    c_tr = acc_tr.to(a_ptr.dtype.element_ty)
    c_tr = gl.convert_layout(c_tr, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_tr_offsets, stored_value=c_tr)

    # Store bottom-right quadrant (final)
    c_br = acc_br.to(a_ptr.dtype.element_ty)
    c_br = gl.convert_layout(c_br, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_br_offsets, stored_value=c_br)


def matmul(a, b, c=None):
    """Launch the 256x128x64 Gluon GEMM kernel.

    Designed for tall-M, short-N shapes where M >> N.
    Uses 4 warps on gfx950 with 8 XCDs.

    Args:
        a: (M, K) input tensor, FP16/BF16, contiguous
        b: (K, N) input tensor, FP16/BF16
        c: optional (M, N) output tensor; allocated if None
    Returns:
        c: (M, N) result of a @ b
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 128, 64
    num_warps = 4
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    # MI350/MI355 (gfx950) has 8 XCDs
    NUM_XCDS = 8
    # GROUP_SIZE_M=4: group 4 consecutive M-tiles so they share B data in L2
    GROUP_SIZE_M = 4
    v9_256x128_kernel[grid](
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
