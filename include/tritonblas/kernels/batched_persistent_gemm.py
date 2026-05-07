# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Batched persistent GEMM kernel — single-launch BMM (K-684).

Motivation
----------
Before this kernel landed, tritonblas.matmul() handled rank-3 inputs by
looping on the host over the batch dimension and launching one single-GEMM
kernel per element. K-654 + K-659 measured this path: for a Z=4 batch of
2048^3 fp16 GEMMs the per-element kernel-launch (Tcold) overhead dominated
end-to-end time, leaving tritonblas at ratio≈0.30 vs torch.bmm.

The fused kernel here closes the *structural* gap by collapsing the batch
loop into the program-id space:

    grid = (BATCH * cdiv(M, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N),)

so all batch×tile work issues from a single launch.

Grid encoding
-------------
We encode (batch_id, tile_in_batch) into the flat program id by integer
division against the per-batch tile count `tiles_per_batch`:

    pid          = tl.program_id(0)
    tiles_per_batch = cdiv(M, BM) * cdiv(N, BN)
    batch_id     = pid // tiles_per_batch        # which batch element
    tile_in_batch = pid - batch_id * tiles_per_batch

Inside each program, `tile_in_batch` is then run through the standard
Triton grouped-tile swizzle (cf. persistent_gemm) to produce (pid_m, pid_n).

Why decode this way (not pid_z = pid % BATCH)?
    Adjacent program ids share the same `batch_id`, so consecutive blocks
    of `tiles_per_batch` programs touch the same A/B matrices. This keeps
    L2/MALL residency hot per-batch instead of striding across all
    batches every step (which would thrash MALL on large Z).

Per-batch base pointers
-----------------------
A is rank-3 of shape (Z, M, K). Its row-stride along the batch axis is
`a.stride(0)`. Inside the kernel we offset:

    A_batch = A + batch_id.to(tl.int64) * stride_az
    B_batch = B + batch_id.to(tl.int64) * stride_bz
    C_batch = C + batch_id.to(tl.int64) * stride_cz

The cast to int64 is REQUIRED for any large problem: stride_az is M*K
elements; for fp16 with M=K=8192, batch_id*stride_az easily exceeds 2^31.
Without the cast Triton emits 32-bit pointer arithmetic and silently
wraps for batch_id ≥ ⌈2^31 / (M·K)⌉.

Chiplet remap scope
-------------------
MI300X has 8 XCDs (chiplets). The single-GEMM kernel applies a
chiplet-aware swizzle (`chiplet_transform_chunked`) so adjacent tiles land
on the same XCD's L2 — improving MALL hit rate. We apply the SAME swizzle
here, but to `tile_in_batch` (within the batch element), NOT to the global
`pid`. Reasoning: each batch element has its own A[z], B[z], C[z]; tiling
across batch boundaries on the same XCD would *not* improve reuse because
the L2-resident matrices change at the batch boundary anyway. Keeping the
swizzle local-to-batch preserves the per-batch L2 reuse the swizzle was
designed for.

Tile selection
--------------
BLOCK_SIZE_{M,N,K} are passed in by the host. The host wrapper picks them
through `_select_batched_tile()` in `tritonblas.kernels.batched_dispatch`,
which chooses larger tiles when total_grid >> num_CUs (i.e. when the batch
dimension already saturates the GPU and arithmetic intensity is the
limiter rather than tile count). This is the K-684 small-MN fix.
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
    """Single-launch batched GEMM body. See module docstring for grid layout."""
    # ─── decode (batch_id, tile_in_batch) from flat program id ────────────────
    # Adjacent pids share batch_id ⇒ per-batch L2 reuse stays hot.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tiles_per_batch = num_pid_m * num_pid_n
    batch_id = pid // tiles_per_batch
    tile_id = pid - batch_id * tiles_per_batch

    # Defensive bound — host launches exactly BATCH*tiles_per_batch programs,
    # so this branch is statically dead in the common path.
    if batch_id >= BATCH:
        return

    # Chiplet swizzle is applied to the WITHIN-batch tile id only — see
    # module docstring "Chiplet remap scope".
    if NUM_XCDS != 1:
        tile_id = chiplet_transform_chunked(tile_id, tiles_per_batch, NUM_XCDS, CHUNK_SIZE)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    # ─── grouped-tile swizzle (standard Triton GEMM) ──────────────────────────
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # ─── per-batch base offsets ───────────────────────────────────────────────
    # int64 cast is required for large Z*M*K — stride_az is M*K elements,
    # which overflows int32 once batch_id*stride_az * elem_size ≥ 2^31.
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

    # ─── main K-loop ──────────────────────────────────────────────────────────
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

    # ─── K-tail (when K is not a multiple of BLOCK_SIZE_K) ────────────────────
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

    # ─── epilogue + store ─────────────────────────────────────────────────────
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
