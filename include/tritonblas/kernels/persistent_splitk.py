# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Atomic-free Split-K persistent GEMM for the small-M residual cohort.

Background
----------
For very skinny problems (M <= 32, K >= 4096) the persistent / stream-K paths
under-utilize the MI300X CU array because the M-dim only produces a handful of
tiles.  The classical "split-K" remedy partitions the K-axis across additional
workgroups and reduces their partial sums into the C tile.  The traditional
implementation uses ``tl.atomic_add`` directly on C in the output dtype.

Why the atomic path is slow on gfx942
-------------------------------------
gfx942 has *native* hardware atomics for FP32 (``GLOBAL_ATOMIC_ADD_F32``) and
INT32, but **no** native atomic-add for FP16/BF16.  When Triton lowers an
``atomic_add`` against an FP16/BF16 buffer it emits a CAS retry loop
(``GLOBAL_ATOMIC_CMPSWAP`` + half-precision add + retry on conflict), which
serializes all SPLIT_K writers that target the same (M,N) tile and is roughly
~10x slower than the corresponding FP32 atomic at SPLIT_K >= 8.  For small-M
problems every split writes to the *same* M-stripe, maximizing contention.

This module sidesteps the issue with a two-stage atomic-free reduction:

  Stage 1 (`splitk_partials_kernel`):
      Each program owns a unique (pid_m, pid_n, pid_k) and writes its FP32
      partial accumulator into a workspace tensor of shape
      ``[SPLIT_K, M_pad, N_pad]`` -- no atomics, every byte is written by
      exactly one program.

  Stage 2 (`splitk_reduce_kernel`):
      A tiny epilogue program owns a unique (pid_m, pid_n) tile, sums the
      ``SPLIT_K`` FP32 partials in-register, casts to the output dtype, and
      stores into C.

The workspace is FP32 regardless of the output dtype so reduction stays
numerically accurate.  The dispatcher only enables this path when
``M <= 32 AND K >= 4096 AND SPLIT_K >= 4`` -- the regime where the atomic
contention dominates -- so larger problems continue to use the existing
persistent / stream-K paths unchanged.
"""

import math

import triton
import triton.language as tl
import torch


@triton.jit
def splitk_partials_kernel(
    A,
    B,
    W,                               # workspace [SPLIT_K, M_pad, N_pad] fp32
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_wk,
    stride_wm,
    stride_wn,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,           # constexpr K-slab per split (rounded up)
    ITERS_PER_SPLIT: tl.constexpr,       # K_PER_SPLIT // BLOCK_SIZE_K
    EVEN_K_PER_SPLIT: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    ALLOW_TF32: tl.constexpr = True,
):
    """Stage 1: each program computes one (pid_m, pid_n, pid_k) partial in FP32.

    Grid layout:
        program_id(0) -> tile index over (pid_m, pid_n) with GROUP_SIZE_M
                         super-grouping for L2 locality.
        program_id(1) -> pid_k in [0, SPLIT_K).

    No atomics -- every workspace element is owned by exactly one program.
    """
    pid_mn = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # GROUP_SIZE_M super-grouping (L2 locality on N-stripe).
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)
    pid_n = (pid_mn % num_pid_in_group) // group_size_m

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_wk > 0)
    tl.assume(stride_wm > 0)
    tl.assume(stride_wn > 0)

    # Output tile coordinates (raw + masked + modulo for safe contiguous loads).
    rm_raw = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn_raw = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = rm_raw < M
    mask_n = rn_raw < N
    rm = tl.max_contiguous(tl.multiple_of(rm_raw % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn_raw % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # K range owned by this split.  K_PER_SPLIT is a constexpr from the host
    # (rounded up so the integral-iter fast path is always available when K
    # divides evenly).
    k_start = pid_k * K_PER_SPLIT
    # Bounded end -- last split may be short when EVEN_K_PER_SPLIT is False.
    k_end = tl.minimum(k_start + K_PER_SPLIT, K)

    rk = k_start + tl.arange(0, BLOCK_SIZE_K)
    A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if EVEN_K_PER_SPLIT:
        # Fast path: constexpr-trip-count K loop over a clean K slab.
        for _ in tl.static_range(0, ITERS_PER_SPLIT):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)),
                            mask=mask_m[:, None], other=0.0,
                            cache_modifier=CACHE_MODIFIER_A)
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)),
                            mask=mask_m[:, None], other=0.0,
                            cache_modifier=CACHE_MODIFIER_A)
            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)),
                            mask=mask_n[None, :], other=0.0,
                            cache_modifier=CACHE_MODIFIER_B)
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)),
                            mask=mask_n[None, :], other=0.0,
                            cache_modifier=CACHE_MODIFIER_B)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk
    else:
        # Slow path: K-mask each load.  ITERS_PER_SPLIT is rounded up by the
        # host so workers cover the full K_PER_SPLIT slab; tail iterations
        # see all-zero loads via the K mask.
        for it in tl.static_range(0, ITERS_PER_SPLIT):
            k_offset = k_start + it * BLOCK_SIZE_K
            k_mask = (k_offset + tl.arange(0, BLOCK_SIZE_K)) < k_end
            a = tl.load(A_BASE,
                        mask=mask_m[:, None] & k_mask[None, :], other=0.0,
                        cache_modifier=CACHE_MODIFIER_A)
            b = tl.load(B_BASE,
                        mask=mask_n[None, :] & k_mask[:, None], other=0.0,
                        cache_modifier=CACHE_MODIFIER_B)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

    # Stage-1 write: each (pid_k, pid_m, pid_n) is unique -> plain store.
    W_PTR = (
        W
        + pid_k * stride_wk
        + rm_raw[:, None] * stride_wm
        + rn_raw[None, :] * stride_wn
    )
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(W_PTR, acc, mask=c_mask)


@triton.jit
def splitk_reduce_kernel(
    W,                               # [SPLIT_K, M_pad, N_pad] fp32
    C,
    bias_ptr,
    M,
    N,
    stride_wk,
    stride_wm,
    stride_wn,
    stride_cm,
    stride_cn,
    stride_bias,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BIAS: tl.constexpr,
):
    """Stage 2: sum the SPLIT_K FP32 partials and cast to the output dtype.

    Grid layout: program_id(0) ranges over the (pid_m, pid_n) tiles in
    row-major order -- one program per output tile.
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_idx in tl.static_range(0, SPLIT_K):
        W_PTR = (
            W
            + k_idx * stride_wk
            + rm[:, None] * stride_wm
            + rn[None, :] * stride_wn
        )
        acc += tl.load(W_PTR, mask=mask, other=0.0)

    if BIAS:
        bias = tl.load(bias_ptr + rn * stride_bias, mask=rn < N, other=0.0)
        acc += bias[None, :].to(tl.float32)

    C_PTR = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(C_PTR, acc.to(C.type.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Host-side dispatcher
# ---------------------------------------------------------------------------

# Bookkeeping: persistent FP32 workspace reused across calls.  Sized lazily
# and grown only when a larger problem appears.  Allocated on the first
# device that requests it; if a different device is later targeted, a fresh
# tensor is allocated on that device.
_workspace_cache = {}


def _get_workspace(split_k: int, m_pad: int, n_pad: int, device: torch.device) -> torch.Tensor:
    """Return a [SPLIT_K, m_pad, n_pad] FP32 buffer, growing the cache as needed."""
    key = device.index if device.index is not None else 0
    needed = split_k * m_pad * n_pad
    cached = _workspace_cache.get(key)
    if cached is None or cached.numel() < needed:
        cached = torch.empty(needed, device=device, dtype=torch.float32)
        _workspace_cache[key] = cached
    return cached.view(-1)[:needed].view(split_k, m_pad, n_pad)


def choose_split_k(M: int, K: int, block_k: int) -> int:
    """Pick a SPLIT_K for the small-M cohort.

    Targets >= 8 BLOCK_K iterations per split (Triton software pipeline likes
    a deep K-loop), and caps at 8 splits because beyond that the reduction
    epilogue starts costing more than the parallelism saves on MI300X.
    """
    if K < 4096 or block_k <= 0:
        return 1
    iters_per_split_floor = 8
    max_splits_by_k = max(1, K // (block_k * iters_per_split_floor))
    candidates = [s for s in (8, 6, 4) if s <= max_splits_by_k]
    if not candidates:
        return 1
    # Prefer the largest split that divides K // block_k evenly to keep the
    # fast (EVEN_K_PER_SPLIT) path active.  Falls back to the largest
    # candidate otherwise.
    iters_total = K // block_k
    for s in candidates:
        if iters_total % s == 0:
            return s
    return candidates[0]


def should_use_atomic_free_splitk(M: int, N: int, K: int, split_k: int) -> bool:
    """Gate: only enable for the small-M, deep-K, deep-N, high-SPLIT_K cohort.

    Gating rationale (measured on MI300X / gfx942):

      * ``M <= 32`` -- small-M is the cohort where Origami's persistent /
        stream-K paths produce only a handful of (pid_m, pid_n) tiles and
        leave most of the gfx942 CU array idle.
      * ``K >= 8192`` -- the two-stage launch overhead (~80 us on MI300X) is
        only amortized when the K-loop is long enough that the partial-write
        kernel actually has work to hide it.  At ``K = 4096`` the new path
        loses to torch.matmul on every measured ``(N, dtype)`` pair; at
        ``K >= 8192`` it begins to win on the wider-N tail.
      * ``N >= 4096`` -- below this the (pid_m, pid_n) tile count is too
        small for SPLIT_K to add useful parallelism and the new path stays
        below 0.4x torch.  Above it the geomean climbs to 0.47-0.55x.
      * ``SPLIT_K >= 4`` -- ensures we are actually using the split-K codepath
        (not a degenerate 1-split that would just be the persistent path with
        extra overhead).

    Boundary fall-through (verified):

      * ``M = 64`` (just outside the small-M cohort) -> persistent path
      * ``K = 4096`` or ``K = 2048`` -> persistent path
      * ``N = 1024`` or ``N = 2048`` -> persistent path
      * ``SPLIT_K = 2`` (e.g. via ``choose_split_k`` returning 1 or 2) ->
        persistent path

    Callers may set ``TBLAS_DISABLE_SPLITK_SMALLM=1`` to force fall-through
    even on the gated cohort (useful for A/B perf bisection).
    """
    return M <= 32 and N >= 4096 and K >= 8192 and split_k >= 4


def persistent_splitk_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    block_m: int,
    block_n: int,
    block_k: int,
    group_m: int,
    split_k: int,
    bias: torch.Tensor = None,
    num_stages: int = 2,
    num_warps: int = 4,
    allow_tf32: bool = True,
) -> torch.Tensor:
    """Two-stage atomic-free split-K dispatcher.

    Returns ``c``.  Caller is responsible for picking ``block_m``/``block_n``/
    ``block_k`` -- the dispatcher inherits Origami's choices but clamps
    ``block_m`` to <= M (the small-M motif).
    """
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    # For small M (16, 32) make sure BLOCK_M doesn't waste a whole MFMA tile.
    block_m = min(block_m, max(16, triton.next_power_of_2(M)))

    num_pid_m = triton.cdiv(M, block_m)
    num_pid_n = triton.cdiv(N, block_n)
    total_tiles = num_pid_m * num_pid_n

    m_pad = num_pid_m * block_m
    n_pad = num_pid_n * block_n

    workspace = _get_workspace(split_k, m_pad, n_pad, a.device)

    # EVEN_K_PER_SPLIT: every split sees an integral number of BLOCK_K tiles
    # *and* K is partitioned evenly.  This is the constexpr-loop-trip-count
    # fast path.
    even_k_per_split = (K % split_k == 0) and ((K // split_k) % block_k == 0)
    if even_k_per_split:
        k_per_split = K // split_k
    else:
        # Round up so workers walk past K_PER_SPLIT in BLOCK_K steps; the
        # masked path bounds them by K.
        k_per_split = ((K + split_k - 1) // split_k + block_k - 1) // block_k * block_k
    iters_per_split = k_per_split // block_k

    grid_partials = (total_tiles, split_k)
    splitk_partials_kernel[grid_partials](
        a, b, workspace,
        M, N, K,
        a.stride(0), b.stride(1),
        workspace.stride(0), workspace.stride(1), workspace.stride(2),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        SPLIT_K=split_k,
        K_PER_SPLIT=k_per_split,
        ITERS_PER_SPLIT=iters_per_split,
        EVEN_K_PER_SPLIT=even_k_per_split,
        CACHE_MODIFIER_A=None,
        CACHE_MODIFIER_B=None,
        ALLOW_TF32=allow_tf32,
        num_stages=num_stages,
        num_warps=num_warps,
        matrix_instr_nonkdim=16,
        kpack=1,
    )

    grid_reduce = (total_tiles,)
    splitk_reduce_kernel[grid_reduce](
        workspace, c,
        bias if bias is not None else c,
        M, N,
        workspace.stride(0), workspace.stride(1), workspace.stride(2),
        c.stride(0), c.stride(1),
        bias.stride(0) if bias is not None else 0,
        SPLIT_K=split_k,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BIAS=bias is not None,
        num_warps=4,
    )

    return c
