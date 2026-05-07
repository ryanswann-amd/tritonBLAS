# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Split-K GEMM kernel for the CU-starved small-M corner.

Background
----------
For tall-skinny GEMM where ``total_tiles = ceil(M/BLOCK_M) * ceil(N/BLOCK_N)``
is far below ``num_cus`` (e.g. M<=16, N<=2048, K>=4096 on MI300X's 304 CUs),
the persistent grid-stride launch in ``persistent_matmul_lt`` still leaves
most of the chip idle even at the smallest legal MFMA tile (16x16):

    e.g. M=16, N=1024, K=8192 -> total_tiles = 64, num_cus = 304
                                 -> ~5x under-subscribed

Closing this gap requires partitioning the K dimension across multiple
workgroups per output tile (split-K).  Each workgroup computes a partial
sum over its K-shard and atomic-adds it into a shared fp32 partial output;
a tiny finaliser cast converts back to fp16/bf16.

Numerical model
---------------
* Accumulator: fp32 (Triton ``tl.float32``).
* Partial output: fp32 (atomic-add accumulation in global memory).  The
  per-shard partial sum has the same K-summation error bound as the
  unsharded fp16 GEMM (both are fp16-load * fp16-load + fp32-accum).
  Atomic-add of fp32 introduces *non-determinism in summation order*
  but the bound on the final result is still O(K * eps_fp32 * |A| * |B|),
  unchanged from the non-split path.  This is the same numerical model
  cuBLAS uses for split-K GEMM.
* Output cast: fp32 -> dtype (fp16/bf16) via a vectorised Triton kernel.

Why not in-place atomic add to fp16 C?
- AMD ROCm 7.x ``tl.atomic_add`` on fp16 lacks an MFMA-fast path; round-off
  also accumulates worse than fp32.  The fp32 scratch is cheap (M*N*4 bytes;
  even at the largest cohort shape M=64 N=8192 it's 2 MB -- below L2 reuse
  threshold).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def split_k_gemm_kernel(
    A,
    B,
    C_partial,            # fp32, M x N, must be pre-zeroed
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """One workgroup computes (BLOCK_M x BLOCK_N) of C over a K-shard of size
    ceil(K / SPLIT_K), then atomic-adds the fp32 partial sum into C_partial.

    Grid layout (1-D, contiguous in pid_k for L2 locality on A and B):
        pid = pid_k + SPLIT_K * (pid_m + num_pid_m * pid_n)
    """
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Decode (pid_m, pid_n, pid_k) from the linearised pid.
    pid_k = pid % SPLIT_K
    pid_mn = pid // SPLIT_K
    pid_m = pid_mn % num_pid_m
    pid_n = pid_mn // num_pid_m

    # K-range for this shard.  Use ceil-division so the union of shards
    # covers [0, K) exactly (the last shard may be short).
    K_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * K_per_split
    k_end = tl.minimum(k_start + K_per_split, K)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # We always mask M and N tails (M tiny, N may not divide BLOCK_N).
    mask_m = offs_m < M
    mask_n = offs_n < N

    a_base = A + offs_m[:, None] * stride_am
    b_base = B + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = k_start
    # Number of full BLOCK_K iters in this shard.  When EVEN_K is True we
    # also know K_per_split is a multiple of BLOCK_K (caller's contract),
    # so we can skip the K-tail mask in the hot path.
    while k < k_end:
        a_ptrs = a_base + (k + offs_k[None, :]) * stride_ak
        b_ptrs = b_base + (k + offs_k[:, None]) * stride_bk
        if EVEN_K:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        else:
            mask_k = (k + offs_k) < k_end
            a = tl.load(
                a_ptrs,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
            )
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
        k += BLOCK_K

    # Atomic-add this shard's partial sum into the fp32 scratch.  The mask
    # protects against M/N tails (BLOCK_M may exceed M, BLOCK_N may exceed N).
    c_ptrs = C_partial + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.atomic_add(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def cast_kernel(
    SRC,           # fp32, M x N
    DST,           # fp16/bf16, M x N
    M,
    N,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Cast fp32 -> output dtype.  Pure memory-bandwidth-bound; we use a
    plain 2-D launch (no XCD-aware swizzle) because the partial buffer
    is at most M*N*4 bytes (a few MB at the cohort's worst case)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    src_ptrs = SRC + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    dst_ptrs = DST + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn
    val = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, val, mask=mask)


def split_k_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
    SPLIT_K: int,
    num_warps: int = 4,
) -> torch.Tensor:
    """Run split-K GEMM C = A @ B.

    A is (M, K) row-major (stride_am=K, stride_ak=1).
    B is (K, N) row-major (stride_bk=N, stride_bn=1).
    C is (M, N) row-major.  C is *overwritten* (the partial scratch is
    pre-zeroed; the final cast does an unconditional store into C).

    EVEN_K is True iff K_per_split is a multiple of BLOCK_K AND
    K is a multiple of BLOCK_K (so every shard is exactly BLOCK_K-aligned
    and the K-tail mask in the kernel can be elided in the hot path).
    """
    assert a.shape[1] == b.shape[0], "shape mismatch"
    M, K = a.shape
    _, N = b.shape
    assert c.shape == (M, N), "C shape mismatch"
    assert SPLIT_K >= 1

    # fp32 partial output, zeroed.  At the small-M cohort's worst case
    # (M=64, N=8192) this is 2 MB -- fits in MI300X L2 (256 MB across XCDs).
    c_partial = torch.zeros((M, N), device=a.device, dtype=torch.float32)

    K_per_split = (K + SPLIT_K - 1) // SPLIT_K
    even_k = (K % BLOCK_K == 0) and (K_per_split % BLOCK_K == 0)

    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid_size = SPLIT_K * num_pid_m * num_pid_n

    split_k_gemm_kernel[(grid_size,)](
        a, b, c_partial,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c_partial.stride(0), c_partial.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        EVEN_K=even_k,
        num_warps=num_warps,
        num_stages=2,
    )

    # Cast fp32 -> output dtype.  Pick block sizes that cover the output
    # in a single 2-D launch -- partial is small.
    BM_CAST = min(64, _next_pow2(M))
    BN_CAST = 128
    cast_kernel[(triton.cdiv(M, BM_CAST), triton.cdiv(N, BN_CAST))](
        c_partial, c,
        M, N,
        c_partial.stride(0), c_partial.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BM_CAST,
        BLOCK_N=BN_CAST,
        num_warps=4,
    )
    return c


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    p = 1
    while p < x:
        p <<= 1
    return p
