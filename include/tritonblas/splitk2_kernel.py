# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""K-795 Split-K=2 kernel and shape/dtype-guarded routed override.

Split-K=2 prototype kernel from K-795 / K-811 with FP32 atomic_add reduction,
mounted behind a strict shape+dtype gate that fires only on the 6 cells:

    M=N=2048, K∈{4096, 8192, 16384}, dtype∈{torch.float16, torch.bfloat16}

This module is intentionally self-contained so the gate is impossible to leak
to other shapes; the dispatch helper returns False (no-op) for anything else.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# K-795 split-K=2 kernel: per-shard accumulator + FP32 atomic_add reduction.
# Tile geometry from K-795's prototype mini-sweep best entry:
#   BM=128, BN=128, BK=64, GROUP_M=8, num_warps=8, num_stages=2, kpack=1.
@triton.jit
def _splitk2_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_sk = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Each split-K shard owns a contiguous K-slab of length K/SPLIT_K.
    K_per_shard = K // SPLIT_K
    k_start = pid_sk * K_per_shard
    k_end = k_start + K_per_shard

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak)
    b_ptrs = b_ptr + ((k_start + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k_start, k_end, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)
    else:
        tl.atomic_add(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


# Strict 6-cell override table.
# (M, N, K, dtype) -> kernel-tile spec.  Only these keys ever route through the
# split-K=2 kernel; every other shape returns False from the dispatch helper.
_SPLITK2_OVERRIDE_TABLE = {
    (2048, 2048, 4096,  torch.float16):  dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=2),
    (2048, 2048, 4096,  torch.bfloat16): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=2),
    (2048, 2048, 8192,  torch.float16):  dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=2),
    (2048, 2048, 8192,  torch.bfloat16): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=2),
    (2048, 2048, 16384, torch.float16):  dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=2),
    (2048, 2048, 16384, torch.bfloat16): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=2),
}

_SPLIT_K = 2


def lookup_splitk2_override(
    M: int, N: int, K: int,
    a_dtype: torch.dtype, b_dtype: torch.dtype, c_dtype: torch.dtype,
) -> Optional[dict]:
    """Return the kernel-tile spec for this (M,N,K,dtype) cell or None.

    The gate is a strict equality match on all of (M, N, K, a_dtype) AND requires
    a_dtype == b_dtype == c_dtype.  No looser predicate, no wildcards.
    """
    if a_dtype is not b_dtype or a_dtype is not c_dtype:
        return None
    return _SPLITK2_OVERRIDE_TABLE.get((M, N, K, a_dtype))


def splitk2_matmul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, spec: dict) -> torch.Tensor:
    """Run the K-795 split-K=2 kernel into `out` for one of the 6 routed cells.

    Caller must have already verified the (M,N,K,dtype) cell is in the override
    table via `lookup_splitk2_override`.  `out` is zeroed in-place because the
    kernel uses atomic_add reduction across split-K shards.
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Incompatible Dimensions"
    assert out.shape == (M, N)
    assert K % _SPLIT_K == 0, "K must be divisible by SPLIT_K=2"

    # atomic_add requires zero-init.
    out.zero_()

    BLOCK_M = spec["BLOCK_M"]
    BLOCK_N = spec["BLOCK_N"]
    BLOCK_K = spec["BLOCK_K"]
    GROUP_M = spec["GROUP_M"]
    num_warps = spec["num_warps"]
    num_stages = spec["num_stages"]

    grid = (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        _SPLIT_K,
    )

    _splitk2_matmul_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M, SPLIT_K=_SPLIT_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# Public dispatch helper used from `tritonblas.matmul`.
def maybe_dispatch_splitk2(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor,
) -> bool:
    """Try to route the matmul through the K-795 split-K=2 kernel.

    Returns True iff the kernel handled the multiply (output written into `out`),
    False otherwise — caller must fall through to its default code path.
    """
    if a.dim() != 2 or b.dim() != 2:
        return False
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        return False
    spec = lookup_splitk2_override(M, N, K, a.dtype, b.dtype, out.dtype)
    if spec is None:
        return False
    splitk2_matmul(a, b, out, spec)
    return True
