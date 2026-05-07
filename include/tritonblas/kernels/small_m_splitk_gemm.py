# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Small-M split-K GEMM kernel.

Why this exists
---------------
At small M (<= 32) the data-parallel persistent GEMM only produces
``ceil(N / BLOCK_N)`` output tiles, which leaves the vast majority of
the GPU's compute units idle. Splitting the K dimension across many
programs turns one M-row tile into ``SPLIT_K`` programs, oversubscribing
the grid and recovering occupancy.

Each program computes its K-slice and **atomically accumulates** into the
output tensor (``tl.atomic_add`` works on BF16/FP16/FP32 on MI300X). The
output is pre-zeroed by the caller; with atomic accumulation we only
need a single GEMM-kernel launch and no FP32 partial workspace, keeping
per-call latency low for the small problems where launch overhead matters.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _small_m_splitk_atomic(
    A,
    B,
    C,
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
    NUM_PID_M: tl.constexpr,
    NUM_PID_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    ATOMIC: tl.constexpr,
):
    """One program = one (m_tile, n_tile, k_slice) → atomic_add into C."""
    pid = tl.program_id(0)
    tiles_per_split = NUM_PID_M * NUM_PID_N
    pid_k = pid // tiles_per_split
    rest = pid % tiles_per_split
    pid_m = rest // NUM_PID_N
    pid_n = rest % NUM_PID_N

    iters_total = tl.cdiv(K, BLOCK_K)
    iters_per_split = iters_total // SPLIT_K
    iters_rem = iters_total % SPLIT_K
    extra = tl.where(pid_k < iters_rem, 1, 0)
    my_iters = iters_per_split + extra
    my_start_iter = pid_k * iters_per_split + tl.minimum(pid_k, iters_rem)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    A_BASE = A + offs_m[:, None] * stride_am + (my_start_iter * BLOCK_K + offs_k[None, :]) * stride_ak
    B_BASE = B + (my_start_iter * BLOCK_K + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for it in range(0, my_iters):
        if EVEN_K:
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)))
            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)))
        else:
            global_k = (my_start_iter + it) * BLOCK_K
            k_in_range = (global_k + offs_k) < K
            a = tl.load(A_BASE, mask=k_in_range[None, :], other=0.0)
            b = tl.load(B_BASE, mask=k_in_range[:, None], other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        A_BASE += BLOCK_K * stride_ak
        B_BASE += BLOCK_K * stride_bk

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    C_BASE = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    out = acc.to(C.type.element_ty)
    if ATOMIC:
        tl.atomic_add(C_BASE, out, mask=mask, sem="relaxed")
    else:
        tl.store(C_BASE, out, mask=mask)


# ─── Heuristics ───────────────────────────────────────────────────────────────

# Target one wave of work per launch — sized to roughly the CU count of
# the gfx942-class GPUs we ship to. Keeps wall-time dominated by the
# wave's work rather than persistent-loop bookkeeping. Querying the
# device count at import time would be nicer but introduces a hard
# CUDA-context dependency at module load.
_TARGET_GRID = 304


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


# Per-shape config cache to avoid recomputing on every call.
_cfg_cache: dict = {}


def _pick_config(M: int, N: int, K: int):
    key = (M, N, K)
    cfg = _cfg_cache.get(key)
    if cfg is not None:
        return cfg
    block_m = max(16, min(32, _next_pow2(max(M, 1))))
    if N <= 64:
        block_n = max(32, _next_pow2(N))
    elif N <= 256:
        block_n = max(64, _next_pow2(N))
    else:
        block_n = 128

    # BLOCK_K must keep LDS pressure under MI300X's 64 KB:
    #   lds = num_stages * (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * dsize
    # For BLOCK_M=32, BLOCK_N=128, num_stages=2, dsize=2 (bf16/fp16):
    #   BLOCK_K=128 → 80 KB (OOM); BLOCK_K=64 → 40 KB (fits).
    block_k = 64

    num_pid_m = max(1, (M + block_m - 1) // block_m)
    num_pid_n = max(1, (N + block_n - 1) // block_n)
    base_tiles = num_pid_m * num_pid_n
    k_iters = max(1, (K + block_k - 1) // block_k)
    desired_split = max(1, _TARGET_GRID // base_tiles)
    split_k = min(desired_split, k_iters, 16)
    split_k = max(split_k, 1)

    cfg = dict(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        NUM_PID_M=num_pid_m,
        NUM_PID_N=num_pid_n,
        num_warps=4,
        num_stages=2,
        kpack=2,
        waves_per_eu=2,
    )
    _cfg_cache[key] = cfg
    return cfg


def small_m_splitk_matmul_lt(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor = None,
):
    """Single-kernel split-K GEMM that atomically accumulates into ``c``."""
    M, K = a.shape
    _, N = b.shape

    cfg = _pick_config(M, N, K)
    BLOCK_M = cfg["BLOCK_M"]
    BLOCK_N = cfg["BLOCK_N"]
    BLOCK_K = cfg["BLOCK_K"]
    SPLIT_K = cfg["SPLIT_K"]
    NUM_PID_M = cfg["NUM_PID_M"]
    NUM_PID_N = cfg["NUM_PID_N"]
    even_k = (K % BLOCK_K) == 0

    use_atomic = SPLIT_K > 1
    if use_atomic:
        c.zero_()

    grid = (NUM_PID_M * NUM_PID_N * SPLIT_K,)

    _small_m_splitk_atomic[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        NUM_PID_M=NUM_PID_M,
        NUM_PID_N=NUM_PID_N,
        EVEN_K=even_k,
        ATOMIC=use_atomic,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        matrix_instr_nonkdim=16,
        kpack=cfg["kpack"],
        waves_per_eu=cfg["waves_per_eu"],
    )

    if bias is not None:
        c.add_(bias.view(1, -1))

    return c


def is_small_m_eligible(
    M: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_dtype: torch.dtype,
    enable_streamk: bool,
    quantized: bool,
    bias: bool,
) -> bool:
    """Eligibility check for routing through the small-M split-K kernel."""
    if enable_streamk or quantized or bias:
        return False
    if M > 32:
        return False
    supported = (torch.float16, torch.bfloat16, torch.float32)
    if a_dtype not in supported or b_dtype not in supported or c_dtype not in supported:
        return False
    return True
