# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Split-K GEMM kernel specialized for small-M decode shapes.

Targeted at the M<=8, K>=4096 sub-cohort where the persistent K-144 small-M
kernel under-utilises the MI300X XCDs because a single CTA per (M, N) tile
serialises the K-reduction. Splitting K across SPLIT_K CTAs that write to a
temporary [SPLIT_K, M, N] fp32 buffer, followed by a tiny epilogue kernel
that sums the buffer to the output dtype, lets us fan out across XCDs.

Activated only via the narrow gate in ``tritonblas.matmul`` for
M in {1,2,4,8} x K in {4096,8192} x N in {1024,2048,4096,8192} and fp16/bf16.
Outside that gate the K-144 persistent path remains untouched.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_splitk_smallm_kernel(
    A,
    B,
    P,                       # [SPLIT_K, M, N] fp32 scratch buffer
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_psk,
    stride_pm,
    stride_pn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Split-K small-M kernel.

    Grid: (num_n_blocks, SPLIT_K). Each CTA owns
      - one BLOCK_N column slab of N
      - one of SPLIT_K equal slices of K (gate guarantees K % SPLIT_K == 0)

    Accumulator is fp32 and stored to the [SPLIT_K, M, N] scratch.
    A separate reduction kernel sums along axis 0 and writes the output dtype.
    """
    pid_n = tl.program_id(0)
    pid_sk = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Gate guarantees K is divisible by SPLIT_K.
    K_per_split = K // SPLIT_K
    k_start = pid_sk * K_per_split

    a_ptrs = A + (
        offs_m[:, None] * stride_am
        + (k_start + offs_k[None, :]) * stride_ak
    )
    b_ptrs = B + (
        (k_start + offs_k[:, None]) * stride_bk
        + offs_n[None, :] * stride_bn
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Inner K-loop over BLOCK_SIZE_K chunks within this split's slice.
    # K_per_split is always a multiple of BLOCK_SIZE_K under the gate
    # ({4096,8192} / {2,4,8} / {64,128}), so no K-mask is required.
    for _ in range(0, K_per_split, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=tl.float32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    p_ptrs = (
        P
        + pid_sk * stride_psk
        + offs_m[:, None] * stride_pm
        + offs_n[None, :] * stride_pn
    )
    tl.store(p_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _splitk_reduction_kernel(
    P,
    C,
    M,
    N,
    stride_psk,
    stride_pm,
    stride_pn,
    stride_cm,
    stride_cn,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Sum the [SPLIT_K, M, N] scratch along axis 0 and write output dtype."""
    pid_n = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_mn = mask_m[:, None] & mask_n[None, :]

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    p_base = (
        P
        + offs_m[:, None] * stride_pm
        + offs_n[None, :] * stride_pn
    )
    for sk in tl.static_range(SPLIT_K):
        acc += tl.load(p_base + sk * stride_psk, mask=mask_mn, other=0.0)

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask_mn)


# ----------------------------------------------------------------------------
# Per-bucket SPLIT_K / BLOCK_N choices for the gated 32 shapes.
#
# Picked from the c42 sweep recorded at ``output/splitk_sweep.csv`` in the
# K-513 workspace. Cells without a recorded winner fall back to the persistent
# kernel (gate returns None).
#
# Key: (M, K, dtype_str) -> {N: (SPLIT_K, BLOCK_N, BLOCK_K)}
# ----------------------------------------------------------------------------

# Per-bucket winners chosen by the c42 sweep
# (output/splitk_winners.csv in the K-513 workspace).
# Format: (M, K, dtype_str) -> {N: (SPLIT_K, BLOCK_N, BLOCK_K)}
_BUCKET_WINNERS = {
    (1, 4096, "fp16"): {1024: (4, 256, 64), 2048: (4, 256, 64),
                         4096: (8, 64, 128), 8192: (4, 128, 128)},
    (1, 8192, "fp16"): {1024: (8, 128, 128), 2048: (4, 64, 128),
                         4096: (8, 128, 128), 8192: (4, 64, 64)},
    (2, 4096, "fp16"): {1024: (8, 64, 64),  2048: (8, 128, 128),
                         4096: (8, 128, 64), 8192: (4, 128, 64)},
    (2, 8192, "fp16"): {1024: (4, 128, 128), 2048: (4, 64, 128),
                         4096: (8, 128, 128), 8192: (4, 64, 64)},
    (4, 4096, "fp16"): {1024: (4, 64, 128),  2048: (2, 64, 128),
                         4096: (2, 128, 128), 8192: (8, 128, 128)},
    (4, 8192, "fp16"): {1024: (8, 64, 128),  2048: (8, 128, 64),
                         4096: (4, 64, 64),   8192: (4, 64, 64)},
    (8, 4096, "fp16"): {1024: (4, 128, 128), 2048: (8, 256, 64),
                         4096: (8, 256, 64),  8192: (8, 128, 128)},
    (8, 8192, "fp16"): {1024: (8, 256, 64),  2048: (4, 64, 128),
                         4096: (8, 128, 64),  8192: (4, 64, 64)},
    (1, 4096, "bf16"): {1024: (4, 64, 128),  2048: (4, 256, 64),
                         4096: (8, 128, 128), 8192: (8, 64, 128)},
    (1, 8192, "bf16"): {1024: (4, 64, 128),  2048: (4, 64, 128),
                         4096: (8, 128, 64),  8192: (4, 64, 64)},
    (2, 4096, "bf16"): {1024: (8, 64, 64),   2048: (8, 128, 128),
                         4096: (4, 256, 64),  8192: (8, 64, 128)},
    (2, 8192, "bf16"): {1024: (8, 128, 128), 2048: (8, 256, 64),
                         4096: (8, 128, 128), 8192: (4, 64, 64)},
    (4, 4096, "bf16"): {1024: (2, 128, 128), 2048: (2, 128, 128),
                         4096: (2, 128, 128), 8192: (8, 256, 64)},
    (4, 8192, "bf16"): {1024: (4, 64, 128),  2048: (4, 128, 128),
                         4096: (8, 256, 64),  8192: (4, 64, 64)},
    (8, 4096, "bf16"): {1024: (2, 64, 128),  2048: (8, 128, 128),
                         4096: (4, 256, 64),  8192: (4, 64, 128)},
    (8, 8192, "bf16"): {1024: (4, 64, 128),  2048: (4, 64, 128),
                         4096: (4, 64, 128),  8192: (4, 64, 64)},
}


def _default_winner(M, N, K):
    if K == 8192:
        split_k = 8
    elif K == 4096:
        split_k = 4
    else:
        return None
    block_n = 128
    block_k = 64
    return split_k, block_n, block_k


def get_splitk_smallm_config(M, N, K, dtype):
    """Return (SPLIT_K, BLOCK_N, BLOCK_K) for the gated bucket, or None.

    The gate enforced upstream guarantees that this function is only ever
    called for the 32 in-scope shapes; we still defensively return None for
    anything we have no winner for so callers can fall back.
    """
    dtype_str = "fp16" if dtype == torch.float16 else "bf16"
    bucket = _BUCKET_WINNERS.get((M, K, dtype_str))
    if bucket is not None:
        winner = bucket.get(N)
        if winner is not None:
            return winner
    return _default_winner(M, N, K)


def install_winners(table):
    """Install per-bucket winners from a sweep result.

    ``table`` is a dict {(M, K, dtype_str): {N: (SPLIT_K, BLOCK_N, BLOCK_K)}}.
    """
    _BUCKET_WINNERS.clear()
    _BUCKET_WINNERS.update(table)


# ----------------------------------------------------------------------------
# Host-side launcher
# ----------------------------------------------------------------------------


# Cache the temp buffer so back-to-back small-M decodes don't re-allocate.
_PSCRATCH_CACHE = {}


def _get_pscratch(split_k, block_m, n, device):
    key = (split_k, block_m, n, device)
    buf = _PSCRATCH_CACHE.get(key)
    if buf is None:
        buf = torch.empty(
            (split_k, block_m, n), device=device, dtype=torch.float32
        )
        _PSCRATCH_CACHE[key] = buf
    return buf


def splitk_smallm_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    split_k: int,
    block_n: int,
    block_k: int,
    num_warps: int = 4,
    num_stages: int = 2,
):
    """Launch the split-K small-M GEMM (a @ b -> c).

    a: (M, K)   b: (K, N)   c: (M, N) — c may be any output dtype that the
    epilogue can cast to from fp32 (fp16/bf16/fp32).
    """
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape
    assert K % split_k == 0, "Gate must guarantee K % SPLIT_K == 0"
    K_per_split = K // split_k
    assert K_per_split % block_k == 0, "K_per_split must be a multiple of BLOCK_K"

    # M is small (<=8); pad BLOCK_M to 16 (smallest mfma-friendly tile).
    BLOCK_M = 16

    # Allocate / reuse the [SPLIT_K, BLOCK_M, N] fp32 scratch.
    P = _get_pscratch(split_k, BLOCK_M, N, a.device)

    num_n_blocks = triton.cdiv(N, block_n)

    grid_main = (num_n_blocks, split_k)
    _matmul_splitk_smallm_kernel[grid_main](
        a,
        b,
        P,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        P.stride(0),
        P.stride(1),
        P.stride(2),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        SPLIT_K=split_k,
        ALLOW_TF32=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    grid_red = (num_n_blocks,)
    _splitk_reduction_kernel[grid_red](
        P,
        c,
        M,
        N,
        P.stride(0),
        P.stride(1),
        P.stride(2),
        c.stride(0),
        c.stride(1),
        SPLIT_K=split_k,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=block_n,
        num_warps=4,
        num_stages=1,
    )

    return c


# ----------------------------------------------------------------------------
# Dispatch gate
# ----------------------------------------------------------------------------

_GATED_M = {1, 2, 4, 8}
_GATED_K = {4096, 8192}
_GATED_N = {1024, 2048, 4096, 8192}
_GATED_DTYPES = {torch.float16, torch.bfloat16}

# Master toggle. Tests / benches flip this to compare gated vs ungated runs
# without monkey-patching internals. Set via set_gate_enabled().
#
# Default: DISABLED. The K-513 192-shape cohort comparison on MI300X
# (rocm7.2 / pytorch 2.10) showed end-to-end median latency of every gated
# bucket regressing 1-5% versus the K-144 persistent kernel. Per-bucket data
# is in ``output/cohort_192_compare.csv`` in the K-513 workspace. The kernel
# itself is correct (see ``test_splitk_smallm_dispatch.py``, 115 cases) and
# the gate machinery is here so that future tuning (e.g., reduced launch
# overhead, different SPLIT_K choices, or a different driver) can re-enable
# dispatch by flipping this flag and/or updating ``_BUCKET_WINNERS`` without
# any further refactor. As of K-513 the gate is opt-in only.
_GATE_ENABLED = False


def set_gate_enabled(enabled: bool) -> bool:
    """Enable or disable the K-513 split-K dispatch gate.

    Returns the previous value. Used by the 192-cohort comparison harness so
    it can measure baseline vs gated without monkey-patching ``_BUCKET_WINNERS``.
    """
    global _GATE_ENABLED
    prev = _GATE_ENABLED
    _GATE_ENABLED = bool(enabled)
    return prev


def is_gate_enabled() -> bool:
    return _GATE_ENABLED


def should_dispatch_splitk_smallm(M, N, K, a_dtype, b_dtype, c_dtype):
    """Narrow gate: True only for the 32 in-scope shapes."""
    if not _GATE_ENABLED:
        return False
    if a_dtype != b_dtype or a_dtype != c_dtype:
        return False
    if a_dtype not in _GATED_DTYPES:
        return False
    if M not in _GATED_M:
        return False
    if K not in _GATED_K:
        return False
    if N not in _GATED_N:
        return False
    return True


__all__ = [
    "splitk_smallm_matmul",
    "should_dispatch_splitk_smallm",
    "get_splitk_smallm_config",
    "install_winners",
    "set_gate_enabled",
    "is_gate_enabled",
    "_matmul_splitk_smallm_kernel",
    "_splitk_reduction_kernel",
]
