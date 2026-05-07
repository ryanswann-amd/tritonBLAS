# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
True batched matmul entrypoint.

Replaces the legacy "Python for-loop over rank-2 matmul" dispatch path for
rank-3+ inputs. A single Triton launch processes all batches by mapping the
batch dimension onto grid axis 1 and biasing the per-batch base pointers
inside the kernel. This eliminates per-batch host launch overhead (the cold-
launch component that dominated batched residuals in the K-654 sweep).

Public API:
    bmm(a, b, out=None) -> Tensor
    batched_matmul(a, b, out=None) -> Tensor
        Both expect rank-3 A (B, M, K) and rank-3 B (B, K, N), with optional
        broadcast of either batch dim to 1.

The rank-2 ``matmul`` entrypoint dispatches here automatically when its
operands carry leading batch dims (ndim > 2) so user code paths through
``tritonblas.matmul(x, y)`` get the speedup without source changes.
"""

import functools
from typing import Optional

import torch
import triton

from .kernels import batched_persistent_matmul
from .origami import OrigamiMatmulSelector


def _broadcast_batch_strides(t: torch.Tensor, batch: int) -> int:
    """Return the per-batch byte stride for ``t``, broadcasting size-1 dims to 0."""
    if t.shape[0] == 1 and batch != 1:
        return 0
    return t.stride(0)


def _resolve_batched_inputs(a: torch.Tensor, b: torch.Tensor):
    """Reshape rank-N inputs to rank-3 (B, M, K)/(B, K, N).

    Supports:
      - Both rank-3 with matching batch (or one batch == 1 -> broadcast).
      - Rank > 3: flatten leading dims into a single batch dim. The two operands
        must agree on the leading dims (or have broadcast compatibility on
        them).

    Returns:
        a3, b3, batch, output_leading_shape

    The caller is responsible for reshaping the rank-3 output back to
    ``output_leading_shape + (M, N)``.
    """
    if a.ndim < 3 and b.ndim < 3:
        raise ValueError(
            f"batched_matmul requires at least one rank-3+ tensor, "
            f"got a.ndim={a.ndim}, b.ndim={b.ndim}"
        )

    # Promote a 2-D operand to rank 3 with a singleton batch (broadcast).
    if a.ndim == 2:
        a = a.unsqueeze(0)
    if b.ndim == 2:
        b = b.unsqueeze(0)

    # Broadcast leading dims (everything except the last 2 axes) using the
    # standard torch broadcasting semantics, then flatten to a single batch.
    a_lead = a.shape[:-2]
    b_lead = b.shape[:-2]
    out_lead = torch.broadcast_shapes(a_lead, b_lead)

    a_full = a.expand(*out_lead, a.shape[-2], a.shape[-1])
    b_full = b.expand(*out_lead, b.shape[-2], b.shape[-1])

    batch = 1
    for d in out_lead:
        batch *= d

    # We need contiguity in the inner two dims (or at least standard 2-D
    # strides) for the kernel; flatten the leading dims into a single batch
    # axis. ``reshape`` materialises a contiguous tensor only if the leading
    # dims aren't already contiguous; broadcasted batch dims with stride 0 are
    # preserved as a stride-0 batch in the rank-3 view by going through
    # ``view`` when possible.
    try:
        a3 = a_full.view(batch, a.shape[-2], a.shape[-1])
    except RuntimeError:
        a3 = a_full.reshape(batch, a.shape[-2], a.shape[-1])
    try:
        b3 = b_full.view(batch, b.shape[-2], b.shape[-1])
    except RuntimeError:
        b3 = b_full.reshape(batch, b.shape[-2], b.shape[-1])

    return a3, b3, batch, tuple(out_lead)


# Origami selector construction is non-trivial (per the matmul.py comment it
# costs "several microseconds" and was originally meant to be LRU-cached).
# Cache on (M, N, K, dtype tuple, device index) so the per-launch overhead
# disappears on the second and subsequent calls for any given problem.
@functools.lru_cache(maxsize=2048)
def _cached_selector(M, N, K, a_dtype, b_dtype, c_dtype, device_idx):
    device = torch.device(f"cuda:{device_idx}")
    return OrigamiMatmulSelector(
        M, N, K, a_dtype, b_dtype, c_dtype, device,
        mx_block_size=0, streamk=False, num_stages=2,
    )


def _make_batched_selector(M, N, K, a_dtype, b_dtype, c_dtype, device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return _cached_selector(M, N, K, a_dtype, b_dtype, c_dtype, idx)


# Number of CUs on a typical MI300X / gfx942 chip. Used as a soft target for
# the per-batch tile count (batch * tiles >= _CU_TARGET means the GPU stays
# saturated with at least one wave of programs).
_CU_TARGET = 304


def _batched_block_override(selector, M, N, K, batch):
    """Override Origami's block selection for batched workloads.

    Origami picks blocks for a single rank-2 GEMM where exposing more tiles
    helps fill the GPU. For batched workloads the batch dimension already
    multiplies the number of programs by ``batch``, so we can afford larger
    per-tile blocks (better MFMA utilisation, less per-tile loop overhead).

    Strategy: pick the *largest* block size that still produces at least one
    full wave of programs (``batch * tiles_per_batch >= _CU_TARGET``) and that
    fits within M/N. This tracks the K-678 MI300X sweep results, where:
      * 256x256x64 wins on shapes large enough that batch * tiles_256 >= ~CU
      * 128x128x64 wins on shapes where 256x256 leaves the GPU under-occupied
      * Origami's choice is preserved otherwise (tiny shapes, batch == 1)

    Returns ``(block_m, block_n, block_k, group_m)``.
    """
    blk_m, blk_n, blk_k, gm = (
        selector.block_m, selector.block_n, selector.block_k, selector.group_m,
    )

    if batch <= 1:
        return blk_m, blk_n, blk_k, gm
    if M < 128 or N < 128:
        return blk_m, blk_n, blk_k, gm

    def tiles(bm, bn):
        return ((M + bm - 1) // bm) * ((N + bn - 1) // bn)

    # Try 256x256x64 first: largest tile that's still well-supported by the
    # MFMA pipeline. Accept if it keeps the GPU saturated.
    if M >= 256 and N >= 256 and batch * tiles(256, 256) >= _CU_TARGET:
        return 256, 256, 64, max(gm, 8)

    # Otherwise 128x128x64: a strong default for medium shapes.
    if batch * tiles(128, 128) >= _CU_TARGET:
        return 128, 128, 64, max(gm, 4)

    # Fall back to Origami's pick if neither candidate saturates the GPU.
    return blk_m, blk_n, blk_k, gm


def _launch_batched(a3: torch.Tensor, b3: torch.Tensor, c3: torch.Tensor):
    """Dispatch one Triton launch over a rank-3 (batch, M, K) x (batch, K, N) GEMM."""
    batch, M, K = a3.shape
    batch_b, K_b, N = b3.shape
    assert K == K_b, f"K mismatch: A has K={K}, B has K={K_b}"
    assert batch == c3.shape[0]
    assert M == c3.shape[1]
    assert N == c3.shape[2]
    # Either batches match or one operand broadcasts from batch=1.
    assert batch == batch_b or batch_b == 1 or batch == 1, (
        f"Incompatible batch dims: a={batch} b={batch_b}"
    )

    selector = _make_batched_selector(M, N, K, a3.dtype, b3.dtype, c3.dtype, a3.device)

    BLK_M, BLK_N, BLK_K, gsize_m = _batched_block_override(
        selector, M, N, K, batch=max(batch, batch_b)
    )
    num_xcds = selector.num_sms

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0

    # Persistent-style: NUM_SMS = total_tiles_per_batch. Grid axis 1 = batch.
    num_stages = getattr(selector, "num_stages", 2)
    num_warps = 8
    waves_per_eu = 0
    mfma_instr_size = 16
    kpack = 1

    chunk_size = gsize_m * gsize_m
    if num_xcds > 0:
        chunk_size = min(chunk_size, max(1, total_tiles // num_xcds))
    else:
        num_xcds = 1

    # Per-batch strides (0 if broadcast).
    stride_az = _broadcast_batch_strides(a3, batch)
    stride_bz = _broadcast_batch_strides(b3, batch)
    stride_cz = c3.stride(0)

    grid = (total_tiles, batch)

    batched_persistent_matmul[grid](
        a3,
        b3,
        c3,
        None,  # A_scale_ptr (unused)
        None,  # B_scale_ptr (unused)
        None,  # bias_ptr (unused)
        M,
        N,
        K,
        stride_az,
        a3.stride(1),
        a3.stride(2),
        stride_bz,
        b3.stride(1),
        b3.stride(2),
        stride_cz,
        c3.stride(1),
        c3.stride(2),
        0,  # stride_bias (unused)
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m,
        NUM_SMS=total_tiles,
        NUM_XCDS=num_xcds,
        CHUNK_SIZE=chunk_size,
        BIAS=False,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=None,
        CACHE_MODIFIER_B=None,
        QUANTIZED=False,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=mfma_instr_size,
        kpack=kpack,
    )

    return c3


def batched_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute C = A @ B where A and B carry one or more leading batch dims.

    Equivalent to ``torch.matmul`` for rank-3+ inputs (and to ``torch.bmm`` for
    exactly rank-3 inputs with matching batch sizes), but executed as a single
    Triton kernel launch instead of one launch per batch slice.

    Args:
        a: Tensor of shape ``(*batch_a, M, K)``.
        b: Tensor of shape ``(*batch_b, K, N)``.
        out: Optional pre-allocated output tensor of broadcast shape
            ``(*broadcast(batch_a, batch_b), M, N)``.

    Returns:
        Tensor of shape ``(*broadcast(batch_a, batch_b), M, N)``.
    """
    if a.dtype != b.dtype:
        raise ValueError(f"a.dtype ({a.dtype}) must equal b.dtype ({b.dtype})")
    if a.device != b.device:
        raise ValueError(f"a.device ({a.device}) must equal b.device ({b.device})")

    a3, b3, batch, out_lead = _resolve_batched_inputs(a, b)
    M = a3.shape[1]
    N = b3.shape[2]
    out_dtype = a.dtype

    if out is None:
        c3 = torch.empty((batch, M, N), device=a.device, dtype=out_dtype)
    else:
        # Validate the user's out tensor and reshape into rank-3.
        expected_shape = tuple(out_lead) + (M, N)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out has shape {tuple(out.shape)}, expected {expected_shape}"
            )
        try:
            c3 = out.view(batch, M, N)
        except RuntimeError:
            c3 = out.reshape(batch, M, N)

    _launch_batched(a3, b3, c3)

    if out is None:
        # Reshape rank-3 back to the user-visible leading dims.
        if len(out_lead) == 1 and out_lead[0] == batch:
            return c3
        return c3.view(*out_lead, M, N)
    return out


# Alias matching torch.bmm semantics (rank-3 only, no broadcasting).
def bmm(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """torch.bmm-compatible entrypoint: rank-3 A and B with matching batch."""
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError(
            f"bmm requires rank-3 inputs; got a.ndim={a.ndim}, b.ndim={b.ndim}"
        )
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"bmm batch dims must match; got a.shape[0]={a.shape[0]}, "
            f"b.shape[0]={b.shape[0]}"
        )
    return batched_matmul(a, b, out=out)
