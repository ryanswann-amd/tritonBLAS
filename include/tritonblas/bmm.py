# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Batched matrix-multiply (single-launch BMM) public entrypoint.

This module owns the entire ``tritonblas.bmm`` dispatch policy so the rank-2
``matmul`` path and the rank-3 batched path can evolve independently.  The
prior implementation lived inside ``matmul.py`` and mixed the two policies
into a single +700-line module; that mixing was flagged in code review as a
god-module / layering violation.

Design summary
--------------

* Public API: ``bmm(a, b, out=None)`` with torch.bmm/torch.matmul shape
  semantics — rank-3, rank-2 broadcast on either operand, rank-1 vector
  promotion.
* Single launch: every BMM invocation produces exactly one Triton kernel
  launch via the ``batched_persistent_matmul`` kernel; the grid is
  ``(tiles_per_batch, B)`` so the batch dimension is parallelised by the
  hardware scheduler rather than serialised by a Python ``for`` loop.  This
  is the entire point of the PR — it eliminates the per-batch launch
  overhead that the prior loop-dispatch path paid.
* Tile selection: delegated to ``OrigamiMatmulSelector`` with
  ``batch=batch`` so the heuristic itself is batch-aware.  No outside-the-
  selector tile fixups live in this module.

There is no fallback to the per-batch loop: by construction the single-launch
path is always at most as expensive as the loop (1 launch <= B launches), and
the bench (`benchmarks/bmm_bench.py`) verifies this on the largest tile shape
in the cohort (B=4 4096^3 fp16 — within measurement noise of the loop).
Re-introducing the loop would re-introduce the very pathology this PR fixes.
"""
from __future__ import annotations

import functools
from typing import Optional

import torch
import triton

from .kernels import batched_persistent_matmul
from .origami import OrigamiMatmulSelector


# ---------------------------------------------------------------------------
# Selector cache
# ---------------------------------------------------------------------------
#
# The non-batched ``matmul`` path explicitly does NOT cache the selector (see
# the commented-out lru_cache in matmul.py); it sees one-shot calls where the
# ~185us Origami pass is amortised over the kernel's runtime.  bmm callers,
# in contrast, reissue identical (M,N,K,B,dtype) problems every step of a
# training/inference loop — for the smallest shapes in the cohort
# (B=32 256x256x256) the per-call kernel time is on the order of the Origami
# pass itself, so caching turns the selector lookup from a hot path into a
# dict probe.
#
# The cache key includes ``B`` because the selector is now batch-aware
# (``OrigamiMatmulSelector(..., batch=B)`` may pick different tiles when B>=2)
# and ``device_str`` because Origami queries hardware properties keyed on
# device index.
@functools.lru_cache(maxsize=1024)
def _bmm_selector_cached(M, N, K, B, a_dtype, b_dtype, c_dtype, device_str):
    return OrigamiMatmulSelector(
        M, N, K, a_dtype, b_dtype, c_dtype,
        torch.device(device_str),
        batch=B,
    )


def _normalize_bmm_strides(t: torch.Tensor, want_b: int):
    """Return ``(stride_b, stride_row, stride_col)`` in elements.

    ``t`` may be rank-2 (broadcast across batches with stride_b == 0) or
    rank-3.  Rank-1 inputs are not handled here — caller upcasts before
    invoking.
    """
    if t.dim() == 2:
        return 0, t.stride(0), t.stride(1)
    if t.dim() == 3:
        if t.shape[0] != want_b:
            raise RuntimeError(
                f"tritonblas.bmm: batch dimension mismatch "
                f"(expected {want_b}, got {t.shape[0]})"
            )
        return t.stride(0), t.stride(1), t.stride(2)
    raise RuntimeError(
        f"tritonblas.bmm: input tensor must be 2D or 3D, got {t.dim()}D"
    )


def _resolve_bmm_shapes(a: torch.Tensor, b: torch.Tensor):
    """Promote rank-1/rank-2 inputs to a canonical 3D BMM problem.

    Returns ``(a3, b3, batch, m, n, k, squeeze_a, squeeze_b)``.  ``a3`` and
    ``b3`` are views suitable for the kernel; ``squeeze_a/squeeze_b``
    indicate whether the corresponding leading dim of the output should be
    squeezed back out (matching torch.matmul rank-promotion semantics).
    """
    if a.dim() == 0 or b.dim() == 0:
        raise RuntimeError("tritonblas.bmm: scalar inputs are not supported")

    squeeze_a = False
    squeeze_b = False
    a3 = a
    b3 = b

    # rank-1 promotion (torch.matmul semantics)
    if a3.dim() == 1:
        a3 = a3.unsqueeze(0)         # (K,) -> (1, K)
        squeeze_a = True
    if b3.dim() == 1:
        b3 = b3.unsqueeze(-1)        # (K,) -> (K, 1)
        squeeze_b = True

    if a3.dim() not in (2, 3) or b3.dim() not in (2, 3):
        raise RuntimeError(
            f"tritonblas.bmm: only 1D/2D/3D inputs supported, got "
            f"{a.dim()}D and {b.dim()}D"
        )

    K_a = a3.shape[-1]
    K_b = b3.shape[-2]
    if K_a != K_b:
        raise RuntimeError(
            f"tritonblas.bmm: incompatible K dimension ({K_a} vs {K_b})"
        )
    M = a3.shape[-2]
    N = b3.shape[-1]
    K = K_a

    batch_a = a3.shape[0] if a3.dim() == 3 else 1
    batch_b = b3.shape[0] if b3.dim() == 3 else 1
    if batch_a != batch_b and batch_a != 1 and batch_b != 1:
        raise RuntimeError(
            f"tritonblas.bmm: incompatible batch dimensions "
            f"({batch_a} vs {batch_b})"
        )
    batch = max(batch_a, batch_b)

    return a3, b3, batch, M, N, K, squeeze_a, squeeze_b


def _bmm_dispatch(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    quantized: bool = False,
):
    """Single-launch batched persistent GEMM kernel call.

    Caller is responsible for shape/dtype validation and ``out`` allocation.
    Tile parameters come from ``OrigamiMatmulSelector(..., batch=B)`` — the
    selector itself is the single owner of the tile-shape policy (rank-2
    and rank-3 alike), so this dispatcher contains no out-of-band tile
    overrides.
    """
    if out.dim() != 3:
        raise RuntimeError("tritonblas.bmm: out tensor must be 3D (B, M, N)")

    B, M, N = out.shape
    K = a.shape[-1]

    if a.shape[-2:] != (M, K):
        raise RuntimeError(
            f"tritonblas.bmm: A trailing dims must be ({M}, {K}), "
            f"got {tuple(a.shape[-2:])}"
        )
    if b.shape[-2:] != (K, N):
        raise RuntimeError(
            f"tritonblas.bmm: B trailing dims must be ({K}, {N}), "
            f"got {tuple(b.shape[-2:])}"
        )

    selector = _bmm_selector_cached(
        M, N, K, B, a.dtype, b.dtype, out.dtype, str(a.device)
    )

    BLK_M = selector.block_m
    BLK_N = selector.block_n
    BLK_K = selector.block_k
    gsize_m = selector.group_m
    num_xcds = selector.num_sms

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0

    chunk_size = gsize_m * gsize_m
    if num_xcds > 0:
        chunk_size = min(chunk_size, max(1, total_tiles // num_xcds))
    else:
        num_xcds = 1

    num_stages = getattr(selector, "num_stages", 2)
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 1

    stride_ab, stride_am, stride_ak = _normalize_bmm_strides(a, B)
    stride_bb, stride_bk, stride_bn = _normalize_bmm_strides(b, B)
    stride_cb, stride_cm, stride_cn = out.stride(0), out.stride(1), out.stride(2)

    if bias is not None:
        if bias.dim() == 1:
            stride_bias_b, stride_bias = 0, bias.stride(0)
        elif bias.dim() == 2:
            stride_bias_b, stride_bias = bias.stride(0), bias.stride(1)
        else:
            raise RuntimeError("tritonblas.bmm: bias must be 1D or 2D")
    else:
        stride_bias_b, stride_bias = 0, 0

    if quantized:
        if a_scale is None or b_scale is None:
            raise RuntimeError(
                "tritonblas.bmm: quantized=True requires a_scale and b_scale"
            )
        stride_a_scale_b = a_scale.stride(0) if a_scale.dim() >= 2 else 0
        stride_b_scale_b = b_scale.stride(0) if b_scale.dim() >= 2 else 0
    else:
        stride_a_scale_b = 0
        stride_b_scale_b = 0

    # Single launch.  The grid is (tiles_per_batch, B) so the batch dim is
    # parallelised by the hardware scheduler — collapsing what was N
    # separate Python-side launches into 1.
    grid = (total_tiles, B)

    batched_persistent_matmul[grid](
        a,
        b,
        out,
        a_scale if quantized else None,
        b_scale if quantized else None,
        bias if bias is not None else None,
        M,
        N,
        K,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        stride_bias_b,
        stride_bias,
        stride_a_scale_b,
        stride_b_scale_b,
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m,
        NUM_SMS=total_tiles,
        NUM_XCDS=num_xcds,
        CHUNK_SIZE=chunk_size,
        BIAS=bias is not None,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=None,
        CACHE_MODIFIER_B=None,
        QUANTIZED=quantized,
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=mfmaInstrSize,
        kpack=kpack,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
    )

    return out


def bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched matrix multiply: ``out[i] = a[i] @ b[i]``.

    Single-launch implementation that replaces the prior per-batch Python
    loop over ``tritonblas.matmul``.  Accepts rank-2 and rank-3 inputs with
    standard broadcasting rules:

        a: (B, M, K) or (M, K)
        b: (B, K, N) or (K, N)

    Rank-1 inputs are promoted via ``torch.matmul`` semantics (the matching
    output dim is squeezed out).  Tile parameters come from
    ``OrigamiMatmulSelector`` with ``batch=B`` so the selector itself owns
    the batched-vs-rank-2 tile policy.

    Args:
        a: Left operand.
        b: Right operand.
        out: Optional pre-allocated output tensor (3D, shape (B, M, N)).
            When provided, autograd is not supported.

    Returns:
        The output tensor.  Shape is ``(B, M, N)`` for the canonical case;
        rank-1 promotions of ``a`` or ``b`` strip the corresponding
        singleton (matching ``torch.matmul`` semantics).
    """
    a3, b3, batch, M, N, K, squeeze_a, squeeze_b = _resolve_bmm_shapes(a, b)

    out_dtype = (out.dtype if out is not None
                 else torch.promote_types(a.dtype, b.dtype))

    if out is None:
        # Use a 3D buffer internally; squeeze before returning if needed.
        out_buf = torch.empty((batch, M, N), dtype=out_dtype, device=a.device)
    else:
        if out.dim() == 2 and (squeeze_a or squeeze_b):
            # Caller passed already-squeezed out; expand the squeezed dim back.
            if squeeze_a and not squeeze_b:
                if out.shape != (batch, N):
                    raise RuntimeError(
                        f"tritonblas.bmm: out shape {tuple(out.shape)} "
                        f"incompatible with broadcast result ({batch}, {N})"
                    )
                out_buf = out.unsqueeze(-2)  # (B, N) -> (B, 1, N)
            elif squeeze_b and not squeeze_a:
                if out.shape != (batch, M):
                    raise RuntimeError(
                        f"tritonblas.bmm: out shape {tuple(out.shape)} "
                        f"incompatible with broadcast result ({batch}, {M})"
                    )
                out_buf = out.unsqueeze(-1)  # (B, M) -> (B, M, 1)
            else:  # both squeezed -> (B,)
                if out.shape != (batch,):
                    raise RuntimeError(
                        f"tritonblas.bmm: out shape {tuple(out.shape)} "
                        f"incompatible with broadcast result ({batch},)"
                    )
                out_buf = out.unsqueeze(-1).unsqueeze(-1)
        elif out.dim() == 3:
            if out.shape != (batch, M, N):
                raise RuntimeError(
                    f"tritonblas.bmm: out shape {tuple(out.shape)} "
                    f"!= expected ({batch}, {M}, {N})"
                )
            out_buf = out
        else:
            raise RuntimeError(
                f"tritonblas.bmm: out must be 2D or 3D, got {out.dim()}D"
            )

    _bmm_dispatch(a3, b3, out_buf)

    if out is None:
        result = out_buf
        if squeeze_a:
            result = result.squeeze(-2)
        if squeeze_b:
            result = result.squeeze(-1)
        return result
    return out
