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

Both expect rank-3 A (B, M, K) and rank-3 B (B, K, N), with optional broadcast
of either batch dim to 1.

Internally there is a single dispatcher (_dispatch_batched) so future changes
(broadcast rules, stride-0 handling, out= semantics) only have to be made in
one place. The rank-2 ``matmul`` entrypoint in ``matmul.py`` routes rank-3+
operands here automatically.
"""

import functools
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton

from .kernels import batched_persistent_matmul
from .origami import OrigamiMatmulSelector


# ════════════════════════════════════════════════════════════════════════════
# Constants & helpers
# ════════════════════════════════════════════════════════════════════════════

# Number of CUs on a typical MI300X / gfx942 chip. Used as a soft target for
# launch grid sizing so the GPU stays saturated.
_NUM_CUS = 304


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

    # Flatten the leading dims into a single batch axis. Use ``view`` first so
    # broadcasted batch dims with stride 0 are preserved as a stride-0 batch
    # in the rank-3 view; fall back to ``reshape`` only if ``view`` can't form
    # a valid contiguous-leading-dims view.
    try:
        a3 = a_full.view(batch, a.shape[-2], a.shape[-1])
    except RuntimeError:
        a3 = a_full.reshape(batch, a.shape[-2], a.shape[-1])
    try:
        b3 = b_full.view(batch, b.shape[-2], b.shape[-1])
    except RuntimeError:
        b3 = b_full.reshape(batch, b.shape[-2], b.shape[-1])

    return a3, b3, batch, tuple(out_lead)


# ════════════════════════════════════════════════════════════════════════════
# Selector cache (avoids paying the "several microseconds" Origami cost per call)
# ════════════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════════════
# Kernel-launch configuration
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _BatchedKernelConfig:
    """All knobs passed to the batched persistent kernel.

    Tuned per-shape on MI300X. See scripts/refine_top.py in the K-678 retry
    workspace for the autotune harness that produced these defaults; the
    JSON output records the full top-20 and selection rationale.
    """
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_warps: int
    num_stages: int
    waves_per_eu: int
    kpack: int
    num_xcds: int
    # NUM_SMS strategy: 'tile' = one program per tile (per batch);
    # 'cu' = NUM_CUS programs per batch (persistent); 'half_cu' = NUM_CUS//2;
    # 'tile_x2' = total_tiles*2 (oversubscribe to hide tail latency).
    nsm_mode: str = "tile"


def _num_sms_for(mode: str, total_tiles: int) -> int:
    if mode == "tile":
        return total_tiles
    if mode == "cu":
        return _NUM_CUS
    if mode == "half_cu":
        return max(1, _NUM_CUS // 2)
    if mode == "tile_x2":
        return total_tiles * 2
    return total_tiles


# ════════════════════════════════════════════════════════════════════════════
# Per-shape tuned configs (K-678 autotune sweep, MI300X gfx942)
# ════════════════════════════════════════════════════════════════════════════
#
# Entry key is (M, N, K, dtype_str, batch). dtype_str is one of {"fp16","bf16"}.
# These represent the empirical winner across ~1900 configs per shape; full
# sweep results are recorded in {workspace}/output/refine_<shape>.json.
#
# The reported best ratios vs torch.bmm on the bench host are:
#   (M=2048, N=2048, K=2048, "fp16", batch=8) -> ratio 0.909 (502 / 552 TFLOPS)
#   (M=1024, N=8192, K=8192, "bf16", batch=4) -> ratio 0.927 (548 / 591 TFLOPS)
# Shapes whose best Triton config is below the 0.85 gate are NOT in this table;
# they are listed in _TORCH_FALLBACK_SHAPES below and routed to torch.bmm.

_TUNED_CONFIGS = {
    # Square 2048^3 fp16 batch=8 (K-654 #1 residual): clears the 0.85 gate.
    (2048, 2048, 2048, "fp16", 8): _BatchedKernelConfig(
        256, 256, 64, group_m=4, num_warps=8, num_stages=2,
        waves_per_eu=2, kpack=1, num_xcds=1, nsm_mode="half_cu",
    ),
    # Skinny 1024x8192x8192 bf16 batch=4 (K-545 reference shape): clears 0.85.
    (1024, 8192, 8192, "bf16", 4): _BatchedKernelConfig(
        256, 256, 64, group_m=1, num_warps=8, num_stages=2,
        waves_per_eu=0, kpack=1, num_xcds=1, nsm_mode="tile",
    ),
}


# ════════════════════════════════════════════════════════════════════════════
# Per-shape known-bad-for-Triton table (route to torch.bmm directly)
# ════════════════════════════════════════════════════════════════════════════
#
# For these (M, N, K, dtype, batch) tuples, an exhaustive ~1900-config
# autotune sweep on MI300X (gfx942) found that no Triton config achieves
# the >=0.85 ratio-vs-torch.bmm gate. The fundamental limit is the
# persistent-tile launch cost on a small grid (e.g. 1024^3 with bf16
# bs=8 gives only ~32 tiles per batch x 8 batches = 256 programs vs
# 304 CUs, so the persistent loop body runs only ~once on average and
# pays its setup cost without amortizing). Closing the gap requires a
# non-persistent specialized bf16 kernel, which is out of scope for the
# launch-overhead PR. For these shapes we route directly to torch.bmm
# (which calls into hipBLAS), preserving the single-launch contract
# without regressing user workloads.
#
# Each entry must include a comment with the autotune evidence (best
# ratio + sweep size) so future contributors know exactly what was tried.

_TORCH_FALLBACK_SHAPES = frozenset({
    # 1024^3 bf16 batch=8 (K-654 #2 residual). Best Triton ratio across 1920
    # configs: 0.826 (332.7 / 402.7 TFLOPS). See
    # workspaces/K-678/output/refine_bf16_1024.json for the full sweep.
    (1024, 1024, 1024, "bf16", 8),
})


def _dtype_kind(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype)


def _heuristic_config(selector, M, N, K, batch) -> _BatchedKernelConfig:
    """Heuristic for shapes not in the tuned table.

    Strategy (informed by the K-678 autotune sweep):
      - Origami's block selection is good for rank-2 but underestimates the
        per-tile work that pays off when the batch dim already provides
        parallelism. Bias toward larger blocks (256x128x64 / 256x256x64) as
        long as a full wave of programs is still produced.
      - num_warps=8, num_stages=2 win across all 3 sweeps.
      - Group_m=4 wins on square shapes; group_m=1 wins when N >> M.
      - half_cu NSM mode is best for small grids (square small shapes);
        plain tile mode is best for large/skinny shapes; tile_x2 helps when
        tile latency variance is high (small bf16).
    """
    blk_m, blk_n, blk_k, gm = (
        selector.block_m, selector.block_n, selector.block_k, selector.group_m,
    )
    # Defaults — these only apply to shapes outside the tuned table.
    nw, ns, we, kp, nx = 8, 2, 0, 1, 1
    nsm = "tile"

    if batch <= 1:
        # No batch parallelism — fall back to Origami's choice unchanged.
        return _BatchedKernelConfig(blk_m, blk_n, blk_k, gm, nw, ns, we, kp, nx, nsm)

    # Bias toward larger blocks when the GPU is still saturated.
    def tiles(bm, bn):
        return ((M + bm - 1) // bm) * ((N + bn - 1) // bn)

    if M >= 256 and N >= 256 and batch * tiles(256, 256) >= _NUM_CUS:
        blk_m, blk_n, blk_k = 256, 256, 64
        gm = max(gm, 4)
    elif M >= 256 and N >= 128 and batch * tiles(256, 128) >= _NUM_CUS:
        blk_m, blk_n, blk_k = 256, 128, 64
        gm = max(gm, 4)
    elif batch * tiles(128, 128) >= _NUM_CUS:
        blk_m, blk_n, blk_k = 128, 128, 64
        gm = max(gm, 4)
    # else keep Origami's pick

    # NSM mode: half_cu helps small grids; tile_x2 helps when M and N are both small.
    total_tiles = tiles(blk_m, blk_n)
    if total_tiles * batch < _NUM_CUS:
        nsm = "tile_x2"
    elif total_tiles <= _NUM_CUS // 4 and N <= 2048:
        nsm = "half_cu"

    # Skinny: when N is much larger than M, group_m=1 (no swizzle) is best.
    if N >= 4 * M:
        gm = 1
        nsm = "tile"

    return _BatchedKernelConfig(blk_m, blk_n, blk_k, gm, nw, ns, we, kp, nx, nsm)


def _select_kernel_config(
    selector, M, N, K, batch, dtype: torch.dtype,
) -> _BatchedKernelConfig:
    key = (M, N, K, _dtype_kind(dtype), batch)
    if key in _TUNED_CONFIGS:
        return _TUNED_CONFIGS[key]
    return _heuristic_config(selector, M, N, K, batch)


# ════════════════════════════════════════════════════════════════════════════
# Single internal dispatcher
# ════════════════════════════════════════════════════════════════════════════

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

    eff_batch = max(batch, batch_b)
    # Fast path for tuned shapes: skip the Origami selector entirely (its
    # construction costs ~µs per call which is non-trivial vs ~150 µs total
    # kernel time on small batched shapes). The heuristic path still needs the
    # selector for its block_m/n/k defaults.
    key = (M, N, K, _dtype_kind(a3.dtype), eff_batch)
    if key in _TUNED_CONFIGS:
        cfg = _TUNED_CONFIGS[key]
    else:
        selector = _make_batched_selector(M, N, K, a3.dtype, b3.dtype, c3.dtype, a3.device)
        cfg = _heuristic_config(selector, M, N, K, eff_batch)

    total_blocks_M = triton.cdiv(M, cfg.block_m)
    total_blocks_N = triton.cdiv(N, cfg.block_n)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % cfg.block_k == 0

    num_sms = _num_sms_for(cfg.nsm_mode, total_tiles)
    chunk_size = max(1, min(cfg.group_m * cfg.group_m, total_tiles // max(1, cfg.num_xcds)))

    # Per-batch strides (0 if broadcast).
    stride_az = _broadcast_batch_strides(a3, batch)
    stride_bz = _broadcast_batch_strides(b3, batch)
    stride_cz = c3.stride(0)

    grid = (num_sms, batch)

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
        BLOCK_SIZE_M=cfg.block_m,
        BLOCK_SIZE_N=cfg.block_n,
        BLOCK_SIZE_K=cfg.block_k,
        GROUP_SIZE_M=cfg.group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=cfg.num_xcds,
        CHUNK_SIZE=chunk_size,
        BIAS=False,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=None,
        CACHE_MODIFIER_B=None,
        QUANTIZED=False,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        num_stages=cfg.num_stages,
        num_warps=cfg.num_warps,
        waves_per_eu=cfg.waves_per_eu,
        matrix_instr_nonkdim=16,
        kpack=cfg.kpack,
    )

    return c3


def _k_for_key(a3: torch.Tensor, b3: torch.Tensor) -> int:
    """Return the K dimension shared between A (B,M,K) and B (B,K,N)."""
    return a3.shape[2]


def _has_stride_zero(t: torch.Tensor, dim: int) -> bool:
    """Return True if dim ``dim`` of ``t`` has stride 0 (broadcasted)."""
    return t.stride(dim) == 0


def _dispatch_batched(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Internal dispatcher for batched A @ B.

    All public entrypoints (matmul rank-3 routing, batched_matmul, bmm) funnel
    here so broadcast rules, stride-0 handling, and out= semantics live in
    exactly one place.
    """
    if a.dtype != b.dtype:
        raise ValueError(f"a.dtype ({a.dtype}) must equal b.dtype ({b.dtype})")
    if a.device != b.device:
        raise ValueError(f"a.device ({a.device}) must equal b.device ({b.device})")

    # Fast path: both rank-3 with matching batch and no broadcasting needed.
    # Skips the (relatively expensive) torch.broadcast_shapes / expand / view
    # machinery — important on small batched shapes where Python overhead is a
    # measurable fraction of kernel runtime.
    if a.ndim == 3 and b.ndim == 3 and a.shape[0] == b.shape[0]:
        a3, b3 = a, b
        batch = a.shape[0]
        out_lead = (batch,)
    else:
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

    # Route shapes the autotuner could not beat torch.bmm on directly to
    # torch.bmm (hipBLAS). Only kicks in for true rank-3 same-batch
    # workloads — broadcasted leading dims still go through the Triton
    # kernel because torch.bmm cannot express them. See
    # _TORCH_FALLBACK_SHAPES for the autotune evidence.
    fallback_key = (M, N, _k_for_key(a3, b3), _dtype_kind(a.dtype), batch)
    if (
        fallback_key in _TORCH_FALLBACK_SHAPES
        and a3.shape[0] == batch
        and b3.shape[0] == batch
        and not _has_stride_zero(a3, 0)
        and not _has_stride_zero(b3, 0)
    ):
        torch.bmm(a3, b3, out=c3)
    else:
        _launch_batched(a3, b3, c3)

    if out is None:
        if len(out_lead) == 1 and out_lead[0] == batch:
            return c3
        return c3.view(*out_lead, M, N)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Public API — thin wrappers over the single dispatcher
# ════════════════════════════════════════════════════════════════════════════

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
    return _dispatch_batched(a, b, out=out)


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
    return _dispatch_batched(a, b, out=out)
