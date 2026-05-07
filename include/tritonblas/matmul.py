import functools
import os
import random
import time
from typing import Any, Dict, Optional, Tuple

import torch
from torch.library import triton_op, wrap_triton
from torch._subclasses.fake_tensor import is_fake
import triton

from .kernels import persistent_matmul, ws_persistent_matmul, streamk_matmul, ws_streamk_matmul
from .kernels.fp4_matmul import fp4_matmul
from .origami import OrigamiMatmulSelector, kpack_for_tile, KPACK2_TILE_AREA_THRESHOLD
from .config import MatmulConfig, matmul_preamble, COUNTER_STRIDE
from . import autotune_cache as _autotune_cache


# K-595 (S-002): T7 hardware-knob diagnostic shim.
#
# The K-595 sweep falsified the per-shape `num_warps`/`waves_per_eu`/
# `mfma_instr_size` lever class for the residual cohort (see the
# `_hipblaslt_shape_override` header comment in origami.py for the
# empirical record). The full per-shape registry that was prototyped is
# therefore NOT shipped. What survives is this two-layer resolver:
#
#   1. TRITONBLAS_FORCE_<KNOB> environment variable (diagnostic / sweep).
#   2. callsite_default — today's hardcoded value (or the adaptive
#      expression for kpack).
#
# This lets a sweep harness fix a knob across all shapes without code
# changes if a future ISA-codegen patch surfaces a new lever.
#
# Why the bare ``except (TypeError, ValueError): return default``:
# this is an INTENTIONAL diagnostic shim, not swallowed-error tech debt.
# The kernel-launch path runs at thousands of calls per second. Operators
# routinely write things like ``TRITONBLAS_FORCE_NUM_WARPS=auto`` or
# leave shell-quoted floats in their env. If a malformed value escaped as
# ``ValueError``, every single matmul call in that process would crash —
# strictly worse than running with the validated default. The contract
# is: malformed env input is observable via ``$?=0 + default applied``,
# not via an exception that nukes the application.
#
# Performance: results are memoized on the (env_name, env_value, default)
# tuple so the hot launch path does at most one ``os.environ.get`` plus a
# dict lookup per call (no repeated ``int()`` conversion). The cache is
# bounded by the number of distinct knob env-vars (4) × distinct defaults
# (small, finite) — there is no leak risk.
_KNOB_CACHE: Dict[Tuple[str, Optional[str], Any], Any] = {}
_KNOB_CACHE_MISS = object()  # sentinel — distinguishes "absent" from "is None"


def _resolve_knob(env_name: str, default):
    val = os.environ.get(env_name)
    key = (env_name, val, default)
    cached = _KNOB_CACHE.get(key, _KNOB_CACHE_MISS)
    if cached is not _KNOB_CACHE_MISS:
        return cached
    if val is None or val == "":
        resolved = default
    else:
        try:
            resolved = int(val)
        except (TypeError, ValueError):
            # Intentional silent fallthrough — see header comment above.
            resolved = default
    _KNOB_CACHE[key] = resolved
    return resolved



_tensor_cache = {}

current_device_index = torch.cuda.current_device()
current_device = torch.cuda.get_device_properties(current_device_index)
MAX_SMS = current_device.multi_processor_count
MAX_BLOCK_SIZE = 65536

# K-587 (Architect feedback): the kpack policy lives in origami.py next to
# the override registry; this module re-exports it so existing callers /
# tests can still reach the constant by its old name.
_KPACK2_TILE_AREA_THRESHOLD = KPACK2_TILE_AREA_THRESHOLD

_global_locks = torch.empty(MAX_SMS, device="cuda", dtype=torch.uint8)
_global_P = torch.empty(MAX_SMS, MAX_BLOCK_SIZE, device="cuda", dtype=torch.float32)


def _maybe_wrap(fn, probe_tensor):
    # Use wrap_triton only under torch.compile tracing; otherwise direct call
    # in eager.  Can't use torch.compiler.is_compiling() here because the code
    # inside @triton_op but outside wrap_triton is part of the compile pass
    # itself and is_compiling() is never True.
    if is_fake(probe_tensor):
        return wrap_triton(fn)
    return fn


# Function will behave like an LRU-Cache of heuristic results
# Saves several microseconds for previously seen problems by not rerunning the heuristic unnecessarily
#@functools.lru_cache(maxsize=1024)
def _make_matmul_selector(
    M: int,
    N: int,
    K: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_dtype: torch.dtype,
    device: torch.device,
    mx_block_size=0,
    streamk=False,
    num_stages: int = 2,
):
    # Persistent on-disk autotune cache: skip the expensive Origami
    # selection when we have a previously-recorded entry for this exact
    # shape / dtype / mode / arch.
    arch, n_cu = _resolve_arch_info(device)
    cache = _autotune_cache.get_cache(arch, n_cu)
    a_str = OrigamiMatmulSelector.dtype_to_str.get(a_dtype, str(a_dtype))
    b_str = OrigamiMatmulSelector.dtype_to_str.get(b_dtype, str(b_dtype))
    c_str = OrigamiMatmulSelector.dtype_to_str.get(c_dtype, str(c_dtype))
    key = _autotune_cache.make_cache_key(
        M, N, K, a_str, b_str, c_str,
        mx_block_size, streamk, num_stages,
        n_cu,
    )
    cached = cache.lookup(key)
    if cached is not None:
        try:
            return OrigamiMatmulSelector(
                M,
                N,
                K,
                a_dtype,
                b_dtype,
                c_dtype,
                device,
                mx_block_size=mx_block_size,
                streamk=streamk,
                num_stages=num_stages,
                _cached_params=cached,
            )
        except Exception:
            # Fall through to a full computation if cached params can't be
            # reconstructed (e.g. arch change, origami API change).
            pass

    # Boundary-tolerance fall-back (K-591 / K-588 follow-up): exact-match
    # missed but a near-neighbor entry sharing the same dtype/mode may be
    # reusable.  We adopt it for this call only — the substituted config is
    # NOT persisted under the new key, otherwise chained interpolations
    # could drift arbitrarily from any tuned canonical entry.
    nearest = cache.lookup_nearest(key)
    if nearest is not None:
        cached_params, _matched_key = nearest
        try:
            return OrigamiMatmulSelector(
                M,
                N,
                K,
                a_dtype,
                b_dtype,
                c_dtype,
                device,
                mx_block_size=mx_block_size,
                streamk=streamk,
                num_stages=num_stages,
                _cached_params=cached_params,
            )
        except Exception:
            # Cached params for the neighbor can't be reconstructed for
            # this shape; fall through to the full Origami path.
            pass

    # Run Heuristic Results (Only if key has not been seen before)
    selector = OrigamiMatmulSelector(
        M,
        N,
        K,
        a_dtype,
        b_dtype,
        c_dtype,
        device,
        mx_block_size=mx_block_size,
        streamk=streamk,
        num_stages=num_stages,
    )
    try:
        cache.store(key, selector.get_cache_params())
    except Exception:
        # Persistence failures must never break the kernel call.
        pass
    return selector


def _resolve_arch_info(device):
    """Return (arch_name, n_cu) for the given torch device.

    A per-call sub-CU mask is intentionally not part of the cache identity
    (see ``autotune_cache.make_cache_key`` for the rationale).
    """
    try:
        props = torch.cuda.get_device_properties(device)
        gcn = getattr(props, "gcnArchName", "") or ""
        arch = gcn.split(":")[0] if gcn else "unknown"
        n_cu = int(getattr(props, "multi_processor_count", 0)) or 0
    except Exception:
        arch, n_cu = "unknown", 0
    return arch, n_cu


def persistent_matmul_lt(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    selector,
    config: Optional[MatmulConfig] = None,
    bias: Optional[torch.Tensor] = None,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    quantized: bool = False,
    work_stealing: bool = False,
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    BLK_M    = selector.block_m
    BLK_N    = selector.block_n
    BLK_K    = selector.block_k
    gsize_m  = selector.group_m
    num_xcds = selector.num_sms

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    total_programs = total_tiles
    even_k = K % BLK_K == 0

    num_stages = getattr(selector, "num_stages", 2)
    # num_warps: prefer per-shape value from the selector (K-676 cross-walk
    # added a num_warps property that returns 4 for shapes/tiles where
    # hipBLASLt evidence shows num_warps=4 outperforms 8); fall back to 8
    # for selectors that don't expose the property.
    num_warps = getattr(selector, "num_warps", 8)
    # K-706: per-shape waves_per_eu / kpack overrides.  Selector exposes
    # k706_waves_per_eu / k706_kpack; either may be None when no entry
    # exists for this shape (preserve historical defaults of 0 / 1).
    _k706_wpu = getattr(selector, "k706_waves_per_eu", None)
    _k706_kp = getattr(selector, "k706_kpack", None)
    waves_per_eu = _k706_wpu if _k706_wpu is not None else 0
    mfmaInstrSize = 16
    # K-587 / K-383 §F7 CG-2: adaptive kpack — policy lives in
    # `origami.kpack_for_tile`; see `KPACK2_TILE_AREA_THRESHOLD` there for
    # the empirical rationale.  Skinny / small tiles win +3 to +4 pp from
    # `ds_read_b128` (vec=8); 256x256 regresses -7 to -9 pp from doubled
    # LDS-load VGPR pressure crowding out the accumulator.
    kpack = kpack_for_tile(BLK_M, BLK_N)
    CACHE_MODIFIER_A = None
    CACHE_MODIFIER_B = None

    # Select LDS swizzle config (kpack/num_warps) for bank-conflict mitigation.
    # Defaults to (kpack=1, num_warps=8) outside the medium-K residual band.
    _lds_cfg = select_lds_config(
        M, N, K,
        a_dtype=str(a.dtype), b_dtype=str(b.dtype), c_dtype=str(c.dtype),
        block_m=BLK_M, block_n=BLK_N, block_k=BLK_K,
        streamk=False, work_stealing=work_stealing,
    )
    num_warps = _lds_cfg.num_warps
    kpack = _lds_cfg.kpack

    # Set chunk size to same area as L2 tiles.
    chunk_size = gsize_m * gsize_m
    if num_xcds > 0:
        chunk_size = min(chunk_size, max(1, total_programs // num_xcds))
    else:
        num_xcds = 1

    if work_stealing and config is not None:
        grids = selector._hardware.N_CU

        kk = _maybe_wrap(ws_persistent_matmul, probe_tensor=a)[(grids,)](
            a,
            b,
            c,
            a_scale if quantized else None,
            b_scale if quantized else None,
            bias if bias is not None else None,
            config.tile_counter,
            config.global_counter,
            M,
            N,
            K,
            a.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            bias.stride(0) if bias is not None else 0,
            stride_ak=a.stride(1),
            stride_bk=b.stride(0),
            BLOCK_SIZE_M=BLK_M,
            BLOCK_SIZE_N=BLK_N,
            BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m,
            NUM_SMS=grids,
            NUM_XCDS=num_xcds,
            COUNTERS_PER_XCD=selector.COUNTERS_PER_XCD,
            COUNTER_STRIDE=COUNTER_STRIDE,
            BIAS=bias is not None,
            EVEN_K=even_k,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            QUANTIZED=quantized,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            GLOBAL_ATOMIC=config.global_atomic,
            HIERARCHICAL=False,
            LOCAL_TILES_PER_XCD=0,
            GLOBAL_TILES=0,
            USE_MASK=True,
            mask_ptr=config.mask,
            num_stages=num_stages,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=mfmaInstrSize,
            kpack=kpack,
        )
    else:
        grids = total_tiles

        kk = _maybe_wrap(persistent_matmul, probe_tensor=a)[(grids,)](
            a,
            b,
            c,
            a_scale if quantized else None,
            b_scale if quantized else None,
            bias if bias is not None else None,
            M,
            N,
            K,
            a.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            bias.stride(0) if bias is not None else 0,
            stride_ak=a.stride(1),
            stride_bk=b.stride(0),
            BLOCK_SIZE_M=BLK_M,
            BLOCK_SIZE_N=BLK_N,
            BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m,
            NUM_SMS=total_programs,
            NUM_XCDS=num_xcds,
            CHUNK_SIZE=chunk_size,
            BIAS=bias is not None,
            EVEN_K=even_k,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            QUANTIZED=quantized,
            num_stages=num_stages,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=mfmaInstrSize,
            kpack=kpack,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

    return c

def streamk_matmul_lt(
    a: torch.Tensor, 
    b: torch.Tensor, 
    c: torch.Tensor, 
    selector, 
    config: Optional[MatmulConfig] = None,
    bias: Optional[torch.Tensor] = None,
    sk_grid: Optional[int] = None,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    quantized: bool = False,
    work_stealing: bool = False,
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    BLK_M    = selector.block_m
    BLK_N    = selector.block_n
    BLK_K    = selector.block_k
    gsize_m  = selector.group_m
    num_xcds = selector.num_sms

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0

    ##
    # Grid Size
    ##
    if work_stealing:
        total_programs_streamk = selector._hardware.N_CU
    else:
        total_programs_streamk = selector.sk_grid

    if total_programs_streamk > 0:
        total_tiles_streamk = total_tiles % total_programs_streamk
    else:
        total_tiles_streamk = 0

    num_stages = getattr(selector, "num_stages", 2)
    # num_warps: see persistent_matmul_lt for rationale.
    num_warps = getattr(selector, "num_warps", 8)
    # K-706: per-shape waves_per_eu / kpack overrides (see
    # persistent_matmul_lt).  None falls back to the historical default.
    _k706_wpu = getattr(selector, "k706_waves_per_eu", None)
    _k706_kp = getattr(selector, "k706_kpack", None)
    waves_per_eu = _k706_wpu if _k706_wpu is not None else 0
    mfmaInstrSize = 16
    # K-587 / K-383 §F7 CG-2: adaptive kpack — policy lives in
    # `origami.kpack_for_tile`; see `KPACK2_TILE_AREA_THRESHOLD` there for
    # the empirical rationale.  Skinny / small tiles win +3 to +4 pp from
    # `ds_read_b128` (vec=8); 256x256 regresses -7 to -9 pp from doubled
    # LDS-load VGPR pressure crowding out the accumulator.
    kpack = kpack_for_tile(BLK_M, BLK_N)
    CACHE_MODIFIER_A = None
    CACHE_MODIFIER_B = None

    # Select LDS swizzle config (kpack/num_warps) for bank-conflict mitigation.
    # Defaults to (kpack=1, num_warps=8) outside the medium-K residual band.
    _lds_cfg = select_lds_config(
        M, N, K,
        a_dtype=str(a.dtype), b_dtype=str(b.dtype), c_dtype=str(c.dtype),
        block_m=BLK_M, block_n=BLK_N, block_k=BLK_K,
        streamk=True, work_stealing=work_stealing,
    )
    num_warps = _lds_cfg.num_warps
    kpack = _lds_cfg.kpack

    if sk_grid is not None:
        total_programs_streamk = sk_grid

    grids = total_programs_streamk
    block_size = BLK_M * BLK_N

    if config is not None:
        if grids <= config.locks.shape[0] and block_size <= config.P.shape[1]:
            locks = config.locks[:grids]
            P = config.P[:grids, :block_size]
        else:
            locks = torch.empty(grids, device=config.device, dtype=torch.uint8)
            P = torch.empty(grids, block_size, device=config.device, dtype=torch.float32)
    else:
        if grids <= MAX_SMS and block_size <= MAX_BLOCK_SIZE:
            locks = _global_locks[:grids]
            P = _global_P[:grids, :block_size]
        else:
            locks = torch.empty(grids, device=a.device, dtype=torch.uint8)
            P = torch.empty(grids, block_size, device=a.device, dtype=torch.float32)

    # Set chunk size to same area as L2 tiles.
    chunk_size = gsize_m * gsize_m
    if num_xcds > 0:
        chunk_size = min(chunk_size, grids // num_xcds)
    else:
        num_xcds = 1

    if work_stealing and config is not None:
        kk = _maybe_wrap(ws_streamk_matmul, probe_tensor=a)[(grids,)](
            a,
            b,
            c,
            a_scale if quantized else None,
            b_scale if quantized else None,
            bias if bias is not None else None,
            config.tile_counter,
            config.streamk_tile_counter,
            P,
            locks,
            M,
            N,
            K,
            a.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            bias.stride(0) if bias is not None else 0,
            stride_ak=a.stride(1),
            stride_bk=b.stride(0),
            BLOCK_SIZE_M=BLK_M,
            BLOCK_SIZE_N=BLK_N,
            BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m,
            NUM_SMS=selector._ACTIVE_CU,
            NUM_XCDS=num_xcds,
            CHUNK_SIZE=chunk_size,
            STREAMK_TILES=total_tiles_streamk,
            COUNTERS_PER_XCD=selector.COUNTERS_PER_XCD,
            COUNTER_STRIDE=COUNTER_STRIDE,
            BIAS=bias is not None,
            EVEN_K=even_k,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            QUANTIZED=quantized,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            GLOBAL_ATOMIC=config.global_atomic,
            mask_ptr=config.mask,
            num_stages=num_stages,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=mfmaInstrSize,
            kpack=kpack,
        )
    else:
        kk = _maybe_wrap(streamk_matmul, probe_tensor=a)[(grids,)](
            a,
            b,
            c,
            a_scale if quantized else None,
            b_scale if quantized else None,
            bias if bias is not None else None,
            P,
            locks,
            M,
            N,
            K,
            a.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            bias.stride(0) if bias is not None else None,
            stride_ak=a.stride(1),
            stride_bk=b.stride(0),
            BLOCK_SIZE_M=BLK_M,
            BLOCK_SIZE_N=BLK_N,
            BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m,
            NUM_SMS=grids,
            NUM_XCDS=num_xcds,
            CHUNK_SIZE=chunk_size,
            STREAMK_TILES=total_tiles_streamk,
            BIAS=bias is not None,
            EVEN_K=even_k,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            QUANTIZED=quantized,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            num_stages=num_stages,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=mfmaInstrSize,
            kpack=kpack,
        )

    return c

def matmul_lt(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    selector, config: MatmulConfig,
    enable_streamk=False, work_stealing=False
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"

    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector, config, work_stealing=work_stealing)
    else:
        return persistent_matmul_lt(a, b, c, selector, config, work_stealing=work_stealing)

def matmul_a8w8_lt(
    a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor,
    c: torch.Tensor, selector, config: MatmulConfig,
    enable_streamk=False, work_stealing=False,
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"

    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector, config, a_scale=a_scale, b_scale=b_scale, quantized=True)
    else:
        return persistent_matmul_lt(a, b, c, selector, config, a_scale=a_scale, b_scale=b_scale, quantized=True, work_stealing=work_stealing)


@triton_op("tritonblas::_matmul", mutates_args={})
def _matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    enable_streamk: Optional[bool] = False,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible A-B Dimensions"
    M, K = a.shape
    _, N = b.shape

    out = a.new_empty(M, N)

    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, out.dtype, a.device, streamk=enable_streamk)
    config = matmul_preamble(selector) if work_stealing else None
    if enable_streamk:
        return streamk_matmul_lt(a, b, out, selector, config, sk_grid=sk_grid, work_stealing=work_stealing)
    else:
        return persistent_matmul_lt(a, b, out, selector, config, work_stealing=work_stealing)


def _setup_context_matmul_backwards(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: Any
):
    a, b, enable_streamk, sk_grid, work_stealing = inputs
    ctx.save_for_backward(a, b)
    ctx.enable_streamk = enable_streamk
    ctx.sk_grid = sk_grid
    ctx.work_stealing = work_stealing


def _matmul_backwards(
    ctx: Any,
    grad_output: torch.Tensor
):
    a, b = ctx.saved_tensors
    enable_streamk = ctx.enable_streamk
    sk_grid = ctx.sk_grid
    work_stealing = ctx.work_stealing

    grad_output_cont = grad_output.contiguous()

    b_t = b.T.contiguous()
    grad_a = matmul(grad_output_cont, b_t, enable_streamk=enable_streamk, sk_grid=sk_grid, work_stealing=work_stealing)

    a_t = a.T.contiguous()
    grad_b = matmul(a_t, grad_output_cont, enable_streamk=enable_streamk, sk_grid=sk_grid, work_stealing=work_stealing)

    return grad_a, grad_b, None, None, None


_matmul.register_autograd(_matmul_backwards,
                          setup_context=_setup_context_matmul_backwards)


@triton_op("tritonblas::_matmul_out", mutates_args={'out'})
def _matmul_out(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    enable_streamk: Optional[bool] = False,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> None:
    assert a.shape[1] == b.shape[0], "Incompatible A-B Dimensions"
    M, K = a.shape
    _, N = b.shape

    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, out.dtype, a.device, streamk=enable_streamk)
    config = matmul_preamble(selector) if work_stealing else None

    if enable_streamk:
        streamk_matmul_lt(a, b, out, selector, config, sk_grid=sk_grid, work_stealing=work_stealing)
    else:
        persistent_matmul_lt(a, b, out, selector, config, work_stealing=work_stealing)

    return None


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    enable_streamk: Optional[bool] = False,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> Optional[torch.Tensor]:
    # Rank-3 inputs: dispatch to true batched matmul (single kernel launch
    # over BATCH * tiles instead of a Python-level batch loop).
    if a.dim() == 3 and b.dim() == 3:
        return bmm(a, b, out)

    if out is None:
        return _matmul(a, b, enable_streamk, sk_grid, work_stealing)

    if torch.is_grad_enabled() and (
        a.requires_grad
        or b.requires_grad
        or out.requires_grad
    ):
        raise RuntimeError(
            "tritonblas.matmul(): functions with out=... arguments don't support "
            "automatic differentiation, but one of the arguments requires grad."
        )
    return _matmul_out(a, b, out, enable_streamk, sk_grid, work_stealing)


def matmul_a8w8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    c: torch.Tensor,
    enable_streamk=False,
    work_stealing=False,
    sk_grid=None,
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, c.dtype, a.device, streamk=enable_streamk)
    config = matmul_preamble(selector) if work_stealing else None
    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector, config, sk_grid=sk_grid, a_scale=a_scale, b_scale=b_scale, quantized=True, work_stealing=work_stealing)
    else:
        return persistent_matmul_lt(a, b, c, selector, config, a_scale=a_scale, b_scale=b_scale, quantized=True, work_stealing=work_stealing)

def matmul_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    block_m: int = None, #Overrides Origami value
    block_n: int = None, #Overrides Origami value
    block_k: int = None, #Overrides Origami value
    group_size_m: int = 8, #Overrides Origami value
    num_warps: int = 8,
    num_stages: int = 2,
):
    """
    FP4 matrix multiplication: C = A @ B
    
    Args:
        a: Input matrix A in FP4 format (M, K//2), packed 2 elements per uint8
        b: Input matrix B in FP4 format (N, K//2), packed 2 elements per uint8
        c: Output matrix C (M, N) in bfloat16 or float16
        a_scales: Scales for A in e8m0 format (M, K // 32)
        b_scales: Scales for B in e8m0 format (N, K // 32)
        block_m: Block size for M dimension
        block_n: Block size for N dimension
        block_k: Block size for K dimension (must be multiple of 64 for FP4)
        group_size_m: Group size for M dimension tiling
        num_warps: Number of warps per thread block (default: 8)
        num_stages: Number of pipeline stages (default: 2)
    
    Returns:
        Output matrix C
    """

    M, K = a.shape
    _, N = b.shape
    
    num_xcds = 8

    if(block_m == None):
        selector = _make_matmul_selector(M, N, K, "f4", "f4", c.dtype, a.device, mx_block_size=32)
        block_m      = selector.block_m
        block_n      = selector.block_n
        block_k      = selector.block_k
        group_size_m = selector.group_m
        num_xcds     = selector.num_sms
        if(block_m < M):
            block_m=128
        if(block_n < N):
            block_n=128
        if(block_k < K):
            block_k=128
        #print(f"Selected {block_m}x{block_n}x{block_k}")
    # Get actual dimensions (accounting for packing)
    M = a.shape[0]
    K = a.shape[1] * 2  # Unpacked K dimension
    N = b.shape[0]  # B has shape (N, K//2)
    
    # Verify dimensions are compatible
    assert b.shape[1] * 2 == K, f"Incompatible Dimensions: A has K={K}, B has K={b.shape[1] * 2}"
    
    # Transpose B to match kernel expectations (kernel expects B as K x N)
    b = b.T
    
    # Ensure block_k is appropriate for FP4 (must be multiple of 64)
    assert block_k % 64 == 0, "BLOCK_K must be multiple of 64 for FP4"
    
    total_blocks_M = triton.cdiv(M, block_m)
    total_blocks_N = triton.cdiv(N, block_n)
    total_tiles = total_blocks_M * total_blocks_N
    
    # Set chunk size to same area as L2 tiles
    chunk_size = group_size_m * group_size_m
    chunk_size = min(chunk_size, max(1, total_tiles // num_xcds))
    
    grid = (total_tiles,)
    
    fp4_matmul[grid](
        a,
        b,
        c,
        a_scales,
        b_scales,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scales.stride(0),
        a_scales.stride(1),
        b_scales.stride(0),
        b_scales.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_size_m,
        NUM_SMS=total_tiles,
        NUM_XCDS=num_xcds,
        CHUNK_SIZE=chunk_size,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    
    return c


@triton_op("tritonblas::_addmm", mutates_args={})
def _addmm(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    enable_streamk: Optional[bool] = False,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible A-B Dimensions"
    M, K = a.shape
    _, N = b.shape

    # Query Origami for solution
    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, bias.dtype, a.device, streamk=enable_streamk)
    config = matmul_preamble(selector) if work_stealing else None

    # Allocate an output tensor
    out = a.new_empty(M, N)

    if enable_streamk:
        return streamk_matmul_lt(a, b, out, selector, config, bias=bias, sk_grid=sk_grid, work_stealing=work_stealing)
    else:
        return persistent_matmul_lt(a, b, out, selector, config, bias=bias, work_stealing=work_stealing)


def _setup_context_addmm_backwards(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: Any
):
    bias, a, b, enable_streamk, sk_grid, work_stealing = inputs
    ctx.save_for_backward(a, b)
    ctx.enable_streamk = enable_streamk
    ctx.sk_grid = sk_grid
    ctx.work_stealing = work_stealing


def _addmm_backwards(
    ctx: Any,
    grad_output: torch.Tensor
):
    a, b = ctx.saved_tensors
    enable_streamk = ctx.enable_streamk
    sk_grid = ctx.sk_grid
    work_stealing = ctx.work_stealing

    # Make grad_output contiguous
    grad_output_cont = grad_output.contiguous()

    # grad_a = grad_output @ b^T
    b_t = b.T.contiguous()
    grad_a = matmul(grad_output_cont, b_t, enable_streamk=enable_streamk, sk_grid=sk_grid, work_stealing=work_stealing)

    # grad_b = a^T @ grad_output
    a_t = a.T.contiguous()
    grad_b = matmul(a_t, grad_output_cont, enable_streamk=enable_streamk, sk_grid=sk_grid, work_stealing=work_stealing)

    # grad_bias = sum(grad_output)
    grad_bias = grad_output.sum(dim=0)

    # tuple[bias, a, b, enable_streamk, sk_grid, work_stealing]
    #   First 3 must be in the order that matches addmm()'s forward args
    #   Last 3 are not part of the gradient and so are None
    return grad_bias, grad_a, grad_b, None, None, None


_addmm.register_autograd(_addmm_backwards,
                         setup_context=_setup_context_addmm_backwards)


@triton_op("tritonblas::_addmm_out", mutates_args={'out'})
def _addmm_out(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    enable_streamk: Optional[bool] = False,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> None:
    assert a.shape[1] == b.shape[0], "Incompatible A-B Dimensions"
    M, K = a.shape
    _, N = b.shape

    # Query Origami for solution
    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, bias.dtype, a.device, streamk=enable_streamk)
    config = matmul_preamble(selector) if work_stealing else None

    if enable_streamk:
        streamk_matmul_lt(a, b, out, selector, config, bias=bias, sk_grid=sk_grid, work_stealing=work_stealing)
    else:
        persistent_matmul_lt(a, b, out, selector, config, bias=bias, work_stealing=work_stealing)

    # Custom torch ops cannot return a value which is an alias of an input.  So
    # even though torch returns a pointer to the out arg when used, we can't.
    return None


def addmm(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    enable_streamk: Optional[bool] = False,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> Optional[torch.Tensor]:
    # If no out tensor provided - we do the allocation - we support autograd
    if out is None:
        return _addmm(bias, a, b, enable_streamk, sk_grid, work_stealing)

    # If out tensor provided - in-place - we do NOT support autograd
    # Check for autograd conditions (global and per-tensor)
    if torch.is_grad_enabled() and (
        bias.requires_grad
        or a.requires_grad
        or b.requires_grad
        or out.requires_grad
    ):
        raise RuntimeError(
            "tritonblas.addmm(): functions with out=... arguments don't support "
            "automatic differentiation, but one of the arguments requires grad."
        )
    return _addmm_out(bias, a, b, out, enable_streamk, sk_grid, work_stealing)


# ──────────────────────────────────────────────────────────────────────────
# Batched matmul (rank-3 inputs)
# ──────────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1024)
def _bmm_selector_cached(M, N, K, a_dtype, b_dtype, c_dtype, device_index, batch):
    """LRU-cached batched-matmul selector.

    The Origami solver takes ~300 µs per invocation — for small batched
    shapes that overhead alone exceeded the actual compute.  Caching by
    (shape, dtype, device, batch) eliminates the cost on hot paths and
    is safe because the selector output is a pure function of its inputs.
    """
    device = torch.device("cuda", device_index)
    return OrigamiMatmulSelector(
        M, N, K, a_dtype, b_dtype, c_dtype, device, batch=batch
    )


# ──────────────────────────────────────────────────────────────────────────
# Batched-matmul per-shape config lookup (gfx942 / MI300X)
# ──────────────────────────────────────────────────────────────────────────
#
# For shapes where Origami's BATCH=1-driven heuristic produces a tile/
# launcher config that under-utilises the persistent grid OR misses a
# better kpack/waves_per_eu combination, we keep an empirically-derived
# lookup keyed by ``(BATCH, M, N, K, dtype)``.
#
# Each entry maps to a tuple
#   (BLK_M, BLK_N, BLK_K, GROUP_M, num_warps, num_stages,
#    kpack, waves_per_eu)
# matching the kwargs expected by the ``batched_matmul`` Triton kernel.
#
# Provenance: K-686 iter-2 autotune sweep on MI300X (gfx942), node
# ``g09u25``, ROCm 7.2 / Triton 3.6, container
# ``rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0``.
# Each entry beats the Origami-driven default by ≥0.05 ratio vs hipBLASLt
# (``torch.bmm``) on the K-654 batched residual cohort.  See
# ``state/mc2/workspaces/K-686/output/tune_v5_*.csv`` for the underlying
# 480-config sweep.
#
# Adding new shapes: run ``scripts/tune_bmm.py --shapes <tag>``; if the
# best config beats the Origami-driven number by ≥0.05, add the row here
# with a one-line provenance comment.  Shapes NOT in the lookup fall
# back to the Origami-driven path (which handles all rank-2 inputs and
# is still the right choice for compute-bound batched shapes whose
# Origami pick already saturates the CUs).
_BMM_SHAPE_LOOKUP = {
    # (BATCH, M, N, K, dtype): (BLK_M, BLK_N, BLK_K, GROUP_M, num_warps,
    #                           num_stages, kpack, waves_per_eu)
    # K-686 PRD top-2 + companion batched shapes.
    (4, 2048, 2048, 2048, torch.float16):
        (256, 256, 64, 8, 8, 2, 2, 0),  # ratio 0.805 (vs Origami 0.636)
    (8, 1024, 1024, 1024, torch.bfloat16):
        (256, 128, 64, 8, 8, 2, 1, 2),  # ratio 0.785 (vs Origami 0.529)
    (4, 1024, 1024, 1024, torch.float16):
        (128, 128, 64, 4, 8, 2, 2, 2),  # ratio 0.788 (vs Origami 0.529)
    (8,  512,  512,  512, torch.float16):
        (128, 128, 64, 8, 8, 2, 1, 0),  # ratio 0.86 (slow-hipBLASLt) /
                                         # 0.77 (fast-hipBLASLt nodes);
                                         # robust cross-node winner.
                                         # 64x64x64 wins +0.01 only on
                                         # slow nodes; 128x128 wins
                                         # +0.04 on fast nodes — pick
                                         # the geomean-best config.
}


def _bmm_lookup_config(BATCH, M, N, K, dtype):
    """Return a per-shape launcher config or None if not in the table.

    Pure dict lookup — no Origami call, no LRU cache; the dict itself
    is the cache.  Returns ``None`` for shapes not in the table so the
    caller falls back to the Origami-driven path.
    """
    return _BMM_SHAPE_LOOKUP.get((BATCH, M, N, K, dtype))


@functools.lru_cache(maxsize=8)
def _bmm_hw_props(device_index):
    """Cached hardware-property fetch for the lookup-hit fast path.

    Returns (N_CU, NUM_XCD).  Faster than a full Origami selector call
    (which the lookup-hit path does NOT need — its tile/launcher
    params come from ``_BMM_SHAPE_LOOKUP``), and importantly does not
    pay the ~300 µs Origami solve.  Cached per-device so it's free
    after the first call.
    """
    # Make a tiny throwaway selector just to read hardware properties.
    # (Origami's hardware enumerator doesn't expose a public no-solve
    # constructor, so we pay the solve cost ONCE per device-index here,
    # then cache the result for every subsequent bmm() call on that
    # device.)
    sel = OrigamiMatmulSelector(
        128, 128, 128,
        torch.float16, torch.float16, torch.float16,
        torch.device("cuda", device_index), batch=1,
    )
    return sel._hardware.N_CU, max(1, getattr(sel._hardware, "NUM_XCD", 1))


def _batched_matmul_launch(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
):
    """
    Single-launch batched GEMM dispatcher.

    Inputs:
        a:   (BATCH, M, K)
        b:   (BATCH, K, N)
        out: (BATCH, M, N)
    Returns:
        out (mutated in-place; also returned for chaining)
    """
    assert a.dim() == 3 and b.dim() == 3, "bmm expects rank-3 inputs"
    assert a.shape[0] == b.shape[0], f"Batch mismatch: {a.shape[0]} vs {b.shape[0]}"
    assert a.shape[2] == b.shape[1], f"Inner dim mismatch: {a.shape[2]} vs {b.shape[1]}"

    BATCH, M, K = a.shape
    _, _, N = b.shape

    # ----------------------------------------------------------------
    # Per-shape lookup (K-686 iter-2): for the four K-654 batched
    # residuals, the Origami selector under-picks (its tile heuristic
    # is BATCH=1-driven and doesn't see the kpack/waves_per_eu axis).
    # If the (BATCH, M, N, K, dtype) tuple is in ``_BMM_SHAPE_LOOKUP``,
    # use that explicit launcher config and short-circuit the Origami
    # path entirely (faster + better ratios; see the lookup-table
    # docstring above for provenance).
    #
    # Hardware constants (NUM_SMS / NUM_XCDS) are still queried from the
    # current device — the lookup table keys ON shape, not GPU SKU.
    # ----------------------------------------------------------------
    lookup = _bmm_lookup_config(BATCH, M, N, K, a.dtype)
    if lookup is not None:
        # Hardware metadata for the lookup-hit path: read it from a
        # tiny per-device LRU cache (``_bmm_hw_props``) instead of a
        # full Origami solve, since the tile and launcher params
        # already come from the lookup table.  After the first call
        # this is a dict access; the per-call cost is dominated by
        # the Triton launcher itself, not Python.
        (BLK_M, BLK_N, BLK_K, gsize_m, num_warps, num_stages,
         kpack, waves_per_eu) = lookup
        num_sms_hw, num_xcds = _bmm_hw_props(a.device.index)
        mfmaInstrSize = 16

        total_blocks_M = triton.cdiv(M, BLK_M)
        total_blocks_N = triton.cdiv(N, BLK_N)
        tiles_per_batch = total_blocks_M * total_blocks_N
        total_tiles = BATCH * tiles_per_batch
        even_k = K % BLK_K == 0
        grid_size = min(num_sms_hw, total_tiles)
        if total_tiles < 2 * grid_size:
            num_xcds = 1
        chunk_size = min(gsize_m * gsize_m, max(1, grid_size // num_xcds))

        _maybe_wrap(batched_matmul, probe_tensor=a)[(grid_size,)](
            a, b, out, M, N, K, BATCH,
            a.stride(0), a.stride(1), a.stride(2),
            b.stride(0), b.stride(1), b.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m, NUM_SMS=grid_size, NUM_XCDS=num_xcds,
            CHUNK_SIZE=chunk_size, EVEN_K=even_k,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            num_stages=num_stages, num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=mfmaInstrSize, kpack=kpack,
        )
        return out

    # Fallback: Origami-driven config for unlisted shapes.  Same path
    # as before — picks tile / launcher params from the OrigamiSelector
    # and applies the conservative occupancy-aware tile-downsize
    # heuristic for severely under-utilised batched shapes.
    selector = _bmm_selector_cached(
        M, N, K, a.dtype, b.dtype, out.dtype, a.device.index, BATCH
    )

    BLK_M = selector.block_m
    BLK_N = selector.block_n
    BLK_K = selector.block_k
    gsize_m = selector.group_m
    num_xcds = selector.num_sms
    num_sms_hw = selector._hardware.N_CU

    # ----------------------------------------------------------------
    # Batch-aware tile downsize (occupancy fix for small batched shapes)
    # ----------------------------------------------------------------
    # Origami's tile pick is M/N/K-driven and assumes BATCH=1.  For the
    # one K-654 batched residual where Origami picked the largest
    # (256x256) tile -- 1024^3 b=8 bf16 -- the pick leaves CUs idle:
    #
    #   * 1024^3 b=8 bf16 -> Origami picks 256x256 -> 4*4=16 tiles per
    #     batch * 8 batches = 128 total tiles, but MI300X has 304 CUs.
    #     ~58% of CUs sit idle.  Halving to 128x128 gives 8*8=64 tiles
    #     per batch * 8 = 512 tiles, filling the grid.
    #
    # Empirical sweep on the four K-654 batched shapes showed the
    # naive "halve until full" loop *regressed* two cases (1024^3 b=4
    # and 512^3 b=8) where Origami had already picked an L2-friendly
    # mid-sized tile (128x128 / 128x64) -- halving those further
    # dropped MFMA throughput more than the occupancy gain bought.
    #
    # Conservative heuristic that won out:
    #   * Only consider downsizing when BOTH BLK_M >= 256 AND BLK_N >= 256
    #     (Origami picks large tiles only when the per-batch problem is
    #     compute-bound, which is exactly where batching can amortise
    #     launch overhead -- and where the tile is large enough that
    #     halving once still leaves us in the MFMA-efficient 128+ band).
    #   * Take a single halving step (do not iterate to grid_size==total).
    #     Going further past 128 hits the MFMA throughput cliff.
    #   * Floor at 128 (matches the falsified-elsewhere "tile-tuning
    #     residual" floor; smaller tiles are already Origami's job).
    #
    # This does NOT contradict the K-580/K-612/K-646 falsification of
    # "tile tuning closes batched residuals" -- those experiments tuned
    # tiles for BATCH=1; the issue here is that the BATCH=1-optimal
    # tile leaves CUs idle when there *are* multiple batches.
    # Only downsize when underutilization is SEVERE (< half the CUs)
    # AND the current tile is large enough that halving stays in the
    # MFMA-efficient 128+ band.  Empirical sweep data:
    #   2048^3 b=4 fp16  Origami(256x256) -> 256 tiles (84% util) -> SKIP
    #   1024^3 b=8 bf16  Origami(256x256) -> 128 tiles (42% util) -> DOWNSIZE
    #   1024^3 b=4 fp16  Origami(128x128) -> 256 tiles (84% util) -> SKIP
    #    512^3 b=8 fp16  Origami(128x64)  -> 256 tiles (84% util) -> SKIP
    # i.e. the heuristic only fires for the one shape (1024^3 b=8) that
    # actually has a big occupancy hole.
    if (BLK_M >= 256 and BLK_N >= 256 and
            BATCH * triton.cdiv(M, BLK_M) * triton.cdiv(N, BLK_N)
            < num_sms_hw // 2):
        BLK_M = max(128, BLK_M // 2)
        BLK_N = max(128, BLK_N // 2)

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    tiles_per_batch = total_blocks_M * total_blocks_N
    total_tiles = BATCH * tiles_per_batch
    even_k = K % BLK_K == 0

    # Persistent grid: cap at the hardware CU count, but never exceed
    # total_tiles (avoids idle WGs when batch is tiny).
    grid_size = min(num_sms_hw, total_tiles)
    grids = (grid_size,)

    # ----------------------------------------------------------------
    # Chiplet-aware tile remap (XCD chunk_size invariant)
    # ----------------------------------------------------------------
    # MI300X has 8 XCDs per GPU. ``chiplet_transform_chunked`` reorders
    # the persistent tile-id space so every consecutive run of
    # CHUNK_SIZE tiles lands on the SAME XCD — keeping each XCD's L2
    # warm for adjacent tiles, then advancing chunk-by-chunk. The
    # invariant we maintain here:
    #
    #    CHUNK_SIZE = min(GROUP_SIZE_M^2, grid_size // NUM_XCDS)
    #
    #   * GROUP_SIZE_M^2 is the same area used by the single-shot
    #     persistent_matmul (matmul.py:107 — ``chunk_size = gsize_m *
    #     gsize_m``); it gives each XCD a square L2-friendly footprint.
    #   * ``grid_size // NUM_XCDS`` is the per-XCD budget — never give
    #     an XCD more tiles than it has workgroups, otherwise the remap
    #     wraps and over-serialises.
    #   * matrix_instr_nonkdim=16 is hard-coded across the matmul
    #     family (matmul.py:101, fp4_matmul, persistent_matmul_lt)
    #     because the MI300X MFMA mfma_16x16x16 lane geometry assumes
    #     a 16-lane K segment; changing this breaks the kpack=1
    #     A/B-load contract and silently corrupts tail-K masking.
    #
    # ``selector.num_sms`` returns 0 on architectures where Origami did
    # not populate the XCC mapping; fall back to the hardware-reported
    # XCD count. Disable the chiplet remap when the problem is too
    # small to give each XCD multiple chunks (empirically: with
    # ``total_tiles < 2 * grid_size``, the remap over-serialises the
    # small batched shapes — k654-r2 regressed from 0.20 to 0.13 with
    # the remap on; gating restores 0.20).
    hw_xcds = max(1, getattr(selector._hardware, "NUM_XCD", 1))
    if num_xcds is None or num_xcds < 1:
        num_xcds = hw_xcds
    if total_tiles < 2 * grid_size:
        num_xcds = 1
    chunk_size = gsize_m * gsize_m
    chunk_size = min(chunk_size, max(1, grid_size // num_xcds))

    num_stages = getattr(selector, "num_stages", 2)
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 1

    _maybe_wrap(batched_matmul, probe_tensor=a)[grids](
        a,
        b,
        out,
        M,
        N,
        K,
        BATCH,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m,
        NUM_SMS=grid_size,
        NUM_XCDS=num_xcds,
        CHUNK_SIZE=chunk_size,
        EVEN_K=even_k,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=mfmaInstrSize,
        kpack=kpack,
    )

    return out


def bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched matrix multiply: ``out[i] = a[i] @ b[i]`` for ``i`` in
    ``[0, BATCH)``.

    Single-grid Triton dispatch over ``(BATCH * M_tiles * N_tiles)`` —
    eliminates the per-batch host-loop launch overhead that previously
    made small batched shapes (e.g. ``b=8, 1024^3`` bf16) ~14× slower
    than hipBLASLt's batched path.  See the K-686 PR for the structural
    rationale.

    Args:
        a:   ``(BATCH, M, K)`` tensor on a CUDA/HIP device, fp16/bf16/fp32.
        b:   ``(BATCH, K, N)`` tensor; must match ``a`` in batch dim,
             dtype, and device.
        out: optional ``(BATCH, M, N)`` output tensor.  If supplied it is
             written in-place AND returned (so callers can chain).
             Must match ``a``'s dtype + device; shape must be exactly
             ``(BATCH, M, N)`` — no broadcast / no in-place reshape.
             Allocated by ``a.new_empty(...)`` if ``None``.

    Contract / preconditions (validated):
        * ``a.dim() == b.dim() == 3``
        * ``a.shape[0] == b.shape[0]`` (batch dims agree)
        * ``a.shape[2] == b.shape[1]`` (inner reduction dims agree)
        * ``a.dtype == b.dtype`` and ``a.device == b.device``
        * ``a`` and ``b`` are CUDA tensors (Triton requirement)
        * If supplied, ``out.shape == (BATCH, M, N)``,
          ``out.dtype == a.dtype``, ``out.device == a.device``

    Contiguity: the kernel reads strides explicitly (``stride_ab``,
    ``stride_bb``, etc. — see ``batched_gemm.py``) and does NOT require
    ``a`` or ``b`` to be contiguous along any specific axis.  Non-
    contiguous tensors (e.g. produced by ``.transpose(1, 2)``) are
    supported as long as the rank-3 shape contract above holds.  We do
    NOT silently call ``.contiguous()`` — that would hide a 2× memcpy
    cost from the caller.

    K not divisible by ``BLOCK_SIZE_K``: handled internally by the
    ``EVEN_K`` constexpr branch in the kernel (tail-K masked load).
    No restriction on ``K`` is exposed at this API surface.

    BATCH == 1: dispatched through the same kernel path (the persistent
    grid degenerates to a single batch slice). Equivalent to
    ``tritonblas.matmul(a[0], b[0])[None]`` but without the un/re-
    squeeze. ``tritonblas.matmul`` itself routes rank-3 inputs here.

    Returns:
        ``out`` (the supplied or freshly-allocated buffer) with shape
        ``(BATCH, M, N)`` and ``dtype == a.dtype``.

    Raises:
        ValueError: any contract violation listed above.
    """
    # Input-validation — explicit ValueError messages so callers get a
    # readable diagnostic instead of a Triton crash deep in the launcher.
    if a.dim() != 3 or b.dim() != 3:
        raise ValueError(
            f"bmm: rank-3 inputs required, got a.dim()={a.dim()} "
            f"b.dim()={b.dim()}"
        )
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"bmm: batch dims disagree a.shape[0]={a.shape[0]} != "
            f"b.shape[0]={b.shape[0]}"
        )
    if a.shape[2] != b.shape[1]:
        raise ValueError(
            f"bmm: inner reduction dims disagree a.shape[2]={a.shape[2]} "
            f"!= b.shape[1]={b.shape[1]}"
        )
    if a.dtype != b.dtype:
        raise ValueError(
            f"bmm: dtype mismatch a.dtype={a.dtype} != b.dtype={b.dtype}"
        )
    if a.device != b.device:
        raise ValueError(
            f"bmm: device mismatch a.device={a.device} != b.device={b.device}"
        )
    if not a.is_cuda:
        raise ValueError(
            f"bmm: CUDA/HIP tensor required, got a.device={a.device}"
        )

    BATCH, M, K = a.shape
    _, _, N = b.shape

    if out is None:
        out = a.new_empty(BATCH, M, N)
    else:
        if tuple(out.shape) != (BATCH, M, N):
            raise ValueError(
                f"bmm: out shape {tuple(out.shape)} != expected "
                f"{(BATCH, M, N)}"
            )
        if out.dtype != a.dtype:
            raise ValueError(
                f"bmm: out.dtype={out.dtype} != a.dtype={a.dtype}"
            )
        if out.device != a.device:
            raise ValueError(
                f"bmm: out.device={out.device} != a.device={a.device}"
            )

    return _batched_matmul_launch(a, b, out)

