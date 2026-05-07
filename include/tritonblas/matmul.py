import functools
import random
import time
from typing import Any, Dict, Optional, Tuple

import torch
from torch.library import triton_op, wrap_triton
from torch._subclasses.fake_tensor import is_fake
import triton

from .kernels import persistent_matmul, ws_persistent_matmul, streamk_matmul, ws_streamk_matmul
from .kernels.fp4_matmul import fp4_matmul
from .origami import OrigamiMatmulSelector
from .config import MatmulConfig, matmul_preamble, COUNTER_STRIDE



_tensor_cache = {}

current_device_index = torch.cuda.current_device()
current_device = torch.cuda.get_device_properties(current_device_index)
MAX_SMS = current_device.multi_processor_count
MAX_BLOCK_SIZE = 65536

_global_locks = torch.empty(MAX_SMS, device="cuda", dtype=torch.uint8)
_global_P = torch.empty(MAX_SMS, MAX_BLOCK_SIZE, device="cuda", dtype=torch.float32)


# ---------------------------------------------------------------------------
# K-212: Skinny-N (N <= 32) wave-quantization + tile-selection fix.
#
# Origami's analytical selector targets square-ish problems and chooses tiles
# (typically BLOCK_M >= 128, BLOCK_N >= 64, num_warps=8) that under-fill
# MI300X's 304 CUs when N is tiny.  For N=16, M=8192, BLOCK_M=128, BLOCK_N=128
# the data-parallel grid is `ceil(8192/128) * ceil(16/128) = 64 * 1 = 64`
# tiles -- only 64 of 304 CUs busy (21% occupancy).  The other 240 CUs sit
# idle every wave.  num_warps=8 = 512 threads per CTA on a 128x16 tile is
# also massively over-warped: a 128x16 tile is one MFMA fragment column, so
# ~96% of the threads have nothing to do.
#
# Fix: when N <= 32, dispatch to the StreamK kernel with a tight skinny-N
# recipe -- BLOCK_M=64, BLOCK_N in {16, 32}, BLOCK_K=128, num_warps=2 --
# and grid = N_CU (304 on MI300X).  StreamK splits along K and spreads work
# across all 304 CUs regardless of the data-parallel grid; the small tile +
# low warp count eliminates partial-warp / partial-wave waste.
#
# Empirical (K-212 sweep, 72 (M,N,K) shapes, MI300X FP16, c42):
#   default tritonblas geomean ratio vs hipBLASLt = 0.110x
#   skinny-N gate     geomean ratio vs hipBLASLt = 0.622x  (5.6x lift)
# Window: M >= 256 and K >= 1024 keeps the gate off launch-bound shapes
# where the StreamK reduction overhead would dominate.
# ---------------------------------------------------------------------------
_SKINNY_N_BLK_M     = 64
_SKINNY_N_BLK_K     = 128
_SKINNY_N_NUM_WARPS = 2
_SKINNY_N_NUM_STAGES = 2


def _resolve_skinny_n(M: int, N: int, K: int, enable_streamk):
    """Decide whether to take the skinny-N StreamK fast path and, if so,
    return the (BLK_M, BLK_N, BLK_K, num_warps) recipe.  Returns None to
    fall through to the Origami-selected default path.

    `enable_streamk=True`  : caller explicitly forces StreamK -> respect it.
    `enable_streamk=False` : caller explicitly forces persistent -> respect it.
    `enable_streamk=None`  : dispatcher decides; consult shape predicate.
    """
    if enable_streamk is True or enable_streamk is False:
        return None  # caller is explicit; do not override
    if not (0 < N <= 32 and M >= 256 and K >= 1024):
        return None
    # BLOCK_N=16 covers N in [1..16]; for N in (16..32] use BLOCK_N=32 so a
    # single N-tile still covers the row (avoids a wasted second N-iteration).
    blk_n = 32 if N > 16 else 16
    return (_SKINNY_N_BLK_M, blk_n, _SKINNY_N_BLK_K, _SKINNY_N_NUM_WARPS)


def _skinny_n_streamk(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    BLK_M: int,
    BLK_N: int,
    BLK_K: int,
    num_warps: int,
    bias: Optional[torch.Tensor] = None,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    quantized: bool = False,
):
    """Direct StreamK dispatch with the skinny-N recipe.  Bypasses Origami;
    grid = MAX_SMS (real CU count) so all CUs participate."""
    M, K = a.shape
    _, N = b.shape
    even_k = K % BLK_K == 0
    grids = MAX_SMS  # use full CU complement (304 on MI300X)
    num_xcds = 8
    gsize_m = 8
    block_size = BLK_M * BLK_N

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    streamk_tiles = total_tiles % grids if grids > 0 else 0

    # Reuse globally-allocated lock + P workspace when small enough.
    if grids <= MAX_SMS and block_size <= MAX_BLOCK_SIZE:
        locks = _global_locks[:grids]
        P = _global_P[:grids, :block_size]
    else:
        locks = torch.empty(grids, device=a.device, dtype=torch.uint8)
        P = torch.empty(grids, block_size, device=a.device, dtype=torch.float32)
    locks.zero_()  # Stream-K requires a clean lock array per call

    chunk_size = max(1, min(gsize_m * gsize_m, grids // num_xcds))

    _maybe_wrap(streamk_matmul, probe_tensor=a)[(grids,)](
        a, b, out,
        a_scale if quantized else None,
        b_scale if quantized else None,
        bias if bias is not None else None,
        P, locks,
        M, N, K,
        a.stride(0), b.stride(1), out.stride(0), out.stride(1),
        bias.stride(0) if bias is not None else 0,
        stride_ak=a.stride(1), stride_bk=b.stride(0),
        BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m, NUM_SMS=grids, NUM_XCDS=num_xcds,
        CHUNK_SIZE=chunk_size, STREAMK_TILES=streamk_tiles,
        BIAS=bias is not None, EVEN_K=even_k,
        CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None,
        QUANTIZED=quantized,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        num_stages=_SKINNY_N_NUM_STAGES, num_warps=num_warps,
        waves_per_eu=0, matrix_instr_nonkdim=16, kpack=1,
    )
    return out


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
    # Run Heuristic Results (Only if key has not been seen before)
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
    )


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
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 1
    CACHE_MODIFIER_A = None
    CACHE_MODIFIER_B = None

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
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 1
    CACHE_MODIFIER_A = None
    CACHE_MODIFIER_B = None

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
    enable_streamk: Optional[bool] = None,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible A-B Dimensions"
    M, K = a.shape
    _, N = b.shape

    out = a.new_empty(M, N)

    # K-212 skinny-N (N<=32) wave-quantization fast path.
    skinny = _resolve_skinny_n(M, N, K, enable_streamk)
    if skinny is not None and not work_stealing and a.dtype == b.dtype:
        return _skinny_n_streamk(a, b, out, *skinny)

    es = bool(enable_streamk) if enable_streamk is not None else False
    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, out.dtype, a.device, streamk=es)
    config = matmul_preamble(selector) if work_stealing else None
    if es:
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
    enable_streamk: Optional[bool] = None,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> None:
    assert a.shape[1] == b.shape[0], "Incompatible A-B Dimensions"
    M, K = a.shape
    _, N = b.shape

    # K-212 skinny-N (N<=32) wave-quantization fast path.
    skinny = _resolve_skinny_n(M, N, K, enable_streamk)
    if skinny is not None and not work_stealing and a.dtype == b.dtype:
        _skinny_n_streamk(a, b, out, *skinny)
        return None

    es = bool(enable_streamk) if enable_streamk is not None else False
    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, out.dtype, a.device, streamk=es)
    config = matmul_preamble(selector) if work_stealing else None

    if es:
        streamk_matmul_lt(a, b, out, selector, config, sk_grid=sk_grid, work_stealing=work_stealing)
    else:
        persistent_matmul_lt(a, b, out, selector, config, work_stealing=work_stealing)

    return None


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    enable_streamk: Optional[bool] = None,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> Optional[torch.Tensor]:
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
    enable_streamk: Optional[bool] = None,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible A-B Dimensions"
    M, K = a.shape
    _, N = b.shape

    # Allocate an output tensor
    out = a.new_empty(M, N)

    # K-212 skinny-N (N<=32) wave-quantization fast path.
    skinny = _resolve_skinny_n(M, N, K, enable_streamk)
    if skinny is not None and not work_stealing and a.dtype == b.dtype:
        return _skinny_n_streamk(a, b, out, *skinny, bias=bias)

    es = bool(enable_streamk) if enable_streamk is not None else False
    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, bias.dtype, a.device, streamk=es)
    config = matmul_preamble(selector) if work_stealing else None

    if es:
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
    enable_streamk: Optional[bool] = None,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> None:
    assert a.shape[1] == b.shape[0], "Incompatible A-B Dimensions"
    M, K = a.shape
    _, N = b.shape

    # K-212 skinny-N (N<=32) wave-quantization fast path.
    skinny = _resolve_skinny_n(M, N, K, enable_streamk)
    if skinny is not None and not work_stealing and a.dtype == b.dtype:
        _skinny_n_streamk(a, b, out, *skinny, bias=bias)
        return None

    es = bool(enable_streamk) if enable_streamk is not None else False
    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, bias.dtype, a.device, streamk=es)
    config = matmul_preamble(selector) if work_stealing else None

    if es:
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
    enable_streamk: Optional[bool] = None,
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

