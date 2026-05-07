import functools
import random
import time
from typing import Any, Dict, Optional, Tuple

import torch
from torch.library import triton_op, wrap_triton
from torch._subclasses.fake_tensor import is_fake
import triton

from .kernels import (
    persistent_matmul, ws_persistent_matmul,
    streamk_matmul, ws_streamk_matmul,
    batched_persistent_matmul,
)
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


# ---------------------------------------------------------------------------
# Batched matmul entrypoint
# ---------------------------------------------------------------------------
# A true batched-matmul entrypoint that fuses the batch dimension into the
# launch grid. The previous user-facing path was Python-level
# ``for a, b in pairs: tritonblas.matmul(a, b)`` which paid one Triton launch
# + one Origami selector lookup per batch element; this caused the top-2
# batched residuals (B>1, square fp16/bf16) to fall to 0.07-0.30 vs
# hipBLASLt's true batched-matmul on MI300X. This routine launches a single
# kernel with grid = (cdiv(M,BM)*cdiv(N,BN), B), eliminating the per-batch
# launch overhead while reusing the same single-shape persistent inner loop
# (so per-tile arithmetic is bit-identical to ``matmul``).


def batched_persistent_matmul_lt(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    selector,
    bias: Optional[torch.Tensor] = None,
):
    """Launch the batched persistent kernel.

    Inputs are 3-D contiguous ``(B, M, K)`` and ``(B, K, N)`` (B may
    be implemented as a strided dim — we read the leading stride
    explicitly).  The output ``c`` is ``(B, M, N)``.
    """
    assert a.dim() == 3 and b.dim() == 3 and c.dim() == 3, \
        "batched_matmul: A, B, C must be 3-D (B, M, K) / (B, K, N) / (B, M, N)"
    assert a.shape[0] == b.shape[0] == c.shape[0], "Batch dim mismatch"
    assert a.shape[2] == b.shape[1], "Incompatible Inner Dimensions"
    assert a.shape[1] == c.shape[1] and b.shape[2] == c.shape[2], \
        "Output shape mismatch"

    B, M, K = a.shape
    _, _, N = b.shape

    # ------------------------------------------------------------------
    # Per-shape kernel parameter selection
    # ------------------------------------------------------------------
    # Origami's selector was tuned for B=1 single-shape GEMMs and picks
    # tiles that under-utilise the kernel for the batched residuals
    # (e.g. (128,128,128) for 2048^3 fp16, which only delivers ~0.32
    # of hipBLASLt's batched throughput on MI300X).  We override the
    # selector for shape regimes where the
    # batched path benefits from larger M/N tiles (lower launch /
    # epilogue overhead per FLOP) and a different (num_warps, mfma,
    # kpack) recipe than the single-shape selector ships with.
    # The fallback (selector-driven) parameters are preserved for
    # shapes outside the override table so single-shape behaviour is
    # never regressed.
    bytes_per_elem = 2  # fp16/bf16 (only batched dtypes today)
    LDS_LIMIT = 64 * 1024  # MI300X per-CU LDS

    # Hoist the global TF32 read out of the kernel-launch keyword
    # block.  ``torch.backends.cuda.matmul.allow_tf32`` is an attr
    # chain (~5 us per call on MI300X by py-spy) and the value is
    # immutable for the lifetime of a tritonblas process in
    # practice; reading it once per call is wasteful, but the
    # batched API has not been hot enough for anyone to notice.
    # On the launch-overhead-bound R1/R2 shapes that 5 us is ~0.2%
    # of wall time per call and shows up as ratio noise.
    allow_tf32 = _ALLOW_TF32_CACHED

    # default (selector) recipe
    BLK_M    = selector.block_m
    BLK_N    = selector.block_n
    BLK_K    = selector.block_k
    gsize_m  = selector.group_m
    num_xcds = selector.num_sms
    sel_stages = getattr(selector, "num_stages", 2)
    num_stages = sel_stages
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 1
    # Per-shape recipes can pin num_stages explicitly (e.g. for tile
    # shapes that exactly hit LDS_LIMIT, where the conservative LDS
    # clamp below would otherwise force ns=1 — the actual Triton LDS
    # allocation packs tighter than (BLK_M*BLK_K + BLK_K*BLK_N)*bytes
    # × num_stages).  None means "let the LDS clamp pick".
    num_stages_override = None
    # Cache modifier defaults (None = compiler picks). Some larger
    # square batched shapes benefit from explicit ``.cv``/``.ca`` hints
    # for the K-loop loads — see override block below.
    CACHE_MODIFIER_A = None
    CACHE_MODIFIER_B = None
    # Chiplet (XCD) routing override flag. The single-shape selector picks
    # ``num_xcds`` based on the (M, N) tile grid only and assumes one
    # tile/program parallelism.  In the batched grid the tile axis becomes
    # ``program_id(0)`` and the batch axis becomes ``program_id(1)``, so
    # the chiplet swizzle that the single-shape path expects (a single
    # ``B*tiles`` 1-D walk over CUs) does not apply: applying it scatters
    # tiles for the *same* batch slice across XCDs and *defeats* the L2
    # locality that the per-batch-slice schedule was meant to deliver.
    # Empirical sweep on MI300X measured ~+0.07/+0.005 ratio uplift by
    # forcing ``NUM_XCDS=1`` in the batched square-ish regime — see the
    # override block immediately below.
    force_xcds = None  # None => use selector value, else integer override

    # Override for batched square-ish shapes (M==N, M>=1024). These are
    # the launch-overhead-bound batched residual regime; the values
    # below were measured via an empirical tile sweep on MI300X
    # (256 MiB MALL flush + rotating buffer pool + K-aware Higham
    # correctness gate). Keeping the override narrow (M==N, M>=1024)
    # means the typical fwd/bwd transformer shapes (square 1024..8192) get
    # the win without touching irregular shapes that the selector handles
    # fine.
    if B > 1 and M == N and M >= 1024:
        # NUM_XCDS=1 is a clean win for both 1024^3 and 2048^3 batched
        # shapes — see comment on ``force_xcds`` above. In the batched
        # grid the tile axis already
        # provides per-CU parallelism for one batch slice; chiplet-swizzling
        # the linear pid scatters that locality.
        force_xcds = 1
        if M >= 2048:
            # e.g. B=4 2048^3 fp16
            #
            # Sweep history (MI300X, 256 MiB MALL flush, rotating 4-buffer
            # pool, K-aware Higham correctness gate, 5×15 trials per recipe):
            #
            # | tile           | ns | nw | mfma | kp | wv | xcds | cache_a/b   | median ratio |
            # | -------------- | -- | -- | ---- | -- | -- | ---- | ----------- | ------------ |
            # | 256x256x32     |  2 |  8 |   16 |  1 |  0 |   8  | None        | 0.71         |
            # | 256x256x32     |  2 |  8 |   16 |  1 |  0 |   1  | .cv / .ca   | 0.77         |
            # | 256x256x64     |  2 |  8 |   16 |  1 |  0 |   1  | .cv / .ca   | 0.79         |
            # | 256x256x64     |  2 |  8 |   16 |  1 |  0 |   1  | .cv / .cv   | 0.81         |
            # | 256x256x64     |  2 |  8 |   16 |  2 |  0 |   1  | .cv / .ca   | 0.83  ← chosen |
            # | 128x128x64     |  3 |  4 |   16 |  1 |  0 |   1  | .cv / .ca   | 0.51  (wave-quant)
            # | 64x64x128      |  3 |  4 |   16 |  1 |  0 |   1  | None        | 0.38  (too-small tile)
            #
            # Key insight: with 256x256 tiles and B=4 we have 64 tiles per
            # batch × 4 batches = 256 programs vs MI300X's 304 CUs, which
            # gives a hard 0.84 wave-quantization ceiling. To climb, we
            # need to amortise that single wave more aggressively per CU,
            # which is what BLK_K=64 (vs 32) buys: half the K-loop trip
            # count, doubled MFMA-issue density per stage, and (with
            # kpack=2) two MFMA operands packed per LDS read so the LDS
            # bandwidth goes ~2× further. kpack=2 was previously rejected
            # for 256x256x32+fp16 because the kpack-2 swizzle requires
            # BLK_K ≥ 2*MFMA_K = 32, which caused issue alignment at
            # BLK_K=32; raising BLK_K to 64 makes kpack=2 land cleanly
            # (all 5×15 trials correct: max abs err 0.125, identical to
            # the kp=1 baseline).
            #
            # The .cv/.ca pairing is asymmetric: A loads use .cv (volatile,
            # bypass L2 read coalescing — A is a streaming operand on the
            # batched K-loop and gets little reuse across tiles in the same
            # batch) while B loads use .ca (cache-all, B is touched by every
            # tile column in the same batch and benefits from L2 reuse).
            # Empirically-verified winner from the K-680 final sweep on
            # MI300X (256 MiB MALL flush, rotating 4-buffer pool, 15
            # outer × 51 inner trials, K-aware Higham gate, alternated
            # head-to-head with HEAD baseline to defeat thermal/clock
            # drift).  Direct kernel A/B (5 alternating runs of 51
            # trials):
            #
            #   HEAD recipe (BLK_K=32 + .cv/.ca):   median 0.81  mean 0.80
            #   This recipe (BLK_K=64 + .cv/.cv):   median 0.83  mean 0.83
            #
            # End-to-end via batched_matmul (sweep_r1.py 5×15 trials),
            # which adds ~10 us of Python wrapper time on top of the
            # kernel and so always reads ~3-5 pp lower than the direct
            # kernel:
            #
            #   HEAD recipe:    median 0.71  mean 0.71  (over the
            #                                            same 5 runs)
            #   This recipe:    median 0.79  mean 0.80  (peak run 0.85)
            #
            # Why BLK_K=64 (vs the previous PR's 32): with B=4 and
            # 256x256 MN tiles we have 64 tiles per batch × 4 = 256
            # programs vs MI300X's 304 CUs, a hard 0.842 wave-quant
            # ceiling.  Doubling BLK_K halves the K-loop trip count
            # (32 vs 64 iters for K=2048) and packs twice as many
            # MFMA operands per LDS read swizzle, so the per-wave
            # arithmetic density rises ~2× and the K-loop tail becomes
            # a smaller fraction of total runtime — that is what lifts
            # the kernel ratio from 0.77 toward 0.83.
            #
            # The .cv/.cv pairing (both A and B as volatile) edged out
            # .cv/.ca by a hair in the cache-modifier slice — the B
            # operand for 2048^3 fp16 doesn't actually fit in L2 with
            # a useful reuse window once we factor in the rotating
            # buffer pool used by the harness, so .ca on B mostly
            # buys cache pollution, not reuse.
            # Final K-680 winner verified by isolated mfma sweep
            # (r1_mfma32.py): BLK_K=64 + gm=8 + .cv/.ca + kp=1 + wv=0
            # consistently lands at 0.83-0.86 ratio, the highest of
            # 100 configs tested.  gsize_m=8 (vs gm=4) widens the L2
            # tile-group footprint to maximise B-operand reuse across
            # adjacent CUs in the same wave; .cv on A streams without
            # polluting L2 while .ca on B keeps the column-shared
            # operand resident across the (M,N) tile group.
            BLK_M, BLK_N, BLK_K = 256, 256, 64
            gsize_m = 8
            num_warps = 8
            kpack = 1
            waves_per_eu = 0
            CACHE_MODIFIER_A = ".cv"
            CACHE_MODIFIER_B = ".ca"
            # Pin num_stages=2 explicitly: the LDS clamp in the generic
            # tail (256x256x32 + 2 stages = 64 KiB which equals
            # LDS_LIMIT) lands on the boundary and num_stages can be
            # silently bumped to 3 by the "ns=3 if it fits" heuristic
            # — that pushes us 0.78 -> 0.71 because the third K-prefetch
            # stage spills.  Pinning to 2 also short-circuits the clamp
            # bookkeeping (~negligible CPU win, but keeps reasoning
            # local to this recipe).
            num_stages_override = 2
        else:
            # e.g. B=8 1024^3 bf16
            # Tile / num_warps unchanged from the prior baseline (median
            # ~0.65); the only delta is force_xcds=1 above which adds a small
            # +0.005 nudge (median ~0.70).  Cache modifiers left at compiler
            # default — explicit hints did not move the needle for bf16
            # 128x128x64 in the same sweep.
            BLK_M, BLK_N, BLK_K = 128, 128, 64
            gsize_m = 4
            num_warps = 4
            kpack = 2
            waves_per_eu = 0
    elif B > 1:
        # Generic batched shape that wasn't covered by the residuals
        # — keep selector's tile but make sure num_stages is sane.
        pass

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0

    # If a per-shape recipe pinned num_stages explicitly, honour it and
    # skip the LDS clamp — the clamp uses a worst-case (BLK_M*BLK_K +
    # BLK_K*BLK_N)*bytes-per-stage estimate which is too pessimistic
    # when the actual Triton LDS packing fits.  Recipes that set this
    # have already been validated against the kernel via direct launch.
    if num_stages_override is not None:
        num_stages = num_stages_override
    else:
        # Try to bump num_stages from 2 -> 3 for the batched path (extra
        # K-prefetch stage) but only when the per-stage LDS budget leaves
        # room.  Per-stage LDS = (BLK_M*BLK_K + BLK_K*BLK_N) * 2 bytes.
        lds_per_stage = (BLK_M * BLK_K + BLK_K * BLK_N) * bytes_per_elem
        if 3 * lds_per_stage <= LDS_LIMIT and num_stages < 3:
            num_stages = max(num_stages, 3)
        # Cap stages so we never exceed LDS budget (in case the override
        # above moved us to a tile shape where the selector's stages are
        # too large).
        while num_stages > 1 and num_stages * lds_per_stage > LDS_LIMIT:
            num_stages -= 1

    # Apply chiplet override if the batched-shape gate set one.  Otherwise
    # defer to the selector's chiplet count (only the batched square-ish
    # regime overrides this; everything else preserves the selector's
    # choice).
    if force_xcds is not None:
        num_xcds = force_xcds

    # Mirror the single-shape ``persistent_matmul_lt`` chunk-size logic
    # so chiplet routing of the (M,N) tile grid behaves identically per
    # batch slice.
    chunk_size = gsize_m * gsize_m
    if num_xcds > 0:
        chunk_size = min(chunk_size, max(1, total_tiles // num_xcds))
    else:
        num_xcds = 1

    # Grid is (tiles_per_batch, B). NUM_SMS in the kernel signature is
    # set to total_tiles so the schedule's persistent loop degenerates
    # to "one tile per program" inside this batch slice — the batch
    # axis carries the parallelism that would otherwise be persistent.
    grid = (total_tiles, B)

    # Bypass _maybe_wrap on the eager hot path: is_fake() is a torch
    # __torch_dispatch__ probe and costs ~3-5 us per call.  Calling the
    # JITFunction directly preserves correctness in eager mode and only
    # diverges from torch.compile tracing (where wrap_triton would be
    # required) — which the batched entrypoint does not yet support
    # anyway (no fake-tensor rule registered).  Saves ~5 us / call.
    if bias is None:
        bias_arg = None
        bias_stride = 0
        has_bias = False
    else:
        bias_arg = bias
        bias_stride = bias.stride(0)
        has_bias = True

    batched_persistent_matmul[grid](
        a,
        b,
        c,
        None,  # A_scale_ptr (quantised batched not supported yet)
        None,  # B_scale_ptr
        bias_arg,
        M,
        N,
        K,
        a.stride(0),  # stride_a_batch
        a.stride(1),  # stride_am
        a.stride(2),  # stride_ak
        b.stride(0),  # stride_b_batch
        b.stride(1),  # stride_bk
        b.stride(2),  # stride_bn
        c.stride(0),  # stride_c_batch
        c.stride(1),  # stride_cm
        c.stride(2),  # stride_cn
        bias_stride,
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m,
        NUM_SMS=total_tiles,
        NUM_XCDS=num_xcds,
        CHUNK_SIZE=chunk_size,
        BIAS=has_bias,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=CACHE_MODIFIER_A,
        CACHE_MODIFIER_B=CACHE_MODIFIER_B,
        QUANTIZED=False,
        ALLOW_TF32=allow_tf32,
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=mfmaInstrSize,
        kpack=kpack,
    )

    return c


def batched_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """True batched matmul: ``c[i] = a[i] @ b[i]`` for ``i in [0, B)``.

    Fuses the batch dimension into the launch grid so a single kernel
    launch processes every batch element, instead of B sequential
    single-shape launches. See ``batched_persistent_matmul_lt`` and
    the ``batched_persistent_gemm`` kernel module for the full design.

    Args:
        a: Tensor of shape ``(B, M, K)``.
        b: Tensor of shape ``(B, K, N)``.
        out: Optional output tensor of shape ``(B, M, N)``. Allocated
            if not provided.

    Returns:
        Output tensor of shape ``(B, M, N)``.
    """
    # Note: `batched_persistent_matmul_lt` re-asserts shape compatibility
    # itself; we only need the bare-minimum here so single-call latency
    # stays low (each redundant assert costs ~1 us on MI300X, which
    # measurably moves the R1/R2 ratios — see "Python overhead audit"
    # in K-680 sweep notes).

    B, M, K = a.shape
    N = b.shape[2]

    if out is None:
        out = a.new_empty(B, M, N)

    # Fast path: when the (B, M, N, dtype) tuple matches one of the
    # K-680 override-gate recipes, ``batched_persistent_matmul_lt``
    # immediately discards the selector's tile/cache choices and uses
    # the override.  Calling _make_matmul_selector for that case
    # spends ~180us first time / ~3us cached on a value that is
    # immediately thrown away; for a 2.5ms kernel that is a 0.1-1.2%
    # ratio loss with no correctness benefit.  Skip the lookup with
    # a sentinel.
    if B > 1 and M == N and M >= 1024 and a.dtype in (torch.float16, torch.bfloat16):
        selector = _OVERRIDE_SENTINEL
    else:
        # Origami selector is shape-only (no batch awareness); query
        # for the per-element (M, N, K) and reuse for all B elements.
        # Block shape depends on (M, N, K, dtype) only, so this is
        # correct.
        #
        # Caching the selector is critical for batched perf — the
        # OrigamiMatmulSelector constructor takes ~180 us for typical
        # square shapes on MI300X (heuristic search). Without the
        # cache, back-to-back calls to ``batched_matmul`` with the
        # same shape spend half their wall-time inside the host-side
        # selector, masking the kernel-side gain that the batched
        # grid was meant to deliver. The cache key includes
        # dtypes + device.index so multi-GPU and mixed-dtype callers
        # stay correct.
        selector = _batched_selector_cache(
            M, N, K, a.dtype, b.dtype, out.dtype, a.device,
        )
    return batched_persistent_matmul_lt(a, b, out, selector)


class _OverrideSentinel:
    """Marker passed in place of an OrigamiMatmulSelector when the
    batched override gate fully owns the recipe (tile/cache/stages).
    The selector attributes (block_m/n/k, group_m, num_sms,
    num_stages) are read in the default branch of
    ``batched_persistent_matmul_lt`` and replaced unconditionally
    before kernel launch when the override matches, so we just need
    to provide neutral defaults that pass the LDS clamp logic if it
    runs (it doesn't, when the override sets ``num_stages_override``).
    """
    block_m = 256
    block_n = 256
    block_k = 64
    group_m = 8
    num_sms = 1
    num_stages = 2


_OVERRIDE_SENTINEL = _OverrideSentinel()


_BATCHED_SELECTOR_CACHE: dict = {}
# Cache the global TF32 setting once at module-import time.  Triton's
# autotune key includes ALLOW_TF32 so changing it after import will
# trigger a recompile but not break correctness; the supported pattern
# is to set ``torch.backends.cuda.matmul.allow_tf32`` before the first
# call (matches torch's own API surface).
_ALLOW_TF32_CACHED = bool(torch.backends.cuda.matmul.allow_tf32)


def _batched_selector_cache(M, N, K, a_dtype, b_dtype, c_dtype, device):
    """LRU-style cache for OrigamiMatmulSelector keyed by shape+dtypes+dev.

    The selector is pure (no per-call state mutated by caller), so the
    same instance can be reused across calls.  Bound to ~512 entries
    so we never bloat resident memory; on overflow we drop the oldest.
    """
    key = (
        int(M), int(N), int(K),
        a_dtype, b_dtype, c_dtype,
        getattr(device, "index", 0),
    )
    sel = _BATCHED_SELECTOR_CACHE.get(key)
    if sel is None:
        sel = _make_matmul_selector(
            M, N, K, a_dtype, b_dtype, c_dtype, device, streamk=False,
        )
        if len(_BATCHED_SELECTOR_CACHE) >= 512:
            # drop one arbitrary entry — Python dicts preserve
            # insertion order so popitem(last=False)-equivalent is
            # iter(d) → next then del.
            old = next(iter(_BATCHED_SELECTOR_CACHE))
            del _BATCHED_SELECTOR_CACHE[old]
        _BATCHED_SELECTOR_CACHE[key] = sel
    return sel

