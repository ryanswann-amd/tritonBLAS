import functools
import random
import time
from typing import Any, Dict, Optional, Tuple

import torch
from torch.library import triton_op, wrap_triton
from torch._subclasses.fake_tensor import is_fake
import triton

from .kernels import persistent_matmul, ws_persistent_matmul, streamk_matmul, ws_streamk_matmul
from .kernels import batched_persistent_matmul
from .kernels.fp4_matmul import fp4_matmul
from .origami import OrigamiMatmulSelector, select_bmm_strategy
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


# ----------------------------------------------------------------------------
# Batched Matrix Multiply (true single-launch BMM)
# ----------------------------------------------------------------------------
#
# Replaces the implicit Python-side per-batch loop that prior tritonblas users
# wrote on top of ``tritonblas.matmul``.  By launching a single Triton grid
# parameterised by ``(NUM_SMS, BATCH)`` we eliminate per-call dispatch overhead
# and expose batch-level parallelism to the chiplet scheduler.  The Origami
# tile-selection heuristic is reused (keyed on ``(M, N, K)`` only — the optimal
# tile shape for a given GEMM is independent of the batch count) so behaviour
# at ``B == 1`` matches the existing ``matmul`` path.
#
# Accepted input shapes (mirroring ``torch.bmm`` / ``torch.matmul``):
#   * ``a`` rank-3 (B, M, K), ``b`` rank-3 (B, K, N)         — standard BMM
#   * ``a`` rank-3 (B, M, K), ``b`` rank-2 (K, N)            — broadcast B over batch
#   * ``a`` rank-2 (M, K),    ``b`` rank-3 (B, K, N)         — broadcast A over batch
#   * ``a`` rank-2 (M, K),    ``b`` rank-2 (K, N)            — degenerate B==1
#
# Rank-1 inputs (vectors) are upcast to rank-2 with a leading singleton (gemv
# is dispatched as a B==1 BMM of (1, K) x (K, N)) — this matches torch.matmul
# rank-1 promotion semantics and removes the previous per-vector launch loop.

def _normalize_bmm_strides(t: torch.Tensor, want_b: int):
    """Return ``(stride_b, stride_row, stride_col)`` in elements.

    ``t`` may be rank-2 (broadcast across batches with stride_b == 0) or rank-3.
    Rank-1 inputs are not handled here — caller upcasts before invoking.
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


# LRU-cached selector for the bmm path.  The non-batched matmul path has an
# explicit policy of NOT caching the selector (see ``_make_matmul_selector``),
# but bmm callers reissue identical (M,N,K,B,dtype) problems for every step of
# a training/inference loop — caching turns a ~185 µs Origami pass into a dict
# lookup, which dominates the per-call cost for small per-batch problems.
#
# The cache key follows the project requirement of keying on ``(M, N, K, B)``.
# In the current implementation the ``OrigamiMatmulSelector`` itself is
# B-independent (the optimal tile shape does not change with batch count) so
# ``B`` does not change the returned object — but it IS part of the key so
# that future B-sensitive policy changes (e.g. tile-fixup decisions that
# depend on total grid size) flow through this cache cleanly without callers
# accidentally hitting a stale entry from a different B.  ``a_dtype/b_dtype/
# c_dtype`` participate because the selector branches on dtype bitsize for
# matrix-instruction selection; ``device_str`` participates because Origami
# queries hardware properties keyed on device index.
@functools.lru_cache(maxsize=1024)
def _bmm_selector_cached(M, N, K, B, a_dtype, b_dtype, c_dtype, device_str):
    del B  # see docstring — included in the cache key for forward-compat
    return OrigamiMatmulSelector(
        M, N, K, a_dtype, b_dtype, c_dtype,
        torch.device(device_str),
    )


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
    Tile parameters are taken from ``OrigamiMatmulSelector(M, N, K, ...)`` —
    the same heuristic used by the non-batched ``matmul`` path.
    """
    if out.dim() != 3:
        raise RuntimeError("tritonblas.bmm: out tensor must be 3D (B, M, N)")

    B, M, N = out.shape
    K = a.shape[-1]

    if a.shape[-2:] != (M, K):
        raise RuntimeError(
            f"tritonblas.bmm: A trailing dims must be ({M}, {K}), got {tuple(a.shape[-2:])}"
        )
    if b.shape[-2:] != (K, N):
        raise RuntimeError(
            f"tritonblas.bmm: B trailing dims must be ({K}, {N}), got {tuple(b.shape[-2:])}"
        )

    selector = _bmm_selector_cached(
        M, N, K, B, a.dtype, b.dtype, out.dtype, str(a.device)
    )

    BLK_M = selector.block_m
    BLK_N = selector.block_n
    BLK_K = selector.block_k
    gsize_m = selector.group_m
    num_xcds = selector.num_sms

    # Batched-aware tile fixup.  Origami's per-shape selection is calibrated
    # for single-shot GEMMs; on the batched residuals identified in the
    # MI300X full sweep it picks 64x64 tiles with BLK_K=256 which leave
    # MFMA pipes under-fed.  Tile-size sweeps on MI300X show BLK 128x128
    # with BLK_K=64 and GROUP_M=4 wins by 1.4-1.8x for B>=2 and
    # M,N>=256 with no regression on smaller shapes.  We only override when
    # both Origami's M and N tiles fall below 128 — for shapes where it
    # already picks >=128 (e.g. 4096^3) we trust its selection.
    if M >= 512 and N >= 512 and (BLK_M < 128 or BLK_N < 128):
        BLK_M = 128
        BLK_N = 128
        BLK_K = 64
        gsize_m = 4

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
            raise RuntimeError("tritonblas.bmm: quantized=True requires a_scale and b_scale")
        stride_a_scale_b = a_scale.stride(0) if a_scale.dim() >= 2 else 0
        stride_b_scale_b = b_scale.stride(0) if b_scale.dim() >= 2 else 0
    else:
        stride_a_scale_b = 0
        stride_b_scale_b = 0

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


def _resolve_bmm_shapes(a: torch.Tensor, b: torch.Tensor):
    """Promote rank-1/rank-2 inputs to a canonical 3D BMM problem.

    Returns ``(a3, b3, batch, m, n, k, squeeze_a, squeeze_b)``.  ``a3`` and
    ``b3`` are views suitable for the kernel; ``squeeze_a/squeeze_b`` indicate
    whether the corresponding leading dim of the output should be squeezed
    back out (matching torch.matmul rank-promotion semantics).
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
            f"tritonblas.bmm: incompatible batch dimensions ({batch_a} vs {batch_b})"
        )
    batch = max(batch_a, batch_b)

    return a3, b3, batch, M, N, K, squeeze_a, squeeze_b


def bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched matrix multiply: out[i] = a[i] @ b[i].

    Single-launch implementation that replaces the prior per-batch Python loop
    over ``tritonblas.matmul``.  Accepts rank-2 and rank-3 inputs with standard
    broadcasting rules:

        a: (B, M, K) or (M, K)
        b: (B, K, N) or (K, N)

    Rank-1 inputs are promoted via ``torch.matmul`` semantics (the matching
    output dim is squeezed out).  Tile parameters come from the existing
    Origami heuristic (keyed on ``(M, N, K)``) so the per-shape selection
    matches the non-batched ``matmul`` path.

    Args:
        a: Left operand.
        b: Right operand.
        out: Optional pre-allocated output tensor (3D, shape (B, M, N)).
            When provided, autograd is not supported.

    Returns:
        The output tensor.  Shape is ``(B, M, N)`` for the canonical case;
        rank-1 promotions of ``a`` or ``b`` strip the corresponding singleton.
    """
    a3, b3, batch, M, N, K, squeeze_a, squeeze_b = _resolve_bmm_shapes(a, b)

    out_dtype = (out.dtype if out is not None
                 else torch.promote_types(a.dtype, b.dtype))

    # Dispatch policy lives in origami.select_bmm_strategy (keyed on M,N,K,B).
    # When that returns "per_batch_loop", each per-batch GEMM is already
    # large enough to saturate the GPU and the well-tuned single-shot
    # ``matmul`` path beats the batched kernel.  See the docstring in
    # ``origami.select_bmm_strategy`` for the calibration data and threshold.
    if select_bmm_strategy(M, N, K, batch) == "per_batch_loop":
        if out is None:
            out_buf = torch.empty(
                (batch, M, N), dtype=out_dtype, device=a3.device
            )
        else:
            out_buf = out if out.dim() == 3 else out.unsqueeze(0)
        # Materialize broadcast inputs into per-batch slices.
        for i in range(batch):
            a_i = a3[i] if a3.dim() == 3 else a3
            b_i = b3[i] if b3.dim() == 3 else b3
            matmul(a_i, b_i, out=out_buf[i])
        if out is None:
            result = out_buf
            if squeeze_a:
                result = result.squeeze(-2)
            if squeeze_b:
                result = result.squeeze(-1)
            return result
        return out

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

