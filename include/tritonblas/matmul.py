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


# ============================================================================
# K-697: tall-skinny FP16/BF16 dispatch override
# ----------------------------------------------------------------------------
# K-668 profiled the (M >= 2048, N <= 256, K >= 1024, dtype in {fp16, bf16})
# cohort on MI300X and found tritonblas runs at 0.51x hipBLASLt geomean
# across 128 shapes. The N=32 slice is the worst (geomean 0.28x): the
# Origami selector picks (BLOCK_M, BLOCK_N, BLOCK_K) = (16, 16, 32) or
# (32, 16, 32) which (a) under-decomposes the K loop (256 mainloop iters
# at K=8192 vs hipBLASLt's 32 with MT_K=256), (b) wastes 50% of MFMA
# N-lanes via padded BLOCK_N=16 against N=32, and (c) routes through the
# persistent-only path despite total_tiles < N_CU.
#
# The K-697 81-config tile sweep on the four worst K-668 shapes
# (rocprof-confirmed VALU-bound, non-MFMA-bound) found a single winner:
# Stream-K with (BLOCK_M=32, BLOCK_N=32, BLOCK_K=128, num_warps=2,
# num_stages=3). Verification on 16 tall-skinny shapes (the four worst
# plus 12 cohort siblings spanning M in {2048..16384}, K in {1024..8192})
# gives geomean +2.91x vs Origami (range 1.20x..7.57x) and lifts the
# cohort from 0.51x to 0.69x vs hipBLASLt with no cell regressing below
# 1.20x Origami. Adjacent-cohort leakage check (K-278 small-M-large-N,
# K-353 small-M-very-large-K, K-518/K-644-P1 M<=8, K-644-P2 large-square,
# K-667 large-K square) shows every shape outside the gate is left alone.
#
# Implementation follows the K-667/K-644 pattern: a strict-equality gate
# on N (no padding when BN=BLOCK_N=32==N), a measured M-lower-bound, and
# a measured K-lower-bound, all conjunctive. The kernel launch bypasses
# the Origami selector entirely (lru_cache ineligible -- selector is ~180
# us/call which alone wipes the 5-25 us hbl baseline at this size) and
# hand-picks the verified Stream-K parameters.
# ============================================================================

# Hand-picked Stream-K tile (verified K-697 winner across 16 cohort shapes).
_K697_BLOCK_M = 32
_K697_BLOCK_N = 32
_K697_BLOCK_K = 128
_K697_NUM_WARPS = 2
_K697_NUM_STAGES = 3
_K697_GROUP_M = 8
_K697_NUM_XCDS = 8

# Cohort gate bounds (measured from the K-697 verify + leakage sweeps).
_K697_M_MIN = 2048   # K-668 cohort lower bound; M=1024 won 6.3x but was outside
                     # the verified envelope, so left in Origami's hands.
_K697_K_MIN = 1024   # K=1024 was the smallest verified-winning K (1.20x..1.41x);
                     # K=512 measured 0.86x Origami (REGRESSION) -> excluded.


def _is_k697_tall_skinny(M, N, K, dtype):
    """K-697 cohort gate: tall-skinny FP16/BF16 GEMM where Origami's
    persistent-mode tile is structurally wrong (under-decomposed K-loop,
    padded MFMA N-lane, total_tiles << N_CU).

    Strict on N (must equal _K697_BLOCK_N=32 to avoid padding); measured
    bounds on M and K from the K-697 verify and leakage sweeps.
    """
    return (
        dtype in (torch.float16, torch.bfloat16)
        and N == _K697_BLOCK_N
        and M >= _K697_M_MIN
        and K >= _K697_K_MIN
    )


def _k697_tall_skinny_streamk(a, b, c):
    """Launch the streamk kernel directly with the K-697 winning tile.
    Bypasses _make_matmul_selector to avoid Origami's ~180us setup cost
    (which alone is >baseline kernel time at these sizes).
    """
    M, K = a.shape
    _, N = b.shape

    BM = _K697_BLOCK_M
    BN = _K697_BLOCK_N
    BK = _K697_BLOCK_K
    grids = MAX_SMS  # 304 on MI300X
    total_blocks_M = triton.cdiv(M, BM)
    total_blocks_N = triton.cdiv(N, BN)
    total_tiles = total_blocks_M * total_blocks_N
    total_tiles_streamk = (total_tiles % grids) if grids > 0 else 0
    even_k = (K % BK) == 0
    block_size = BM * BN
    if grids <= MAX_SMS and block_size <= MAX_BLOCK_SIZE:
        locks = _global_locks[:grids]
        P = _global_P[:grids, :block_size]
    else:
        locks = torch.empty(grids, device=a.device, dtype=torch.uint8)
        P = torch.empty(grids, block_size, device=a.device, dtype=torch.float32)
    chunk_size = _K697_GROUP_M * _K697_GROUP_M
    if _K697_NUM_XCDS > 0:
        chunk_size = min(chunk_size, max(1, grids // _K697_NUM_XCDS))

    _maybe_wrap(streamk_matmul, probe_tensor=a)[(grids,)](
        a, b, c, None, None, None,
        P, locks,
        M, N, K,
        a.stride(0), b.stride(1), c.stride(0), c.stride(1),
        0,
        stride_ak=a.stride(1), stride_bk=b.stride(0),
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
        GROUP_SIZE_M=_K697_GROUP_M,
        NUM_SMS=grids, NUM_XCDS=_K697_NUM_XCDS,
        CHUNK_SIZE=chunk_size,
        STREAMK_TILES=total_tiles_streamk,
        BIAS=False, EVEN_K=even_k,
        CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None,
        QUANTIZED=False,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        num_stages=_K697_NUM_STAGES, num_warps=_K697_NUM_WARPS,
        waves_per_eu=0, matrix_instr_nonkdim=16, kpack=1,
    )
    return c


# ============================================================================
# K-725: tall-skinny FP16/BF16 dispatch override -- N=64 sibling of K-697
# ----------------------------------------------------------------------------
# K-668 also flagged the N=64 column of the tall-skinny cohort as
# underperforming (geomean 0.49x vs hipBLASLt across 16 N=64 cells), with the
# worst (M=16384, K=8192) at 0.20x.  The K-725 81-config tile sweep
# (BM in {32,64,128} x BK in {64,128,256} x num_warps in {2,4,8}
# x (num_stages,kpack) in {(2,1),(2,2),(3,1)}) on the four worst K-668 N=64
# shapes (M=16384,K=8192 fp16 and bf16; M=16384,K=4096 bf16; M=4096,K=8192
# fp16) found a single dominating winner: Stream-K with
# (BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, num_warps=8, num_stages=2, kpack=2),
# geomean 3.07x over Origami across the 4 worst-gap shapes.
#
# Verification on the full 32-shape N=64 cohort
# (M in {2048..16384} x K in {1024..8192} x {fp16, bf16}, paired ON/OFF on
# MI300X gfx942) shows the candidate wins decisively when (M,K) is
# data-volume-rich, and loses to Origami on the smallest cells where
# dispatch tax dominates.  The shipped gate is therefore tightened to the
# clean-win envelope (zero shape regressions across the 32-cell cohort,
# 18/32 cells fired, geomean +2.16x within the gate, cohort-vs-hipBLASLt
# geomean lifts from 0.35x to 0.76x on the 18 fired cells).
#
# K-683 audit cohort leakage check (61 shapes spanning K-104, K-278, K-353,
# K-518, K-539, K-545, K-644, K-667, splitK PRD): only 2 audit shapes match
# the gate (4096x64x4096 fp16 and 8192x64x4096 fp16, both K-496 markers --
# they live in the K-668 cohort by construction); both improve (+1.40x and
# +1.70x).  Every other audit shape misses the gate by predicate (different
# N, smaller M, or smaller K).
#
# Implementation mirrors K-697 (single tile, hand-launched Stream-K, bypass
# Origami selector to skip the ~180us setup overhead which would otherwise
# wipe the win on the smaller-K cells of the cohort).  Cohort gate is
# strict on N (must equal _K725_BLOCK_N=64 to avoid any padding) and uses
# a disjunctive (M,K) envelope measured from the verify sweep.
# ============================================================================

# Hand-picked Stream-K tile (verified K-725 winner across 4 worst-gap shapes
# and on every shape covered by the cohort gate below).
_K725_BLOCK_M = 64
_K725_BLOCK_N = 64
_K725_BLOCK_K = 128
_K725_NUM_WARPS = 8
_K725_NUM_STAGES = 2
_K725_KPACK = 2
_K725_GROUP_M = 8
_K725_NUM_XCDS = 8


def _is_k725_n64_tall_skinny(M, N, K, dtype):
    """K-725 cohort gate: tall-skinny FP16/BF16 GEMM with N=64 strict.

    Tightened envelope from the K-725 32-cell verify sweep -- captures
    every (M,K) cell where the K-725 candidate beat Origami by >=1.20x
    (zero regressions in the gated subset, vs 13/32 regressions if the
    full K-668 N=64 envelope is used).
    """
    if dtype not in (torch.float16, torch.bfloat16):
        return False
    if N != _K725_BLOCK_N:        # strict-equality: N must equal BLOCK_N to avoid padding
        return False
    # Disjunctive (M, K) envelope measured from the verify sweep
    if M >= 16384 and K >= 1024:
        return True
    if M >= 4096 and K >= 4096:
        return True
    if M >= 2048 and K >= 8192:
        return True
    return False


def _k725_n64_tall_skinny_streamk(a, b, c):
    """Launch the streamk kernel directly with the K-725 winning tile.
    Bypasses _make_matmul_selector to avoid Origami's ~180us setup cost
    (which alone is greater than baseline kernel time at the smaller-K
    cells of the cohort).
    """
    M, K = a.shape
    _, N = b.shape

    BM = _K725_BLOCK_M
    BN = _K725_BLOCK_N
    BK = _K725_BLOCK_K
    grids = MAX_SMS  # 304 on MI300X
    total_blocks_M = triton.cdiv(M, BM)
    total_blocks_N = triton.cdiv(N, BN)
    total_tiles = total_blocks_M * total_blocks_N
    total_tiles_streamk = (total_tiles % grids) if grids > 0 else 0
    even_k = (K % BK) == 0
    block_size = BM * BN
    if grids <= MAX_SMS and block_size <= MAX_BLOCK_SIZE:
        locks = _global_locks[:grids]
        P = _global_P[:grids, :block_size]
    else:
        locks = torch.empty(grids, device=a.device, dtype=torch.uint8)
        P = torch.empty(grids, block_size, device=a.device, dtype=torch.float32)
    chunk_size = _K725_GROUP_M * _K725_GROUP_M
    if _K725_NUM_XCDS > 0:
        chunk_size = min(chunk_size, max(1, grids // _K725_NUM_XCDS))

    _maybe_wrap(streamk_matmul, probe_tensor=a)[(grids,)](
        a, b, c, None, None, None,
        P, locks,
        M, N, K,
        a.stride(0), b.stride(1), c.stride(0), c.stride(1),
        0,
        stride_ak=a.stride(1), stride_bk=b.stride(0),
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
        GROUP_SIZE_M=_K725_GROUP_M,
        NUM_SMS=grids, NUM_XCDS=_K725_NUM_XCDS,
        CHUNK_SIZE=chunk_size,
        STREAMK_TILES=total_tiles_streamk,
        BIAS=False, EVEN_K=even_k,
        CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None,
        QUANTIZED=False,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        num_stages=_K725_NUM_STAGES, num_warps=_K725_NUM_WARPS,
        waves_per_eu=0, matrix_instr_nonkdim=16, kpack=_K725_KPACK,
    )
    return c



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

    # K-697: tall-skinny FP16/BF16 dispatch override.  Skips the Origami
    # selector entirely (its ~180us setup is >baseline kernel time at
    # these sizes) and routes to a hand-picked Stream-K tile that the
    # K-697 sweep proved beats Origami by 1.20x..7.57x (geomean 2.91x)
    # and lifts cohort-vs-hipBLASLt from 0.51x to 0.69x.  See the gate
    # docstring above for cohort definition and verification protocol.
    if not work_stealing and _is_k697_tall_skinny(M, N, K, a.dtype):
        return _k697_tall_skinny_streamk(a, b, out)

    # K-725: tall-skinny FP16/BF16 N=64 dispatch override.  Sibling of
    # K-697 with a different tile (BM=BN=64, BK=128, NW=8, NS=2, KP=2).
    # Sub-cohort selected by the tightened (M,K) envelope to guarantee
    # zero regressions vs Origami across every shape that fires (geomean
    # +2.16x within the gate; cohort-vs-hipBLASLt 0.35x -> 0.76x).  See
    # the gate docstring above for cohort definition and verify protocol.
    if not work_stealing and _is_k725_n64_tall_skinny(M, N, K, a.dtype):
        return _k725_n64_tall_skinny_streamk(a, b, out)

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

    # K-697: tall-skinny FP16/BF16 dispatch override (see _matmul above).
    if not work_stealing and _is_k697_tall_skinny(M, N, K, a.dtype):
        _k697_tall_skinny_streamk(a, b, out)
        return None

    # K-725: tall-skinny FP16/BF16 N=64 dispatch override (see _matmul above).
    if not work_stealing and _is_k725_n64_tall_skinny(M, N, K, a.dtype):
        _k725_n64_tall_skinny_streamk(a, b, out)
        return None

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

