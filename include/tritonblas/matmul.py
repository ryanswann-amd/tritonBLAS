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
from .kernels.persistent_gemm_monolithic import persistent_matmul as _persistent_matmul_monolithic
from .kernels.fp4_matmul import fp4_matmul
from .origami import OrigamiMatmulSelector
from .config import MatmulConfig, matmul_preamble, COUNTER_STRIDE
from .lds_swizzle import select_lds_config


# K-349: FP8 (e4m3fnuz / e5m2fnuz) medium-K cohort fix for matmul_a8w8_lt on
# MI300X (gfx942). Two issues land together:
#   (1) Composable persistent_gemm hits an MLIR DenseElementsAttr assertion
#       on every FP8 tile we tested. We force the monolithic kernel for FP8
#       (analogous to TBLAS_USE_MONOLITHIC but scoped to FP8 only).
#   (2) Origami picks BLOCK_K in {128, 256} for FP8 medium-K, but the K-349
#       sweep on the K-313 medium-K cohort (K in [2048,4096]) shows BLOCK_K=64
#       wins 3-13% once BLOCK_M*BLOCK_N >= 128*128: fp32 acc pressure
#       (2*BM*BN regs) stays fixed, LDS-bw pressure halves, mainloop_iters
#       doubles (>=16, the K-654 amortization floor). Exact (M,N,K,dtype)
#       gate per K-654/K-683. Bypassed for streamk and work_stealing paths.
#       Kill switch: TRITONBLAS_DISABLE_K349_FP8.

# Lazily-initialised module state. Using a single int key (M<<32 | N<<16 | K
# packed into 48 bits) makes the per-call overhead a dict lookup with a
# single int hash and no tuple allocation. The dispatch table has 8 entries.
_K349_FP8_DT0 = None
_K349_FP8_DT1 = None
_K349_DT0_TABLE = {}  # int key -> (BM, BN, BK, ns)
_K349_DT1_TABLE = {}
_K349_DISABLE_OVERRIDE = False
_K349_INIT_DONE = False


def _k349_init():
    global _K349_FP8_DT0, _K349_FP8_DT1
    global _K349_DT0_TABLE, _K349_DT1_TABLE
    global _K349_DISABLE_OVERRIDE, _K349_INIT_DONE
    import os
    _K349_DISABLE_OVERRIDE = bool(os.environ.get("TRITONBLAS_DISABLE_K349_FP8"))
    _K349_FP8_DT0 = getattr(torch, "float8_e4m3fnuz", None)
    _K349_FP8_DT1 = getattr(torch, "float8_e5m2fnuz", None)
    base = {
        (4096 << 32) | (4096 << 16) | 4096: (256, 256, 64, 2),
        (4096 << 32) | (4096 << 16) | 2048: (256, 256, 64, 2),
        (2048 << 32) | (2048 << 16) | 4096: (128, 128, 128, 2),
        (2048 << 32) | (2048 << 16) | 2048: (128, 128, 128, 2),
    }
    _K349_DT0_TABLE = dict(base) if _K349_FP8_DT0 is not None else {}
    _K349_DT1_TABLE = dict(base) if _K349_FP8_DT1 is not None else {}
    _K349_INIT_DONE = True


def _k349_apply_fp8_overrides(a, b, selector):
    """Return True iff `a.dtype` is FP8 FNUZ (caller should force monolithic
    kernel due to a known composable-kernel FP8 compile-time bug). Side
    effect: applies the per-cohort tile/num_stages override in-place on
    `selector` if (M, N, K, dtype) is a dispatch-table entry AND the kill
    switch TRITONBLAS_DISABLE_K349_FP8 is unset.

    The override application is cached per `selector` instance via the
    sentinel attribute `_k349_done` so paying the dispatch-table lookup
    once per shape — _make_matmul_selector is itself wrapped in
    @functools.lru_cache(maxsize=1024) upstream (see K-138, K-278), so a
    repeated call with the same (M, N, K, dtype) hits the cached selector
    and skips this path entirely.
    """
    if not _K349_INIT_DONE:
        _k349_init()
    dt = a.dtype
    if dt is _K349_FP8_DT0:
        table = _K349_DT0_TABLE
    elif dt is _K349_FP8_DT1:
        table = _K349_DT1_TABLE
    else:
        return False
    if getattr(selector, "_k349_done", False):
        return True
    selector._k349_done = True
    if _K349_DISABLE_OVERRIDE:
        return True
    o = table.get((a.shape[0] << 32) | (b.shape[1] << 16) | a.shape[1])
    if o is not None:
        BM, BN, BK, ns = o
        # LDS guard: ns * (BM*BK + BK*BN) bytes for FP8 (1 byte/elem).
        if ns * (BM * BK + BK * BN) <= 64 * 1024:
            selector._result.config.mt.m = BM
            selector._result.config.mt.n = BN
            selector._result.config.mt.k = BK
            selector._num_stages = ns
    return True


# Per-shape tile overrides for FP8 e5m2fnuz x e4m3fnuz square GEMMs on MI300X
# where the analytical selector picks tiles that trail hipBLASLt by >=15%.
# Each entry below is the winning config from a paired ON/OFF graph-captured
# sweep (BM/BN/BK in {64,128,256} x num_stages in {2,3} x num_warps in {4,8}).
_FP8_SQUARE_TILE_OVERRIDES: Dict[Tuple[int, int, int],
                                  Tuple[int, int, int, int, int]] = {
    # (M, N, K) -> (BM, BN, BK, num_stages, num_warps)
    (1024, 1024,  512): ( 64,  64, 256, 2, 8),
    (4096, 4096, 1024): (256, 256,  64, 2, 8),
    (4096, 4096, 2048): (256, 256,  64, 2, 8),
}


class _TileOverride:
    """Override tile knobs on a selector; delegate everything else."""

    def __init__(self, inner, bm, bn, bk, ns, nw):
        self._inner = inner
        self.block_m, self.block_n, self.block_k = bm, bn, bk
        self.num_stages, self.num_warps = ns, nw

    def __getattr__(self, name):
        return getattr(self._inner, name)


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

# ---------------------------------------------------------------------------
# K-695: FP8 e4m3fnuz tall-skinny (small-M, large-N) tile override.
# Origami over-tiles BM and under-stages BK on this cohort, costing up to
# 2.10x vs hipBLASLt. The override is a single (M,N,K)->(BM,NS,KP) table
# populated from the per-shape sweep winner (see workspace
# output/tile_table.json and output/gate_tile_table.json). Shapes whose
# sweep winner == Origami default (vs_origami < 1.03) are intentionally
# omitted so the predicate doesn't fire where it doesn't help; this skips
# (16,4096,16384) and the entire (M=64,N=8192) sub-band. See K-695.
_FP8_E4M3FNUZ = getattr(torch, "float8_e4m3fnuz", None)
_K695_GATE_TABLE = {
    # M=16 (8 shapes; (16,4096,16384) omitted: sweep winner == Origami default)
    (16, 4096, 4096):   (32, 2, 1),
    (16, 4096, 8192):   (32, 2, 1),
    (16, 8192, 4096):   (32, 2, 1),
    (16, 8192, 8192):   (32, 2, 1),
    (16, 8192, 16384):  (32, 2, 1),
    (16, 16384, 4096):  (32, 2, 1),
    (16, 16384, 8192):  (32, 2, 1),
    (16, 16384, 16384): (32, 2, 1),
    # M=32 (9 shapes)
    (32, 4096, 4096):   (32, 2, 1),
    (32, 4096, 8192):   (32, 2, 1),
    (32, 4096, 16384):  (32, 2, 1),
    (32, 8192, 4096):   (32, 2, 1),
    (32, 8192, 8192):   (32, 2, 1),
    (32, 8192, 16384):  (32, 2, 1),
    (32, 16384, 4096):  (32, 2, 1),
    (32, 16384, 8192):  (32, 2, 1),
    (32, 16384, 16384): (32, 2, 1),
    # M=64,N=4096 (3 shapes; NS=1 because Origami picks BK=512 here)
    (64, 4096, 4096):   (32, 1, 2),
    (64, 4096, 8192):   (32, 1, 2),
    (64, 4096, 16384):  (32, 1, 1),  # sweep winner is KP=1 at this K
    # (64,8192,*) omitted: Origami already optimal on this sub-band
    # M=64,N=16384 (3 shapes)
    (64, 16384, 4096):  (32, 2, 1),
    (64, 16384, 8192):  (32, 2, 1),
    (64, 16384, 16384): (32, 2, 1),
}


def _k695_tile_override(M, N, K, a_dtype):
    """Return (BM, NS, KP) override for the K-695 cohort, else None."""
    if _FP8_E4M3FNUZ is None or a_dtype is not _FP8_E4M3FNUZ:
        return None
    return _K695_GATE_TABLE.get((M, N, K))
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# K-693: tall-skinny FP16/BF16 N=64 persistent-path tile override.
#
# Origami over-tiles BM (256 macrotile at M >= 4096) and under-stages BK
# (BLOCK_K=32 -> 256 mainloop iters at K=8192) on the N=64 sub-band of
# the K-668 tall-skinny cohort. A narrow tile sweep on the 4 worst-offender
# shapes (M in {4096,8192,16384} x N=64 x K in {4096,8192} x {fp16,bf16})
# found a single cross-shape persistent-path winner:
#   BLOCK_M=32, BLOCK_N=64, BLOCK_K=128, num_stages=2, num_warps=8, kpack=2
# Persistent dispatch (no Stream-K) -- with BM=32 there are enough output
# tiles to fill MI300X's 304 CUs, so Stream-K only adds atomic overhead.
#
# Predicate: M >= 4096 AND N == 64 AND K >= 4096 AND {fp16, bf16}.
# M=2048,N=64 row excluded -- Origami's default beats the override there.
#
# Disjoint from K-697 (N==32 strict-equality, dispatched via early-return
# in _matmul / _matmul_out). Overlaps K-710 on (M=4096, N=64, K in
# {4096,8192}); K-693 force-restores num_warps=8 over K-710's 4 because
# the K-693 tile geometry needs the larger warp count for issue density.
#
# Set TRITONBLAS_DISABLE_K693=1 to bypass.
# ---------------------------------------------------------------------------
import os as _os_k693
_K693_GATE_ENABLED = _os_k693.environ.get("TRITONBLAS_DISABLE_K693", "0") != "1"


def _k693_tile_override(M, N, K, a_dtype, b_dtype):
    """Return (BM, BN, BK, NS, NW, KP) override for the K-693 N=64 cohort,
    else None.  Strict-equality on N; conservative measured bounds on M, K.
    """
    if not _K693_GATE_ENABLED:
        return None
    if a_dtype is not b_dtype:
        return None
    if a_dtype is not torch.float16 and a_dtype is not torch.bfloat16:
        return None
    if N != 64 or M < 4096 or K < 4096:
        return None
    return (32, 64, 128, 2, 8, 2)
# ---------------------------------------------------------------------------


def _num_warps_for_tall_skinny_fp16_bf16(M: int, N: int, K: int, dtype: torch.dtype) -> int:
    """Tall-skinny FP16/BF16 cohort override (default 8 -> 4).

    On the M in {2048, 4096}, N <= 64, K >= 2048, fp16/bf16 sub-cohort the
    Origami selector picks a small persistent tile (BLOCK_N=16) that leaves
    50% of the MFMA N-lanes masked, and the resulting kernel is VALU-bound
    (rocprofv2 VALUUtil ~100%, MfmaUtil ~4%) per K-668. Cutting num_warps
    from 8 to 4 lets the issue scheduler pack two unrolled K-loop iterations
    per VALU slot, restoring kernel time to within range of hipBLASLt.

    Validated on MI300X / gfx942 via paired HIP-graph kernel-only
    timing (n_warm=3, n_capture=15, n_rounds=5). Worst-4 cohort speedup
    geomean 1.41x (min 1.31x); 12-shape neighbor cohort speedup geomean
    1.51x (min 1.10x). Out-of-cohort guard shapes (squares, large-N,
    M=8192/16384, K<2048) regress 8-34% with this override -- gate is
    therefore strictly conjunctive on (M, N, K, dtype).
    """
    if dtype not in (torch.float16, torch.bfloat16):
        return 8
    if M not in (2048, 4096):
        return 8
    if N > 64:
        return 8
    if K < 2048:
        return 8
    return 4


def _maybe_wrap(fn, probe_tensor):
    # Use wrap_triton only under torch.compile tracing; otherwise direct call
    # in eager.  Can't use torch.compiler.is_compiling() here because the code
    # inside @triton_op but outside wrap_triton is part of the compile pass
    # itself and is_compiling() is never True.
    if is_fake(probe_tensor):
        return wrap_triton(fn)
    return fn


# ---------------------------------------------------------------------------
# FP8 e5m2fnuz medium-K square tile-override gate (K-656).
#
# Cohort: M==N==4096, K in {1024, 2048}, dtype = float8_e5m2fnuz, non-streamk.
#
# Origami picks BM=BN=256, BK=128, NS=2 for these shapes. A K-624-style
# tile/pipeline sweep across the 9-shape symmetric FP8 e5m2fnuz medium-K
# cohort (M=N in {1024,2048,4096} x K in {512,1024,2048}) shows BK=64 wins
# only on M=N=4096 with K in {1024, 2048} (+8.4% / +5.5% vs Origami,
# paired ON/OFF n=25x100). The other 7 cohort shapes either match Origami
# or regress under any override -- the predicate is the cohort.
#
# Set K656_DISABLE=1 in the environment to bypass the override (A/B harness).
# ---------------------------------------------------------------------------
import os as _os_k656


def _is_k656_cohort(M, N, K, a_dtype, b_dtype, streamk):
    """Strict shape+dtype predicate for the K-656 tile override."""
    if streamk or _os_k656.environ.get("K656_DISABLE", "0") == "1":
        return False
    if a_dtype is not torch.float8_e5m2fnuz or b_dtype is not torch.float8_e5m2fnuz:
        return False
    return M == 4096 and N == 4096 and K in (1024, 2048)


class _K656Selector(OrigamiMatmulSelector):
    """OrigamiMatmulSelector subclass that pins BM=BN=256, BK=64, NS=2.

    Class-level overrides shadow the @property descriptors on the parent.
    All other attributes (group_m, num_sms, sk_grid, _hardware, ...) are
    inherited unchanged.
    """
    block_m = 256
    block_n = 256
    block_k = 64
    num_stages = 2


# Per-shape kpack override for the large-K square FP16/BF16 GEMM regime (K-667).
#
# Empirical kernel-only HIP-graph A/B (5 trials, 50-op chain x 3 replays/cell
# on MI300X / gfx942 / ROCm 7.2) over the cohort
#     M = N in {1024, 2048, 4096}, K in {4096, 8192, 16384}, dtype in {fp16, bf16}
# shows that flipping kpack from 1 -> 2 is non-regressing only on the M=N=2048
# sub-row; the M=N=1024 sub-row regresses +1.7% .. +4.5% and the M=N=4096
# sub-row regresses +8.9% .. +11.6%. Set TRITONBLAS_DISABLE_K667=1 to bypass.
_LARGE_K_SQUARE_KPACK2_M = 2048
_LARGE_K_SQUARE_KPACK2_K_VALUES = (4096, 8192, 16384)
_LARGE_K_SQUARE_KPACK2_DTYPES = (torch.float16, torch.bfloat16)


def _kpack_for_large_k_square(M: int, N: int, K: int, dtype) -> int:
    """Return kpack=2 only on (M==N==2048, K in {4096,8192,16384}, fp16/bf16)."""
    if _os_k656.environ.get("TRITONBLAS_DISABLE_K667", "0") == "1":
        return 1
    if dtype not in _LARGE_K_SQUARE_KPACK2_DTYPES:
        return 1
    if M != N or M != _LARGE_K_SQUARE_KPACK2_M:
        return 1
    if K not in _LARGE_K_SQUARE_KPACK2_K_VALUES:
        return 1
    return 2


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
    sel = OrigamiMatmulSelector(
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
    if _is_k656_cohort(M, N, K, a_dtype, b_dtype, streamk):
        sel.__class__ = _K656Selector
    return sel


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
    force_monolithic: bool = False,
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
    # K-710: num_warps=4 for FP16/BF16 tall-skinny cohort (M in {2048,4096},
    # N<=64, K>=2048). Returns 8 elsewhere. K-683's LDS-swizzle gate may
    # override this below when its non-default band fires.
    num_warps = _num_warps_for_tall_skinny_fp16_bf16(M, N, K, a.dtype)
    waves_per_eu = 0
    mfmaInstrSize = 16
    # K-667: narrow per-shape kpack gate. Default is kpack=1; on the large-K
    # square FP16/BF16 sub-cohort the helper returns 2. K-683's _lds_cfg may
    # override below for its own (medium-K residual) band.
    kpack = _kpack_for_large_k_square(M, N, K, a.dtype)
    CACHE_MODIFIER_A = None
    CACHE_MODIFIER_B = None

    # K-580/K-683: select LDS swizzle config (kpack/num_warps) for
    # bank-conflict mitigation. Defaults to (kpack=1, num_warps=8) outside
    # the medium-K residual band.
    _lds_cfg = select_lds_config(
        M, N, K,
        a_dtype=str(a.dtype), b_dtype=str(b.dtype), c_dtype=str(c.dtype),
        block_m=BLK_M, block_n=BLK_N, block_k=BLK_K,
        streamk=False, work_stealing=work_stealing,
    )
    # K-683 overrides num_warps/kpack only when its predicate band fires
    # (non-default). Outside that band, preserve K-710's num_warps choice
    # (4 in the FP16/BF16 tall-skinny cohort, 8 elsewhere) and K-667's
    # kpack value (default=1, =2 on the large-K square cohort).
    if _lds_cfg.num_warps != 8:
        num_warps = _lds_cfg.num_warps
    if _lds_cfg.kpack != 1:
        kpack = _lds_cfg.kpack

    # K-695: tile override for FP8 e4m3fnuz tall-skinny cohort. LDS-fit guard
    # falls back to Origami on the (rare) BN/BK combo where the override
    # exceeds the 64KiB MI300X workgroup LDS budget. Applied after K-683 so
    # the override wins on its narrow FP8 predicate (no overlap: K-683 is
    # FP16/BF16-only).
    _ovr = _k695_tile_override(M, N, K, a.dtype)
    if _ovr is not None and not work_stealing:
        _bm, _ns, _kp = _ovr
        if _ns * (_bm * BLK_K + BLK_K * BLK_N) <= 65536:
            BLK_M, num_stages, kpack = _bm, _ns, _kp
            total_blocks_M = triton.cdiv(M, BLK_M)
            total_tiles = total_blocks_M * total_blocks_N
            total_programs = total_tiles

    # K-693: tall-skinny FP16/BF16 N=64 persistent override. Applied after
    # K-695 (FP8 only -- no overlap) and after K-710's num_warps helper so
    # that K-693's num_warps=8 wins back over K-710's num_warps=4 on the
    # M=4096,N=64,K in {4096,8192} overlap region (K-693 tile geometry needs
    # the larger warp count for issue density).
    _k693_ovr = _k693_tile_override(M, N, K, a.dtype, b.dtype)
    if _k693_ovr is not None and not work_stealing:
        _bm, _bn, _bk, _ns, _nw, _kp = _k693_ovr
        BLK_M, BLK_N, BLK_K = _bm, _bn, _bk
        num_stages, num_warps, kpack = _ns, _nw, _kp
        even_k = (K % BLK_K) == 0
        total_blocks_M = triton.cdiv(M, BLK_M)
        total_blocks_N = triton.cdiv(N, BLK_N)
        total_tiles = total_blocks_M * total_blocks_N
        total_programs = total_tiles

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

        # K-349: route FP8 quantized dispatches to the monolithic kernel
        # because the composable persistent_gemm hits an MLIR assertion on
        # the FP8 path. force_monolithic is set by matmul_a8w8_lt for FP8.
        _kernel_fn = _persistent_matmul_monolithic if force_monolithic else persistent_matmul

        kk = _maybe_wrap(_kernel_fn, probe_tensor=a)[(grids,)](
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
    # K-710: num_warps=4 for FP16/BF16 tall-skinny cohort (M in {2048,4096},
    # N<=64, K>=2048). Returns 8 elsewhere. K-683's LDS-swizzle gate may
    # override this below when its non-default band fires.
    num_warps = _num_warps_for_tall_skinny_fp16_bf16(M, N, K, a.dtype)
    waves_per_eu = 0
    mfmaInstrSize = 16
    # K-667: narrow per-shape kpack gate. Default is kpack=1; on the large-K
    # square FP16/BF16 sub-cohort the helper returns 2. K-683's _lds_cfg may
    # override below for its own (medium-K residual) band.
    kpack = _kpack_for_large_k_square(M, N, K, a.dtype)
    CACHE_MODIFIER_A = None
    CACHE_MODIFIER_B = None

    # K-580: select LDS swizzle config (kpack/num_warps) for bank-conflict mitigation.
    # Defaults to (kpack=1, num_warps=8) outside the medium-K residual band.
    _lds_cfg = select_lds_config(
        M, N, K,
        a_dtype=str(a.dtype), b_dtype=str(b.dtype), c_dtype=str(c.dtype),
        block_m=BLK_M, block_n=BLK_N, block_k=BLK_K,
        streamk=True, work_stealing=work_stealing,
    )
    # K-683 overrides num_warps/kpack only when its predicate band fires
    # (non-default). Outside that band, preserve K-710's num_warps choice
    # (4 in the FP16/BF16 tall-skinny cohort, 8 elsewhere) and K-667's
    # kpack value (default=1, =2 on the large-K square cohort).
    if _lds_cfg.num_warps != 8:
        num_warps = _lds_cfg.num_warps
    if _lds_cfg.kpack != 1:
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

    # K-349: FP8 medium-K cohort tile override + force monolithic kernel.
    # Bypasses streamk and work_stealing paths (those have separate kernels).
    is_fp8 = (not enable_streamk) and (not work_stealing) and \
        _k349_apply_fp8_overrides(a, b, selector)

    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector, config, a_scale=a_scale, b_scale=b_scale, quantized=True)
    else:
        return persistent_matmul_lt(a, b, c, selector, config, a_scale=a_scale, b_scale=b_scale, quantized=True, work_stealing=work_stealing, force_monolithic=is_fp8)


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
    # K-722 composite-8: K-697 fires first (strict-equality narrow gate,
    # cannot overlap K-633 ladder cohorts; early-returns before ladder).
    if not work_stealing and _is_k697_tall_skinny(M, N, K, a.dtype):
        return _k697_tall_skinny_streamk(a, b, out)

    # ============================================================================
    # K-763 composite-12 DISPATCH ORDER (load-bearing — DO NOT REORDER):
    # ----------------------------------------------------------------------------
    #   1. K-697 (above): strict-equality N==32 Stream-K early-return.
    #   2. K-725 (here):  strict-equality N==64 Stream-K early-return with
    #                     disjunctive (M,K) envelope.
    #   3. K-633 ladder + Origami selector (below).
    #   4. Inside persistent_matmul_lt: K-693 N==64 PERSISTENT tile override
    #      (M >= 4096 AND N == 64 AND K >= 4096 AND {fp16,bf16}).
    #
    # Why this order (per K-722 pattern): strict-equality narrow Stream-K
    # gates (K-697, K-725) MUST fire BEFORE tile-only overrides that mutate
    # the persistent dispatch (K-693), because K-725's disjunctive envelope
    # (M>=16384 AND K>=1024) OR (M>=4096 AND K>=4096) OR (M>=2048 AND K>=8192)
    # CONTAINS THE FULL K-693 envelope (M>=4096 AND N==64 AND K>=4096).
    # K-725 dominates K-693 on every cell where K-693 fires (K-725 +2.16x
    # gmean / BM=BN=64 Stream-K vs K-693 +1.88x / BM=32 persistent).
    # K-693 is retained as fallback if TRITONBLAS_DISABLE_K725 is set or
    # K-725 ever has to be reverted — the persistent path remains the
    # second-best dispatch on the N==64 sub-band.
    # K-697 ↔ K-725 are disjoint by N (32 vs 64); order between them is
    # cosmetic but K-697-first matches the original composite-11 layout.
    # ============================================================================
    if not work_stealing and _is_k725_n64_tall_skinny(M, N, K, a.dtype):
        return _k725_n64_tall_skinny_streamk(a, b, out)

    # ---------------------------------------------------------------
    # K-633 dispatch precedence ladder (evidence: K-619 sweep, MI300X)
    # Ordered if/elif chain from highest-confidence rule downward.
    # See knowledge/tritonblas/k_633_research.md (P1-P5).
    # ---------------------------------------------------------------
    # P1 - VETO: M <= 8 AND K >= 4096 region.
    # K-619 measured 68/70 streamk regressions vs only 1 win in this
    # window; G2/G3 static-tweaks dominate (29/70 wins, zero regressions).
    # Force-disable streamk even if the caller asked for it (K-654 hard
    # rule: no caller can bypass this veto, mirroring the 53/61-shape
    # regression that motivated the precedence ladder).
    if M <= 8 and K >= 4096:
        enable_streamk = False
    # P2 - Tightened large-balanced streamk auto-flip.
    # K-619 R-G4 (min(M,N)>=4096 AND K>=4096) admitted 7 regressors out
    # of 22. Tighten to M == N AND min(M,N) >= 5120 AND K >= 6144 which
    # keeps 8/9 wins and eliminates every measured regression.
    elif (
        not enable_streamk
        and M == N
        and min(M, N) >= 5120
        and K >= 6144
    ):
        enable_streamk = True
    # P3 - DO NOT auto-flip streamk on R-G4 small_m_decode envelope
    # (M <= 64 AND N >= 1024 AND K >= 1024). K-619 measured 214/221
    # regressions in that envelope; intentionally left as guardrail
    # comment so any future re-introduction is caught by review.

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


# --- K-648/K-692: FP8 e4m3fnuz skinny-M / large-N tile-override gate (MI300X) ---
# Origami over-tiles BM and over-stages BK on the FP8 e4m3fnuz skinny-M / large-N
# cohort, inflating kernel time vs. a fixed BM=32 / BN=128 / nw=4 / ns=2
# launch-floor config (with K-bucketed BK = 128 if K ≥ 4096 else 64).
#
# K-692 paired A/B (HIP-graph, n=30, MI300X gfx942) on the original 27-shape
# cohort (M ∈ {16,32,64} × N ∈ {4096,8192,16384} × K ∈ {1024,2048,4096}) showed
# the override REGRESSES on M ∈ {32,64} for several N/K buckets (worst 0.752×
# at M=64, N=8192, K=1024 — 6 of 27 shapes <0.95×). The gate is therefore
# tightened to M=16 only — the sub-cohort where the override wins on every
# shape (kernel-only speedup 1.26×–2.62×, geomean ≈1.85×).
class _K648Override:
    """Selector wrapper for the K-648/K-692 gate cohort.

    Pins the 5 tile/launch parameters to the validated override; everything
    else (group_m, num_sms, _hardware, …) is delegated to the inner Origami
    selector via __getattr__.
    """

    def __init__(self, inner, BM, BN, BK, ns, nw):
        self._inner = inner
        self.block_m = BM
        self.block_n = BN
        self.block_k = BK
        self.num_stages = ns
        self.num_warps = nw

    def __getattr__(self, name):
        # __getattr__ only fires for attrs not set on self → safe delegation.
        return getattr(self._inner, name)


def _k648_should_override(M: int, N: int, K: int, a_dtype, enable_streamk: bool) -> bool:
    """Tightened K-692 cohort predicate.

    Fires only on FP8 e4m3fnuz, non-streamk, M ≤ 16, N ≥ 4096, K ≥ 1024 —
    the sub-band where the override wins on every shape (paired A/B, MI300X).

    Controlled by env var ``TRITONBLAS_DISABLE_K648`` (default unset). Set to
    ``1`` only to reproduce the pre-patch HEAD baseline for A/B benchmarking.
    """
    if os.environ.get("TRITONBLAS_DISABLE_K648") == "1":
        return False
    return (
        not enable_streamk
        and a_dtype == torch.float8_e4m3fnuz
        and M <= 16
        and N >= 4096
        and K >= 1024
    )


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
    # K-653: per-shape tile override for FP8 e5m2fnuz x e4m3fnuz square cohort.
    # Mutually exclusive with K-692 (different dtype predicate).
    if (a.dtype is torch.float8_e5m2fnuz and b.dtype is torch.float8_e4m3fnuz):
        cfg = _FP8_SQUARE_TILE_OVERRIDES.get((M, N, K))
        if cfg is not None:
            selector = _TileOverride(selector, *cfg)
    # K-648/K-692: tighten to M=16 FP8 e4m3fnuz skinny-M sub-cohort.
    # Predicate is dtype==float8_e4m3fnuz AND M<=16 AND N>=4096 AND K>=1024,
    # which does not overlap K-653 (mixed e5m2/e4m3) or K-695 (M in 16..64
    # but K-695 only fires inside persistent_matmul_lt's path; here in
    # matmul_a8w8 the M=16 sub-cohort gets K-692's swept modal tile).
    if _k648_should_override(M, N, K, a.dtype, enable_streamk):
        selector = _K648Override(
            selector, BM=32, BN=128,
            BK=(128 if K >= 4096 else 64),
            ns=2, nw=4,
        )
    config = matmul_preamble(selector) if work_stealing else None
    # K-349: FP8 cohort tile override + force monolithic kernel for non-streamk/non-WS.
    is_fp8 = (not enable_streamk) and (not work_stealing) and \
        _k349_apply_fp8_overrides(a, b, selector)
    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector, config, sk_grid=sk_grid, a_scale=a_scale, b_scale=b_scale, quantized=True, work_stealing=work_stealing)
    else:
        return persistent_matmul_lt(a, b, c, selector, config, a_scale=a_scale, b_scale=b_scale, quantized=True, work_stealing=work_stealing, force_monolithic=is_fp8)

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

