# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
GemmContext aggregate for tritonblas shards.

Provides the K-loop execution context for GEMM operations, managing
the accumulator and iteration over the reduction dimension. Also bundles
all GEMM configuration parameters (block sizes, scheduling, computation options).
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from .tile import Tile
from .matrix_view import InputView


@aggregate
class GemmContext:
    """
    GEMM context with all configuration parameters and accumulator management.
    
    Bundles together all compile-time GEMM parameters:
    
    * Block sizes (M, N, K)
    * Hardware configuration (NUM_SMS, NUM_XCDS)
    * Scheduling parameters (GROUP_SIZE_M, CHUNK_SIZE)
    * Cache modifiers
    * Computation options (acc_dtype, allow_tf32, even_k, quantized)
    
    Provides two execution modes:
    
    * ``reduce_tile()``: Single BLOCK_K iteration (one dot product)
    * ``reduce_axis()``: Full K loop
    
    **Tile Creation Convention**
    
    For A [M, K] and B [K, N]:
    
    * A tiles: (pid_m, k_idx) with shape (BLOCK_M, BLOCK_K)
    * B tiles: (k_idx, pid_n) with shape (BLOCK_K, BLOCK_N)
    
    The :class:`InputView` handles the pointer arithmetic based on its stored layout.
    
    Example
    -------
    .. code-block:: python
    
        tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
        tensorB = make_tensor_view(B, K, N, stride_bk, stride_bn)
        
        ctx = GemmContext(
            block_m=128, block_n=256, block_k=64,
            num_sms=NUM_SMS, num_xcds=NUM_XCDS,
            group_size_m=8, even_k=EVEN_K,
        )
        
        # Use in ScheduleContext
        sched = ScheduleContext(M, N, K, ctx)
        
        acc = ctx.reduce_axis(tensorA, tensorB, out_tile)
    """
    
    # Block sizes
    block_m: tl.constexpr
    block_n: tl.constexpr
    block_k: tl.constexpr
    
    # Hardware config
    num_sms: tl.constexpr
    num_xcds: tl.constexpr
    
    # Scheduling
    group_size_m: tl.constexpr
    chunk_size: tl.constexpr
    
    # Cache modifiers
    cache_modifier_a: tl.constexpr
    cache_modifier_b: tl.constexpr
    
    # Computation options
    acc_dtype: tl.constexpr
    allow_tf32: tl.constexpr
    even_k: tl.constexpr
    quantized: tl.constexpr
    
    @triton.constexpr_function
    def __init__(
        self,
        block_m,
        block_n,
        block_k,
        num_sms,
        num_xcds=1,
        group_size_m=8,
        chunk_size=1,
        cache_modifier_a=".cg",
        cache_modifier_b=".cg",
        acc_dtype=tl.float32,
        allow_tf32=True,
        even_k=True,
        quantized=False,
    ):
        """
        Create a GEMM context with all configuration parameters.
        
        Args:
            block_m: Block size M (constexpr)
            block_n: Block size N (constexpr)
            block_k: Block size K (constexpr)
            num_sms: Number of SMs/CUs (constexpr)
            num_xcds: Number of XCDs for chiplet transform (default: 1)
            group_size_m: Group size for tile scheduling (default: 8)
            chunk_size: Chunk size for chiplet scheduling (default: 1)
            cache_modifier_a: Cache modifier for A loads (default: ".cg")
            cache_modifier_b: Cache modifier for B loads (default: ".cg")
            acc_dtype: Accumulator dtype (default: tl.float32)
            allow_tf32: Allow TF32 for matmul (default: True)
            even_k: Whether K is evenly divisible by BLOCK_K (default: True)
            quantized: Use int32 accumulation for quantized inputs (default: False)
        """
        self.block_m = tl.constexpr(block_m)
        self.block_n = tl.constexpr(block_n)
        self.block_k = tl.constexpr(block_k)
        self.num_sms = tl.constexpr(num_sms)
        self.num_xcds = tl.constexpr(num_xcds)
        self.group_size_m = tl.constexpr(group_size_m)
        self.chunk_size = tl.constexpr(chunk_size)
        self.cache_modifier_a = tl.constexpr(cache_modifier_a)
        self.cache_modifier_b = tl.constexpr(cache_modifier_b)
        self.acc_dtype = tl.constexpr(acc_dtype)
        self.allow_tf32 = tl.constexpr(allow_tf32)
        self.even_k = tl.constexpr(even_k)
        self.quantized = tl.constexpr(quantized)
    
    @triton.jit
    def init_accumulator(self):
        """
        Initialize and return a zero accumulator.
        
        Returns:
            Accumulator tensor [BLOCK_M, BLOCK_N] initialized to zeros
        
        Example::
        
            acc = ctx.init_accumulator()
        """
        return tl.zeros((self.block_m, self.block_n), dtype=self.acc_dtype)
    
    @triton.jit
    def reduce_tile(
        self,
        A: InputView,
        B: InputView,
        out_tile: Tile,
        k_idx,
        acc,
        boundary: tl.constexpr = False,
    ):
        """
        Execute a single K step (one BLOCK_K iteration).
        
        Creates tiles for A and B at the given K index and loads them using
        the InputView's tile_ptrs method (which handles layout internally).
        
        Args:
            A: InputView for matrix A [M, K] with strides already stored
            B: InputView for matrix B [K, N] with strides already stored
            out_tile: Output Tile with (pid_m, pid_n, BLOCK_M, BLOCK_N)
            k_idx: Current K tile index
            acc: Accumulator to add to
            boundary: Whether this is a boundary iteration needing masking
        
        Returns:
            Updated accumulator tensor [BLOCK_M, BLOCK_N]
        
        Example::
        
            A = make_tensor_view(A_ptr, M, K, stride_am, stride_ak)
            B = make_tensor_view(B_ptr, K, N, stride_bk, stride_bn)
            acc = ctx.init_accumulator()
            for k_idx in range(num_k_tiles):
                acc = ctx.reduce_tile(A, B, out_tile, k_idx, acc)
        """
        pid_m = out_tile.pid_m
        pid_n = out_tile.pid_n
        
        # ═══════════════════════════════════════════════════════════════════
        # TILE CREATION: Create tiles for A and B at this K iteration
        # ═══════════════════════════════════════════════════════════════════
        # A [M, K]: tile at (pid_m, k_idx) with shape (BLOCK_M, BLOCK_K)
        # B [K, N]: tile at (k_idx, pid_n) with shape (BLOCK_K, BLOCK_N)
        a_tile = Tile(pid_m, k_idx, self.block_m, self.block_k)
        b_tile = Tile(k_idx, pid_n, self.block_k, self.block_n)
        
        # ═══════════════════════════════════════════════════════════════════
        # POINTER COMPUTATION: InputView handles layout internally
        # ═══════════════════════════════════════════════════════════════════
        # The InputView's tile_ptrs method uses its stored stride_row and
        # stride_col to compute correct pointers for any layout.
        a_ptrs, a_mask = A.tile_ptrs(a_tile)
        b_ptrs, b_mask = B.tile_ptrs(b_tile)
        
        # ═══════════════════════════════════════════════════════════════════
        # LOAD TILES
        # ═══════════════════════════════════════════════════════════════════
        if boundary:
            # Use masks for K boundary
            a = tl.load(a_ptrs, mask=a_mask, other=0.0, cache_modifier=self.cache_modifier_a)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=self.cache_modifier_b)
        else:
            a = tl.load(a_ptrs, cache_modifier=self.cache_modifier_a)
            b = tl.load(b_ptrs, cache_modifier=self.cache_modifier_b)
        
        # ═══════════════════════════════════════════════════════════════════
        # ACCUMULATE
        # ═══════════════════════════════════════════════════════════════════
        if self.quantized:
            acc += tl.dot(a, b, out_dtype=tl.int32)
        else:
            acc += tl.dot(a, b, allow_tf32=self.allow_tf32)
        
        return acc
    
    @triton.jit
    def reduce_axis(
        self,
        A: InputView,
        B: InputView,
        out_tile: Tile,
    ):
        """
        Execute the full GEMM K loop and return the accumulator.

        Iterates over all K tiles, loading from A and B using their stored
        layout information, and accumulates the dot products.

        K-1653: B-tile (and A-tile) base pointers are hoisted out of the
        inner K loop and advanced via ``+= BLOCK_K * stride_k`` instead of
        being recomputed from scratch every K iteration. This mirrors the
        monolithic kernel's pointer-increment pattern, gives the compiler
        a more predictable address-stream, and shrinks the per-tile
        residency window K-1634 PMC identified as the dominant L2-thrash
        mechanism for the N=16384 long-K cohort. Net effect on the 6
        K-1634 cells is geomean +6.5 pp speedup vs hipBLASLt.

        Args:
            A: InputView for matrix A [M, K] with strides already stored
            B: InputView for matrix B [K, N] with strides already stored
            out_tile: Output Tile with (pid_m, pid_n, BLOCK_M, BLOCK_N)

        Returns:
            Accumulator tensor [BLOCK_M, BLOCK_N]
        """
        # Initialize accumulator
        acc = self.init_accumulator()

        # Compute K loop bounds (K dimension is A.cols or B.rows)
        num_k_tiles = tl.cdiv(A.cols, self.block_k)
        if not self.even_k:
            num_k_tiles -= 1
        tl.assume(num_k_tiles > 0)

        # Hoist A/B base pointers ONCE per output tile (k_idx == 0). The K
        # loop then advances the pointer arrays via += stride_k * BLOCK_K.
        # `zero_k` is a tensor-typed 0 (Tile() requires tensor pid args).
        zero_k = out_tile.pid_m * 0
        a_tile0 = Tile(out_tile.pid_m, zero_k, self.block_m, self.block_k)
        b_tile0 = Tile(zero_k, out_tile.pid_n, self.block_k, self.block_n)
        a_ptrs, _ = A.tile_ptrs(a_tile0)
        b_ptrs, _ = B.tile_ptrs(b_tile0)
        a_step = self.block_k * A.stride_col
        b_step = self.block_k * B.stride_row

        # Main K loop -- pointer-increment form
        for _k_idx in range(num_k_tiles):
            a = tl.load(a_ptrs, cache_modifier=self.cache_modifier_a)
            b = tl.load(b_ptrs, cache_modifier=self.cache_modifier_b)
            if self.quantized:
                acc += tl.dot(a, b, out_dtype=tl.int32)
            else:
                acc += tl.dot(a, b, allow_tf32=self.allow_tf32)
            a_ptrs += a_step
            b_ptrs += b_step

        # Handle K tail if needed -- defer to reduce_tile which builds the
        # masked pointer set freshly (boundary case is rare and not worth a
        # second hoisted path).
        if not self.even_k:
            k_idx = num_k_tiles
            acc = self.reduce_tile(A, B, out_tile, k_idx, acc, boundary=True)

        return acc
