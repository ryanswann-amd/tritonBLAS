# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Tile aggregate for tritonblas shards.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate


@aggregate
class Tile:
    """
    2D tile with coordinates and shape.
    
    Stores runtime coordinates (pid_m, pid_n) and compile-time block sizes.
    
    Use :class:`ScheduleContext` methods to create tiles::
    
        # From linear tile_id
        out_tile = sched.get_tile_from_idx(tile_id)
        
        # From 2D coordinates
        out_tile = sched.get_tile_from_coord(pid_m, pid_n)
        
        # Direct construction is also supported
        out_tile = Tile(pid_m, pid_n, BLOCK_M, BLOCK_N)
    
    Example
    -------
    .. code-block:: python
    
        # Output tile from scheduler
        out_tile = sched.get_tile_from_idx(tile_id)
        
        # Input A tile at k offset
        a_tile = sched.get_tile_from_coord(pid_m, k // BLOCK_K)
        
        # Input B tile at k offset  
        b_tile = sched.get_tile_from_coord(k // BLOCK_K, pid_n)
    """
    
    pid_m: tl.tensor  # Tile coordinate in M dimension
    pid_n: tl.tensor  # Tile coordinate in N dimension
    block_m: tl.constexpr  # Block size M
    block_n: tl.constexpr  # Block size N
    even_m: tl.constexpr   # True iff row dim is divisible by block_m
    even_n: tl.constexpr   # True iff col dim is divisible by block_n

    @triton.constexpr_function
    def __init__(self, pid_m, pid_n, block_m, block_n, even_m=False, even_n=False):
        """
        Create a tile with runtime coordinates and compile-time sizes.

        Args:
            pid_m: Tile coordinate in M dimension
            pid_n: Tile coordinate in N dimension
            block_m: Block size in M dimension (constexpr)
            block_n: Block size in N dimension (constexpr)
            even_m: True iff the row dim later passed to ``layout()`` is
                divisible by ``block_m``; enables the fast path that skips
                the bounds mask and Barrett modulo (constexpr, default False)
            even_n: True iff the col dim later passed to ``layout()`` is
                divisible by ``block_n``; same fast-path gate (default False)
        """
        self.pid_m = pid_m
        self.pid_n = pid_n
        self.block_m = tl.constexpr(block_m)
        self.block_n = tl.constexpr(block_n)
        self.even_m = tl.constexpr(even_m)
        self.even_n = tl.constexpr(even_n)

    @triton.jit
    def indices(self):
        """
        Compute row and column indices for this tile.
        
        Returns:
            rm, rn: Row indices [BLOCK_M], column indices [BLOCK_N]
        """
        rm = self.pid_m * self.block_m + tl.arange(0, self.block_m)
        rn = self.pid_n * self.block_n + tl.arange(0, self.block_n)
        return rm, rn
    
    @triton.jit
    def layout(self, M, N):
        """
        Compute memory layout with bounds checking.
        
        Args:
            M: Total rows
            N: Total columns
        
        Returns:
            rm, rn, mask: Row indices, column indices, bounds mask
        
        Note:
            The mask is computed from RAW indices before modulo wrapping.
            This is critical for correct boundary handling - e.g., when K
            doesn't divide evenly by BLOCK_K, the boundary tile needs proper
            masking to avoid reading garbage data.
        """
        rm, rn = self.indices()
        # ═══════════════════════════════════════════════════════════════════
        # FAST PATH: when both dims are exactly divisible by their block
        # sizes, every `rm < M` / `rn < N` is True and `rm % M`, `rn % N`
        # are identity ops. Skip both the bounds mask and the Barrett-style
        # modulo at compile time to remove the per-tile epilogue VALU tax.
        # ═══════════════════════════════════════════════════════════════════
        if self.even_m and self.even_n:
            mask = tl.full((self.block_m, self.block_n), 1, tl.int1)
            rm = tl.max_contiguous(tl.multiple_of(rm, self.block_m), self.block_m)
            rn = tl.max_contiguous(tl.multiple_of(rn, self.block_n), self.block_n)
        else:
            # ═══════════════════════════════════════════════════════════════
            # SLOW PATH: original behaviour for partially-aligned dims.
            # The mask must be computed from raw indices BEFORE modulo so it
            # correctly identifies out-of-bounds elements; after modulo every
            # index would be < M and < N, making the mask useless.
            # ═══════════════════════════════════════════════════════════════
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            # The modulo + max_contiguous optimization helps with memory
            # access patterns, but must come AFTER mask computation.
            rm = tl.max_contiguous(tl.multiple_of(rm % M, self.block_m), self.block_m)
            rn = tl.max_contiguous(tl.multiple_of(rn % N, self.block_n), self.block_n)
        return rm, rn, mask
    
    @triton.jit
    def scale(self, acc, A_scale_ptr, B_scale_ptr, M, N, stride_a=1, stride_b=1):
        """
        Apply quantization scales to accumulator.
        
        Args:
            acc: Accumulator tensor [BLOCK_M, BLOCK_N]
            A_scale_ptr: Pointer to A scales (per-row)
            B_scale_ptr: Pointer to B scales (per-column)
            M, N: Matrix dimensions for bounds checking
            stride_a: Stride for A scales (default: 1)
            stride_b: Stride for B scales (default: 1)
        
        Returns:
            Scaled accumulator as float32
        
        Example::
        
            acc = tile.scale(acc, A_scale_ptr, B_scale_ptr, M, N)
        """
        rm, rn = self.indices()
        a_scales = tl.load(A_scale_ptr + rm * stride_a, mask=rm < M, other=1.0)
        b_scales = tl.load(B_scale_ptr + rn * stride_b, mask=rn < N, other=1.0)
        acc = acc.to(tl.float32)
        acc = acc * a_scales[:, None]
        acc = acc * b_scales[None, :]
        return acc
    
    @triton.jit
    def bias(self, acc, bias_ptr, M, stride_bias=1):
        """
        Add bias vector to accumulator.
        
        Args:
            acc: Accumulator tensor [BLOCK_M, BLOCK_N]
            bias_ptr: Pointer to bias vector
            M: Matrix dimension for bounds checking
            stride_bias: Stride for bias vector (default: 1)
        
        Returns:
            Accumulator with bias added
        
        Example::
        
            acc = tile.bias(acc, bias_ptr, M)
        """
        rm, _ = self.indices()
        bias_vector = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
        acc = acc + bias_vector[:, None]
        return acc
