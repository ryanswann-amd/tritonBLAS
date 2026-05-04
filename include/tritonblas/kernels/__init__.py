"""
Kernel implementations for tritonblas.

This package contains specific GEMM kernel implementations:
- persistent_gemm: Persistent (data-parallel) GEMM kernel using composable stages
- persistent_gemm_monolithic: Monolithic persistent GEMM kernel (legacy, for debugging)
- streamk_gemm: Stream-K GEMM kernel for load balancing
- stages: Composable kernel building blocks

Environment Variables:
- TBLAS_USE_MONOLITHIC: Set to '1' or 'true' to use the monolithic persistent kernel instead of the composable stages version
"""

import os

# Check environment variable to determine which persistent kernel to use.
# Default to monolithic (compatible with all Triton versions).
# Set TBLAS_USE_STAGES=1 to use the composable stages version (requires
# Triton with aggregate @triton.jit support in __init__ methods).
_use_stages = os.environ.get('TBLAS_USE_STAGES', '').lower() in ('1', 'true', 'yes')

if _use_stages:
    from .persistent_gemm import persistent_matmul
else:
    from .persistent_gemm_monolithic import persistent_matmul

# Stream-K kernel is always the same
from .streamk_gemm import streamk_matmul

# FP4 kernel
from .fp4_matmul import fp4_matmul

# Export stages submodule
from . import stages

__all__ = ['persistent_matmul', 'streamk_matmul', 'fp4_matmul', 'stages']
