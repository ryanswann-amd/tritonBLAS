"""
Kernel implementations for tritonblas.

This package contains specific GEMM kernel implementations:
- persistent_gemm: Persistent (data-parallel) GEMM kernel using composable stages
- persistent_gemm_monolithic: Monolithic persistent GEMM kernel (legacy, for debugging)
- persistent_gemm_work_stealing: Work-stealing persistent GEMM kernel
- streamk_gemm: Stream-K GEMM kernel for load balancing
- stages: Composable kernel building blocks

Environment Variables:
- TBLAS_USE_MONOLITHIC: Set to '1' or 'true' to use the monolithic persistent kernel instead of the composable stages version
"""

import os

# Check environment variable to determine which persistent kernel to use
_use_monolithic = os.environ.get('TBLAS_USE_MONOLITHIC', '').lower() in ('1', 'true', 'yes')

if _use_monolithic:
    # Use monolithic version (legacy implementation)
    from .persistent_gemm_monolithic import persistent_matmul
else:
    # Use composable stages version (default)
    from .persistent_gemm import persistent_matmul

# Work-stealing kernel (opt-in via work_stealing=True in matmul calls)
from .persistent_gemm_work_stealing import ws_persistent_matmul

# Stream-K kernel is always the same
from .streamk_gemm import streamk_matmul

# Work-stealing kernel (opt-in via work_stealing=True in matmul calls)
from .streamk_gemm_work_stealing import ws_streamk_matmul

# FP4 kernel
from .fp4_matmul import fp4_matmul

# Atomic-free split-K dispatcher for the small-M residual cohort
# (M <= 32, K >= 4096, SPLIT_K >= 4).  See persistent_splitk.py for the
# rationale -- gfx942 has no native FP16/BF16 atomic-add, so the classic
# split-K path serializes via a CAS retry loop on the C tile; this dispatcher
# replaces it with a workspace + epilogue reduction.
from .persistent_splitk import (
    persistent_splitk_matmul,
    splitk_partials_kernel,
    splitk_reduce_kernel,
    choose_split_k,
    should_use_atomic_free_splitk,
)

# Export stages submodule
from . import stages

__all__ = ['persistent_matmul', 'ws_persistent_matmul',
           'streamk_matmul', 'ws_streamk_matmul',
           'fp4_matmul', 'stages',
           'persistent_splitk_matmul', 'splitk_partials_kernel',
           'splitk_reduce_kernel', 'choose_split_k',
           'should_use_atomic_free_splitk']
