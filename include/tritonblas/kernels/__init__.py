"""
Kernel implementations for tritonblas.

This package contains specific GEMM kernel implementations:
- persistent_gemm: Persistent (data-parallel) GEMM kernel using composable stages
- persistent_gemm_monolithic: Monolithic persistent GEMM kernel (fallback)
- persistent_gemm_work_stealing: Work-stealing persistent GEMM kernel
- streamk_gemm: Stream-K GEMM kernel for load balancing
- stages: Composable kernel building blocks

Environment Variables:
- TBLAS_USE_MONOLITHIC: Set to '1' or 'true' to force the monolithic persistent
  kernel.  When unset the composable-stages kernel is tried first; if the
  installed Triton does not support the required aggregate types the monolithic
  kernel is selected automatically.
"""

import os
import logging

_log = logging.getLogger(__name__)

# Check environment variable to determine which persistent kernel to use
_use_monolithic = os.environ.get('TBLAS_USE_MONOLITHIC', '').lower() in ('1', 'true', 'yes')

def _triton_supports_aggregates() -> bool:
    """Return True if the installed Triton supports user-defined aggregates.

    The composable-stages kernel relies on factory functions that return
    Triton ``noinline`` structs (aggregates).  Older Triton builds raise
    at IR-generation time when they encounter these constructs.  We probe
    for the required capability by checking whether the ``noinline``
    decorator exists in ``triton.language`` — its presence is a reliable
    proxy for aggregate support.
    """
    try:
        import triton.language as tl
        return hasattr(tl, "noinline")
    except Exception:
        return False


if _use_monolithic:
    # Use monolithic version (explicitly requested)
    from .persistent_gemm_monolithic import persistent_matmul
elif not _triton_supports_aggregates():
    _log.info(
        "Composable-stages kernel unavailable (Triton lacks aggregate support); "
        "using monolithic persistent kernel."
    )
    from .persistent_gemm_monolithic import persistent_matmul
else:
    from .persistent_gemm import persistent_matmul

# Work-stealing kernel (opt-in via work_stealing=True in matmul calls)
from .persistent_gemm_work_stealing import ws_persistent_matmul

# Stream-K kernel is always the same
from .streamk_gemm import streamk_matmul

# Work-stealing kernel (opt-in via work_stealing=True in matmul calls)
from .streamk_gemm_work_stealing import ws_streamk_matmul

# FP4 kernel
from .fp4_matmul import fp4_matmul

# Export stages submodule
from . import stages

__all__ = ['persistent_matmul', 'ws_persistent_matmul',
           'streamk_matmul', 'ws_streamk_matmul',
           'fp4_matmul', 'stages']
