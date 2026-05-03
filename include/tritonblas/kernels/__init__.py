"""
Kernel implementations for tritonblas.

This package contains specific GEMM kernel implementations:
- persistent_gemm: Persistent (data-parallel) GEMM kernel using composable stages
- persistent_gemm_monolithic: Monolithic persistent GEMM kernel (fallback)
- streamk_gemm: Stream-K GEMM kernel for load balancing
- stages: Composable kernel building blocks

Environment Variables:
- TBLAS_USE_MONOLITHIC: Set to '1' or 'true' to force the monolithic persistent
  kernel instead of the composable stages version
"""

import os

# Check environment variable to determine which persistent kernel to use
_use_monolithic = os.environ.get('TBLAS_USE_MONOLITHIC', '').lower() in ('1', 'true', 'yes')


def _triton_supports_jit_aggregates() -> bool:
    """Detect whether the installed Triton supports @triton.jit on classes.

    The composable kernel (persistent_gemm.py) uses JIT-compiled aggregate
    types (InputView, GemmContext, etc.) which require Triton to support
    ``@triton.jit`` on class definitions.  Older Triton builds (e.g. the
    ROCm 6.x system Triton ≤ 3.5) do not support this and will crash at
    kernel compile time with ``hipErrorNotFound`` because the compiler
    cannot produce a valid binary.

    This function probes support cheaply at import time so we can fall back
    to the monolithic kernel before any GPU work is attempted.
    """
    import triton
    import triton.language as tl
    try:
        # Minimal probe: decorate a trivial class with @triton.jit.
        # On unsupported Triton versions this raises AttributeError or
        # similar before any GPU code is generated.
        @triton.jit
        class _Probe:
            def __init__(self, x: tl.constexpr):
                self.x = x
        return True
    except Exception:
        return False


if _use_monolithic:
    # Use monolithic version (explicitly requested)
    from .persistent_gemm_monolithic import persistent_matmul
elif _triton_supports_jit_aggregates():
    # Triton supports JIT aggregates — use the composable stages kernel
    from .persistent_gemm import persistent_matmul
else:
    # Fall back to the monolithic kernel which is functionally identical
    # but uses only plain Triton primitives (no JIT class aggregates).
    from .persistent_gemm_monolithic import persistent_matmul

# Stream-K kernel is always the same
from .streamk_gemm import streamk_matmul

# FP4 kernel
from .fp4_matmul import fp4_matmul

# Export stages submodule
from . import stages

__all__ = ['persistent_matmul', 'streamk_matmul', 'fp4_matmul', 'stages']
