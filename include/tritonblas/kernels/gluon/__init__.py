"""
Gluon kernel backend for gfx950 (MI350/MI355).

Provides hand-optimized GEMM kernels using the Gluon Triton dialect with
explicit layout control, MFMA scheduling, and XCD-aware PID remapping.

Requires:
  - triton built from the gfx950-tutorial branch (triton-lang/triton)
  - Environment: TRITONBLAS_ENABLE_GLUON=1 to opt in

The scheduler env vars TRITON_ENABLE_LLIR_SCHED and TRITON_ENABLE_AMDGCN_AS
are set automatically when the backend activates.
"""

_AVAILABLE = False

try:
    from triton.experimental import gluon  # noqa: F401
    _AVAILABLE = True
except ImportError:
    pass


def is_available():
    return _AVAILABLE
