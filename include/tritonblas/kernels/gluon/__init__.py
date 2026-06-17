"""
Gluon kernel backend for gfx950 (MI350/MI355).

Provides hand-optimized GEMM kernels using the Gluon Triton dialect with
explicit layout control, MFMA scheduling, and XCD-aware PID remapping.

Requires: triton.experimental.gluon (gfx950-tutorial-v0.2 Triton fork or later)

Environment variables set automatically when this backend is active:
- TRITON_ENABLE_LLIR_SCHED=1  (LLIR instruction scheduler)
- TRITON_ENABLE_AMDGCN_AS=1   (AMDGCN assembly post-pass)
Both are required for peak performance and must be used together.
"""

import os

_GLUON_AVAILABLE = False

try:
    from triton.experimental import gluon  # noqa: F401
    from triton.experimental.gluon import language as gl  # noqa: F401
    _GLUON_AVAILABLE = True
except ImportError:
    pass


def is_available():
    return _GLUON_AVAILABLE


def ensure_scheduler_env():
    os.environ.setdefault("TRITON_ENABLE_LLIR_SCHED", "1")
    os.environ.setdefault("TRITON_ENABLE_AMDGCN_AS", "1")
