"""
Gluon kernel backend for gfx950 (MI350/MI355).

Provides hand-optimized GEMM kernels using the Gluon Triton dialect with
explicit layout control, MFMA scheduling, and XCD-aware PID remapping.

Requires: triton >= 3.7.0 with triton.experimental.gluon

Environment variables set automatically when this backend is active:
- TRITON_ENABLE_LLIR_SCHED=1  (LLIR instruction scheduler)
- TRITON_ENABLE_AMDGCN_AS=1   (AMDGCN assembly post-pass)
Both are required for peak performance and must be used together.
"""

import os

_GLUON_AVAILABLE = False
_PATCHED = False

try:
    from triton.experimental import gluon  # noqa: F401
    from triton.experimental.gluon import language as gl  # noqa: F401
    _GLUON_AVAILABLE = True
except ImportError:
    pass


def _patch_async_copy():
    """Widen buffer_load_to_shared layout check to accept DistributedLayout.

    Triton <= 3.7.1 restricts offsets to BlockedLayout|SliceLayout. Gluon v3+ kernels
    use DistributedLinearLayout. The gfx950-tutorial branch already has this fix.
    We apply it at runtime by patching the source file.
    """
    try:
        import triton.experimental.gluon.language.amd.cdna4.async_copy as _ac
        path = _ac.__file__
        with open(path) as f:
            src = f.read()
        if "BlockedLayout, SliceLayout" not in src:
            return
        src = src.replace(
            "from ..._layouts import BlockedLayout, SliceLayout",
            "from ..._layouts import DistributedLayout",
        ).replace(
            "(BlockedLayout, SliceLayout)",
            "DistributedLayout",
        )
        with open(path, "w") as f:
            f.write(src)
        import importlib
        importlib.reload(_ac)
    except Exception:
        pass


def is_available():
    return _GLUON_AVAILABLE


def ensure_scheduler_env():
    os.environ.setdefault("TRITON_ENABLE_LLIR_SCHED", "1")
    os.environ.setdefault("TRITON_ENABLE_AMDGCN_AS", "1")
    _patch_async_copy()
