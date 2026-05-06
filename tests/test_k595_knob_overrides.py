"""K-595 (S-002) — unit tests for the T7 hardware-knob override mechanism.

These tests validate the mechanism additions in origami.py and matmul.py.
They run on CPU (no GPU required) by exercising the resolution path
directly rather than launching kernels.
"""
import importlib
import os
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# origami.py side: the knob registry + lookup helper.
# ---------------------------------------------------------------------------

def test_knob_overrides_registry_is_dict():
    """Registry exists and is a dict so external callers can read it."""
    from tritonblas.origami import _HIPBLASLT_KNOB_OVERRIDES
    assert isinstance(_HIPBLASLT_KNOB_OVERRIDES, dict)


def test_knob_lookup_returns_none_for_unknown_shape():
    from tritonblas.origami import _hipblaslt_knob_override
    assert _hipblaslt_knob_override(13, 17, 19, "fp16") is None


def test_knob_lookup_returns_none_for_unsupported_dtype():
    """fp8 / fp4 paths use a different selector and must not match."""
    from tritonblas.origami import _hipblaslt_knob_override, _HIPBLASLT_KNOB_OVERRIDES
    # Inject a synthetic entry to prove the dtype filter excludes it.
    _HIPBLASLT_KNOB_OVERRIDES[(1, 2, 3, "fp16")] = {"num_warps": 4}
    try:
        assert _hipblaslt_knob_override(1, 2, 3, "fp8e4nv") is None
    finally:
        _HIPBLASLT_KNOB_OVERRIDES.pop((1, 2, 3, "fp16"), None)


def test_knob_lookup_returns_dict_for_known_shape():
    from tritonblas.origami import _hipblaslt_knob_override, _HIPBLASLT_KNOB_OVERRIDES
    _HIPBLASLT_KNOB_OVERRIDES[(123, 456, 789, "fp16")] = {"num_warps": 4, "waves_per_eu": 2}
    try:
        out = _hipblaslt_knob_override(123, 456, 789, "fp16")
        assert out == {"num_warps": 4, "waves_per_eu": 2}
    finally:
        _HIPBLASLT_KNOB_OVERRIDES.pop((123, 456, 789, "fp16"), None)


def test_knob_lookup_disabled_via_env(monkeypatch):
    from tritonblas.origami import _hipblaslt_knob_override, _HIPBLASLT_KNOB_OVERRIDES
    _HIPBLASLT_KNOB_OVERRIDES[(7, 8, 9, "bf16")] = {"num_warps": 4}
    try:
        monkeypatch.setenv("TRITONBLAS_DISABLE_KNOB_OVERRIDES", "1")
        assert _hipblaslt_knob_override(7, 8, 9, "bf16") is None
        monkeypatch.setenv("TRITONBLAS_DISABLE_KNOB_OVERRIDES", "0")
        assert _hipblaslt_knob_override(7, 8, 9, "bf16") == {"num_warps": 4}
    finally:
        _HIPBLASLT_KNOB_OVERRIDES.pop((7, 8, 9, "bf16"), None)


def test_dtype_normalization_handles_f16_and_fp16():
    """Selector emits 'f16'; the registry uses 'fp16'. Both must resolve."""
    from tritonblas.origami import _hipblaslt_knob_override, _HIPBLASLT_KNOB_OVERRIDES
    _HIPBLASLT_KNOB_OVERRIDES[(11, 22, 33, "fp16")] = {"num_warps": 8}
    try:
        assert _hipblaslt_knob_override(11, 22, 33, "f16") == {"num_warps": 8}
        assert _hipblaslt_knob_override(11, 22, 33, "fp16") == {"num_warps": 8}
    finally:
        _HIPBLASLT_KNOB_OVERRIDES.pop((11, 22, 33, "fp16"), None)


# ---------------------------------------------------------------------------
# matmul.py side: the env-var-aware resolver.
# ---------------------------------------------------------------------------

class _DummySelector:
    """Stand-in for OrigamiMatmulSelector that exposes only the K-595 hints."""
    def __init__(self, **hints):
        self._hints = hints
    @property
    def num_warps_hint(self):       return self._hints.get("num_warps")
    @property
    def waves_per_eu_hint(self):    return self._hints.get("waves_per_eu")
    @property
    def mfma_instr_size_hint(self): return self._hints.get("mfma_instr_size")
    @property
    def kpack_hint(self):           return self._hints.get("kpack")


def test_resolve_knob_falls_back_to_default_when_no_hint_no_env(monkeypatch):
    from tritonblas.matmul import _resolve_knob
    monkeypatch.delenv("TRITONBLAS_FORCE_NUM_WARPS", raising=False)
    sel = _DummySelector()
    assert _resolve_knob("TRITONBLAS_FORCE_NUM_WARPS", sel, "num_warps_hint", 8) == 8


def test_resolve_knob_uses_selector_hint_when_env_unset(monkeypatch):
    from tritonblas.matmul import _resolve_knob
    monkeypatch.delenv("TRITONBLAS_FORCE_NUM_WARPS", raising=False)
    sel = _DummySelector(num_warps=4)
    assert _resolve_knob("TRITONBLAS_FORCE_NUM_WARPS", sel, "num_warps_hint", 8) == 4


def test_resolve_knob_env_overrides_selector_hint(monkeypatch):
    from tritonblas.matmul import _resolve_knob
    monkeypatch.setenv("TRITONBLAS_FORCE_NUM_WARPS", "16")
    sel = _DummySelector(num_warps=4)
    assert _resolve_knob("TRITONBLAS_FORCE_NUM_WARPS", sel, "num_warps_hint", 8) == 16


def test_resolve_knob_invalid_env_falls_through_gracefully(monkeypatch):
    from tritonblas.matmul import _resolve_knob
    monkeypatch.setenv("TRITONBLAS_FORCE_NUM_WARPS", "not-a-number")
    sel = _DummySelector(num_warps=4)
    # invalid env value silently falls back to the next layer (selector hint)
    assert _resolve_knob("TRITONBLAS_FORCE_NUM_WARPS", sel, "num_warps_hint", 8) == 4


def test_resolve_knob_empty_env_falls_through(monkeypatch):
    from tritonblas.matmul import _resolve_knob
    monkeypatch.setenv("TRITONBLAS_FORCE_NUM_WARPS", "")
    sel = _DummySelector(num_warps=4)
    assert _resolve_knob("TRITONBLAS_FORCE_NUM_WARPS", sel, "num_warps_hint", 8) == 4


def test_resolve_knob_default_used_when_selector_lacks_attr(monkeypatch):
    """An old-style selector without the hint property must not crash."""
    from tritonblas.matmul import _resolve_knob
    monkeypatch.delenv("TRITONBLAS_FORCE_KPACK", raising=False)
    class _OldSelector: pass
    assert _resolve_knob("TRITONBLAS_FORCE_KPACK", _OldSelector(), "kpack_hint", 1) == 1


# ---------------------------------------------------------------------------
# K-545 / K-587 backward-compat: the existing _HIPBLASLT_SHAPE_OVERRIDES
# must continue to work unchanged.
# ---------------------------------------------------------------------------

def test_k587_shape_overrides_table_intact():
    from tritonblas.origami import _HIPBLASLT_SHAPE_OVERRIDES
    # Sanity: the K-587 entries from the prior PR must still be there.
    assert (1024, 8192, 8192, "fp16") in _HIPBLASLT_SHAPE_OVERRIDES
    assert (8192, 1024, 8192, "bf16") in _HIPBLASLT_SHAPE_OVERRIDES
