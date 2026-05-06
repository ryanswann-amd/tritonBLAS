"""K-595 (S-002) — unit tests for the surviving T7 env-var diagnostic shim.

The K-595 sweep falsified the per-shape hardware-knob registry that was
prototyped (see the `_hipblaslt_shape_override` header comment in
origami.py for the empirical record). What survived is a single env-var
resolver in matmul.py, `_resolve_knob`, used by both the persistent and
Stream-K matmul paths to read the four diagnostic env vars
(TRITONBLAS_FORCE_NUM_WARPS / WAVES_PER_EU / MFMA_INSTR_SIZE / KPACK).

These tests cover ONLY the surviving code. They run on CPU (no GPU
required) by exercising the resolver directly. The critical safety
property is that malformed env-var values MUST NOT crash the kernel
launch path — they must silently fall through to the call-site default.
"""

import pytest


def test_resolve_knob_returns_default_when_env_unset(monkeypatch):
    from tritonblas.matmul import _resolve_knob
    monkeypatch.delenv("TRITONBLAS_FORCE_NUM_WARPS", raising=False)
    assert _resolve_knob("TRITONBLAS_FORCE_NUM_WARPS", 8) == 8


def test_resolve_knob_env_overrides_default(monkeypatch):
    from tritonblas.matmul import _resolve_knob
    monkeypatch.setenv("TRITONBLAS_FORCE_NUM_WARPS", "4")
    assert _resolve_knob("TRITONBLAS_FORCE_NUM_WARPS", 8) == 4


def test_resolve_knob_empty_env_returns_default(monkeypatch):
    from tritonblas.matmul import _resolve_knob
    monkeypatch.setenv("TRITONBLAS_FORCE_KPACK", "")
    assert _resolve_knob("TRITONBLAS_FORCE_KPACK", 2) == 2


@pytest.mark.parametrize("bad_value", [
    "auto",          # common typo: word instead of int
    "default",       # common typo: requested behavior name
    "not-a-number",  # arbitrary non-numeric
    "8.5",           # float (int(...) raises)
    "0x10",          # hex literal (int(..., base=10) raises)
    "8 ",            # trailing whitespace — int() actually accepts this,
    " 8",            # leading whitespace — int() also accepts this,
])
def test_resolve_knob_malformed_env_does_not_crash(monkeypatch, bad_value):
    """The kernel-launch path must NEVER raise on malformed env input."""
    from tritonblas.matmul import _resolve_knob
    monkeypatch.setenv("TRITONBLAS_FORCE_MFMA_INSTR_SIZE", bad_value)
    # Must return *something* (the default OR the parsed int from
    # whitespace-tolerant strings) — what matters is no exception escapes.
    out = _resolve_knob("TRITONBLAS_FORCE_MFMA_INSTR_SIZE", 16)
    assert isinstance(out, int)


def test_resolve_knob_strict_malformed_falls_through_to_default(monkeypatch):
    """For a clearly non-int value, the resolver returns the default."""
    from tritonblas.matmul import _resolve_knob
    monkeypatch.setenv("TRITONBLAS_FORCE_WAVES_PER_EU", "default")
    assert _resolve_knob("TRITONBLAS_FORCE_WAVES_PER_EU", 0) == 0


def test_resolve_knob_does_not_consult_selector_or_registry(monkeypatch):
    """Skeptic guard: confirm there is NO selector-side hint layer.

    The previous (deleted) implementation consulted a per-shape registry
    via `selector.<knob>_hint`. That mechanism was falsified and removed.
    `_resolve_knob` must take exactly two arguments now (env_name,
    default) — anything more would mean the dispatch-layer plumbing
    crept back in.
    """
    import inspect
    from tritonblas.matmul import _resolve_knob
    sig = inspect.signature(_resolve_knob)
    assert list(sig.parameters) == ["env_name", "default"], (
        f"_resolve_knob signature regressed: {sig}. The K-595 falsification "
        f"requires the resolver to remain env-var-only — no selector or "
        f"hint_attr arguments."
    )


def test_no_knob_registry_in_origami(monkeypatch):
    """Skeptic guard: the empty registry must stay deleted, not re-emerge.

    A stale `_HIPBLASLT_KNOB_OVERRIDES` symbol or `_hipblaslt_knob_override`
    helper would be dead infrastructure shipped on falsified science.
    """
    import tritonblas.origami as origami_mod
    assert not hasattr(origami_mod, "_HIPBLASLT_KNOB_OVERRIDES"), (
        "_HIPBLASLT_KNOB_OVERRIDES should not exist — see K-595 falsification "
        "in the _hipblaslt_shape_override header comment."
    )
    assert not hasattr(origami_mod, "_hipblaslt_knob_override"), (
        "_hipblaslt_knob_override should not exist — the lever was falsified."
    )


def test_no_knob_hint_properties_on_selector():
    """Skeptic guard: the four `*_hint` selector properties must stay deleted."""
    from tritonblas.origami import OrigamiMatmulSelector
    for attr in ("num_warps_hint", "waves_per_eu_hint",
                 "mfma_instr_size_hint", "kpack_hint"):
        assert not hasattr(OrigamiMatmulSelector, attr), (
            f"OrigamiMatmulSelector.{attr} should not exist — K-595 deleted "
            f"the dispatch-layer plumbing after empirical falsification."
        )


def test_k587_shape_overrides_table_intact():
    """Backward-compat: K-545/K-587 shape overrides must remain unchanged."""
    from tritonblas.origami import _HIPBLASLT_SHAPE_OVERRIDES
    assert (1024, 8192, 8192, "fp16") in _HIPBLASLT_SHAPE_OVERRIDES
    assert (8192, 1024, 8192, "bf16") in _HIPBLASLT_SHAPE_OVERRIDES
