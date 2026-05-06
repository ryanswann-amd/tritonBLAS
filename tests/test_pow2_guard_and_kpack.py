"""Pow2 tile-dim guard for `_HIPBLASLT_SHAPE_OVERRIDES` and centralized
kpack policy.

These tests lock in two invariants that prior empirical work surfaced:

1.  Every entry in the hipBLASLt shape-override registry must use
    power-of-2 tile dimensions, because Triton's
    `tl.zeros((BLOCK_M, BLOCK_N))` accumulator allocation in
    `include/tritonblas/kernels/stages/gemm_context.py::init_accumulator`
    rejects non-pow2 shapes with `ValueError: Shape element N must be a
    power of 2` at compile time. A historical extension attempt seeded
    `{160, 192, 224}` tiles from hipBLASLt's per-shape top-1 picks; all
    those candidate tiles fail the Triton constraint. The registry-side
    guard ensures (a) the predicate is correct, (b) every override
    entry passes it, and (c) the lookup-site guard fails *closed*
    (returns None → native selector handles the shape) when fed a
    synthetic non-pow2 entry, instead of crashing the whole package at
    import time.

2.  `kpack_for_dtype` is the single source of truth for the kpack
    setting passed to both `persistent_matmul_lt` and
    `streamk_matmul_lt`; the two launch paths must not drift. The
    helper returns 2 for fp16 / bf16 (denser dword-packed mfma operand
    layout, measured +3-6pp same-tile lift on residual cohort) and 1
    for everything else (fp8 / fp4 / fp32 paths kept on the existing
    setting).

These tests do NOT require a GPU — they exercise pure Python selector
logic.
"""

import pytest
import torch

from tritonblas.constraints import (
    is_pow2 as _is_pow2,
    is_triton_valid_block_tile as _is_triton_valid_block_tile,
    kpack_for_dtype,
)
from tritonblas.origami import (
    _HIPBLASLT_SHAPE_OVERRIDES,
    _hipblaslt_shape_override,
)


def test_constraints_module_is_single_owner():
    """The shared predicates MUST live in `tritonblas.constraints` so
    matmul.py / origami.py / the test suite cannot drift. Re-exports
    from `tritonblas.origami` are kept for back-compat but are not the
    canonical home.
    """
    import tritonblas.constraints as ctr
    import tritonblas.origami as og

    assert ctr.is_pow2 is _is_pow2
    assert ctr.is_triton_valid_block_tile is _is_triton_valid_block_tile
    assert ctr.kpack_for_dtype is kpack_for_dtype
    # origami.py re-exports under the underscore-prefixed back-compat names
    assert og._is_pow2 is ctr.is_pow2
    assert og._is_triton_valid_block_tile is ctr.is_triton_valid_block_tile
    assert og.kpack_for_dtype is ctr.kpack_for_dtype


# ---------------------------------------------------------------------------
# Pow2 predicate correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n,expected",
    [
        (1, True), (2, True), (4, True), (8, True), (16, True),
        (32, True), (64, True), (128, True), (256, True), (512, True),
        # Non-pow2 tile widths hipBLASLt uses for the residual cohort:
        (160, False), (192, False), (224, False),
        # Edge cases:
        (0, False), (-1, False), (3, False), (255, False), (257, False),
    ],
)
def test_is_pow2(n, expected):
    assert _is_pow2(n) is expected


@pytest.mark.parametrize(
    "bm,bn,bk,expected",
    [
        # In-table entries (must be valid):
        (128, 256, 64, True),
        (256, 128, 64, True),
        (256, 256, 64, True),
        # hipBLASLt top-1 candidates that were ruled out (must be invalid):
        (128, 224, 64, False),  # 1024x8192x8192
        (192, 160, 64, False),  # 4096x2048x4096
        (256, 224, 64, False),  # 8192x8192x4096
        # Non-pow2 BK also blocked:
        (128, 128, 96, False),
    ],
)
def test_is_triton_valid_block_tile(bm, bn, bk, expected):
    assert _is_triton_valid_block_tile(bm, bn, bk) is expected


# ---------------------------------------------------------------------------
# Registry invariant: every shipped override entry is Triton-legal
# ---------------------------------------------------------------------------

def test_all_overrides_are_pow2():
    """Every entry in `_HIPBLASLT_SHAPE_OVERRIDES` MUST satisfy
    `_is_triton_valid_block_tile`. Catches the historical mistake of
    seeding (128, 224, 64) etc. before re-attempt cycles burn cluster
    cycles on what is fundamentally a Triton-AMD compiler constraint.
    """
    bad = []
    for key, entry in _HIPBLASLT_SHAPE_OVERRIDES.items():
        bm, bn, bk = entry["tile"]
        if not _is_triton_valid_block_tile(bm, bn, bk):
            bad.append((key, (bm, bn, bk)))
    assert not bad, (
        f"Non-pow2 override entries: {bad}. Triton's `init_accumulator` "
        f"rejects non-pow2 BLOCK_M/BLOCK_N/BLOCK_K."
    )


def test_all_overrides_have_bucket_label():
    """Every shipped override entry MUST carry a `bucket` provenance tag
    so future audits can categorize the registry without re-deriving
    intent. Today's recognized buckets:

      * "TILE-ORIENTATION" — flip the asymmetric tile so the larger dim
        aligns with the long axis of the workload.
      * "RANKING-OVERRIDE" — pin a tile to suppress the post-hoc
        symmetric 256x256 fallback for shapes where the post-hoc
        dispatch is wrong.
    """
    allowed = {"TILE-ORIENTATION", "RANKING-OVERRIDE"}
    missing = []
    bad_label = []
    for key, entry in _HIPBLASLT_SHAPE_OVERRIDES.items():
        if "bucket" not in entry:
            missing.append(key)
        elif entry["bucket"] not in allowed:
            bad_label.append((key, entry["bucket"]))
    assert not missing, f"Override entries missing bucket tag: {missing}"
    assert not bad_label, (
        f"Override entries with unknown bucket label: {bad_label}. "
        f"Allowed: {sorted(allowed)}."
    )


# ---------------------------------------------------------------------------
# Lookup-site behavior: fail-closed, do NOT crash the package
# ---------------------------------------------------------------------------

def test_lookup_skips_non_pow2_override(monkeypatch):
    """If a future contributor injects a non-pow2 entry into the
    override table, the lookup MUST return None (so the native selector
    picks a Triton-legal tile) instead of returning the bad entry to
    the dispatcher (which would surface deep inside Triton's
    `init_accumulator` with no pointer back to the override table).
    """
    # Inject a synthetic bad entry without mutating the shipped table.
    bad_key = (1024, 8192, 8192, "fp16")
    bad_tile = (128, 224, 64)
    patched = dict(_HIPBLASLT_SHAPE_OVERRIDES)
    patched[bad_key] = {"tile": bad_tile, "bucket": "TILE-ORIENTATION"}
    monkeypatch.setattr(
        "tritonblas.origami._HIPBLASLT_SHAPE_OVERRIDES",
        patched,
    )
    out = _hipblaslt_shape_override(
        m=1024, n=8192, k=8192,
        a_dtype_str="f16", b_dtype_str="f16",
        bytes_a=2.0, bytes_b=2.0,
        lds_cap=64 * 1024,
        num_stages=2,
    )
    assert out is None, (
        f"Fail-closed pow2 guard at lookup site failed — "
        f"returned {out} for synthetic non-pow2 override {bad_tile}. "
        f"Expected None so the native selector handles the shape."
    )


def test_lookup_returns_pow2_tile_for_shipped_entry():
    """The shipped 1024x8192x8192 entry must round-trip through the
    lookup as the canonical (128, 256, 64) pow2 tile. Locks in the
    schema (`{"tile": (...), "bucket": ...}`) so a future contributor
    refactoring the registry cannot silently break the lookup contract.
    """
    out = _hipblaslt_shape_override(
        m=1024, n=8192, k=8192,
        a_dtype_str="f16", b_dtype_str="f16",
        bytes_a=2.0, bytes_b=2.0,
        lds_cap=64 * 1024,
        num_stages=2,
    )
    assert out == (128, 256, 64)


def test_import_does_not_raise_on_existing_table():
    """Importing tritonblas.origami must succeed under the shipped
    override table. The module explicitly chooses lookup-site
    fail-closed over import-time fail-fast so a future bad entry cannot
    brick the whole package for every downstream caller.
    """
    import importlib

    import tritonblas.origami as mod  # noqa: F401

    importlib.reload(mod)  # exercise the import path explicitly


# ---------------------------------------------------------------------------
# Native-selector pow2 enforcement: final-pick assertion in selector __init__
# ---------------------------------------------------------------------------

def test_selector_init_asserts_final_pick_is_pow2():
    """`OrigamiMatmulSelector.__init__` MUST sanity-check the final
    `(BLOCK_M, BLOCK_N, BLOCK_K)` it returns to the dispatcher against
    `is_triton_valid_block_tile`. The assertion exists so that if a
    future code path (override extension, new heuristic, autotune path)
    mutates the final pick to a non-pow2 tile, the failure surfaces at
    dispatch time with a clear message rather than as a `ValueError:
    Shape element N must be a power of 2` raised from inside `tl.zeros`
    deep in the Triton compiler. We verify the assertion is structurally
    present rather than running a GPU kernel (which would require a
    live device and can't easily synthesize the failure mode).
    """
    import inspect
    from tritonblas.origami import OrigamiMatmulSelector

    src = inspect.getsource(OrigamiMatmulSelector.__init__)
    # Both the predicate call and the explanatory raise must be present.
    assert "_is_triton_valid_block_tile(" in src, (
        "Selector __init__ must call `_is_triton_valid_block_tile(...)` "
        "on the final pick before returning."
    )
    assert "raise ValueError" in src, (
        "Selector __init__ must `raise ValueError` if the final pick "
        "fails the pow2 predicate."
    )


# ---------------------------------------------------------------------------
# Centralized kpack policy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dtype,bm,bn,expected",
    [
        # Asymmetric tile + fp16/bf16 → kpack=2 (the only case that
        # produced a clean win in the paired bench).
        (torch.float16,  256, 128, 2),
        (torch.float16,  128, 256, 2),
        (torch.bfloat16, 256, 128, 2),
        (torch.bfloat16, 128, 256, 2),
        # Symmetric tile + fp16/bf16 → kpack=1 (the symmetric tile
        # regressed -10pp on 8192x8192x8192 in the paired bench; gated
        # back to the existing default).
        (torch.float16,  256, 256, 1),
        (torch.float16,  128, 128, 1),
        (torch.bfloat16, 256, 256, 1),
        (torch.bfloat16, 128, 128, 1),
        # Non-fp16/bf16 dtypes are always kpack=1 regardless of tile.
        (torch.float32,  256, 128, 1),
        (torch.int8,     256, 256, 1),
    ],
)
def test_kpack_for_dtype_asymmetric_gate(dtype, bm, bn, expected):
    assert kpack_for_dtype(dtype, bm, bn) == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kpack_for_dtype_falls_back_to_one_without_tile(dtype):
    """When `block_m`/`block_n` are not supplied (legacy or external
    caller), the helper MUST fall back to the safe `kpack=1` setting
    rather than guessing `kpack=2` based on dtype alone (which would
    re-introduce the symmetric-tile regression for any caller that
    happens to land on a 256x256x64 tile)."""
    assert kpack_for_dtype(dtype) == 1
    assert kpack_for_dtype(dtype, None, 256) == 1
    assert kpack_for_dtype(dtype, 256, None) == 1


def test_kpack_force_kpack1_env(monkeypatch):
    """`TRITONBLAS_FORCE_KPACK1=1` clamps every dispatch to `kpack=1`,
    used by the paired bench harness to isolate the kpack contribution
    without checking out an older revision."""
    monkeypatch.setenv("TRITONBLAS_FORCE_KPACK1", "1")
    # Even the case that would normally pick kpack=2 must clamp to 1.
    assert kpack_for_dtype(torch.float16, 256, 128) == 1
    assert kpack_for_dtype(torch.bfloat16, 128, 256) == 1


def test_kpack_helper_used_at_both_launch_sites():
    """`kpack_for_dtype` is the single source of truth for the kpack
    setting; both `persistent_matmul_lt` and `streamk_matmul_lt` must
    call it instead of inlining their own dtype check (or the two paths
    will silently drift on a future dtype addition). We assert this
    structurally by reading the source rather than running a kernel.
    """
    import importlib
    import inspect

    # `tritonblas.matmul` is shadowed at the package top level by the
    # public matmul() function (re-exported in __init__.py). Reach the
    # *module* explicitly via importlib so we can read the launcher
    # function sources.
    _matmul_mod = importlib.import_module("tritonblas.matmul")

    src_persistent = inspect.getsource(_matmul_mod.persistent_matmul_lt)
    src_streamk = inspect.getsource(_matmul_mod.streamk_matmul_lt)

    for label, src in (("persistent_matmul_lt", src_persistent),
                       ("streamk_matmul_lt", src_streamk)):
        assert "kpack_for_dtype(" in src, (
            f"{label} must obtain kpack via `kpack_for_dtype(...)` so the "
            f"two launch paths cannot drift; inlining the dtype check is "
            f"explicitly disallowed."
        )
