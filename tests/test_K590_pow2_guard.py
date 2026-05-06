"""K-590 / S-002 — Pow2 tile-dim guard for `_HIPBLASLT_SHAPE_OVERRIDES`.

These tests lock in the K-590 finding: every entry in the hipBLASLt
shape-override registry must use power-of-2 tile dimensions, because
Triton's `tl.zeros((BLOCK_M, BLOCK_N))` accumulator allocation in
`include/tritonblas/kernels/stages/gemm_context.py::init_accumulator`
rejects non-pow2 shapes with `ValueError: Shape element N must be a
power of 2` at compile time.

The K-579/K-581 dispatch design proposed seeding `{160, 192, 224}`
tiles from hipBLASLt's per-shape top-1 picks; all 12 candidate tiles
fail the Triton constraint. This test suite ensures (a) the predicate
is correct, (b) every override entry passes it, and (c) the
lookup-site guard fails *closed* (returns None → native selector
handles the shape) when fed a synthetic non-pow2 entry, instead of
crashing the whole package at import time.

These tests do NOT require a GPU — they exercise pure Python
selector logic.
"""

import pytest

from tritonblas.origami import (
    _HIPBLASLT_SHAPE_OVERRIDES,
    _hipblaslt_shape_override,
    _is_pow2,
    _is_triton_valid_block_tile,
)


# ---------------------------------------------------------------------------
# K-590 predicate correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n,expected",
    [
        (1, True), (2, True), (4, True), (8, True), (16, True),
        (32, True), (64, True), (128, True), (256, True), (512, True),
        # Non-pow2 tile widths hipBLASLt uses for the K-579 cohort:
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
        # K-545 in-table entries (must be valid):
        (128, 256, 64, True),
        (256, 128, 64, True),
        (256, 256, 64, True),
        # K-579 hipBLASLt top-1 candidates (must be invalid):
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
# K-590 invariant: every shipped override entry is Triton-legal
# ---------------------------------------------------------------------------

def test_all_overrides_are_pow2():
    """Every entry in `_HIPBLASLT_SHAPE_OVERRIDES` MUST satisfy
    `_is_triton_valid_block_tile`. Catches the K-579 / K-581 mistake
    of seeding (128, 224, 64) etc. before re-attempt cycles burn
    cluster cycles on what is fundamentally a Triton-AMD compiler
    constraint. See K-590 finding in `include/tritonblas/origami.py`.
    """
    bad = []
    for key, (bm, bn, bk) in _HIPBLASLT_SHAPE_OVERRIDES.items():
        if not _is_triton_valid_block_tile(bm, bn, bk):
            bad.append((key, (bm, bn, bk)))
    assert not bad, (
        f"Non-pow2 override entries: {bad}. Triton's `init_accumulator` "
        f"rejects non-pow2 BLOCK_M/BLOCK_N/BLOCK_K. See K-590 finding."
    )


# ---------------------------------------------------------------------------
# K-590 lookup-site behavior: fail-closed, do NOT crash the package
# ---------------------------------------------------------------------------

def test_lookup_skips_non_pow2_override(monkeypatch):
    """If a future contributor injects a non-pow2 entry into the
    override table, the lookup MUST return None (so the native
    selector picks a Triton-legal tile) instead of returning the
    bad entry to the dispatcher (which would surface deep inside
    Triton's `init_accumulator` with no pointer back to the override
    table).
    """
    # Inject a synthetic K-579-style bad entry without mutating the
    # shipped table on disk.
    bad_key = (1024, 8192, 8192, "fp16")
    bad_tile = (128, 224, 64)
    patched = dict(_HIPBLASLT_SHAPE_OVERRIDES)
    patched[bad_key] = bad_tile
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
        f"K-590 fail-closed pow2 guard at lookup site failed — "
        f"returned {out} for synthetic non-pow2 override {bad_tile}. "
        f"Expected None so the native selector handles the shape."
    )


def test_import_does_not_raise_on_existing_table():
    """Importing tritonblas.origami must succeed under the shipped
    override table. K-590 explicitly chose lookup-site fail-closed
    over import-time fail-fast so a future bad entry cannot brick
    the whole package for every downstream caller (per Devil's
    Advocate review feedback).
    """
    import importlib

    import tritonblas.origami as mod  # noqa: F401

    importlib.reload(mod)  # exercise the import path explicitly
