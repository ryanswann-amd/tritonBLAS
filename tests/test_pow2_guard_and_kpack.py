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
    helper is whitelist-driven: kpack=2 ships ONLY for the validated
    `(BLOCK_M, BLOCK_N, BLOCK_K, a_dtype)` tuples in
    `KPACK2_TILE_DTYPE_WHITELIST`. Every other dispatch — symmetric
    tile, asymmetric tile that failed the paired-bench gate (e.g.
    128x256x64 / T2-a, where bf16 regressed -1.56pp), and all
    non-fp16/bf16 dtypes — gets kpack=1.

    The whitelist tests below explicitly lock in the regressions that
    must NOT recur: the symmetric `8192^3` 256x256x64 case (-8.8pp
    when kpack=2 was unconditional) and the asymmetric T2-a 128x256x64
    case (which failed the +1.5pp noise-floor gate even though the
    earlier asymmetry-only heuristic would have allowed it).

These tests do NOT require a GPU — they exercise pure Python selector
logic.
"""

import pytest
import torch

from tritonblas.constraints import (
    KPACK2_TILE_DTYPE_WHITELIST,
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
    "dtype,bm,bn,bk,expected",
    [
        # ---- Whitelist hits: long-M skinny 256x128x64 fp16/bf16 →
        # kpack=2. These are the ONLY two (tile, dtype) combinations
        # validated by the paired bench (+2 to +4pp lift across two
        # independent runs at 8192x1024x8192, both above the +1.5pp
        # noise floor with zero rows regressing).
        (torch.float16,  256, 128, 64, 2),
        (torch.bfloat16, 256, 128, 64, 2),

        # ---- 128x256x64 EXCLUSION (regression lock).
        # An earlier asymmetry-only heuristic would have allowed this
        # tile, but the paired bench measured a median Δ inside noise
        # with bf16 regressing — failing the falsification gate. The
        # whitelist correctly excludes it.
        (torch.float16,  128, 256, 64, 1),
        (torch.bfloat16, 128, 256, 64, 1),

        # ---- 256x256x64 symmetric tile → kpack=1. Whitelist excludes
        # 256x256x64 entirely (no symmetric-tile entry shipped — see
        # the dedicated regression-lock test below).
        (torch.float16,  256, 256, 64, 1),
        (torch.bfloat16, 256, 256, 64, 1),

        # ---- Other symmetric / control-bucket tiles → kpack=1.
        (torch.float16,  128, 128, 128, 1),
        (torch.bfloat16, 128, 128, 128, 1),

        # ---- Whitelist-tile + wrong BK → kpack=1. The validation was
        # at BK=64 specifically; widening to BK=128/32 has no
        # supporting paired-bench data.
        (torch.float16,  256, 128, 32, 1),
        (torch.float16,  256, 128, 128, 1),
        (torch.bfloat16, 256, 128, 32, 1),

        # ---- Whitelist-tile + wrong dtype → kpack=1.
        (torch.float32,  256, 128, 64, 1),
        (torch.int8,     256, 128, 64, 1),

        # ---- Non-fp16/bf16 dtypes are always kpack=1 regardless of
        # tile (fp8 path is gfx950-clamped at codegen, fp4 has its
        # own kernel entry).
        (torch.float32,  256, 256, 64, 1),
        (torch.int8,     256, 256, 64, 1),
    ],
)
def test_kpack_for_dtype_whitelist(dtype, bm, bn, bk, expected):
    assert kpack_for_dtype(dtype, bm, bn, bk) == expected


def test_kpack2_whitelist_contents_are_locked():
    """The whitelist is the audit-trail-bearing surface for kpack=2.
    Lock its EXACT contents so that any widening — even a one-line
    addition — forces the contributor to also update the bench
    harness, the docstring, and this test together. Both a previous
    unconditional kpack=2 policy and a subsequent overly broad
    asymmetry-only gate shipped real regressions; a tightly locked
    whitelist is the structural defense against repeating that miss.
    """
    expected = frozenset({
        (256, 128, 64, torch.float16),
        (256, 128, 64, torch.bfloat16),
    })
    assert KPACK2_TILE_DTYPE_WHITELIST == expected, (
        f"KPACK2_TILE_DTYPE_WHITELIST has been changed. Current: "
        f"{sorted(KPACK2_TILE_DTYPE_WHITELIST, key=str)}. "
        f"If this is intentional, re-run the paired-bench harness on "
        f"the candidate tile and update both this test and the "
        f"docstring in `tritonblas.constraints` together."
    )


def test_kpack_8192_cubed_symmetric_tile_returns_kpack1():
    """REGRESSION LOCK.

    A historical unconditional kpack=2 measured -8.8 to -10.1pp on
    `8192x8192x8192` at the symmetric `256x256x64` tile (well outside
    the ±1.5pp noise floor; reproduced across two paired runs after
    clearing `~/.triton/cache/`). This test pins the post-fix
    behavior so a future helper rewrite cannot silently re-enable
    kpack=2 on this exact `(tile, dtype)` and re-introduce the
    regression.
    """
    # Both fp16 and bf16 must return kpack=1 at the symmetric tile.
    assert kpack_for_dtype(torch.float16, 256, 256, 64) == 1
    assert kpack_for_dtype(torch.bfloat16, 256, 256, 64) == 1
    # And the same exclusion holds for any other symmetric pow2 tile
    # at the same (BK=64, dtype) combination, since none have been
    # paired-benched.
    assert kpack_for_dtype(torch.float16, 128, 128, 64) == 1
    assert kpack_for_dtype(torch.bfloat16, 128, 128, 64) == 1


def test_kpack_long_n_skinny_128x256x64_returns_kpack1():
    """REGRESSION LOCK.

    The ``128x256x64`` tile (long-N skinny, 1024x8192x8192) is
    asymmetric, so an earlier "block_m != block_n" heuristic would
    have shipped kpack=2 here — but the paired bench measured bf16
    regressing with a median Δ inside noise, falsifying the
    heuristic. The whitelist correctly excludes this tile; this test
    locks that exclusion in so a future broadening proposal cannot
    silently re-enable it without re-validating the paired bench.
    """
    assert kpack_for_dtype(torch.float16, 128, 256, 64) == 1
    assert kpack_for_dtype(torch.bfloat16, 128, 256, 64) == 1


def test_kpack_for_dtype_falls_back_to_one_without_tile():
    """When ANY of `block_m`/`block_n`/`block_k` is not supplied (legacy
    or external caller), the helper MUST fall back to the safe
    `kpack=1` setting rather than guessing `kpack=2`. Without all
    three tile dims, the whitelist lookup cannot run and the safe
    default is the only option that cannot re-introduce a regression.
    """
    for dtype in (torch.float16, torch.bfloat16):
        assert kpack_for_dtype(dtype) == 1
        assert kpack_for_dtype(dtype, 256, 128) == 1                # missing BK
        assert kpack_for_dtype(dtype, None, 128, 64) == 1           # missing BM
        assert kpack_for_dtype(dtype, 256, None, 64) == 1           # missing BN
        assert kpack_for_dtype(dtype, 256, 128, None) == 1          # missing BK


def test_kpack_force_kpack1_env(monkeypatch):
    """`TRITONBLAS_FORCE_KPACK1=1` clamps every dispatch to `kpack=1`,
    used by the paired bench harness to isolate the kpack contribution
    without checking out an older revision."""
    monkeypatch.setenv("TRITONBLAS_FORCE_KPACK1", "1")
    # Even the whitelist hits must clamp to kpack=1.
    assert kpack_for_dtype(torch.float16, 256, 128, 64) == 1
    assert kpack_for_dtype(torch.bfloat16, 256, 128, 64) == 1


def test_kpack_helper_call_passes_block_k():
    """The launchers MUST pass `BLK_K` to the helper (the whitelist is
    keyed on the full `(BM, BN, BK, dtype)` tuple). A 3-arg call
    would silently fall back to kpack=1 for every dispatch — losing
    the T1 win — so we assert structurally that both launchers pass
    four positional args.
    """
    import importlib
    import inspect

    _matmul_mod = importlib.import_module("tritonblas.matmul")
    src_persistent = inspect.getsource(_matmul_mod.persistent_matmul_lt)
    src_streamk = inspect.getsource(_matmul_mod.streamk_matmul_lt)

    for label, src in (("persistent_matmul_lt", src_persistent),
                       ("streamk_matmul_lt", src_streamk)):
        assert "kpack_for_dtype(a.dtype, BLK_M, BLK_N, BLK_K)" in src, (
            f"{label} must call `kpack_for_dtype(a.dtype, BLK_M, BLK_N, BLK_K)` "
            f"to consult the whitelist; a 3-arg call would silently "
            f"fall back to kpack=1 for every dispatch."
        )


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
