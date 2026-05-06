"""K-587 — Tests for the extended hipBLASLt-derived override registry,
adaptive `kpack` threshold, and per-entry provenance schema.

Covers:
  1. Every override-table entry is a power-of-2 tile inside the canonical
     `_block_mn_range x _block_k_range` search space (Triton requires
     power-of-2 BLOCK_M / BLOCK_N — non-pow2 raises CompilationError;
     this test pins the avoidance documented in the K-587 PR description).
  2. The provenance side-table is in lock-step with the override table.
  3. The adaptive `kpack` threshold gates correctly at the
     `_KPACK2_TILE_AREA_THRESHOLD` boundary (one shape just inside, one
     just outside).
  4. The override-lookup memo round-trips correctly (cache hit returns the
     same tile; cache stores `None` for misses without re-scanning the
     underlying dict).
  5. The dtype-string normalization accepts both human-readable spellings
     ("fp16" / "bf16") and the internal `dtype_to_str` codes ("f16" /
     "bf16") used by the matmul dispatcher.

These tests do NOT require a GPU — they only exercise the registry,
provenance, memo, and threshold-constant levers that K-587 adds.
"""

import os
import importlib

import pytest

# Module-level imports of registry / lookup / constants. We import the
# matmul module by attribute to avoid pulling in the GPU-only side effects
# of `tritonblas/__init__.py` on test workstations without a GPU.
from tritonblas.origami import (
    _HIPBLASLT_SHAPE_OVERRIDES,
    _HIPBLASLT_SHAPE_OVERRIDE_REGISTRY,
    _HIPBLASLT_SHAPE_OVERRIDE_PROVENANCE,
    _OVERRIDE_LOOKUP_CACHE,
    _hipblaslt_shape_override,
)


VALID_BLOCK_MN = {16, 32, 64, 128, 256}
VALID_BLOCK_K = {16, 32, 64, 128, 256, 512}


# -----------------------------------------------------------------------------
# 1. Tile-shape validity & non-pow2 avoidance
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("entry", _HIPBLASLT_SHAPE_OVERRIDE_REGISTRY)
def test_every_override_entry_is_power_of_two(entry):
    """Triton's persistent_matmul kernel raises CompilationError when
    BLOCK_M / BLOCK_N is not in `_block_mn_range` (16 / 32 / 64 / 128 / 256).
    Pin every override entry inside that search space — and document the
    non-pow2 substitution explicitly via the entry's `rationale` text.
    """
    m, n, k, dtype, bm, bn, bk, source, rationale = entry
    assert bm in VALID_BLOCK_MN, f"{(m,n,k,dtype)} BM={bm} not power-of-2"
    assert bn in VALID_BLOCK_MN, f"{(m,n,k,dtype)} BN={bn} not power-of-2"
    assert bk in VALID_BLOCK_K, f"{(m,n,k,dtype)} BK={bk} outside block_k_range"
    # When the rationale documents a non-pow2 hipBLASLt pick (224, 192, 160),
    # the substituted tile must be 128 or 256 — never the rejected 224 / 192.
    if any(tok in rationale for tok in ("224", "192", "160")):
        assert bm in (128, 256) and bn in (128, 256), (
            f"non-pow2 rationale at {(m,n,k,dtype)} but tile {(bm,bn)} is not in {{128,256}}"
        )


def test_provenance_in_sync_with_override_dict():
    """The flat `_HIPBLASLT_SHAPE_OVERRIDES` lookup dict and the
    `_HIPBLASLT_SHAPE_OVERRIDE_PROVENANCE` side-table must have identical
    keysets — otherwise an entry was added/removed in only one place.
    """
    assert set(_HIPBLASLT_SHAPE_OVERRIDES.keys()) == set(
        _HIPBLASLT_SHAPE_OVERRIDE_PROVENANCE.keys()
    )
    for key, (source, rationale) in _HIPBLASLT_SHAPE_OVERRIDE_PROVENANCE.items():
        assert isinstance(source, str) and source.startswith("K-"), (
            f"{key} provenance source {source!r} must be a K-NNN ticket id"
        )
        assert isinstance(rationale, str) and len(rationale) >= 20, (
            f"{key} rationale text too terse — explain WHY this tile was chosen"
        )


# -----------------------------------------------------------------------------
# 2. Adaptive kpack threshold pin
# -----------------------------------------------------------------------------

def test_kpack_threshold_constant_value():
    """The named constant must be 32768 (=128*256), the empirically-derived
    inflection point between the +3 to +4 pp `ds_read_b128` win on
    skinny / small tiles and the -7 to -9 pp VGPR-pressure regression on
    256x256x64. Re-tuning this value belongs in the named constant, not
    inline in the kernel-launch path."""
    matmul_mod = importlib.import_module("tritonblas.matmul")
    assert matmul_mod._KPACK2_TILE_AREA_THRESHOLD == 32768


@pytest.mark.parametrize(
    "blk_m,blk_n,expected_kpack,why",
    [
        (128, 128, 2, "16384 < 32768 — small square keeps ds_read_b128 win"),
        (128, 256, 2, "32768 = inclusive boundary — skinny tile keeps win"),
        (256, 128, 2, "32768 = inclusive boundary — transposed skinny tile"),
        (64,  256, 2, "16384 < 32768 — narrow-M tile gets the win"),
        (256, 256, 1, "65536 > 32768 — symmetric square avoids VGPR pressure"),
        (256, 512, 1, "131072 > 32768 — large tiles always kpack=1"),
    ],
)
def test_kpack_threshold_gating(blk_m, blk_n, expected_kpack, why):
    """Pin the boundary behavior by replicating the exact gate expression
    used in `persistent_matmul_lt` / `streamk_matmul_lt`. If the threshold
    constant moves, this test surfaces the regression."""
    matmul_mod = importlib.import_module("tritonblas.matmul")
    threshold = matmul_mod._KPACK2_TILE_AREA_THRESHOLD
    kpack = 2 if (blk_m * blk_n) <= threshold else 1
    assert kpack == expected_kpack, why


# -----------------------------------------------------------------------------
# 3. Lookup memo correctness
# -----------------------------------------------------------------------------

def _clear_cache_and_env():
    """Reset the override memo and env switch between tests so cache state
    from one test does not leak into another."""
    _OVERRIDE_LOOKUP_CACHE.clear()
    os.environ.pop("TRITONBLAS_DISABLE_SHAPE_OVERRIDES", None)


def test_lookup_memo_hits_on_repeat():
    _clear_cache_and_env()
    args = dict(m=1024, n=8192, k=8192,
                a_dtype_str="f16", b_dtype_str="f16",
                bytes_a=2.0, bytes_b=2.0,
                lds_cap=64 * 1024, num_stages=2)
    first = _hipblaslt_shape_override(**args)
    assert first == (128, 256, 64)
    # cache should now contain the (m,n,k,dtype,bytes_a,bytes_b,lds_cap,ns) key
    assert len(_OVERRIDE_LOOKUP_CACHE) == 1
    second = _hipblaslt_shape_override(**args)
    assert second is first  # identity — cached object reused


def test_lookup_memo_caches_misses():
    _clear_cache_and_env()
    args = dict(m=999, n=999, k=999,
                a_dtype_str="f16", b_dtype_str="f16",
                bytes_a=2.0, bytes_b=2.0,
                lds_cap=64 * 1024, num_stages=2)
    assert _hipblaslt_shape_override(**args) is None
    assert len(_OVERRIDE_LOOKUP_CACHE) == 1  # miss is cached too
    assert _hipblaslt_shape_override(**args) is None
    assert len(_OVERRIDE_LOOKUP_CACHE) == 1  # second call did NOT re-add


def test_lookup_lds_safety_guard_caches_none():
    """An override that does not fit in LDS must return None AND cache
    that decision so the LDS check is not paid on every dispatch."""
    _clear_cache_and_env()
    args = dict(m=1024, n=8192, k=8192,
                a_dtype_str="f16", b_dtype_str="f16",
                bytes_a=2.0, bytes_b=2.0,
                lds_cap=1024,  # 1KB, way too small for 128x256x64
                num_stages=2)
    assert _hipblaslt_shape_override(**args) is None
    assert len(_OVERRIDE_LOOKUP_CACHE) == 1


# -----------------------------------------------------------------------------
# 4. dtype-string normalization
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("a_dtype_str", ["fp16", "f16"])
def test_fp16_dtype_spellings_normalized(a_dtype_str):
    _clear_cache_and_env()
    out = _hipblaslt_shape_override(
        m=1024, n=8192, k=8192,
        a_dtype_str=a_dtype_str, b_dtype_str=a_dtype_str,
        bytes_a=2.0, bytes_b=2.0,
        lds_cap=64 * 1024, num_stages=2,
    )
    assert out == (128, 256, 64)


def test_bf16_dtype_normalized():
    _clear_cache_and_env()
    out = _hipblaslt_shape_override(
        m=1024, n=8192, k=8192,
        a_dtype_str="bf16", b_dtype_str="bf16",
        bytes_a=2.0, bytes_b=2.0,
        lds_cap=64 * 1024, num_stages=2,
    )
    assert out == (128, 256, 64)


def test_unsupported_dtype_returns_none():
    """fp8 / fp32 paths use different selectors and must not match this
    fp16/bf16-only override registry."""
    _clear_cache_and_env()
    out = _hipblaslt_shape_override(
        m=1024, n=8192, k=8192,
        a_dtype_str="f8", b_dtype_str="f8",
        bytes_a=1.0, bytes_b=1.0,
        lds_cap=64 * 1024, num_stages=2,
    )
    assert out is None


def test_env_disable_short_circuits_lookup():
    _clear_cache_and_env()
    os.environ["TRITONBLAS_DISABLE_SHAPE_OVERRIDES"] = "1"
    try:
        out = _hipblaslt_shape_override(
            m=1024, n=8192, k=8192,
            a_dtype_str="f16", b_dtype_str="f16",
            bytes_a=2.0, bytes_b=2.0,
            lds_cap=64 * 1024, num_stages=2,
        )
        assert out is None
        # disabled-by-env path should NOT pollute the cache
        assert len(_OVERRIDE_LOOKUP_CACHE) == 0
    finally:
        del os.environ["TRITONBLAS_DISABLE_SHAPE_OVERRIDES"]
