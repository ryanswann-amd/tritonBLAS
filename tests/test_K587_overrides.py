"""K-587 — Tests for the extended hipBLASLt-derived override registry,
adaptive `kpack` policy (co-located with the registry), per-entry
provenance schema, and end-to-end correctness through `tritonblas.matmul`.

Covers:
  1. Every override-table entry is a power-of-2 tile inside the canonical
     `_block_mn_range x _block_k_range` search space (Triton requires
     power-of-2 BLOCK_M / BLOCK_N — non-pow2 raises CompilationError;
     this test pins the avoidance documented in the registry header).
  2. The provenance side-table is in lock-step with the override table.
  3. The adaptive `kpack_for_tile` policy gates correctly at the
     `KPACK2_TILE_AREA_THRESHOLD` boundary (one shape just inside, one
     just outside).  Both the origami-side and the matmul-side re-export
     of the constant are pinned so they cannot drift apart silently.
  4. The dtype-string normalization accepts both human-readable spellings
     ("fp16" / "bf16") and the internal `dtype_to_str` codes ("f16" /
     "bf16") used by the matmul dispatcher.
  5. The `TRITONBLAS_DISABLE_SHAPE_OVERRIDES=1` escape hatch returns None.
  6. End-to-end numerical correctness — `tritonblas.matmul` against
     `torch.matmul` on an override shape — confirms the dispatch path
     (override → kernel-launch with adaptive kpack) does not corrupt
     results. Skipped when no GPU is present.

These tests run CPU-only EXCEPT the e2e correctness test, which is
skipped when CUDA is unavailable.
"""

import os

import pytest

# Module-level imports of registry / lookup / constants. We import the
# matmul module by attribute to avoid pulling in the GPU-only side effects
# of `tritonblas/__init__.py` on test workstations without a GPU.
from tritonblas.origami import (
    _HIPBLASLT_SHAPE_OVERRIDES,
    _HIPBLASLT_SHAPE_OVERRIDE_REGISTRY,
    _HIPBLASLT_SHAPE_OVERRIDE_PROVENANCE,
    _hipblaslt_shape_override,
    kpack_for_tile,
    KPACK2_TILE_AREA_THRESHOLD,
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
# 2. Adaptive kpack policy — co-located with registry, owned by origami.py
# -----------------------------------------------------------------------------

def test_kpack_threshold_constant_value():
    """The named constant must be 32768 (=128*256), the empirically-derived
    inflection point between the +3 to +4 pp `ds_read_b128` win on
    skinny / small tiles and the -7 to -9 pp VGPR-pressure regression on
    256x256x64. Re-tuning this value belongs in the named constant, not
    inline in the kernel-launch path."""
    assert KPACK2_TILE_AREA_THRESHOLD == 32768


def test_matmul_re_exports_kpack_threshold():
    """Architect-feedback regression test: the kpack policy must live in
    one place (origami.py).  matmul.py re-exports the constant by its old
    name `_KPACK2_TILE_AREA_THRESHOLD` so existing callers / external tests
    keep working — verify the re-export equals the source-of-truth."""
    import importlib
    matmul_mod = importlib.import_module("tritonblas.matmul")
    assert matmul_mod._KPACK2_TILE_AREA_THRESHOLD == KPACK2_TILE_AREA_THRESHOLD


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
def test_kpack_for_tile_gating(blk_m, blk_n, expected_kpack, why):
    """Pin the boundary behavior of the policy function consumed by
    `persistent_matmul_lt` / `streamk_matmul_lt`.  If the policy moves,
    this test surfaces the regression."""
    assert kpack_for_tile(blk_m, blk_n) == expected_kpack, why


# -----------------------------------------------------------------------------
# 3. Lookup correctness (no memo — the bare dict.get path is the hot path)
# -----------------------------------------------------------------------------

def _clear_env():
    """Reset the override env switch between tests so state from one test
    does not leak into another."""
    os.environ.pop("TRITONBLAS_DISABLE_SHAPE_OVERRIDES", None)


def test_lookup_returns_registered_tile():
    _clear_env()
    args = dict(m=1024, n=8192, k=8192,
                a_dtype_str="f16", b_dtype_str="f16",
                bytes_a=2.0, bytes_b=2.0,
                lds_cap=64 * 1024, num_stages=2)
    assert _hipblaslt_shape_override(**args) == (128, 256, 64)


def test_lookup_returns_none_for_unregistered_shape():
    _clear_env()
    args = dict(m=999, n=999, k=999,
                a_dtype_str="f16", b_dtype_str="f16",
                bytes_a=2.0, bytes_b=2.0,
                lds_cap=64 * 1024, num_stages=2)
    assert _hipblaslt_shape_override(**args) is None


def test_lookup_lds_safety_guard_returns_none():
    """An override that does not fit in LDS must return None so the
    caller falls back to Origami's natively-selected tile."""
    _clear_env()
    args = dict(m=1024, n=8192, k=8192,
                a_dtype_str="f16", b_dtype_str="f16",
                bytes_a=2.0, bytes_b=2.0,
                lds_cap=1024,  # 1KB, way too small for 128x256x64
                num_stages=2)
    assert _hipblaslt_shape_override(**args) is None


# -----------------------------------------------------------------------------
# 4. dtype-string normalization
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("a_dtype_str", ["fp16", "f16"])
def test_fp16_dtype_spellings_normalized(a_dtype_str):
    _clear_env()
    out = _hipblaslt_shape_override(
        m=1024, n=8192, k=8192,
        a_dtype_str=a_dtype_str, b_dtype_str=a_dtype_str,
        bytes_a=2.0, bytes_b=2.0,
        lds_cap=64 * 1024, num_stages=2,
    )
    assert out == (128, 256, 64)


def test_bf16_dtype_normalized():
    _clear_env()
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
    _clear_env()
    out = _hipblaslt_shape_override(
        m=1024, n=8192, k=8192,
        a_dtype_str="f8", b_dtype_str="f8",
        bytes_a=1.0, bytes_b=1.0,
        lds_cap=64 * 1024, num_stages=2,
    )
    assert out is None


def test_env_disable_short_circuits_lookup():
    _clear_env()
    os.environ["TRITONBLAS_DISABLE_SHAPE_OVERRIDES"] = "1"
    try:
        out = _hipblaslt_shape_override(
            m=1024, n=8192, k=8192,
            a_dtype_str="f16", b_dtype_str="f16",
            bytes_a=2.0, bytes_b=2.0,
            lds_cap=64 * 1024, num_stages=2,
        )
        assert out is None
    finally:
        del os.environ["TRITONBLAS_DISABLE_SHAPE_OVERRIDES"]


# -----------------------------------------------------------------------------
# 5. End-to-end correctness through `tritonblas.matmul` on an override shape
#    (Testing Zealot feedback — pin that the dispatch + adaptive-kpack path
#    does not corrupt numerical results on a real GPU run.)
# -----------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="end-to-end correctness test requires CUDA / ROCm GPU",
)
@pytest.mark.parametrize("dtype_str,torch_dtype", [
    ("fp16", "float16"),
    ("bf16", "bfloat16"),
])
def test_matmul_override_e2e_correctness(dtype_str, torch_dtype):
    """Run the smallest override shape (1024x8192x8192 fp16 / bf16) through
    `tritonblas.matmul` and verify the result matches torch.matmul to
    bf16-GEMM accumulation tolerance.  This is the only test in this file
    that hits the kernel-launch path; it pins that the override dispatch +
    `kpack_for_tile` do not silently corrupt results.
    """
    import torch
    import tritonblas

    torch.manual_seed(0)
    M, N, K = 1024, 8192, 8192
    dt = getattr(torch, torch_dtype)
    A = torch.randn(M, K, device="cuda", dtype=dt)
    B = torch.randn(K, N, device="cuda", dtype=dt)
    C_tb = tritonblas.matmul(A, B)
    C_ref = torch.matmul(A, B)
    # bf16 / fp16 GEMM accumulation tolerance — same gate the K-587 bench
    # harness uses; well inside random-init GEMM noise.
    err = (C_tb.float() - C_ref.float()).abs().max().item()
    assert err < 5.0, (
        f"override-path e2e correctness failed for {dtype_str}: max_abs_err={err}"
    )
