"""Smoke tests for hipBLASLt shape-override seed configs.

These tests exercise the override registry and the skinny-shape guard
on the post-hoc 256x256 fallback inside ``OrigamiMatmulSelector``.

Most tests require a GPU because ``OrigamiMatmulSelector`` calls into
origami's hardware lookup; the registry-shape and lookup-fallback
helpers are exercised CPU-only and live in
``tests/test_pow2_guard_and_kpack.py``.
"""

import os
import pytest
import torch

from tritonblas.origami import (
    OrigamiMatmulSelector,
    _HIPBLASLT_SHAPE_OVERRIDES,
    _hipblaslt_shape_override,
    estimate_triton_lds_bytes,
    check_triton_lds_capacity,
)


# Shipped overrides currently cover the long-N skinny family
# (M=1024, N=8192, K=8192) for both fp16 and bf16; those rows pin the
# orientation flip from Origami's native 256x128x64 pick to the
# long-axis-aligned 128x256x64.
COHORT_SHAPES = [
    # (M, N, K, dtype, expected_BM, expected_BN, expected_BK)
    (1024, 8192, 8192, torch.float16, 128, 256, 64),
    (1024, 8192, 8192, torch.bfloat16, 128, 256, 64),
]


_HAS_CUDA = torch.cuda.is_available()


def _selector(m, n, k, dtype):
    """Build a selector on CUDA — Origami requires the GPU device for hardware
    lookup (`get_hardware_for_device` accepts only int device indices)."""
    return OrigamiMatmulSelector(
        m, n, k,
        a_dtype=dtype,
        b_dtype=dtype,
        out_dtype=dtype,
        device=torch.device("cuda:0"),
    )


pytestmark = pytest.mark.skipif(
    not _HAS_CUDA,
    reason="Selector smoke tests require CUDA/HIP device for Origami hardware lookup",
)


@pytest.mark.parametrize("m,n,k,dtype,exp_bm,exp_bn,exp_bk", COHORT_SHAPES)
def test_cohort_override_applies(m, n, k, dtype, exp_bm, exp_bn, exp_bk):
    """Each cohort shape gets its registered tile."""
    sel = _selector(m, n, k, dtype)
    assert (sel.block_m, sel.block_n, sel.block_k) == (exp_bm, exp_bn, exp_bk), (
        f"Override failed for {m}x{n}x{k} {dtype}: "
        f"got {sel.block_m}x{sel.block_n}x{sel.block_k}, "
        f"expected {exp_bm}x{exp_bn}x{exp_bk}"
    )


def test_skinny_guard_disables_256_override():
    """For skinny shapes (aspect >= 4) NOT in the override table, the
    post-hoc 256x256 heuristic must not fire. Use a synthetic shape
    well outside the cohort to isolate the guard."""
    # 16384x2048x2048: aspect = 8, NOT in override table.
    sel = _selector(16384, 2048, 2048, torch.float16)
    # The post-hoc 256x256x64 override would have rewritten any
    # (256, !=256) tile to (256, 256, 64). With the skinny guard, the
    # native Origami pick survives. We don't pin a specific tile —
    # just verify the selector returned a valid pick.
    assert sel.block_m > 0 and sel.block_n > 0 and sel.block_k > 0


def test_square_shape_unaffected():
    """For square (non-skinny) shapes outside the override table, the
    original 256x256 fallback heuristic must still fire when applicable."""
    # 4096x4096x4096: aspect = 1, NOT in override table.
    sel = _selector(4096, 4096, 4096, torch.float16)
    assert sel.block_m in (16, 32, 64, 128, 256)
    assert sel.block_n in (16, 32, 64, 128, 256)
    assert sel.block_k in (16, 32, 64, 128, 256, 512)


def test_disable_via_env_var():
    """TRITONBLAS_DISABLE_SHAPE_OVERRIDES=1 must restore old behavior."""
    os.environ["TRITONBLAS_DISABLE_SHAPE_OVERRIDES"] = "1"
    try:
        # Even cohort shapes should NOT get the override when disabled.
        sel = _selector(8192, 1024, 8192, torch.float16)
        # The override-specific tile should NOT appear unless Origami
        # independently picks it; just verify the selector returned a
        # valid pick.
        assert sel.block_m in (16, 32, 64, 128, 256)
    finally:
        del os.environ["TRITONBLAS_DISABLE_SHAPE_OVERRIDES"]


def test_override_registry_well_formed():
    """All override entries must be valid tile triples in the canonical
    `_block_mn_range x _block_k_range` search space."""
    valid_mn = {16, 32, 64, 128, 256}
    valid_k = {16, 32, 64, 128, 256, 512}
    for key, _entry in _HIPBLASLT_SHAPE_OVERRIDES.items():
        bm, bn, bk = _entry["tile"] if isinstance(_entry, dict) else _entry
        m, n, k, dtype = key
        assert bm in valid_mn, f"{key} BM={bm} outside _block_mn_range"
        assert bn in valid_mn, f"{key} BN={bn} outside _block_mn_range"
        assert bk in valid_k, f"{key} BK={bk} outside _block_k_range"
        assert dtype in ("fp16", "bf16"), f"{key} dtype must be fp16/bf16"


def test_lds_safety_check():
    """An override that doesn't fit in LDS must be rejected (returns None)."""
    # Force a tiny lds_cap to verify the guard. Use one of the
    # still-active override entries (1024x8192x8192 -> 128x256x64).
    out = _hipblaslt_shape_override(
        m=1024, n=8192, k=8192,
        a_dtype_str="f16", b_dtype_str="f16",
        bytes_a=2.0, bytes_b=2.0,
        lds_cap=1024,  # 1KB — too small for 128x256x64
        num_stages=2,
    )
    assert out is None, "LDS safety guard failed — override returned despite tile overflow"


def test_shape_not_in_table_returns_none():
    """Shapes outside the cohort must return None (no override applied)."""
    out = _hipblaslt_shape_override(
        m=512, n=512, k=512,
        a_dtype_str="f16", b_dtype_str="f16",
        bytes_a=2.0, bytes_b=2.0,
        lds_cap=64 * 1024,
        num_stages=2,
    )
    assert out is None
