"""K-695: unit tests for the FP8 e4m3fnuz tall-skinny tile-override gate.

These tests guard the predicate in `tritonblas.matmul._k695_tile_override`
against future regressions. They do NOT execute the kernel — they assert the
predicate's pure-Python decisions for each (M, N, K, dtype) so a re-tightening
of the table or a typo in the dispatch logic trips a CI failure.

Concretely they check:
  * The 23 shapes that are supposed to fire return the exact (BM, NS, KP) we
    landed (sweep winner per shape).
  * The 4 cohort shapes that are supposed to skip — (16,4096,16384) and the
    three (M=64, N=8192) shapes — return None (the K-654-style 61-shape
    regression sweep showed the override either no-ops or regresses there).
  * Out-of-cohort shapes (M, N, or K outside {16,32,64} x {4096,8192,16384}^2)
    do not fire.
  * Wrong-dtype shapes (FP16, BF16, FP8 e5m2fnuz) do not fire even on
    in-cohort (M, N, K).
"""

import pytest
import torch

from tritonblas.matmul import _k695_tile_override, _K695_GATE_TABLE


FP8 = getattr(torch, "float8_e4m3fnuz", None)
FP8_E5M2 = getattr(torch, "float8_e5m2fnuz", None)


# Expected fires (23 shapes, sweep winners)
_EXPECTED_FIRES = {
    # M=16 (8): uniform (32, 2, 1)
    (16, 4096, 4096):   (32, 2, 1),
    (16, 4096, 8192):   (32, 2, 1),
    (16, 8192, 4096):   (32, 2, 1),
    (16, 8192, 8192):   (32, 2, 1),
    (16, 8192, 16384):  (32, 2, 1),
    (16, 16384, 4096):  (32, 2, 1),
    (16, 16384, 8192):  (32, 2, 1),
    (16, 16384, 16384): (32, 2, 1),
    # M=32 (9): uniform (32, 2, 1)
    (32, 4096, 4096):   (32, 2, 1),
    (32, 4096, 8192):   (32, 2, 1),
    (32, 4096, 16384):  (32, 2, 1),
    (32, 8192, 4096):   (32, 2, 1),
    (32, 8192, 8192):   (32, 2, 1),
    (32, 8192, 16384):  (32, 2, 1),
    (32, 16384, 4096):  (32, 2, 1),
    (32, 16384, 8192):  (32, 2, 1),
    (32, 16384, 16384): (32, 2, 1),
    # M=64,N=4096 (3): NS=1; KP=1 only at K=16384 (sweep winner)
    (64, 4096, 4096):   (32, 1, 2),
    (64, 4096, 8192):   (32, 1, 2),
    (64, 4096, 16384):  (32, 1, 1),
    # M=64,N=16384 (3)
    (64, 16384, 4096):  (32, 2, 1),
    (64, 16384, 8192):  (32, 2, 1),
    (64, 16384, 16384): (32, 2, 1),
}

# Cohort shapes that MUST skip (return None). Future "tightening" of the
# predicate (e.g. extending KP=2 across the M=64,N=4096 row) must not silently
# re-introduce these shapes.
_EXPECTED_SKIPS = [
    (16, 4096, 16384),  # sweep winner == Origami default; override no-op/regress
    (64, 8192, 4096),   # Origami already optimal on this sub-band
    (64, 8192, 8192),
    (64, 8192, 16384),
]


@pytest.mark.skipif(FP8 is None, reason="float8_e4m3fnuz not available")
@pytest.mark.parametrize("shape,expected", sorted(_EXPECTED_FIRES.items()))
def test_k695_gate_fires_with_correct_tile(shape, expected):
    M, N, K = shape
    got = _k695_tile_override(M, N, K, FP8)
    assert got == expected, (
        f"K-695 gate at {shape} returned {got}, expected {expected}. "
        f"If you changed the override table, update the expected values."
    )


@pytest.mark.skipif(FP8 is None, reason="float8_e4m3fnuz not available")
@pytest.mark.parametrize("shape", _EXPECTED_SKIPS)
def test_k695_gate_skips_cohort_shape(shape):
    M, N, K = shape
    assert _k695_tile_override(M, N, K, FP8) is None, (
        f"K-695 gate must NOT fire on {shape}: this shape was skipped because "
        f"the override regressed or no-op'd vs Origami in the K-654-style sweep. "
        f"Re-tightening the predicate here will re-introduce a real perf regression."
    )


@pytest.mark.skipif(FP8 is None, reason="float8_e4m3fnuz not available")
@pytest.mark.parametrize("shape", [
    # M outside cohort
    (8, 8192, 8192), (128, 8192, 8192), (256, 8192, 8192), (512, 8192, 8192),
    # N outside cohort
    (32, 2048, 8192), (32, 32768, 8192),
    # K outside cohort
    (32, 8192, 2048), (32, 8192, 32768),
])
def test_k695_gate_does_not_fire_out_of_cohort(shape):
    M, N, K = shape
    assert _k695_tile_override(M, N, K, FP8) is None


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float8_e5m2fnuz"])
@pytest.mark.parametrize("shape", [
    (16, 8192, 8192), (32, 8192, 8192), (64, 4096, 8192),
])
def test_k695_gate_only_fires_for_e4m3fnuz(dtype_name, shape):
    dt = getattr(torch, dtype_name, None)
    if dt is None:
        pytest.skip(f"{dtype_name} not available")
    M, N, K = shape
    assert _k695_tile_override(M, N, K, dt) is None, (
        f"K-695 gate must only fire for float8_e4m3fnuz; fired for {dtype_name} at {shape}"
    )


def test_k695_gate_table_size():
    # Sanity: 8 (M=16) + 9 (M=32) + 6 (M=64) = 23 entries.
    assert len(_K695_GATE_TABLE) == 23, (
        f"_K695_GATE_TABLE has {len(_K695_GATE_TABLE)} entries, expected 23. "
        f"If you intentionally added/removed entries, update this test."
    )


def test_k695_gate_table_matches_expected():
    # Defense in depth: the table itself matches what we test individually
    # above. Catches the case where someone edits the table but not the tests.
    assert dict(_K695_GATE_TABLE) == _EXPECTED_FIRES
