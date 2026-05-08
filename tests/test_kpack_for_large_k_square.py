"""Unit tests for the per-shape kpack override gate in tritonblas.matmul.

The helper `_kpack_for_large_k_square(M, N, K, dtype)` returns 2 only on the
empirically non-regressing slice of the large-K square FP16/BF16 GEMM cohort
(M == N == 2048, K in {4096, 8192, 16384}, dtype in {fp16, bf16}); for every
other shape / dtype it must return the historical default of 1 so that the
adjacent regression-guard cohorts (medium-K, intermediate-K, non-square,
FP32, FP8, ...) execute on a bit-for-bit identical kernel-arg payload.

These tests intentionally exercise both the positive cases (the gate fires)
and a comprehensive battery of negative cases (the gate does not fire) so
that any future widening or accidental regression of the predicate is
caught by CI before it touches the kernel.
"""

import pytest
import torch

from tritonblas.matmul import _kpack_for_large_k_square


# --- Positive cases: the gate must fire ----------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k", [4096, 8192, 16384])
def test_gate_fires_on_large_k_square_2048(dtype, k):
    """All 6 (dtype x K) cells on the M=N=2048 row must return kpack=2."""
    assert _kpack_for_large_k_square(2048, 2048, k, dtype) == 2


# --- Negative cases: dtype --------------------------------------------------

@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int32,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ],
)
@pytest.mark.parametrize("k", [4096, 8192, 16384])
def test_gate_skips_non_fp16_bf16_dtypes(dtype, k):
    """Even on the otherwise-matching M=N=2048 row, non-FP16/BF16 dtypes
    must take the default kpack=1 path."""
    assert _kpack_for_large_k_square(2048, 2048, k, dtype) == 1


# --- Negative cases: M / N --------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [1024, 4096])
@pytest.mark.parametrize("k", [4096, 8192, 16384])
def test_gate_skips_other_square_rows(dtype, m, k):
    """The M=N=1024 row regresses +1.7..+4.5% under kpack=2 and the
    M=N=4096 row regresses +8.9..+11.6% — both must take kpack=1."""
    assert _kpack_for_large_k_square(m, m, k, dtype) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "m, n",
    [
        (2048, 1024),  # non-square (N short)
        (1024, 2048),  # non-square (M short)
        (2048, 4096),  # non-square (N long)
        (4096, 2048),  # non-square (M long)
        (2048, 2049),  # off-by-one
        (2049, 2048),  # off-by-one
    ],
)
@pytest.mark.parametrize("k", [4096, 8192, 16384])
def test_gate_skips_non_square_shapes(dtype, m, n, k):
    """Any M != N must skip the gate, even when one of M/N equals 2048."""
    assert _kpack_for_large_k_square(m, n, k, dtype) == 1


# --- Negative cases: K outside the empirical envelope ----------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "k",
    [
        # Adjacent K-451 medium-K cohort
        128, 256, 512,
        # Adjacent K-570 intermediate-K cohort
        1024, 2048,
        # Off-by-one above and below the gated K set
        4095, 4097,
        8191, 8193,
        16383, 16385,
        # Far above the gated K set
        32768, 65536,
    ],
)
def test_gate_skips_out_of_envelope_K(dtype, k):
    """K values outside {4096, 8192, 16384} must take the default path."""
    assert _kpack_for_large_k_square(2048, 2048, k, dtype) == 1


# --- Spot-check the entire 18-shape S-002 large-K square cohort ------------

def test_full_18_cell_cohort_routing():
    """All 18 cohort cells: 6 must route to kpack=2 (M=N=2048 row); the
    other 12 (M=N=1024 and M=N=4096 rows) must stay on kpack=1."""
    m_values = (1024, 2048, 4096)
    k_values = (4096, 8192, 16384)
    dtypes = (torch.float16, torch.bfloat16)
    fired = 0
    skipped = 0
    for m in m_values:
        for k in k_values:
            for dt in dtypes:
                kpack = _kpack_for_large_k_square(m, m, k, dt)
                if kpack == 2:
                    fired += 1
                    assert m == 2048
                else:
                    skipped += 1
                    assert kpack == 1
                    assert m in (1024, 4096)
    assert fired == 6
    assert skipped == 12


# --- Spot-check that the helper is referentially transparent ---------------

def test_helper_is_pure():
    """Call the helper twice with the same args; result must be identical."""
    a = _kpack_for_large_k_square(2048, 2048, 8192, torch.float16)
    b = _kpack_for_large_k_square(2048, 2048, 8192, torch.float16)
    assert a == b == 2

    c = _kpack_for_large_k_square(4096, 4096, 8192, torch.float16)
    d = _kpack_for_large_k_square(4096, 4096, 8192, torch.float16)
    assert c == d == 1
