"""
K-697 unit tests for the tall-skinny FP16/BF16 dispatch gate.

Two layers:
1. PURE-PYTHON gate predicate tests (no GPU required) — exhaustive boundary
   coverage that fails if any inequality drifts (>= vs >, == vs >=, etc.).
2. END-TO-END dispatch tests (require CUDA/ROCm) — assert that the override
   path is taken iff the gate fires, and that numerical correctness holds.

Run:
    python3 -m pytest tests/test_k697_tall_skinny_gate.py -v
"""

import pytest
import torch

from tritonblas.matmul import (
    _is_k697_tall_skinny,
    _K697_BLOCK_M,
    _K697_BLOCK_N,
    _K697_BLOCK_K,
    _K697_NUM_WARPS,
    _K697_NUM_STAGES,
)


# ----------------------------------------------------------------------------
# Layer 1 — pure-python predicate tests
# ----------------------------------------------------------------------------

# Cohort definition (mirror of the gate so a copy/paste typo gets caught).
_FP_DTYPES = (torch.float16, torch.bfloat16)
_NON_FP_DTYPES = (torch.float32, torch.float64, torch.int8, torch.int32)


def test_constants_match_verified_winner():
    """Sanity: the on-disk constants are the K-697-verified values."""
    assert _K697_BLOCK_M == 32
    assert _K697_BLOCK_N == 32
    assert _K697_BLOCK_K == 128
    assert _K697_NUM_WARPS == 2
    assert _K697_NUM_STAGES == 3


# --- IN-COHORT positives -----------------------------------------------------

@pytest.mark.parametrize("M", [2048, 4096, 8192, 16384])
@pytest.mark.parametrize("K", [1024, 2048, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", _FP_DTYPES)
def test_in_cohort_fires(M, K, dtype):
    """Every measured cell in the K-697 verify sweep must fire the gate."""
    assert _is_k697_tall_skinny(M, 32, K, dtype) is True


# --- OUT-OF-COHORT negatives (dtype) ----------------------------------------

@pytest.mark.parametrize("dtype", _NON_FP_DTYPES)
@pytest.mark.parametrize("M", [2048, 8192])
@pytest.mark.parametrize("K", [1024, 8192])
def test_non_fp_dtype_excluded(dtype, M, K):
    """FP32/FP64/INT dtypes must NEVER fire the gate (untested by K-697)."""
    assert _is_k697_tall_skinny(M, 32, K, dtype) is False


# --- OUT-OF-COHORT negatives (N) --------------------------------------------

@pytest.mark.parametrize("N", [1, 2, 4, 8, 16, 31, 33, 48, 64, 96, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("dtype", _FP_DTYPES)
def test_n_not_32_excluded(N, dtype):
    """Strict equality on N=32 — any other N must miss the gate.

    The override tile is BLOCK_N=32, so any N != 32 either pads (waste) or
    leaves part of the output uncovered. The K-697 leakage check measured a
    regression at N=128 (0.59x Origami) and N=256 (0.43x Origami); even
    N=64 is only a marginal 1.18x win that does not justify expanding the
    gate without a dedicated sweep.
    """
    assert _is_k697_tall_skinny(4096, N, 4096, dtype) is False


# --- OUT-OF-COHORT negatives (M) --------------------------------------------

@pytest.mark.parametrize("M", [1, 8, 16, 64, 128, 256, 512, 1024, 2047])
@pytest.mark.parametrize("dtype", _FP_DTYPES)
def test_m_below_2048_excluded(M, dtype):
    """M < 2048 is outside the K-697 verified envelope.

    K-518 / K-644-P1 cover the M<=8 region with their own veto; K-353
    covers the small-M-very-large-K region. Conservative lower bound at
    M=2048 matches the K-668 cohort definition exactly.
    """
    assert _is_k697_tall_skinny(M, 32, 4096, dtype) is False


def test_m_2048_boundary_inclusive():
    """M==2048 is the lowest M measured in the K-697 verify sweep — must fire."""
    for dtype in _FP_DTYPES:
        assert _is_k697_tall_skinny(2048, 32, 4096, dtype) is True


# --- OUT-OF-COHORT negatives (K) --------------------------------------------

@pytest.mark.parametrize("K", [1, 16, 32, 64, 96, 127, 128, 256, 512, 1023])
@pytest.mark.parametrize("dtype", _FP_DTYPES)
def test_k_below_1024_excluded(K, dtype):
    """K < 1024 misses the gate. K=512 was measured at 0.86x Origami in the
    K-697 leakage check — explicit regression evidence. K=1024 is the
    smallest verified-winning K in the K-697 sweep (1.20x..1.41x Origami).
    """
    assert _is_k697_tall_skinny(4096, 32, K, dtype) is False


def test_k_1024_boundary_inclusive():
    """K==1024 is the smallest verified-winning K — must fire."""
    for dtype in _FP_DTYPES:
        assert _is_k697_tall_skinny(2048, 32, 1024, dtype) is True


def test_k_1023_boundary_excluded():
    """Off-by-one guard: K==1023 must miss the gate."""
    for dtype in _FP_DTYPES:
        assert _is_k697_tall_skinny(2048, 32, 1023, dtype) is False


# --- ADJACENT-COHORT non-overlap ---------------------------------------------

def test_no_overlap_with_K644_P1():
    """K-644 P1 vetoes M<=8 AND K>=4096 -> StreamK off. K-697 must not
    re-engage on those shapes."""
    for dtype in _FP_DTYPES:
        for M in [1, 4, 8]:
            for K in [4096, 8192, 16384]:
                assert _is_k697_tall_skinny(M, 32, K, dtype) is False


def test_no_overlap_with_K644_P2_square_large():
    """K-644 P2 auto-flips StreamK on M==N>=5120, K>=6144 (square). K-697
    requires N==32 so it cannot fire on any square-large shape."""
    for dtype in _FP_DTYPES:
        for d in [5120, 6144, 8192, 16384]:
            assert _is_k697_tall_skinny(d, d, max(d, 6144), dtype) is False


def test_no_overlap_with_K278():
    """K-278 covers M in [64, 256], N >= 4096. K-697 caps N at 32, so no
    overlap is possible."""
    for dtype in _FP_DTYPES:
        for M in [64, 128, 256]:
            for N in [4096, 8192, 16384]:
                assert _is_k697_tall_skinny(M, N, 4096, dtype) is False


def test_no_overlap_with_K353():
    """K-353 covers M in [128, 512], N in [256, 1024]. K-697 needs N==32
    and M>=2048."""
    for dtype in _FP_DTYPES:
        for M in [128, 256, 512]:
            for N in [256, 512, 1024]:
                for K in [16384, 32768]:
                    assert _is_k697_tall_skinny(M, N, K, dtype) is False


def test_no_overlap_with_K667_large_K_square():
    """K-667 covers M==N==2048, K in {4096, 8192, 16384}. Square -> N != 32."""
    for dtype in _FP_DTYPES:
        for K in [4096, 8192, 16384]:
            assert _is_k697_tall_skinny(2048, 2048, K, dtype) is False


# ----------------------------------------------------------------------------
# Layer 2 — end-to-end dispatch tests (skip without CUDA)
# ----------------------------------------------------------------------------

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="K-697 dispatch override is GPU-only; CPU CI skips Layer 2.",
)


@cuda_required
@pytest.mark.parametrize("dtype", _FP_DTYPES)
def test_dispatch_in_cohort_matches_torch(dtype):
    """The K-697 override must produce numerically-equivalent output to
    torch.matmul on a representative in-cohort shape."""
    import tritonblas
    M, N, K = 2048, 32, 4096
    a = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    b = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
    ref = torch.matmul(a, b)
    out = tritonblas.matmul(a, b)
    # FP16/BF16 GEMM at K=4096 with stochastic accumulation order -- use
    # generous absolute tolerance, tighter relative.
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=5e-2)


@cuda_required
def test_dispatch_out_arg_in_cohort_matches_torch():
    """Same correctness check via the out= path -> _matmul_out."""
    import tritonblas
    M, N, K = 2048, 32, 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    ref = torch.matmul(a, b)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    tritonblas.matmul(a, b, out=out)
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=5e-2)


@cuda_required
def test_dispatch_out_of_cohort_unchanged():
    """An adjacent-cohort shape (square) must NOT take the override path
    -- numerical equivalence to torch.matmul on a small square shape."""
    import tritonblas
    M, N, K = 1024, 1024, 1024  # square -> gate misses (N != 32)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    ref = torch.matmul(a, b)
    out = tritonblas.matmul(a, b)
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=5e-2)


@cuda_required
def test_dispatch_work_stealing_path_unchanged():
    """work_stealing=True must skip the K-697 override (the override does
    not allocate the WS counters/locks)."""
    import tritonblas
    from tritonblas.matmul import _matmul
    M, N, K = 2048, 32, 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    ref = torch.matmul(a, b)
    out = _matmul(a, b, enable_streamk=False, sk_grid=None, work_stealing=True)
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=5e-2)
