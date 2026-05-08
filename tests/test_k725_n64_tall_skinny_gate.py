"""
K-725 unit tests for the tall-skinny FP16/BF16 N=64 dispatch gate.

Sibling of K-697 — same two-layer structure:
  Layer 1: PURE-PYTHON gate predicate tests (boundary coverage).
  Layer 2: END-TO-END dispatch tests (require CUDA/ROCm).

Run:
    python3 -m pytest tests/test_k725_n64_tall_skinny_gate.py -v
"""

import pytest
import torch

from tritonblas.matmul import (
    _is_k725_n64_tall_skinny,
    _K725_BLOCK_M,
    _K725_BLOCK_N,
    _K725_BLOCK_K,
    _K725_NUM_WARPS,
    _K725_NUM_STAGES,
    _K725_KPACK,
)


# ----------------------------------------------------------------------------
# Layer 1 — pure-python predicate tests
# ----------------------------------------------------------------------------

_FP_DTYPES = (torch.float16, torch.bfloat16)
_NON_FP_DTYPES = (torch.float32, torch.float64, torch.int8, torch.int32)


def test_constants_match_verified_winner():
    """Sanity: the on-disk constants are the K-725-verified values."""
    assert _K725_BLOCK_M == 64
    assert _K725_BLOCK_N == 64
    assert _K725_BLOCK_K == 128
    assert _K725_NUM_WARPS == 8
    assert _K725_NUM_STAGES == 2
    assert _K725_KPACK == 2


# --- IN-COHORT positives (composite-13 retains only 6 K-725-fire cells) ----
#
# Composite-13 (K-767) tightens the K-725 predicate to be mutually exclusive
# with K-693's persistent-path N=64 override (M >= 4096 AND K >= 4096).
# After mutex, K-725 fires on the disjoint outer cells only:
#     M = 16384 AND K in {1024, 2048}    (M=16384 K<4096)
#   + M = 2048  AND K  = 8192            (M=2048 K=8192)

_IN_COHORT_CELLS = [
    (16384, 1024), (16384, 2048),   # M=16384, K<4096 (outside K-693)
    (2048,  8192),                  # M=2048,  K=8192 (outside K-693)
]

@pytest.mark.parametrize("M,K", _IN_COHORT_CELLS)
@pytest.mark.parametrize("dtype", _FP_DTYPES)
def test_in_cohort_fires(M, K, dtype):
    """Every (M, K) cell K-725 owns in composite-13 must fire."""
    assert _is_k725_n64_tall_skinny(M, 64, K, dtype) is True


# --- K-693 OWNED cells: K-725 must NOT fire (composite-13 mutex) -----------

_K693_OWNED_CELLS = [
    (4096,  4096), (4096,  8192),
    (8192,  4096), (8192,  8192),
    (16384, 4096), (16384, 8192),
]

@pytest.mark.parametrize("M,K", _K693_OWNED_CELLS)
@pytest.mark.parametrize("dtype", _FP_DTYPES)
def test_k693_owned_cells_excluded(M, K, dtype):
    """Composite-13 mutex: K-725 must defer to K-693 persistent override on
    (M >= 4096 AND K >= 4096).  This prevents double-routing — the K-725
    early-return at the top of _matmul would otherwise pre-empt K-693."""
    assert _is_k725_n64_tall_skinny(M, 64, K, dtype) is False


# --- OUT-OF-ENVELOPE negatives (cells that regressed -> excluded) -----------

# Per the K-725 verify CSV (the cells that regressed -> outside the envelope):
_OUT_OF_ENVELOPE_CELLS = [
    (2048, 1024), (2048, 2048), (2048, 4096),  # M=2048 with K<8192 (regress)
    (4096, 1024), (4096, 2048),                # M=4096 with K<4096 (regress)
    (8192, 1024), (8192, 2048),                # M=8192 with K<4096 (regress)
]

@pytest.mark.parametrize("M,K", _OUT_OF_ENVELOPE_CELLS)
@pytest.mark.parametrize("dtype", _FP_DTYPES)
def test_out_of_envelope_excluded(M, K, dtype):
    """Cells that measured below 1.0x speedup must NOT fire the gate.

    These are the dispatch-tax-bound regime where the K-725 candidate's
    ~30 us minimum kernel time exceeds Origami's ~7-22 us baseline.
    """
    assert _is_k725_n64_tall_skinny(M, 64, K, dtype) is False


# --- OUT-OF-COHORT negatives (dtype) ----------------------------------------

@pytest.mark.parametrize("dtype", _NON_FP_DTYPES)
@pytest.mark.parametrize("M,K", [(16384, 8192), (4096, 4096)])
def test_non_fp_dtype_excluded(dtype, M, K):
    """FP32/FP64/INT dtypes must NEVER fire (untested by K-725)."""
    assert _is_k725_n64_tall_skinny(M, 64, K, dtype) is False


# --- OUT-OF-COHORT negatives (N) --------------------------------------------

@pytest.mark.parametrize("N", [1, 2, 4, 8, 16, 31, 32, 33, 48, 63, 65, 96, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("dtype", _FP_DTYPES)
def test_n_not_64_excluded(N, dtype):
    """Strict equality on N=64 — any other N misses the gate.

    The override tile is BLOCK_N=64; any other N either pads (waste) or
    leaves part of the output uncovered. N=32 belongs to K-697 (different
    tile, BM=BN=32). N>=128 was measured by K-668 to have geomean ratio
    >=0.63x, which is too good to need an override.
    """
    assert _is_k725_n64_tall_skinny(16384, N, 8192, dtype) is False


# --- BOUNDARY ON-OFF GUARDS --------------------------------------------------

def test_M_2048_K_8192_boundary_inclusive():
    """The lowest-M corner of the third disjunct: M=2048, K=8192."""
    for dtype in _FP_DTYPES:
        assert _is_k725_n64_tall_skinny(2048, 64, 8192, dtype) is True


def test_M_2048_K_4096_excluded():
    """One-K-step inside the third disjunct: K=4096 < 8192 -> excluded."""
    for dtype in _FP_DTYPES:
        assert _is_k725_n64_tall_skinny(2048, 64, 4096, dtype) is False


def test_M_4096_K_4096_excluded_in_composite13():
    """Composite-13: M=4096 K=4096 is owned by K-693 persistent path.
    K-725 standalone fired here (lowest-M corner of its second disjunct);
    composite-13 mutex defers to K-693 to avoid double-routing."""
    for dtype in _FP_DTYPES:
        assert _is_k725_n64_tall_skinny(4096, 64, 4096, dtype) is False


def test_M_4096_K_2048_excluded():
    """One-K-step inside the second disjunct: K=2048 < 4096 -> excluded."""
    for dtype in _FP_DTYPES:
        assert _is_k725_n64_tall_skinny(4096, 64, 2048, dtype) is False


def test_M_16384_K_1024_boundary_inclusive():
    """Lowest-K corner of the first disjunct: M=16384, K=1024."""
    for dtype in _FP_DTYPES:
        assert _is_k725_n64_tall_skinny(16384, 64, 1024, dtype) is True


def test_M_16383_K_1024_excluded():
    """Off-by-one M guard on the first disjunct (M < 16384, K = 1024)."""
    for dtype in _FP_DTYPES:
        assert _is_k725_n64_tall_skinny(16383, 64, 1024, dtype) is False


def test_M_2048_K_8191_excluded():
    """Off-by-one K guard on the third disjunct (K < 8192 at M = 2048)."""
    for dtype in _FP_DTYPES:
        assert _is_k725_n64_tall_skinny(2048, 64, 8191, dtype) is False


# --- ADJACENT-COHORT non-overlap --------------------------------------------

def test_no_overlap_with_K697_n32():
    """K-697 fires on N==32; K-725 must not fire on N==32 (would conflict)."""
    for dtype in _FP_DTYPES:
        for M in [2048, 4096, 8192, 16384]:
            for K in [1024, 4096, 8192]:
                assert _is_k725_n64_tall_skinny(M, 32, K, dtype) is False


def test_no_overlap_with_K644_P1():
    """K-644 P1 vetoes M<=8 AND K>=4096. K-725 requires M>=2048 (>= 4096
    in second disjunct), so no overlap is possible."""
    for dtype in _FP_DTYPES:
        for M in [1, 4, 8]:
            for K in [4096, 8192, 16384]:
                assert _is_k725_n64_tall_skinny(M, 64, K, dtype) is False


def test_no_overlap_with_K644_P2_square_large():
    """K-644 P2 fires on M==N>=5120 square. K-725 requires N==64 strict,
    so cannot fire on any square shape with M>=5120."""
    for dtype in _FP_DTYPES:
        for d in [5120, 6144, 8192, 16384]:
            assert _is_k725_n64_tall_skinny(d, d, max(d, 6144), dtype) is False


def test_no_overlap_with_K278():
    """K-278 covers M in [64, 256], N >= 4096. K-725 requires M>=2048."""
    for dtype in _FP_DTYPES:
        for M in [64, 128, 256]:
            for N in [4096, 8192, 16384]:
                assert _is_k725_n64_tall_skinny(M, N, 4096, dtype) is False


def test_no_overlap_with_K353():
    """K-353 covers M in [128, 512], N in [256, 1024]. K-725 requires
    N==64 and M>=2048."""
    for dtype in _FP_DTYPES:
        for M in [128, 256, 512]:
            for N in [256, 512, 1024]:
                for K in [16384, 32768]:
                    assert _is_k725_n64_tall_skinny(M, N, K, dtype) is False


def test_no_overlap_with_K667_large_K_square():
    """K-667 covers M==N==2048, K in {4096, 8192, 16384}. Square -> N != 64."""
    for dtype in _FP_DTYPES:
        for K in [4096, 8192, 16384]:
            assert _is_k725_n64_tall_skinny(2048, 2048, K, dtype) is False


# ----------------------------------------------------------------------------
# Layer 2 — end-to-end dispatch tests (skip without CUDA)
# ----------------------------------------------------------------------------

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="K-725 dispatch override is GPU-only; CPU CI skips Layer 2.",
)


@cuda_required
@pytest.mark.parametrize("dtype", _FP_DTYPES)
def test_dispatch_in_cohort_matches_torch(dtype):
    """The K-725 override must produce numerically-equivalent output to
    torch.matmul on a representative in-cohort shape."""
    import tritonblas
    M, N, K = 4096, 64, 4096
    a = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    b = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
    ref = torch.matmul(a, b)
    out = tritonblas.matmul(a, b)
    # FP16/BF16 GEMM at K=4096 with stochastic accumulation order — generous
    # absolute tolerance, tighter relative.
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=5e-2)


@cuda_required
def test_dispatch_out_arg_in_cohort_matches_torch():
    """Same correctness check via the out= path -> _matmul_out."""
    import tritonblas
    M, N, K = 16384, 64, 8192
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    ref = torch.matmul(a, b)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    tritonblas.matmul(a, b, out=out)
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=5e-2)


@cuda_required
def test_dispatch_out_of_envelope_unchanged():
    """A K-725-cohort shape that misses the tightened envelope (M=2048
    K=4096) must take the default Origami path and remain numerically
    equivalent."""
    import tritonblas
    M, N, K = 2048, 64, 4096  # gate misses (third disjunct needs K>=8192 at M=2048)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    ref = torch.matmul(a, b)
    out = tritonblas.matmul(a, b)
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=5e-2)


@cuda_required
def test_dispatch_n32_not_k725_path():
    """N=32 in-cohort shape goes through K-697, not K-725 (different tile)."""
    import tritonblas
    M, N, K = 4096, 32, 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
    ref = torch.matmul(a, b)
    out = tritonblas.matmul(a, b)
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=5e-2)


@cuda_required
def test_dispatch_work_stealing_path_unchanged():
    """work_stealing=True must skip the K-725 override (the override does
    not allocate the WS counters/locks)."""
    import tritonblas
    from tritonblas.matmul import _matmul
    M, N, K = 4096, 64, 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    ref = torch.matmul(a, b)
    out = _matmul(a, b, enable_streamk=False, sk_grid=None, work_stealing=True)
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=5e-2)
