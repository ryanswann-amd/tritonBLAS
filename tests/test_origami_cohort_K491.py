"""K-491: regression guard for the medium-K square FP16/BF16 cohort on MI300X.

Empirical sweep on MI300X (12 in-cohort shapes,
M=N in {1024,2048,4096}, K in {256,512}, dtype in {fp16,bf16}) showed:

  * `OrigamiMatmulSelector` already picks the empirically-best BM/BN per
    shape: BM=BN=64 for M=1024, BM=BN=128 for M=2048, BM=BN=256 for M=4096.
  * Forcing the K-442-derived BM=BN=256 winner on M<4096 regresses 2048^2
    kernel time by ~17% and 1024^2 by ~5%; the K-442 result was a single-
    tile sweep, not a multi-tile dispatch.
  * In-cohort hipBLASLt ratio geomean: 0.59 (cohort gap is real but is NOT
    closed by tile changes — it is rooted elsewhere; see PR description).

Conclusion: the production fix is to leave Origami in charge. This test
exists to (a) lock the empirically-best per-shape tiles so any future
Origami heuristic drift on this exact cohort fails CI, (b) cover the
negative paths (fp32 / non-square / out-of-range K must NOT be coerced
into the cohort tile), and (c) verify numerical correctness of one
in-cohort shape vs torch.matmul. The test is plain regression scaffolding
— it does NOT exercise an override (none exists in production).
"""

from __future__ import annotations

import pytest
import torch

from tritonblas.matmul import _make_matmul_selector
from tritonblas.origami import OrigamiMatmulSelector


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="K-491 cohort guard requires a CUDA/ROCm device for OrigamiMatmulSelector",
)


# Empirically validated MI300X selections (c42, May 2026 sweep):
#   M=N=1024 -> BM=BN=64
#   M=N=2048 -> BM=BN=128
#   M=N=4096 -> BM=BN=256
# Off-cohort tiles (forcing BM=BN=256 for M=N=2048) regress ~17%; do not change
# without re-running scripts/sweep61_kernel.py.
_EXPECTED_BM = {1024: 64, 2048: 128, 4096: 256}
_COHORT_DTYPES = (torch.float16, torch.bfloat16)
_COHORT_K = (256, 512)


def _make_selector(M, N, K, dtype):
    return OrigamiMatmulSelector(
        M, N, K, dtype, dtype, dtype, torch.device("cuda:0"),
    )


# ----------------------------------------------------------------------------
# Positive: cohort shapes get the empirically-best per-shape tile from Origami.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("M", [1024, 2048, 4096])
@pytest.mark.parametrize("K", _COHORT_K)
@pytest.mark.parametrize("dtype", _COHORT_DTYPES)
def test_origami_picks_expected_tile_for_cohort(M, K, dtype):
    sel = _make_selector(M, M, K, dtype)
    expected = _EXPECTED_BM[M]
    assert sel.block_m == expected, (
        f"K-491 regression: Origami picked BM={sel.block_m} for M={M}, K={K}, "
        f"dtype={dtype}; the May-2026 c42 MI300X sweep showed BM={expected} is "
        f"empirically fastest. If Origami's heuristic intentionally moved, "
        f"re-run scripts/sweep61_kernel.py and update _EXPECTED_BM."
    )
    assert sel.block_n == expected, (
        f"K-491 regression: Origami picked BN={sel.block_n} (expected {expected})"
    )
    # BK is delegated to Origami (BK=64/128 both occur in-cohort and are within
    # ~1% of each other on MI300X). We only assert it is a power-of-two divisor
    # of K, the Origami invariant.
    assert sel.block_k > 0
    assert (sel.block_k & (sel.block_k - 1)) == 0, (
        f"K-491: BK={sel.block_k} is not a power of two for M={M}, K={K}"
    )
    assert K % sel.block_k == 0, (
        f"K-491: BK={sel.block_k} does not divide K={K}"
    )


# ----------------------------------------------------------------------------
# Negative / over-fire guards: the production wrapper `_make_matmul_selector`
# must NOT transform Origami's choice — for any input. The strongest possible
# guard is: build the selector both directly and through the wrapper, and
# assert the tile triple matches. Any future override (cohort-targeted or
# otherwise) breaks these tests immediately.
# ----------------------------------------------------------------------------


def _wrapper_matches_origami(M, N, K, dtype):
    direct = _make_selector(M, N, K, dtype)
    wrapped = _make_matmul_selector(
        M, N, K, dtype, dtype, dtype, torch.device("cuda:0"),
    )
    return (
        (direct.block_m, direct.block_n, direct.block_k)
        == (wrapped.block_m, wrapped.block_n, wrapped.block_k)
    )


@pytest.mark.parametrize("K", _COHORT_K)
def test_wrapper_no_override_on_fp32(K):
    """fp32 inputs: production wrapper must not coerce tiles."""
    assert _wrapper_matches_origami(2048, 2048, K, torch.float32), (
        "K-491 over-fire guard: _make_matmul_selector must not transform "
        "Origami's choice for fp32 inputs. A divergent tile triple proves an "
        "out-of-scope override was introduced."
    )


@pytest.mark.parametrize("M,N", [(1024, 2048), (2048, 1024), (2048, 4096), (4096, 2048)])
def test_wrapper_no_override_on_nonsquare(M, N):
    """Non-square shapes: production wrapper must not coerce tiles."""
    assert _wrapper_matches_origami(M, N, 256, torch.float16), (
        f"K-491 over-fire guard: _make_matmul_selector must not transform "
        f"Origami's choice for non-square shape M={M}, N={N}."
    )


@pytest.mark.parametrize("K", [64, 128, 1024, 2048])
def test_wrapper_no_override_on_out_of_range_K(K):
    """K outside {256,512}: production wrapper must not coerce tiles."""
    assert _wrapper_matches_origami(2048, 2048, K, torch.float16), (
        f"K-491 over-fire guard: _make_matmul_selector must not transform "
        f"Origami's choice for out-of-cohort K={K}."
    )


@pytest.mark.parametrize("M", [1024, 2048, 4096])
@pytest.mark.parametrize("K", _COHORT_K)
@pytest.mark.parametrize("dtype", _COHORT_DTYPES)
def test_wrapper_no_override_in_cohort(M, K, dtype):
    """Even for in-cohort shapes the wrapper must not transform Origami's
    pick (we removed the no-op override to keep the wrapper trivial; this
    test catches anyone re-introducing one)."""
    assert _wrapper_matches_origami(M, M, K, dtype), (
        f"K-491: _make_matmul_selector must be a thin wrapper around "
        f"OrigamiMatmulSelector for cohort shape M={M}, K={K}, dtype={dtype}."
    )


# ----------------------------------------------------------------------------
# Numerical correctness: one in-cohort GEMM must match torch.matmul.
# Catches silent miscompiles even if the selector picks an unexpected tile.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _COHORT_DTYPES)
def test_in_cohort_numerical_correctness(dtype):
    """Run one in-cohort GEMM through tritonblas.matmul and compare to a
    higher-precision reference. Catches silent miscompiles independent of
    which tile Origami picks."""
    import tritonblas  # imported inside the test so collection works on CPU

    M, N, K = 2048, 2048, 256
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)

    out = tritonblas.matmul(a, b)
    assert out.shape == (M, N)
    assert out.dtype == dtype

    # fp32 reference for an honest comparison; tolerances are tuned to the
    # accumulated rounding error of K=256 fp16/bf16 matmuls on MI300X.
    ref = (a.float() @ b.float()).to(dtype)
    if dtype is torch.float16:
        atol, rtol = 5e-2, 5e-2
    else:  # bfloat16 has ~3 fewer mantissa bits
        atol, rtol = 1e-1, 1e-1
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
