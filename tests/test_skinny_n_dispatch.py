"""K-212: Tests for the skinny-N (N<=32) wave-quantization fast path.

Covers:
  * the `_resolve_skinny_n` shape predicate (no GPU required)
  * end-to-end correctness of `tritonblas.matmul` on representative skinny-N
    shapes vs `torch.matmul` (FP16, MI300X)
"""
import pytest
import torch

import tritonblas
from tritonblas.matmul import (
    _resolve_skinny_n,
    _SKINNY_N_BLK_M,
    _SKINNY_N_BLK_K,
    _SKINNY_N_NUM_WARPS,
)


# ---------------------------------------------------------------------------
# Predicate tests (no GPU)
# ---------------------------------------------------------------------------

class TestResolveSkinnyN:
    def test_typical_skinny_n_fires(self):
        # M=4096, N=8, K=8192 -- canonical skinny-N PRD shape
        recipe = _resolve_skinny_n(4096, 8, 8192, None)
        assert recipe is not None
        blk_m, blk_n, blk_k, num_warps = recipe
        assert blk_m == _SKINNY_N_BLK_M
        assert blk_n == 16  # N <= 16 -> BLK_N=16
        assert blk_k == _SKINNY_N_BLK_K
        assert num_warps == _SKINNY_N_NUM_WARPS

    def test_n_above_threshold_does_not_fire(self):
        # N=64 is above the N<=32 cutoff
        assert _resolve_skinny_n(4096, 64, 8192, None) is None
        # N=33 (just above) does not fire
        assert _resolve_skinny_n(4096, 33, 8192, None) is None

    def test_n_at_threshold_fires(self):
        # N=32 (boundary) fires with BLK_N=32
        recipe = _resolve_skinny_n(4096, 32, 8192, None)
        assert recipe is not None
        assert recipe[1] == 32  # BLK_N

    def test_n_in_17_to_32_uses_blkn_32(self):
        # N=17..32 -> BLK_N=32 (single tile in N)
        for n in (17, 24, 32):
            r = _resolve_skinny_n(4096, n, 8192, None)
            assert r is not None and r[1] == 32, f"N={n}: {r}"

    def test_n_le_16_uses_blkn_16(self):
        # N <= 16 -> BLK_N=16
        for n in (1, 2, 4, 8, 16):
            r = _resolve_skinny_n(4096, n, 8192, None)
            assert r is not None and r[1] == 16, f"N={n}: {r}"

    def test_small_m_does_not_fire(self):
        # M < 256 means there's not enough M-parallelism for StreamK to win
        assert _resolve_skinny_n(128, 8, 8192, None) is None

    def test_small_k_does_not_fire(self):
        # K < 1024 keeps the gate off launch-bound shapes
        assert _resolve_skinny_n(4096, 8, 512, None) is None

    def test_explicit_streamk_true_skips_gate(self):
        # User asked for StreamK explicitly -> respect it (do not override)
        assert _resolve_skinny_n(4096, 8, 8192, True) is None

    def test_explicit_streamk_false_skips_gate(self):
        # User asked for persistent explicitly -> respect it (do not override)
        assert _resolve_skinny_n(4096, 8, 8192, False) is None

    def test_zero_or_negative_dims_do_not_fire(self):
        assert _resolve_skinny_n(0, 8, 8192, None) is None
        assert _resolve_skinny_n(4096, 0, 8192, None) is None


# ---------------------------------------------------------------------------
# End-to-end correctness (GPU required)
# ---------------------------------------------------------------------------

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for skinny-N e2e correctness",
)


def _close(out, ref, atol=0.5, rtol=0.05):
    return torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)


@requires_gpu
@pytest.mark.parametrize(
    "M,N,K",
    [
        (2048, 1, 4096),
        (4096, 8, 8192),
        (8192, 16, 16384),
        (4096, 32, 8192),
        (2048, 17, 4096),  # N just above 16, exercises BLK_N=32 branch
    ],
)
def test_skinny_n_matmul_correctness(M, N, K):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    out = tritonblas.matmul(a, b)
    ref = torch.matmul(a, b)
    assert out.shape == (M, N)
    assert out.dtype == torch.float16
    assert _close(out, ref), \
        f"max-abs-err={(out.float()-ref.float()).abs().max().item():.3f}"


@requires_gpu
@pytest.mark.parametrize("N", [1, 8, 16, 32])
def test_skinny_n_matmul_out_correctness(N):
    """The out= entrypoint should also take the skinny-N path."""
    torch.manual_seed(1)
    M, K = 4096, 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    out = torch.empty(M, N, device="cuda", dtype=torch.float16)
    tritonblas.matmul(a, b, out=out)
    ref = torch.matmul(a, b)
    assert _close(out, ref), \
        f"N={N}: max-abs-err={(out.float()-ref.float()).abs().max().item():.3f}"


@requires_gpu
def test_explicit_persistent_does_not_take_skinny_path():
    """Passing enable_streamk=False must hit the persistent kernel
    (the gate must respect explicit user intent)."""
    M, N, K = 4096, 8, 8192
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    out = tritonblas.matmul(a, b, enable_streamk=False)
    ref = torch.matmul(a, b)
    assert _close(out, ref)


@requires_gpu
def test_large_n_unaffected():
    """Shapes with N > 32 must not take the skinny-N path -- regression
    guard for the K-654 61-shape sweep."""
    M, N, K = 8192, 8192, 8192
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    out = tritonblas.matmul(a, b)
    ref = torch.matmul(a, b)
    assert _close(out, ref, atol=2.0, rtol=0.05), \
        f"max-abs-err={(out.float()-ref.float()).abs().max().item():.3f}"
