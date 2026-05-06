"""
Correctness regression guards for the large-K MFMA-utilisation fix series.

These tests cover:

1. The auto-streamk promotion path triggered by ``_maybe_promote_to_streamk``
   for tail-wave-bad and under-subscribed-with-K-work shapes — checks that
   the auto-promoted dispatch produces numerically equivalent output to a
   torch.matmul reference.

2. The EVEN_M / EVEN_N constexpr fast path in ``Tile.layout()`` — checks
   that the cleanly-tiled kernel still produces correct output (the fast
   path elides the bounds mask, so any off-by-one would surface as a
   wrong column/row of garbage).

3. The 192-tile addition to ``OrigamiMatmulSelector._block_mn_range`` —
   checks one shape that should now be reachable with a 192-wide tile.

The shapes here are intentionally small (so they run fast in CI) but
hit each of the dispatch branches exercised by the four-commit series.
Larger shapes are validated via the benchmark harness in the workspace.
"""

import pytest
import torch

import tritonblas


# (M, N, K, dtype): each shape exercises a specific code path added by
# the large-K MFMA fix series.
_LARGE_K_SHAPES = [
    # Tail-wave-bad: tiles>N_CU, partial last wave triggers auto-streamk.
    # 2048x12288 with 256x256 tiles = 384 tiles; on MI300X N_CU=304 so
    # last_wave = 80 < 0.6*304 = 182 -> promoted.
    pytest.param(2048, 12288, 12288, torch.bfloat16,
                 id="tail_wave_bad_bf16_2k_12k_12k"),
    # Under-subscribed with real K work: tiles<N_CU, iters_per_tile>=32.
    # 1024x1024 = 16 tiles, K=4096/64 = 64 iters_per_tile -> eligible.
    pytest.param(1024, 1024, 4096, torch.float16,
                 id="under_subscribed_fp16_1k_1k_4k"),
    # 192-tile reachable: 2048 = 192*~10.7 (selector may pick 192-wide).
    pytest.param(2048, 12288, 4096, torch.bfloat16,
                 id="tile_192_reachable_bf16_2k_12k_4k"),
    # EVEN_M / EVEN_N fast path: M and N exactly divisible by tile.
    pytest.param(4096, 4096, 4096, torch.bfloat16,
                 id="even_mn_fast_path_bf16_4k_4k_4k"),
    pytest.param(4096, 4096, 4096, torch.float16,
                 id="even_mn_fast_path_fp16_4k_4k_4k"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs ROCm GPU")
@pytest.mark.parametrize("M, N, K, dtype", _LARGE_K_SHAPES)
def test_large_k_auto_dispatch_correctness(M, N, K, dtype):
    """The default tritonblas.matmul dispatch (which now invokes
    ``_maybe_promote_to_streamk`` and the EVEN_M/EVEN_N fast path) must
    produce numerically equivalent output to torch.matmul / hipBLASLt.
    """
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    b = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1

    ref = torch.matmul(a, b)
    out = tritonblas.matmul(a, b)

    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    # bf16/fp16 GEMM on K=4k..12k: per-element relative error around 1e-2;
    # use an absolute tolerance scaled by the expected magnitude so the
    # test catches dispatch / mask bugs (which produce orders-of-magnitude
    # errors) without being noisy on the bottom bits.
    atol = 0.05 if dtype is torch.float16 else 0.1
    rtol = 0.05 if dtype is torch.float16 else 0.1
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs ROCm GPU")
def test_streamk_promotion_picks_streamk_for_tail_wave_bad():
    """Verify that the new auto-promotion helper does in fact route the
    canonical tail-wave-bad shape (2048x12288x12288 bf16) through the
    Stream-K kernel, not the persistent kernel.  This is a structural
    guard: if the heuristic stops firing for the shape that motivated
    it, the perf win silently regresses.
    """
    from tritonblas.matmul import _maybe_promote_to_streamk, _make_matmul_selector

    M, N, K = 2048, 12288, 12288
    a = torch.empty(1, device="cuda", dtype=torch.bfloat16)
    sel_p = _make_matmul_selector(M, N, K, a.dtype, a.dtype, a.dtype, a.device,
                                  streamk=False)
    sel_sk = _maybe_promote_to_streamk(M, N, K, a.dtype, a.dtype, a.dtype,
                                       a.device, sel_p)
    assert sel_sk is not None, (
        "tail-wave-bad shape (2048x12288x12288 bf16) must auto-promote "
        "to Stream-K; got persistent dispatch"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs ROCm GPU")
def test_streamk_promotion_skips_well_subscribed_small_k():
    """Verify the auto-promotion helper does NOT route well-subscribed
    small-K shapes through Stream-K, where the reduction overhead would
    swamp any gain.  1024x1024x1024 bf16 has only 16 tiles and K=1024
    (16 iters_per_tile of 64) — below the iters_per_tile floor of 32.
    """
    from tritonblas.matmul import _maybe_promote_to_streamk, _make_matmul_selector

    M, N, K = 1024, 1024, 1024
    a = torch.empty(1, device="cuda", dtype=torch.bfloat16)
    sel_p = _make_matmul_selector(M, N, K, a.dtype, a.dtype, a.dtype, a.device,
                                  streamk=False)
    sel_sk = _maybe_promote_to_streamk(M, N, K, a.dtype, a.dtype, a.dtype,
                                       a.device, sel_p)
    # iters_per_tile is small, so promotion should bail out (return None).
    assert sel_sk is None, (
        "small-K shape (1024^3 bf16, K=1024) should NOT auto-promote to "
        "Stream-K; reduction overhead would swamp any gain"
    )
