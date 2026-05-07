"""Routing + correctness tests for the K-162 auto-Stream-K dispatcher.

The auto-router lives in :mod:`tritonblas.dispatch` and is wired into the
eager matmul path through :func:`tritonblas.matmul._select_dispatch`.

Two layers of coverage:

1. **Routing layer**.  Asserts the gate fires on representative
   *under-occupied K>=8192 / N>=1024* shapes from the K-162 cohort and
   stays silent on (a) large shapes that already saturate the device,
   (b) low-K shapes where Stream-K's atomic reduction outweighs the
   occupancy gain, and (c) the headline (M=16, N=1024, K=1024) PRD shape
   which is documented as a residual case (see ``dispatch.py`` module
   docstring for the empirical justification).

2. **Numerical layer**.  For each gate-firing shape we run
   ``tritonblas.matmul`` once with auto-routing enabled (Stream-K under
   the hood) and once with ``TBLAS_DISABLE_AUTO_STREAMK=1`` (pure
   persistent), compare against an fp32 reference, and check both paths
   stay within tolerance.  This guards against future regressions where
   the gate misfires or the Stream-K kernel diverges from persistent.
"""
import math
import os

import pytest
import torch

import tritonblas
from tritonblas.dispatch import _MIN_K, _MIN_N, _num_cus, should_use_streamk
from tritonblas.matmul import _make_matmul_selector


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="dispatch tests require a CUDA device",
)


def _bm_bn(M, N, K, dtype, device):
    sel = _make_matmul_selector(M, N, K, dtype, dtype, dtype, device)
    return sel.block_m, sel.block_n


# Skinny shapes that satisfy all three gate conditions
# (under-occupied + K >= _MIN_K + N >= _MIN_N).
SKINNY_TRIGGER_SHAPES = [
    (8, 1024, 8192),
    (8, 2048, 8192),
    (16, 1024, 8192),
    (16, 2048, 8192),
    (32, 1024, 8192),
    (32, 2048, 8192),
]

# Shapes that fail at least one secondary condition and must NOT trigger:
#   - (16, 1024, 1024): the K-162 PRD's named shape, but K < _MIN_K -- the
#     atomic reduction overhead beats the persistent kernel's idle CUs.
#   - (16, 1024, 4096): same reason, K < _MIN_K.
#   - (16, 512,  8192): N < _MIN_N -- sk grid can't fill the device.
#   - (2048, 2048, 4096), (4096, 4096, 4096), (8192, 8192, 8192):
#     persistent kernel already saturates the device.
NO_TRIGGER_SHAPES = [
    (16, 1024, 1024),
    (16, 1024, 4096),
    (16, 512, 8192),
    (2048, 2048, 4096),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]


def test_num_cus_returns_positive_int():
    n = _num_cus(torch.device("cuda", torch.cuda.current_device()))
    assert isinstance(n, int)
    assert n > 0


def test_module_constants_match_dispatch_logic():
    """Sanity: the module-level constants exist and are positive ints --
    a typo here would make the gate pass everything."""
    assert isinstance(_MIN_K, int) and _MIN_K > 0
    assert isinstance(_MIN_N, int) and _MIN_N > 0


@pytest.mark.parametrize("M,N,K", SKINNY_TRIGGER_SHAPES)
def test_skinny_shapes_route_to_streamk(M, N, K):
    """Under-occupied shapes that meet all three gate conditions must fire."""
    device = torch.device("cuda", torch.cuda.current_device())
    BM, BN = _bm_bn(M, N, K, torch.float16, device)
    tiles = math.ceil(M / BM) * math.ceil(N / BN)
    assert tiles * 2 < _num_cus(device), (
        f"test premise broken: tiles={tiles} BM={BM} BN={BN} but "
        f"num_cus={_num_cus(device)} -- shape isn't under-occupied"
    )
    assert K >= _MIN_K and N >= _MIN_N, (
        f"test premise broken: gate would refuse ({M},{N},{K}) on its "
        f"K/N floors before reaching the tile check"
    )
    assert should_use_streamk(M, N, K, BM, BN, device), (
        f"gate should fire on ({M},{N},{K}) BM={BM} BN={BN} tiles={tiles}"
    )


@pytest.mark.parametrize("M,N,K", NO_TRIGGER_SHAPES)
def test_no_trigger_shapes_stay_persistent(M, N, K):
    """Saturated shapes, low-K shapes, and tiny-N shapes must NOT fire."""
    device = torch.device("cuda", torch.cuda.current_device())
    BM, BN = _bm_bn(M, N, K, torch.float16, device)
    assert not should_use_streamk(M, N, K, BM, BN, device), (
        f"gate must not fire on ({M},{N},{K}) BM={BM} BN={BN}"
    )


def test_env_var_disables_routing():
    """``TBLAS_DISABLE_AUTO_STREAMK=1`` short-circuits the gate."""
    device = torch.device("cuda", torch.cuda.current_device())
    M, N, K = 16, 2048, 8192
    BM, BN = _bm_bn(M, N, K, torch.float16, device)
    assert should_use_streamk(M, N, K, BM, BN, device)
    prev = os.environ.get("TBLAS_DISABLE_AUTO_STREAMK")
    os.environ["TBLAS_DISABLE_AUTO_STREAMK"] = "1"
    try:
        assert not should_use_streamk(M, N, K, BM, BN, device)
    finally:
        if prev is None:
            del os.environ["TBLAS_DISABLE_AUTO_STREAMK"]
        else:
            os.environ["TBLAS_DISABLE_AUTO_STREAMK"] = prev


def _atol_rtol(dtype, K):
    """fp16 / bf16 GEMM error bound, K-scaled.

    Per-element forward error in a K-element dot product accumulated in
    low precision is bounded by ~ ``K * eps * max(|a|) * max(|b|)``.
    For inputs drawn from ``N(0, 1)``:

        fp16 (eps ≈ 9.8e-4):   atol ≈ K * 1e-3,   rtol ≈ 1e-2
        bf16 (eps ≈ 7.8e-3):   atol ≈ K * 8e-3,   rtol ≈ 2e-2

    These are tight enough that a wrong tile / wrong kernel routing /
    single flipped reduction will fail the assertion (a ~50 % output
    error on K=8192 would be ~45 in absolute, vs the bound of ~8).
    """
    if dtype == torch.float16:
        return max(1e-2, K * 1e-3), 1e-2
    return max(1e-2, K * 8e-3), 2e-2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M,N,K", SKINNY_TRIGGER_SHAPES)
def test_streamk_routed_path_matches_reference(M, N, K, dtype):
    """When the gate fires we run the Stream-K kernel.  Verify it matches
    an fp32 reference within the same tolerances the rest of the suite
    uses."""
    torch.manual_seed(0)
    device = "cuda"
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    out = tritonblas.matmul(a, b)
    ref = a.to(torch.float32) @ b.to(torch.float32)
    atol, rtol = _atol_rtol(dtype, K)
    torch.testing.assert_close(out.to(torch.float32), ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M,N,K", SKINNY_TRIGGER_SHAPES)
def test_persistent_baseline_matches_streamk_routed(M, N, K, dtype):
    """The auto-routed Stream-K result must agree with the persistent
    baseline (env-disabled) within fp16/bf16 noise -- so the dispatcher
    can never silently degrade numerics by flipping kernels."""
    torch.manual_seed(0)
    device = "cuda"
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)

    out_routed = tritonblas.matmul(a, b)

    prev = os.environ.get("TBLAS_DISABLE_AUTO_STREAMK")
    os.environ["TBLAS_DISABLE_AUTO_STREAMK"] = "1"
    try:
        out_persistent = tritonblas.matmul(a, b)
    finally:
        if prev is None:
            del os.environ["TBLAS_DISABLE_AUTO_STREAMK"]
        else:
            os.environ["TBLAS_DISABLE_AUTO_STREAMK"] = prev

    atol, rtol = _atol_rtol(dtype, K)
    torch.testing.assert_close(
        out_routed.to(torch.float32), out_persistent.to(torch.float32),
        atol=atol, rtol=rtol,
    )
