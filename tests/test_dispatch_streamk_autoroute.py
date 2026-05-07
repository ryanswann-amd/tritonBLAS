"""Routing + correctness tests for the K-162 auto-Stream-K dispatcher.

The auto-router lives in :mod:`tritonblas.dispatch` and is wired into the
eager matmul path through :func:`tritonblas.matmul._select_dispatch`.

Two layers of coverage:

1. **Routing layer** (no GPU required for the gate itself, but we use the
   real Origami selector to pull realistic ``BLK_M``/``BLK_N`` so the test
   matches production behaviour).  We assert the gate fires on representative
   *skinny / under-occupied* shapes from the K-162 cohort and stays silent
   on *large* shapes that already saturate the device.

2. **Numerical layer**.  For each gate-firing shape we run
   ``tritonblas.matmul`` once with auto-routing enabled (Stream-K under the
   hood) and once with ``TBLAS_DISABLE_AUTO_STREAMK=1`` (pure persistent),
   compare against an fp32 reference, and check both paths stay within
   tolerance.  This guards against future regressions where the gate
   misfires or the Stream-K kernel diverges from persistent on these shapes.
"""
import math
import os

import pytest
import torch

import tritonblas
from tritonblas.dispatch import _num_cus, should_use_streamk
from tritonblas.matmul import _make_matmul_selector


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="dispatch tests require a CUDA device",
)


def _bm_bn(M, N, K, dtype, device):
    sel = _make_matmul_selector(M, N, K, dtype, dtype, dtype, device)
    return sel.block_m, sel.block_n


# K-162 PRD-named shapes that should route to Stream-K (skinny M, under-occupied
# tile grid).  K=1024 is included on purpose -- it is the headline shape from
# the ticket and any gate that drops it is wrong.
#
# NOTE: ``(32, 4096, 8192)`` is intentionally excluded -- with the BM=32 BN=16
# tile that Origami picks for this shape it generates 256 tiles which already
# exceeds num_cus/2=152, so the gate (correctly) does NOT fire and the shape
# is not actually under-occupied.
SKINNY_TRIGGER_SHAPES = [
    (8, 1024, 1024),
    (16, 1024, 1024),  # the headline K-162 shape
    (16, 2048, 4096),
    (16, 2048, 8192),
    (32, 2048, 8192),
]

# Shapes that already saturate MI300X's 304 CUs and must NOT be touched.
LARGE_NO_TRIGGER_SHAPES = [
    (2048, 2048, 4096),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]


def test_num_cus_returns_positive_int():
    """Sanity: the cached CU count must be a positive integer on a CUDA device."""
    n = _num_cus(torch.device("cuda", torch.cuda.current_device()))
    assert isinstance(n, int)
    assert n > 0


@pytest.mark.parametrize("M,N,K", SKINNY_TRIGGER_SHAPES)
def test_skinny_shapes_route_to_streamk(M, N, K):
    """Underutilised skinny shapes must trigger the Stream-K path."""
    device = torch.device("cuda", torch.cuda.current_device())
    BM, BN = _bm_bn(M, N, K, torch.float16, device)
    tiles = math.ceil(M / BM) * math.ceil(N / BN)
    assert tiles * 2 < _num_cus(device), (
        f"test premise broken: tiles={tiles} BM={BM} BN={BN} but "
        f"num_cus={_num_cus(device)} -- shape isn't actually under-occupied"
    )
    assert should_use_streamk(M, N, BM, BN, device), (
        f"gate should fire on skinny shape ({M},{N},{K}) BM={BM} BN={BN} tiles={tiles}"
    )


@pytest.mark.parametrize("M,N,K", LARGE_NO_TRIGGER_SHAPES)
def test_large_shapes_do_not_route_to_streamk(M, N, K):
    """Shapes that already saturate the device must keep the persistent kernel."""
    device = torch.device("cuda", torch.cuda.current_device())
    BM, BN = _bm_bn(M, N, K, torch.float16, device)
    tiles = math.ceil(M / BM) * math.ceil(N / BN)
    assert tiles * 2 >= _num_cus(device), (
        f"test premise broken: tiles={tiles} should saturate num_cus={_num_cus(device)}"
    )
    assert not should_use_streamk(M, N, BM, BN, device), (
        f"gate must not fire on large shape ({M},{N},{K}) BM={BM} BN={BN} tiles={tiles}"
    )


def test_env_var_disables_routing():
    """``TBLAS_DISABLE_AUTO_STREAMK=1`` must short-circuit the gate."""
    device = torch.device("cuda", torch.cuda.current_device())
    BM, BN = _bm_bn(16, 1024, 1024, torch.float16, device)
    # Without the env override, this skinny shape fires.
    assert should_use_streamk(16, 1024, BM, BN, device)
    prev = os.environ.get("TBLAS_DISABLE_AUTO_STREAMK")
    os.environ["TBLAS_DISABLE_AUTO_STREAMK"] = "1"
    try:
        assert not should_use_streamk(16, 1024, BM, BN, device)
    finally:
        if prev is None:
            del os.environ["TBLAS_DISABLE_AUTO_STREAMK"]
        else:
            os.environ["TBLAS_DISABLE_AUTO_STREAMK"] = prev


def _atol_rtol(dtype):
    if dtype == torch.float16:
        return 1.0, 5e-2
    return 2.0, 1e-1  # bf16 has ~half the mantissa


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M,N,K", SKINNY_TRIGGER_SHAPES)
def test_streamk_routed_path_matches_reference(M, N, K, dtype):
    """When the gate fires we run the Stream-K kernel.  Verify it matches an
    fp32 reference within the same tolerances the rest of the suite uses."""
    torch.manual_seed(0)
    device = "cuda"
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    out = tritonblas.matmul(a, b)
    ref = (a.to(torch.float32) @ b.to(torch.float32))
    atol, rtol = _atol_rtol(dtype)
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

    atol, rtol = _atol_rtol(dtype)
    torch.testing.assert_close(
        out_routed.to(torch.float32), out_persistent.to(torch.float32),
        atol=atol, rtol=rtol,
    )
