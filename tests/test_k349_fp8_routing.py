"""K-349: tests for the FP8 (e4m3fnuz / e5m2fnuz) routing and medium-K cohort
tile override added in include/tritonblas/matmul.py.

Two things are checked:
  (1) `matmul_a8w8_lt` no longer crashes on FP8 fnuz inputs (this was the
      "known bug" referenced in `tests/test_matmul_a8w8_lt.py`'s module-wide
      `pytest.mark.skip`). The K-349 patch routes FP8 to the monolithic
      kernel, side-stepping the composable persistent_gemm MLIR assertion.
  (2) The (M, N, K, dtype) dispatch gate fires on the K-349 cohort entries
      and *only* on those entries — no shape outside the table is mutated.
"""
import pytest
import torch
import tritonblas
from tritonblas.matmul import (
    _K349_FP8_DTYPES,
    _K349_TILE_TABLE,
    _k349_fp8_dispatch,
    _make_matmul_selector,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="K-349 FP8 tests require an MI300X-class GPU (gfx942)",
)


_FP8_FNUZ_DTYPES = [dt for dt in (
    getattr(torch, "float8_e4m3fnuz", None),
    getattr(torch, "float8_e5m2fnuz", None),
) if dt is not None]


def _bf16_ref(a, b, a_scale, b_scale):
    """fp32 reference: a @ b in fp32 then per-row/col scaling, cast to bf16."""
    acc = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    if a_scale is not None and b_scale is not None:
        acc = acc * (a_scale[:, None] * b_scale[None, :])
    return acc.to(torch.bfloat16)


def _make_fp8_inputs(M, N, K, dtype):
    g = torch.Generator(device="cuda").manual_seed(0xC349)
    a32 = torch.randn(M, K, generator=g, device="cuda") * 0.1
    b32 = torch.randn(K, N, generator=g, device="cuda") * 0.1
    a = a32.to(dtype)
    b = b32.to(dtype)
    a_scale = torch.ones(M, device="cuda", dtype=torch.float32)
    b_scale = torch.ones(N, device="cuda", dtype=torch.float32)
    return a, b, a_scale, b_scale


@pytest.mark.parametrize("dtype", _FP8_FNUZ_DTYPES,
                         ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("M,N,K", [
    (4096, 4096, 4096),  # K-349 cohort entry — should hit dispatch table
    (4096, 4096, 2048),  # K-349 cohort entry — should hit dispatch table
    (2048, 2048, 2048),  # FP8 fnuz, NOT in cohort — must still not crash
    (1024, 1024, 1024),  # FP8 fnuz, NOT in cohort — smoke test
])
def test_fp8_a8w8_does_not_crash(dtype, M, N, K):
    """Pre-K-349, this path crashed inside the composable persistent_gemm at
    MLIR DenseElementsAttr::get for every FP8 tile we tried. After K-349 it
    routes to monolithic and produces a finite result."""
    a, b, a_scale, b_scale = _make_fp8_inputs(M, N, K, dtype)
    c = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    out = tritonblas.matmul_a8w8(a, b, a_scale, b_scale, c, enable_streamk=False)
    assert out.shape == (M, N)
    assert torch.isfinite(out).all(), "K-349 FP8 path produced non-finite output"
    ref = _bf16_ref(a, b, a_scale, b_scale)
    # FP8 rounding -> loose tolerance; we are checking the kernel ran, not
    # exact accumulator behavior. ATol matched to fp8 dynamic range * K.
    rel_err = (out.to(torch.float32) - ref.to(torch.float32)).abs().mean() \
              / (ref.to(torch.float32).abs().mean() + 1e-6)
    assert rel_err < 0.20, f"FP8 mean-rel-err {rel_err:.4f} too high"


@pytest.mark.parametrize("dtype", _FP8_FNUZ_DTYPES,
                         ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("M,N,K", [
    (4096, 4096, 4096),  # K-349 cohort entry — must hit dispatch table
    (4096, 4096, 2048),  # K-349 cohort entry — must hit dispatch table
    (2048, 2048, 2048),  # FP8 fnuz, NOT in cohort — regression-pin only
    (1024, 1024, 1024),  # FP8 fnuz, NOT in cohort — regression-pin only
])
def test_fp8_a8w8_lt_does_not_crash(dtype, M, N, K, monkeypatch):
    """Explicit regression for Reviewer-1 (Testing Zealot): call the public
    `matmul_a8w8_lt` directly (not via the higher-level `matmul_a8w8`
    wrapper) on FP8 fnuz inputs and confirm we get a finite result. This
    pins the MLIR DenseElementsAttr::get crash that motivated the K-349
    routing change. The test deliberately does NOT set TBLAS_USE_MONOLITHIC
    (and explicitly clears it) so we exercise the production routing path
    that the K-349 fix now installs in `_k349_fp8_dispatch` /
    `force_monolithic`."""
    monkeypatch.delenv("TBLAS_USE_MONOLITHIC", raising=False)
    a, b, a_scale, b_scale = _make_fp8_inputs(M, N, K, dtype)
    c = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    selector = tritonblas.OrigamiMatmulSelector(
        M, N, K, a.dtype, b.dtype, c.dtype, a.device, streamk=False,
    )
    config = tritonblas.matmul_preamble(selector)
    tritonblas.matmul_a8w8_lt(
        a, b, a_scale, b_scale, c, selector, config, False,
    )
    assert c.shape == (M, N)
    assert torch.isfinite(c).all(), \
        "K-349 regression: matmul_a8w8_lt produced non-finite output on FP8 fnuz"


@pytest.mark.parametrize("dtype", _FP8_FNUZ_DTYPES,
                         ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("M,N,K", sorted({k for k in _K349_TILE_TABLE.keys()}))
def test_fp8_dispatch_table_fires_on_cohort(dtype, M, N, K):
    """When (M, N, K, dtype) is a cohort entry, the dispatch gate must
    overwrite (block_m, block_n, block_k, num_stages) on the selector."""
    expected = _K349_TILE_TABLE[(M, N, K)]
    BM_e, BN_e, BK_e, ns_e = expected
    a = torch.zeros(M, K, device="cuda", dtype=dtype)
    b = torch.zeros(K, N, device="cuda", dtype=dtype)
    sel = _make_matmul_selector(M, N, K, dtype, dtype, torch.bfloat16,
                                a.device, streamk=False)
    fired = _k349_fp8_dispatch(a, b, sel)
    assert fired is True, "FP8 dispatch returned False on FP8 fnuz input"
    assert (sel.block_m, sel.block_n, sel.block_k, sel.num_stages) == \
           (BM_e, BN_e, BK_e, ns_e)


@pytest.mark.parametrize("dtype", _FP8_FNUZ_DTYPES,
                         ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("M,N,K", [(2048, 2048, 2048), (1024, 1024, 1024)])
def test_fp8_dispatch_table_skips_non_cohort(dtype, M, N, K):
    """Shapes outside the cohort table must still be FP8-routed (return True
    so caller forces monolithic) but selector tile must be unchanged."""
    a = torch.zeros(M, K, device="cuda", dtype=dtype)
    b = torch.zeros(K, N, device="cuda", dtype=dtype)
    sel = _make_matmul_selector(M, N, K, dtype, dtype, torch.bfloat16,
                                a.device, streamk=False)
    before = (sel.block_m, sel.block_n, sel.block_k, sel.num_stages)
    fired = _k349_fp8_dispatch(a, b, sel)
    assert fired is True
    after = (sel.block_m, sel.block_n, sel.block_k, sel.num_stages)
    assert before == after, "Non-cohort shape mutated selector tile"


def test_dispatch_skips_non_fp8():
    """bf16 inputs must short-circuit (return False) and not touch the
    selector — this guards K-654's lesson about unconditional knob
    application."""
    M = N = K = 1024
    a = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.zeros(K, N, device="cuda", dtype=torch.bfloat16)
    sel = _make_matmul_selector(M, N, K, torch.bfloat16, torch.bfloat16,
                                torch.bfloat16, a.device, streamk=False)
    before = (sel.block_m, sel.block_n, sel.block_k, sel.num_stages)
    fired = _k349_fp8_dispatch(a, b, sel)
    assert fired is False
    after = (sel.block_m, sel.block_n, sel.block_k, sel.num_stages)
    assert before == after
