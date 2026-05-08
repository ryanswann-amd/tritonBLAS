import pytest
import torch  # type: ignore
import triton  # type: ignore
import tritonblas  # type: ignore
from tritonblas.utils import generate_matmul_inputs  # type: ignore


@pytest.mark.parametrize(
    "m, n, k",
    [
        (8192, 8192, 8192),
        (4864, 8192, 4160),
        (4096, 4096, 4096),
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        # (torch.float8_e4m3fn, torch.float8_e4m3fn),
        # (torch.float8_e5m2, torch.float8_e5m2),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
    ],
)
@pytest.mark.parametrize(
    "transA, transB",
    [
        ("T", "T"),  # A^T @ B^T
        ("N", "N"),  # A @ B
        ("T", "N"),  # A^T @ B
        ("N", "T"),  # A @ B^T
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "persistent",
        "streamk",
        "work_stealing",
    ],
)
def test_matmul(m, n, k, in_dtype, out_dtype, transA, transB, mode):
    """Test non-quantized matmul with all transpose combinations using shared input generation utilities."""
    init_type = "randn"
    enable_streamk = mode == "streamk"
    work_stealing = mode == "work_stealing"

    inputs = generate_matmul_inputs(m, n, k, in_dtype, out_dtype, transA, transB, init_type)

    tritonblas.matmul(inputs.A, inputs.B, inputs.C, enable_streamk=enable_streamk,
                      work_stealing=work_stealing)

    torch_c = torch.matmul(inputs.A, inputs.B)
    torch.testing.assert_close(inputs.C.to(out_dtype), torch_c, atol=1, rtol=1)


# =============================================================================
# K-424: FP8 small-M dispatch gate
# =============================================================================
# These tests lock in the routing rule from
# `tritonblas.matmul._fp8_smallm_streamk_route` so that future refactors
# cannot silently regress (a) the M=1 crash that the gate fixes, or (b) the
# (M<=32, K=8192, N!=4096) stall corner that the gate accelerates. The gate
# also intentionally LEAVES (M<=32, K=8192, N=4096) on the persistent path
# because streamk regresses there empirically on MI300X.

@pytest.mark.parametrize(
    "dtype_name",
    ["float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"],
)
@pytest.mark.parametrize(
    "M, N, K, expect_streamk",
    [
        # M==1: rerouted (persistent crashes in make_scale_view).
        (1, 2048, 2048, True),
        (1, 4096, 4096, True),
        (1, 8192, 8192, True),
        (1, 4096, 8192, True),
        # M<=32 + K==8192 + N!=4096: rerouted (persistent K-loop stalls).
        (4, 2048, 8192, True),
        (8, 8192, 8192, True),
        (32, 2048, 8192, True),
        # M<=32 + K==8192 + N==4096: NOT rerouted (streamk regresses here).
        (4, 4096, 8192, False),
        (16, 4096, 8192, False),
        (32, 4096, 8192, False),
        # M<=32 + K<8192: NOT rerouted (persistent path is faster).
        (8, 4096, 4096, False),
        (16, 8192, 2048, False),
        # Larger M: NOT rerouted.
        (64, 2048, 8192, False),
        (128, 8192, 8192, False),
        # Out-of-cohort (large) shapes: NOT rerouted.
        (4096, 4096, 4096, False),
    ],
)
def test_fp8_smallm_streamk_route_predicate(dtype_name, M, N, K, expect_streamk):
    """Routing-rule unit test (no GPU required): asserts the static
    predicate that decides whether an FP8 shape is rerouted to streamk."""
    from tritonblas.matmul import _fp8_smallm_streamk_route
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        pytest.skip(f"{dtype_name} not available in this PyTorch build")
    assert _fp8_smallm_streamk_route(dtype, M, N, K) is expect_streamk


def test_fp8_smallm_streamk_route_non_fp8_unchanged():
    """Non-FP8 dtypes must NOT be rerouted by this gate."""
    from tritonblas.matmul import _fp8_smallm_streamk_route
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int8):
        assert not _fp8_smallm_streamk_route(dtype, 1, 4096, 4096)
        assert not _fp8_smallm_streamk_route(dtype, 8, 8192, 8192)


@pytest.mark.parametrize(
    "M, N, K",
    [
        # M=1 was a hard crash on the persistent path (K-251 A17). The gate
        # must keep this shape running end-to-end via streamk.
        (1, 4096, 4096),
        # M<=32, K=8192, N!=4096 was the stall corner that the gate
        # accelerates by routing to streamk; verify it still runs cleanly.
        (8, 2048, 8192),
    ],
)
def test_fp8_smallm_dispatch_gate_runs_end_to_end(M, N, K):
    """End-to-end smoke test: the M=1 crash regression in particular must
    not reappear undetected. Uses torch._scaled_mm as the reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    dtype = getattr(torch, "float8_e4m3fnuz", None) or getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        pytest.skip("FP8 dtype not available in this PyTorch build")

    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda").to(dtype)
    b = torch.randn(K, N, device="cuda").to(dtype)
    a_scale = torch.ones(M, device="cuda", dtype=torch.float32)
    b_scale = torch.ones(N, device="cuda", dtype=torch.float32)
    c = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)

    # Default kwargs: gate must trigger and reroute internally.
    tritonblas.matmul_a8w8(a, b, a_scale, b_scale, c)

    # Reference: torch._scaled_mm requires column-major B.
    if hasattr(torch, "_scaled_mm"):
        b_t = b.t().contiguous().t()
        ref = torch._scaled_mm(
            a, b_t,
            scale_a=a_scale.view(M, 1),
            scale_b=b_scale.view(1, N),
            out_dtype=torch.bfloat16,
        )
        torch.testing.assert_close(c, ref, atol=1.0, rtol=0.05)

