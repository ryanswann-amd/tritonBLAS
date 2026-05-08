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
# K-424: FP8 small-M dispatch gate (dispatch-wiring tests)
# =============================================================================
# The routing predicate is inlined in matmul_a8w8. These tests intercept the
# two kernel-launch shims (streamk_matmul_lt / persistent_matmul_lt) to verify
# the *wiring* — i.e. that the gate is actually consulted at the dispatch layer
# and that an explicit caller-supplied enable_streamk is respected.

class _RouteSpy:
    """Monkey-patches the two dispatch shims and records which one was called.

    `tritonblas.matmul` resolves to the *function* of that name, so we get
    the module via sys.modules and patch attributes there. matmul_a8w8
    looks up streamk_matmul_lt / persistent_matmul_lt in module globals at
    call time, so attribute patching is sufficient."""
    def __init__(self, monkeypatch):
        import sys
        mm = sys.modules["tritonblas.matmul"]
        self.calls = []
        monkeypatch.setattr(mm, "_make_matmul_selector",
                            lambda *a, **k: object())
        monkeypatch.setattr(mm, "matmul_preamble",
                            lambda selector: None)
        monkeypatch.setattr(mm, "streamk_matmul_lt",
                            lambda *a, **k: self.calls.append("streamk"))
        monkeypatch.setattr(mm, "persistent_matmul_lt",
                            lambda *a, **k: self.calls.append("persistent"))


def _fp8_dtype_or_skip():
    dtype = (getattr(torch, "float8_e4m3fnuz", None) or
             getattr(torch, "float8_e4m3fn",   None))
    if dtype is None:
        pytest.skip("no FP8 dtype available in this PyTorch build")
    return dtype


def _fake_inputs(M, N, K, dtype):
    # meta tensors: no allocation, no GPU required.
    return (torch.empty(M, K, dtype=dtype,         device="meta"),
            torch.empty(K, N, dtype=dtype,         device="meta"),
            torch.empty(M,    dtype=torch.float32, device="meta"),
            torch.empty(N,    dtype=torch.float32, device="meta"),
            torch.empty(M, N, dtype=torch.bfloat16,device="meta"))


@pytest.mark.parametrize(
    "M, N, K, expect",
    [
        # M=1: gated -> streamk (persistent crashes in make_scale_view).
        (1, 2048, 2048, "streamk"),
        (1, 4096, 4096, "streamk"),
        (1, 8192, 8192, "streamk"),
        # M<=32, K=8192, N!=4096: gated -> streamk (persistent K-loop stalls).
        (4,  2048, 8192, "streamk"),
        (8,  8192, 8192, "streamk"),
        (32, 2048, 8192, "streamk"),
        # M<=32, K=8192, N==4096: NOT gated (streamk regresses there).
        (4,  4096, 8192, "persistent"),
        (16, 4096, 8192, "persistent"),
        (32, 4096, 8192, "persistent"),
        # M<=32, K<8192: NOT gated (persistent is faster).
        (8,  4096, 4096, "persistent"),
        (16, 8192, 2048, "persistent"),
        # M>32: NOT gated.
        (64,  2048, 8192, "persistent"),
        (128, 8192, 8192, "persistent"),
    ],
)
def test_matmul_a8w8_routes_per_gate(monkeypatch, M, N, K, expect):
    """Dispatch-level: with no caller override (enable_streamk=None), the
    gate decides which kernel shim is invoked."""
    dtype = _fp8_dtype_or_skip()
    spy = _RouteSpy(monkeypatch)
    a, b, sa, sb, c = _fake_inputs(M, N, K, dtype)
    tritonblas.matmul_a8w8(a, b, sa, sb, c)  # default enable_streamk=None
    assert spy.calls == [expect]


def test_matmul_a8w8_explicit_streamk_true_respected(monkeypatch):
    """Caller-supplied enable_streamk=True wins over the gate even when the
    gate would have chosen persistent."""
    dtype = _fp8_dtype_or_skip()
    spy = _RouteSpy(monkeypatch)
    a, b, sa, sb, c = _fake_inputs(64, 4096, 4096, dtype)  # outside gate
    tritonblas.matmul_a8w8(a, b, sa, sb, c, enable_streamk=True)
    assert spy.calls == ["streamk"]


def test_matmul_a8w8_explicit_streamk_false_respected(monkeypatch):
    """Caller-supplied enable_streamk=False wins over the gate even on a
    shape the gate would otherwise reroute. (Caller accepts the M=1 crash
    risk by passing it explicitly.)"""
    dtype = _fp8_dtype_or_skip()
    spy = _RouteSpy(monkeypatch)
    a, b, sa, sb, c = _fake_inputs(8, 2048, 8192, dtype)  # gated shape
    tritonblas.matmul_a8w8(a, b, sa, sb, c, enable_streamk=False)
    assert spy.calls == ["persistent"]


def test_matmul_a8w8_non_fp8_not_gated(monkeypatch):
    """Non-FP8 dtypes must NEVER be rerouted by the FP8 small-M gate."""
    spy = _RouteSpy(monkeypatch)
    a, b, sa, sb, c = _fake_inputs(1, 4096, 4096, torch.bfloat16)
    tritonblas.matmul_a8w8(a, b, sa, sb, c)
    assert spy.calls == ["persistent"]


@pytest.mark.parametrize("M, N, K", [(1, 4096, 4096), (8, 2048, 8192)])
def test_fp8_smallm_dispatch_gate_runs_end_to_end(M, N, K):
    """E2E smoke test on a real GPU: the M=1 crash (K-251 A17) and the
    M<=32/K=8192 stall corner must keep running cleanly through the gate."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    dtype = _fp8_dtype_or_skip()
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda").to(dtype)
    b = torch.randn(K, N, device="cuda").to(dtype)
    sa = torch.ones(M, device="cuda", dtype=torch.float32)
    sb = torch.ones(N, device="cuda", dtype=torch.float32)
    c = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    tritonblas.matmul_a8w8(a, b, sa, sb, c)  # gate fires, no crash
    if hasattr(torch, "_scaled_mm"):
        b_col = b.t().contiguous().t()
        ref = torch._scaled_mm(a, b_col, scale_a=sa.view(M, 1),
                               scale_b=sb.view(1, N),
                               out_dtype=torch.bfloat16)
        torch.testing.assert_close(c, ref, atol=1.0, rtol=0.05)

