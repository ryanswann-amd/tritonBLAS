"""
Correctness tests for fused activation epilogues in tritonblas.matmul / addmm.

Activations exercised: relu, gelu, gelu_tanh, silu (SWISH(beta=1)), sigmoid.
Each test compares the fused tritonblas output against the unfused PyTorch
reference (matmul + activation, or addmm + activation).
"""

import pytest
import torch
import tritonblas
import torch.nn.functional as F

from conftest import DTYPES, STANDARD_DIMS


# Map of activation name -> (torch reference, atol, rtol)
# Tolerances are intentionally loose for low-precision dtypes; activations
# (especially gelu/silu) compose with fp16 matmul rounding noise.
#
# Note: tritonblas "gelu" uses the Hendrycks tanh approximation (matches
# F.gelu(approximate="tanh") exactly) — same approximation hipBLASLt's
# HIPBLASLT_EPILOGUE_GELU emits. Use "gelu_exact" if you specifically need
# the F.gelu(approximate="none") erf semantics.
_ACTIVATIONS = {
    "relu":       (lambda x: F.relu(x),                          1e-1, 1e-1),
    "gelu":       (lambda x: F.gelu(x, approximate="tanh"),      2e-1, 2e-1),
    "gelu_tanh":  (lambda x: F.gelu(x, approximate="tanh"),      2e-1, 2e-1),
    "gelu_exact": (lambda x: F.gelu(x, approximate="none"),      2e-1, 2e-1),
    "silu":       (lambda x: F.silu(x),                          2e-1, 2e-1),
    "sigmoid":    (lambda x: torch.sigmoid(x),                   1e-1, 1e-1),
}

# Subset of standard shapes — cover medium + square + skinny regimes.
_SHAPES = [
    (128, 256, 512),
    (256, 256, 256),
    (512, 1024, 768),
    (1024, 2048, 512),
]

_STREAMK = [False, True]


@pytest.mark.parametrize("activation", list(_ACTIVATIONS.keys()))
@pytest.mark.parametrize("enable_streamk", _STREAMK)
@pytest.mark.parametrize("m, n, k", _SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matmul_activation_forward(m, n, k, dtype, enable_streamk, activation):
    """tritonblas.matmul(activation=X) matches torch reference."""
    torch.manual_seed(42)

    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)

    ref_act, atol, rtol = _ACTIVATIONS[activation]

    out = tritonblas.matmul(a, b, enable_streamk=enable_streamk, activation=activation)
    expected = ref_act(torch.matmul(a, b))

    torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("activation", list(_ACTIVATIONS.keys()))
@pytest.mark.parametrize("enable_streamk", _STREAMK)
@pytest.mark.parametrize("m, n, k", _SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_addmm_activation_forward(m, n, k, dtype, enable_streamk, activation):
    """tritonblas.addmm(bias, a, b, activation=X) matches torch reference."""
    torch.manual_seed(43)

    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    bias = torch.randn(n, device="cuda", dtype=dtype)

    ref_act, atol, rtol = _ACTIVATIONS[activation]

    out = tritonblas.addmm(bias, a, b, enable_streamk=enable_streamk, activation=activation)
    expected = ref_act(torch.addmm(bias, a, b))

    torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)


def test_activation_default_unchanged():
    """Default activation='none' must yield byte-identical results to the legacy path."""
    torch.manual_seed(44)
    m, n, k = 256, 256, 256
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)

    legacy = tritonblas.matmul(a, b)
    explicit = tritonblas.matmul(a, b, activation="none")
    torch.testing.assert_close(legacy, explicit, atol=0.0, rtol=0.0)


def test_unknown_activation_rejected():
    """Bad activation strings must be rejected at the Python boundary."""
    a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="Unsupported activation"):
        tritonblas.matmul(a, b, activation="not_a_real_activation")


def test_work_stealing_with_activation_rejected():
    """Activation + work_stealing must raise NotImplementedError, not silently drop."""
    a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="work-stealing"):
        tritonblas.matmul(a, b, work_stealing=True, activation="gelu")


def test_addmm_autograd_with_activation_rejected():
    """Backward + activation should explicitly fail rather than silently mis-grad."""
    a = torch.randn(64, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    b = torch.randn(64, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    bias = torch.randn(64, device="cuda", dtype=torch.float16, requires_grad=True)
    out = tritonblas.addmm(bias, a, b, activation="gelu")
    with pytest.raises(NotImplementedError, match="activation"):
        out.sum().backward()
