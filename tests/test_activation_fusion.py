"""
Correctness tests for the fused activation epilogue.

Validates :func:`tritonblas.matmul_activation` and
:func:`tritonblas.addmm_activation` against eager-mode torch references.

The activation matrix is:

    relu       <-> torch.nn.functional.relu
    gelu       <-> torch.nn.functional.gelu(approximate="none")
    gelu_tanh  <-> torch.nn.functional.gelu(approximate="tanh")
    swish      <-> torch.nn.functional.silu      (== x * sigmoid(x))
    sigmoid    <-> torch.sigmoid

Each test compares the fused kernel against the *unfused* tritonblas chain
(matmul/addmm + torch activation) to verify that fusion does not introduce
new numerical drift beyond what the bare matmul already has.
"""

import pytest
import torch
import tritonblas
import torch.nn.functional as F


# Shapes chosen to exercise multiple tile multiples and at least one
# transformer-FFN-shaped tile (Llama-2-7B FFN1 is 4096x11008x4096 — too big
# for a unit test, so we shrink to 256x256x256 and 512x1024x768).
_SHAPES = [
    (128, 256, 512),
    (256, 256, 256),
    (512, 1024, 768),
    (32, 32, 32),
    (64, 16, 128),
]

_DTYPES = [torch.bfloat16, torch.float16]

_ACTIVATIONS = ["relu", "gelu", "gelu_tanh", "swish", "sigmoid"]


def _torch_act(name: str, x: torch.Tensor) -> torch.Tensor:
    """Torch reference matching tritonblas's activation semantics."""
    if name == "relu":
        return F.relu(x)
    if name == "gelu":
        return F.gelu(x, approximate="none")
    if name == "gelu_tanh":
        return F.gelu(x, approximate="tanh")
    if name == "swish":
        return F.silu(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    raise ValueError(name)


@pytest.mark.parametrize("activation", _ACTIVATIONS)
@pytest.mark.parametrize("m, n, k", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_matmul_activation_correctness(m, n, k, dtype, activation):
    """Fused matmul+activation matches unfused chain to within bare-matmul tolerance."""
    torch.manual_seed(42)
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)

    fused = tritonblas.matmul_activation(a, b, activation=activation)

    # Unfused reference uses the SAME tritonblas matmul so any per-tile FMA
    # ordering quirk cancels and we are only measuring the activation hook.
    bare = tritonblas.matmul(a, b)
    expected = _torch_act(activation, bare)

    # Activation can amplify noise (GELU/SWISH near 0), so loosen for low precision.
    tol = {torch.bfloat16: 5e-2, torch.float16: 1e-2}[dtype]
    torch.testing.assert_close(fused, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("activation", _ACTIVATIONS)
@pytest.mark.parametrize("m, n, k", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_addmm_activation_correctness(m, n, k, dtype, activation):
    """Fused bias+matmul+activation matches unfused chain."""
    torch.manual_seed(123)
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    bias = torch.randn(n, device="cuda", dtype=dtype)

    fused = tritonblas.addmm_activation(bias, a, b, activation=activation)

    bare = tritonblas.addmm(bias, a, b)
    expected = _torch_act(activation, bare)

    tol = {torch.bfloat16: 5e-2, torch.float16: 1e-2}[dtype]
    torch.testing.assert_close(fused, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("activation", _ACTIVATIONS)
def test_matmul_activation_out_kwarg(activation):
    """Pre-allocated `out=` tensor is mutated and returned."""
    torch.manual_seed(7)
    m, n, k = 128, 128, 256
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    ret = tritonblas.matmul_activation(a, b, activation=activation, out=out)
    assert ret.data_ptr() == out.data_ptr(), "out= should not allocate"

    bare = tritonblas.matmul(a, b)
    expected = _torch_act(activation, bare)
    torch.testing.assert_close(out, expected, atol=5e-2, rtol=5e-2)


def test_activation_none_matches_bare_matmul():
    """`activation="none"` is a no-op."""
    torch.manual_seed(0)
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)

    fused = tritonblas.matmul_activation(a, b, activation="none")
    bare = tritonblas.matmul(a, b)
    torch.testing.assert_close(fused, bare, atol=0.0, rtol=0.0)


def test_unknown_activation_raises():
    """Typos in the activation kwarg are caught at the host boundary."""
    a = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="unknown activation"):
        tritonblas.matmul_activation(a, b, activation="GELU")  # caps wrong


def test_addmm_activation_bias_shape_validation():
    """A bias of the wrong shape raises a clear error before kernel launch."""
    a = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16)
    bad_bias = torch.randn(7, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="bias must be 1-D"):
        tritonblas.addmm_activation(bad_bias, a, b, activation="relu")
