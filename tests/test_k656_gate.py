"""K-656: tile-override gate predicate tests.

Guards `_is_k656_cohort` (which shapes/dtypes activate the override) and the
`K656_DISABLE=1` env-var kill switch. These run on CPU import only -- no GPU
required -- so they catch silent regressions of the predicate (e.g. someone
loosens the dtype check, or the env-var kill switch stops being honored).
"""

import importlib

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float8_e5m2fnuz"),
    reason="float8_e5m2fnuz is not available in this torch build",
)


@pytest.fixture
def matmul_module(monkeypatch):
    """Import tritonblas.matmul with K656_DISABLE unset; clear it for each test.

    `tritonblas.matmul` is shadowed by the matmul *function* re-exported from
    the package __init__, so we reach for the submodule via importlib.
    """
    monkeypatch.delenv("K656_DISABLE", raising=False)
    return importlib.import_module("tritonblas.matmul")


# Shapes that MUST trigger the override.
_FIRES = [
    (4096, 4096, 1024),
    (4096, 4096, 2048),
]

# Shapes that MUST NOT trigger the override (cohort passthroughs + guards).
_PASSES = [
    # Same dtype, but wrong M/N
    (1024, 1024, 1024),
    (2048, 2048, 1024),
    (2048, 2048, 2048),
    # Same M=N=4096, but K outside {1024, 2048}
    (4096, 4096, 256),
    (4096, 4096, 512),
    (4096, 4096, 4096),
    (4096, 4096, 8192),
    # Asymmetric M/N
    (4096, 2048, 1024),
    (2048, 4096, 1024),
]


@pytest.mark.parametrize("M,N,K", _FIRES)
def test_predicate_fires_on_cohort(matmul_module, M, N, K):
    assert matmul_module._is_k656_cohort(
        M, N, K, torch.float8_e5m2fnuz, torch.float8_e5m2fnuz, streamk=False
    ) is True


@pytest.mark.parametrize("M,N,K", _PASSES)
def test_predicate_passes_outside_cohort(matmul_module, M, N, K):
    assert matmul_module._is_k656_cohort(
        M, N, K, torch.float8_e5m2fnuz, torch.float8_e5m2fnuz, streamk=False
    ) is False


@pytest.mark.parametrize(
    "a_dtype,b_dtype",
    [
        # Wrong dtype on either operand must NOT fire.
        pytest.param(
            getattr(torch, "float8_e4m3fnuz", None),
            getattr(torch, "float8_e4m3fnuz", None),
            id="e4m3fnuz",
            marks=pytest.mark.skipif(
                not hasattr(torch, "float8_e4m3fnuz"),
                reason="e4m3fnuz unavailable",
            ),
        ),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        # Asymmetric dtype must NOT fire.
        pytest.param(
            torch.float16,
            getattr(torch, "float8_e5m2fnuz", None),
            id="mixed-fp16-e5m2",
        ),
    ],
)
def test_predicate_dtype_guard(matmul_module, a_dtype, b_dtype):
    assert matmul_module._is_k656_cohort(
        4096, 4096, 1024, a_dtype, b_dtype, streamk=False
    ) is False


def test_predicate_streamk_guard(matmul_module):
    """Streamk dispatch path must NOT trigger the override."""
    assert matmul_module._is_k656_cohort(
        4096, 4096, 1024,
        torch.float8_e5m2fnuz, torch.float8_e5m2fnuz,
        streamk=True,
    ) is False


def test_env_var_kill_switch(matmul_module, monkeypatch):
    """K656_DISABLE=1 must disable the override even on a cohort shape."""
    args = (
        4096, 4096, 1024,
        torch.float8_e5m2fnuz, torch.float8_e5m2fnuz,
    )
    # Default (env unset): predicate fires.
    monkeypatch.delenv("K656_DISABLE", raising=False)
    assert matmul_module._is_k656_cohort(*args, streamk=False) is True

    # Kill switch on: predicate must return False.
    monkeypatch.setenv("K656_DISABLE", "1")
    assert matmul_module._is_k656_cohort(*args, streamk=False) is False

    # Any other value (e.g. "0") must NOT disable.
    monkeypatch.setenv("K656_DISABLE", "0")
    assert matmul_module._is_k656_cohort(*args, streamk=False) is True


def test_override_class_overrides_tile():
    """_K656Selector class attributes must shadow OrigamiMatmulSelector @properties."""
    from tritonblas.matmul import _K656Selector
    from tritonblas.origami import OrigamiMatmulSelector

    assert issubclass(_K656Selector, OrigamiMatmulSelector)
    # Class-level shadow values used by persistent_matmul_lt.
    assert _K656Selector.block_m == 256
    assert _K656Selector.block_n == 256
    assert _K656Selector.block_k == 64
    assert _K656Selector.num_stages == 2
