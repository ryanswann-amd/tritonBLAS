"""K-656: tile-override gate predicate + numerical correctness tests.

Guards `_is_k656_cohort` (which shapes/dtypes activate the override) and the
`K656_DISABLE=1` env-var kill switch. Predicate tests run on CPU import only
(no GPU required) so they catch silent regressions of the predicate (e.g.
someone loosens the dtype check, or the env-var kill switch stops being
honored).

`test_override_matches_baseline_numerics` requires a GPU and proves the tile
override produces output equivalent to the default Origami pick on a firing
shape, within FP8-matmul tolerance. This catches the failure mode where the
override accidentally picks a tile that produces wrong results.
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Origami requires a real CUDA device.index for hardware lookup")
def test_make_selector_applies_tile_override(matmul_module, monkeypatch):
    """Sanity check on the override mechanism: with the gate ON, the selector
    returned by `_make_matmul_selector` reads back BM=BN=256, BK=64, NS=2 on a
    firing shape; with K656_DISABLE=1 it returns the default Origami pick.

    Catches the case where the override mutates the wrong attribute or the
    @property descriptors stop reflecting the underlying mutation. Selector
    construction does not launch a kernel but does call into Origami, which
    requires a CUDA device.
    """
    device = torch.device("cuda:0")

    monkeypatch.delenv("K656_DISABLE", raising=False)
    sel_on = matmul_module._make_matmul_selector(
        4096, 4096, 1024,
        torch.float8_e5m2fnuz, torch.float8_e5m2fnuz, torch.float16,
        device,
    )
    assert sel_on.block_m == 256
    assert sel_on.block_n == 256
    assert sel_on.block_k == 64
    assert sel_on.num_stages == 2

    monkeypatch.setenv("K656_DISABLE", "1")
    sel_off = matmul_module._make_matmul_selector(
        4096, 4096, 1024,
        torch.float8_e5m2fnuz, torch.float8_e5m2fnuz, torch.float16,
        device,
    )
    # Origami's default for this shape is BK=128 -- the override changes BK.
    # Guard that the override is not silently a no-op.
    assert (sel_off.block_m, sel_off.block_n, sel_off.block_k, sel_off.num_stages) != \
           (256, 256, 64, 2), "Default Origami pick coincidentally equals override; predicate would be a no-op."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_override_matches_baseline_numerics(monkeypatch):
    """Numerical correctness on a firing shape: override-on output must match
    override-off (default Origami pick) output within FP8-matmul tolerance.

    Catches the failure mode where the override picks a tile that silently
    produces wrong results (e.g. mis-tiled K-loop accumulator, partial sums
    overflowing FP16).
    """
    import tritonblas
    matmul_mod = importlib.import_module("tritonblas.matmul")

    M, N, K = 4096, 4096, 1024
    dtype = torch.float8_e5m2fnuz
    out_dtype = torch.float16

    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(M, K, generator=g, device="cuda").to(dtype)
    b = torch.randn(N, K, generator=g, device="cuda").to(dtype)
    a_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    b_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    # Sanity: predicate state must agree with the env we set.
    monkeypatch.delenv("K656_DISABLE", raising=False)
    assert matmul_mod._is_k656_cohort(M, N, K, dtype, dtype, streamk=False) is True
    out_on = torch.empty(M, N, device="cuda", dtype=out_dtype)
    tritonblas.matmul_a8w8(a, b.t(), a_scale, b_scale, out_on)
    torch.cuda.synchronize()

    monkeypatch.setenv("K656_DISABLE", "1")
    assert matmul_mod._is_k656_cohort(M, N, K, dtype, dtype, streamk=False) is False
    out_off = torch.empty(M, N, device="cuda", dtype=out_dtype)
    tritonblas.matmul_a8w8(a, b.t(), a_scale, b_scale, out_off)
    torch.cuda.synchronize()

    # Both code paths feed the same persistent_matmul kernel; only BLOCK_SIZE_*
    # and num_stages differ. Output should match within FP16 round-off of an
    # FP8xFP8->FP16 dot product over K=1024 partial sums.
    torch.testing.assert_close(out_on, out_off, atol=1.0, rtol=1e-2)
