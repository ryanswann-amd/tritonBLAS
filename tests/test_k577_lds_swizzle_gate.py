"""K-577 unit tests: assert the LDS-bank-conflict-fixed tile (BM=128, BN=128,
BK=64, NS=2) is applied STRICTLY inside the K-451 medium-K square fp16/bf16
gate, and the Origami-selected pick is preserved everywhere else.

These tests do not require a GPU. They monkey-patch the per-kernel launcher
to capture the launch kwargs, then exercise the public `persistent_matmul_lt`
/ `streamk_matmul_lt` entry points with cohort and out-of-cohort shapes.

Why this matters: the K-654 lesson is that an unconditional swizzle/tile swap
regresses out-of-cohort shapes (53/61 in the 61-shape regression sweep). The
gate predicate `_k451_match()` is the contract; this test enforces it at the
unit boundary so a future refactor can't silently widen its scope.
"""
from __future__ import annotations

import types
from typing import Any, Dict, List

import importlib

import pytest
import torch

# `tritonblas.matmul` is shadowed in __init__.py by the `matmul` function
# (`from .matmul import matmul, ...`), so we have to import the underlying
# module via importlib to monkeypatch the kernel-launch globals.
matmul_module = importlib.import_module("tritonblas.matmul")
_K451_COHORT_DTYPES = matmul_module._K451_COHORT_DTYPES
_K451_COHORT_K = matmul_module._K451_COHORT_K
_K451_COHORT_M = matmul_module._K451_COHORT_M
_k451_match = matmul_module._k451_match
persistent_matmul_lt = matmul_module.persistent_matmul_lt
streamk_matmul_lt = matmul_module.streamk_matmul_lt


# --- Predicate-level tests -------------------------------------------------


@pytest.mark.parametrize("M", sorted(_K451_COHORT_M))
@pytest.mark.parametrize("K", sorted(_K451_COHORT_K))
@pytest.mark.parametrize("dtype", sorted(_K451_COHORT_DTYPES, key=str))
def test_gate_matches_full_cohort(M, K, dtype):
    """Every (M=N, K, dtype) in the 12-shape K-451 cohort must match."""
    assert _k451_match(M, M, K, dtype) is True


@pytest.mark.parametrize(
    "M,N,K,dtype",
    [
        # Different M and N (asymmetric) -> must NOT match.
        (1024, 2048, 512, torch.float16),
        # Out-of-cohort M -> must NOT match.
        (8192, 8192, 512, torch.float16),
        (768, 768, 512, torch.float16),
        # K-577 narrowing: M=4096 was in the original K-451 cohort but is
        # excluded here because the K-577 fix tile loses ~12% on it and the
        # K-451 winner tile is itself ~5-10% slower than Origami's pick on
        # the current ROCm/Triton stack -- so M=4096 must fall through to
        # the selector default.
        (4096, 4096, 256, torch.float16),
        (4096, 4096, 512, torch.bfloat16),
        # Out-of-cohort K (exactly the K-654 / K-539 small-K and large-K bands).
        (1024, 1024, 128, torch.float16),
        (2048, 2048, 1024, torch.bfloat16),
        (4096, 4096, 8192, torch.float16),
        # Out-of-cohort dtype -> must NOT match.
        (1024, 1024, 256, torch.float32),
        (2048, 2048, 512, torch.float8_e4m3fnuz),
    ],
)
def test_gate_rejects_out_of_cohort(M, N, K, dtype):
    assert _k451_match(M, N, K, dtype) is False


# --- Launch-site capture tests --------------------------------------------


class _CapturedLaunch(Exception):
    """Raised by the patched kernel proxy to short-circuit the dispatch."""

    def __init__(self, kwargs: Dict[str, Any]):
        self.kwargs = kwargs


def _install_kernel_capture(monkeypatch, kernel_attr: str) -> List[Dict[str, Any]]:
    """Patch the named kernel symbol so its first launch raises with kwargs."""

    captured: List[Dict[str, Any]] = []

    class _Proxy:
        def __getitem__(self, _grid):  # mimic triton's `kernel[(grid,)]` syntax
            def _call(*args, **kwargs):
                captured.append(dict(kwargs))
                raise _CapturedLaunch(kwargs)
            return _call

    monkeypatch.setattr(matmul_module, kernel_attr, _Proxy())
    # _maybe_wrap returns the kernel unchanged when triton_aggregates is absent.
    # If it does wrap, the wrapper still routes through our proxy because we
    # patched the module-level symbol.
    return captured


class _FakeSelector:
    """Minimal selector mimicking the OrigamiMatmulSelector @property surface.

    Defaults are intentionally chosen NOT to coincide with the K-577 gate tile
    (BM=128, BN=128, BK=64, NS=2) so out-of-gate tests can detect a leak.
    """

    def __init__(
        self,
        block_m=64,
        block_n=64,
        block_k=32,
        group_m=4,
        num_sms=304,
        num_stages=3,
        num_warps=4,
        waves_per_eu=1,
    ):
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.group_m = group_m
        self.num_sms = num_sms
        self.num_stages = num_stages
        self.num_warps = num_warps
        self.waves_per_eu = waves_per_eu


def _meta_tensor(rows, cols, dtype):
    """Return a `meta` device tensor; dispatch decisions never touch its data."""
    return torch.empty((rows, cols), dtype=dtype, device="meta")


def _run_persistent(monkeypatch, M, N, K, dtype):
    captured = _install_kernel_capture(monkeypatch, "persistent_matmul")
    a = _meta_tensor(M, K, dtype)
    b = _meta_tensor(K, N, dtype)
    c = _meta_tensor(M, N, dtype)
    selector = _FakeSelector()
    with pytest.raises(_CapturedLaunch):
        persistent_matmul_lt(a, b, c, selector)
    assert captured, "persistent_matmul launch was never reached"
    return captured[-1]


def _run_streamk(monkeypatch, M, N, K, dtype):
    captured = _install_kernel_capture(monkeypatch, "streamk_matmul")
    a = _meta_tensor(M, K, dtype)
    b = _meta_tensor(K, N, dtype)
    c = _meta_tensor(M, N, dtype)
    selector = _FakeSelector()
    # streamk_matmul_lt requires the selector to expose an sk_grid attribute and
    # an `_ACTIVE_CU` field; provide minimal stubs.
    selector.sk_grid = 0
    selector._hardware = types.SimpleNamespace(N_CU=304)
    selector._ACTIVE_CU = 304
    selector.COUNTERS_PER_XCD = 1
    with pytest.raises(_CapturedLaunch):
        streamk_matmul_lt(a, b, c, selector)
    assert captured, "streamk_matmul launch was never reached"
    return captured[-1]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [1024, 2048])
@pytest.mark.parametrize("K", [256, 512])
def test_persistent_k577_tile_for_M_1024_2048(monkeypatch, dtype, M, K):
    """For M in {1024, 2048} the persistent launcher must dispatch with the
    K-577 LDS-bank-conflict-fix tile (BM=128, BN=128, BK=64, NS=2)."""
    kwargs = _run_persistent(monkeypatch, M, M, K, dtype)
    assert kwargs["BLOCK_SIZE_M"] == 128, kwargs
    assert kwargs["BLOCK_SIZE_N"] == 128, kwargs  # K-577: BN narrowed for LDS budget.
    assert kwargs["BLOCK_SIZE_K"] == 64, kwargs   # K-577: BK widened to one full bank cycle.
    assert kwargs["num_warps"] == 8, kwargs
    assert kwargs["waves_per_eu"] == 2, kwargs
    assert kwargs["num_stages"] == 2, kwargs       # K-577: NS=2 to fit 64 KiB LDS.
    assert kwargs["kpack"] == 1, kwargs


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("K", [256, 512])
def test_persistent_M_4096_falls_through_to_selector(monkeypatch, dtype, K):
    """K-577 narrowing: M=4096 was in the K-451 cohort but is now excluded
    from the gate (both K-451 winner and K-577 fix tiles lose to Origami's
    natural pick at M=4096 on the current ROCm/Triton stack). The launch
    site must therefore use the FakeSelector defaults (64x64x32) here."""
    kwargs = _run_persistent(monkeypatch, 4096, 4096, K, dtype)
    assert kwargs["BLOCK_SIZE_M"] == 64, kwargs
    assert kwargs["BLOCK_SIZE_N"] == 64, kwargs
    assert kwargs["BLOCK_SIZE_K"] == 32, kwargs
    assert kwargs["num_stages"] == 3, kwargs
    assert kwargs["kpack"] == 1, kwargs


@pytest.mark.parametrize(
    "M,N,K,dtype",
    [
        # Out-of-cohort M (K-539 small-square cohort lower edge).
        (512, 512, 512, torch.float16),
        # Out-of-cohort K (large-K residual band).
        (2048, 2048, 4096, torch.bfloat16),
        # Asymmetric (skinny) shape.
        (1024, 4096, 512, torch.float16),
        # Out-of-cohort dtype (fp32).
        (2048, 2048, 256, torch.float32),
    ],
)
def test_persistent_origami_pick_outside_k451_gate(monkeypatch, M, N, K, dtype):
    """Outside the cohort, the launch site must use the selector's pick (NOT
    the K-577 gate tile) and keep kpack at the historical default of 1.
    With our FakeSelector defaults (64x64x32, NS=3, NW=4, WPEU=1) any leak
    of the gate tile would be immediately visible."""
    kwargs = _run_persistent(monkeypatch, M, N, K, dtype)
    assert kwargs["kpack"] == 1, (
        f"K-577: out-of-cohort dispatch (M={M}, N={N}, K={K}, {dtype}) must "
        f"keep kpack=1 (K-654 lesson: do NOT widen the swizzle); "
        f"got kpack={kwargs['kpack']!r}"
    )
    # Verify we got the selector's pick, not the gate tile.
    assert kwargs["BLOCK_SIZE_M"] == 64, kwargs
    assert kwargs["BLOCK_SIZE_N"] == 64, kwargs
    assert kwargs["BLOCK_SIZE_K"] == 32, kwargs
    assert kwargs["num_stages"] == 3, kwargs
    assert kwargs["num_warps"] == 4, kwargs
    assert kwargs["waves_per_eu"] == 1, kwargs


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [1024, 2048])
@pytest.mark.parametrize("K", [256, 512])
def test_streamk_k577_tile_for_M_1024_2048(monkeypatch, dtype, M, K):
    """The streamk launch site must mirror persistent's K-577 sub-gate."""
    kwargs = _run_streamk(monkeypatch, M, M, K, dtype)
    assert kwargs["BLOCK_SIZE_M"] == 128, kwargs
    assert kwargs["BLOCK_SIZE_N"] == 128, kwargs
    assert kwargs["BLOCK_SIZE_K"] == 64, kwargs
    assert kwargs["num_stages"] == 2, kwargs
    assert kwargs["kpack"] == 1, kwargs


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("K", [256, 512])
def test_streamk_M_4096_falls_through_to_selector(monkeypatch, dtype, K):
    """For M=4096 the streamk launch site must also fall through to the
    selector pick (FakeSelector defaults: 64x64x32)."""
    kwargs = _run_streamk(monkeypatch, 4096, 4096, K, dtype)
    assert kwargs["BLOCK_SIZE_M"] == 64, kwargs
    assert kwargs["BLOCK_SIZE_N"] == 64, kwargs
    assert kwargs["BLOCK_SIZE_K"] == 32, kwargs
    assert kwargs["num_stages"] == 3, kwargs
    assert kwargs["kpack"] == 1, kwargs


@pytest.mark.parametrize(
    "M,N,K,dtype",
    [
        (512, 512, 512, torch.float16),
        (2048, 2048, 4096, torch.bfloat16),
        (1024, 4096, 512, torch.float16),
    ],
)
def test_streamk_origami_pick_outside_k451_gate(monkeypatch, M, N, K, dtype):
    kwargs = _run_streamk(monkeypatch, M, N, K, dtype)
    assert kwargs["kpack"] == 1, (
        f"K-577: streamk out-of-cohort dispatch (M={M}, N={N}, K={K}, {dtype}) "
        f"must keep kpack=1; got kpack={kwargs['kpack']!r}"
    )
    # Verify selector pick (FakeSelector defaults), not the gate tile.
    assert kwargs["BLOCK_SIZE_M"] == 64, kwargs
    assert kwargs["BLOCK_SIZE_N"] == 64, kwargs
    assert kwargs["BLOCK_SIZE_K"] == 32, kwargs
