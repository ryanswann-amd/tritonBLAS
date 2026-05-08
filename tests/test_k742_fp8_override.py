"""Unit tests for the K-742 FP8 e4m3fnuz tall-skinny tile-shrink override.

The override lives in ``tritonblas.matmul`` and is gated by
``_k742_should_override``.  These tests pin the predicate's behaviour so
future Origami changes (or accidental edits to the gate) cannot silently
regress the FP8 tall-skinny cohort to the over-tiled selection.

The predicate logic is dtype/shape/stride checks only and does not require
a GPU; the substitution test stubs the kernel launch and asserts the
selector tile is replaced with ``(64, 32, 256, 4, 2)``.
"""

import logging
import types
from unittest import mock

import pytest
import torch

import sys
import tritonblas.matmul  # noqa: F401  (registers module in sys.modules)
# tritonblas/__init__.py rebinds the attribute `tritonblas.matmul` to the
# `matmul()` function, so we must pull the module object from sys.modules.
tb_matmul = sys.modules["tritonblas.matmul"]
from tritonblas.matmul import (  # noqa: E402
    _K742_FP8_TS_SHAPES,
    _K742_FP8_TS_TILE,
    _k742_should_override,
)


# ---------------------------------------------------------------------------
# Cohort membership
# ---------------------------------------------------------------------------


def test_cohort_size_is_34():
    """36 enumerated cells minus 2 known regressions == 34 in-cohort tuples."""
    assert len(_K742_FP8_TS_SHAPES) == 34


def test_cohort_excludes_known_regressions():
    """K-717 sweep showed Origami wins on these two cells; gate must not fire."""
    assert (8192, 128, 1024) not in _K742_FP8_TS_SHAPES
    assert (8192, 128, 2048) not in _K742_FP8_TS_SHAPES


@pytest.mark.parametrize("shape", [
    (4096, 16, 1024), (8192, 64, 4096), (16384, 128, 4096),
    (4096, 128, 1024), (8192, 32, 2048),
])
def test_cohort_includes_representative_shapes(shape):
    assert shape in _K742_FP8_TS_SHAPES


def test_tile_tuple_is_pinned():
    """Override tile must remain (BM, BN, BK, num_warps, num_stages)."""
    assert _K742_FP8_TS_TILE == (64, 32, 256, 4, 2)


# ---------------------------------------------------------------------------
# Predicate behaviour (no GPU required)
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Minimal duck-typed tensor for predicate testing without CUDA."""

    def __init__(self, dtype, strides, device="cuda:0"):
        self.dtype = dtype
        self._stride = tuple(strides)
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"

    def stride(self, dim=None):
        if dim is None:
            return self._stride
        return self._stride[dim]


FP8 = torch.float8_e4m3fnuz


def _row_major_inputs(M, N, K, a_dtype=FP8, b_dtype=FP8, c_dtype=torch.bfloat16,
                      device="cuda:0"):
    """Strides that match how the verified benchmark feeds the kernel:
    a (M,K) row-major, b (K,N) K-contig (transposed view), c (M,N) row-major."""
    a = _FakeTensor(a_dtype, (K, 1), device=device)
    b = _FakeTensor(b_dtype, (1, K), device=device)
    c = _FakeTensor(c_dtype, (N, 1), device=device)
    return a, b, c


def test_happy_path_returns_true_for_in_cohort_shape():
    a, b, c = _row_major_inputs(8192, 64, 4096)
    assert _k742_should_override(8192, 64, 4096, a, b, c, work_stealing=False)


@pytest.mark.parametrize("shape", [
    (8192, 128, 1024),     # explicit exclusion 1
    (8192, 128, 2048),     # explicit exclusion 2
])
def test_excluded_cohort_cells_return_false(shape):
    M, N, K = shape
    a, b, c = _row_major_inputs(M, N, K)
    assert not _k742_should_override(M, N, K, a, b, c, work_stealing=False)


@pytest.mark.parametrize("shape", [
    (1024, 1024, 1024),    # generic square
    (4096, 256, 4096),     # adjacent N out of cohort
    (32768, 64, 4096),     # M out of cohort
    (4096, 16, 512),       # K out of cohort
])
def test_out_of_cohort_returns_false_silently(caplog, shape):
    M, N, K = shape
    a, b, c = _row_major_inputs(M, N, K)
    with caplog.at_level(logging.WARNING, logger="tritonblas.matmul.k742"):
        assert not _k742_should_override(M, N, K, a, b, c, work_stealing=False)
    # Out-of-cohort shapes must not log — we don't want noise on every kernel.
    assert caplog.records == []


def test_non_fp8_dtype_returns_false_silently(caplog):
    """Non-FP8 callers are not in scope — silent reject, no warning spam."""
    a, b, c = _row_major_inputs(4096, 16, 1024,
                                a_dtype=torch.float16, b_dtype=torch.float16)
    with caplog.at_level(logging.WARNING, logger="tritonblas.matmul.k742"):
        assert not _k742_should_override(4096, 16, 1024, a, b, c, work_stealing=False)
    assert caplog.records == []


def test_work_stealing_skip_logs_warning(caplog):
    a, b, c = _row_major_inputs(4096, 16, 1024)
    with caplog.at_level(logging.WARNING, logger="tritonblas.matmul.k742"):
        assert not _k742_should_override(4096, 16, 1024, a, b, c, work_stealing=True)
    assert any("work_stealing=True" in r.message for r in caplog.records), (
        f"expected work_stealing warning, got {[r.message for r in caplog.records]}"
    )


def test_non_bf16_output_skip_logs_warning(caplog):
    a, b, c = _row_major_inputs(4096, 16, 1024, c_dtype=torch.float16)
    with caplog.at_level(logging.WARNING, logger="tritonblas.matmul.k742"):
        assert not _k742_should_override(4096, 16, 1024, a, b, c, work_stealing=False)
    assert any("c.dtype" in r.message for r in caplog.records)


def test_non_row_major_strides_skip_logs_warning(caplog):
    """B with stride(0)!=1 (e.g. row-major K x N instead of K-contig) is rejected."""
    a, b, c = _row_major_inputs(4096, 16, 1024)
    # Force B to row-major K x N: stride(0)=N, stride(1)=1.
    b._stride = (16, 1)
    with caplog.at_level(logging.WARNING, logger="tritonblas.matmul.k742"):
        assert not _k742_should_override(4096, 16, 1024, a, b, c, work_stealing=False)
    assert any("non-row-major strides" in r.message for r in caplog.records)


def test_device_mismatch_skip_logs_warning(caplog):
    a, b, c = _row_major_inputs(4096, 16, 1024)
    c.device = torch.device("cpu")
    c.is_cuda = False
    with caplog.at_level(logging.WARNING, logger="tritonblas.matmul.k742"):
        assert not _k742_should_override(4096, 16, 1024, a, b, c, work_stealing=False)
    assert any("device" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Substitution test: override must actually replace the kernel launch tile
# ---------------------------------------------------------------------------


def test_override_substitutes_kernel_launch_tile():
    """When the predicate fires, BLOCK_SIZE_{M,N,K}/num_warps/num_stages
    passed into the persistent_matmul launch must equal _K742_FP8_TS_TILE,
    not the (deliberately wrong) values returned by the selector."""
    M, N, K = 8192, 64, 4096
    captured = {}

    def fake_launcher(*args, **kwargs):
        captured.update(kwargs)
        return None

    class _FakeKernel:
        def __getitem__(self, grid):
            return fake_launcher

    a, b, c = _row_major_inputs(M, N, K)
    # Fill in the bits persistent_matmul_lt reads off the Tensor.
    a.shape = (M, K)
    b.shape = (K, N)
    c.shape = (M, N)

    selector = types.SimpleNamespace(
        block_m=256, block_n=256, block_k=64,    # over-tiled (Origami-style)
        group_m=8, num_sms=8, num_stages=3,
        COUNTERS_PER_XCD=8,
    )

    with mock.patch.object(tb_matmul, "persistent_matmul", _FakeKernel()), \
         mock.patch.object(tb_matmul, "_maybe_wrap", lambda fn, probe_tensor: fn):
        tb_matmul.persistent_matmul_lt(a, b, c, selector, work_stealing=False)

    bm, bn, bk, nw, ns = _K742_FP8_TS_TILE
    assert captured["BLOCK_SIZE_M"] == bm
    assert captured["BLOCK_SIZE_N"] == bn
    assert captured["BLOCK_SIZE_K"] == bk
    assert captured["num_warps"] == nw
    assert captured["num_stages"] == ns


def test_override_does_not_fire_for_excluded_cell():
    """Excluded (8192, 128, 1024) must keep the selector's choice intact."""
    M, N, K = 8192, 128, 1024
    captured = {}

    def fake_launcher(*args, **kwargs):
        captured.update(kwargs)
        return None

    class _FakeKernel:
        def __getitem__(self, grid):
            return fake_launcher

    a, b, c = _row_major_inputs(M, N, K)
    a.shape = (M, K); b.shape = (K, N); c.shape = (M, N)
    selector = types.SimpleNamespace(
        block_m=128, block_n=128, block_k=128,   # selector's pick — must survive
        group_m=8, num_sms=8, num_stages=3,
        COUNTERS_PER_XCD=8,
    )
    with mock.patch.object(tb_matmul, "persistent_matmul", _FakeKernel()), \
         mock.patch.object(tb_matmul, "_maybe_wrap", lambda fn, probe_tensor: fn):
        tb_matmul.persistent_matmul_lt(a, b, c, selector, work_stealing=False)

    assert captured["BLOCK_SIZE_M"] == 128
    assert captured["BLOCK_SIZE_N"] == 128
    assert captured["BLOCK_SIZE_K"] == 128
