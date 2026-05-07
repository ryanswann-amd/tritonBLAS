# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + performance gate tests for tritonblas.bmm
(true batched matmul)."""

import math
import os

import pytest
import torch

import tritonblas


def _ktol(dtype: torch.dtype, K: int, scale: float = 8.0) -> float:
    """K-aware tolerance: scale * sqrt(K) * eps(dtype)."""
    eps = torch.finfo(dtype).eps if dtype.is_floating_point else 1.0
    return scale * math.sqrt(K) * eps


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch,M,N,K",
    [
        (4, 128, 128, 128),
        (4, 1024, 1024, 1024),
        (8, 512, 512, 512),
        (8, 1024, 1024, 1024),
        (4, 2048, 2048, 2048),
        (2, 256, 512, 1024),  # non-square, sanity
    ],
)
def test_bmm_correctness(batch, M, N, K, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    a = torch.randn(batch, M, K, device="cuda", dtype=dtype)
    b = torch.randn(batch, K, N, device="cuda", dtype=dtype)

    out = tritonblas.bmm(a, b)
    ref = torch.bmm(a, b)

    diff = (out.float() - ref.float()).abs().max().item()
    tol = _ktol(dtype, K)
    assert diff < tol, (
        f"bmm correctness failed: shape=({batch},{M},{N},{K}) dtype={dtype} "
        f"max_abs_err={diff} tol={tol}"
    )


def test_bmm_dispatch_through_matmul():
    """tritonblas.matmul should dispatch rank-3 inputs to bmm()."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    a = torch.randn(2, 64, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(2, 128, 64, device="cuda", dtype=torch.float16)
    out = tritonblas.matmul(a, b)
    ref = torch.bmm(a, b)
    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < _ktol(torch.float16, 128), f"matmul rank-3 dispatch failed (err={diff})"


def test_bmm_out_argument():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    a = torch.randn(3, 64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(3, 64, 64, device="cuda", dtype=torch.float16)
    out = torch.empty(3, 64, 64, device="cuda", dtype=torch.float16)
    ret = tritonblas.bmm(a, b, out=out)
    assert ret is out, "bmm should return the supplied out tensor"
    ref = torch.bmm(a, b)
    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < _ktol(torch.float16, 64)


# ---------------------------------------------------------------------------
# Adversarial / contract tests (reviewer feedback K-686 retry)
# ---------------------------------------------------------------------------


def test_bmm_batch_one():
    """BATCH=1 must produce identical numerics to the unbatched path."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    a = torch.randn(1, 256, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(1, 128, 256, device="cuda", dtype=torch.float16)
    out_bmm = tritonblas.bmm(a, b)
    out_ref = torch.bmm(a, b)
    diff = (out_bmm.float() - out_ref.float()).abs().max().item()
    assert diff < _ktol(torch.float16, 128)
    # Also verify it's not silently squeezing the batch dim
    assert out_bmm.shape == (1, 256, 256), f"expected (1,256,256), got {tuple(out_bmm.shape)}"


def test_bmm_non_contiguous_a():
    """Non-contiguous A (post-transpose, square) must produce the same
    result as a contiguous-A call. The kernel reads strides explicitly
    (``stride_am``, ``stride_ak``, ``stride_ab``) so this must work
    without the caller having to ``.contiguous()``."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    # Build a non-contig view by transposing (M, K) on a square block —
    # contents differ from the source so we can compare to .contiguous().
    a_t = torch.randn(2, 128, 128, device="cuda", dtype=torch.float16)
    a = a_t.transpose(1, 2)  # (2, 128, 128) but stride_am < stride_ak
    assert not a.is_contiguous(), "test setup: A should be non-contiguous"
    b = torch.randn(2, 128, 64, device="cuda", dtype=torch.float16)
    out_nc = tritonblas.bmm(a, b)
    out_c = tritonblas.bmm(a.contiguous(), b)
    diff = (out_nc.float() - out_c.float()).abs().max().item()
    # Same kernel, same inputs (modulo layout) → must be bitwise-equal
    # within FP rounding noise: use tighter tol than K-aware bound.
    assert diff < 1e-3, f"non-contig vs contig disagree: {diff}"


def test_bmm_non_contiguous_b():
    """Same as above for B — exercises stride_bk and stride_bn."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    a = torch.randn(2, 64, 128, device="cuda", dtype=torch.float16)
    b_t = torch.randn(2, 256, 128, device="cuda", dtype=torch.float16)
    b = b_t.transpose(1, 2)  # (2, 128, 256), non-contig
    assert not b.is_contiguous()
    out_nc = tritonblas.bmm(a, b)
    out_c = tritonblas.bmm(a, b.contiguous())
    diff = (out_nc.float() - out_c.float()).abs().max().item()
    assert diff < 1e-3, f"non-contig B vs contig disagree: {diff}"


@pytest.mark.parametrize("dtype,K", [
    (torch.float16, 130),   # K not divisible by typical BLOCK_SIZE_K (64/128/256)
    (torch.bfloat16, 257),  # forces EVEN_K=False tail-K mask path
    (torch.float16, 33),    # very small odd K
])
def test_bmm_k_not_divisible_by_tile(dtype, K):
    """Exercise the EVEN_K=False tail-K mask branch in the kernel."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    a = torch.randn(3, 128, K, device="cuda", dtype=dtype)
    b = torch.randn(3, K, 128, device="cuda", dtype=dtype)
    out = tritonblas.bmm(a, b)
    ref = torch.bmm(a, b)
    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < _ktol(dtype, K), (
        f"tail-K mask broken for K={K} dtype={dtype}: err={diff}"
    )


def test_bmm_rejects_rank_2():
    """``bmm`` should reject rank-2 inputs with a clear error, not crash."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="rank-3"):
        tritonblas.bmm(a, b)


def test_bmm_rejects_batch_mismatch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    a = torch.randn(2, 64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(3, 64, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="batch"):
        tritonblas.bmm(a, b)


def test_bmm_rejects_inner_dim_mismatch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    a = torch.randn(2, 64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(2, 32, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="inner"):
        tritonblas.bmm(a, b)


def test_bmm_rejects_dtype_mismatch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    a = torch.randn(2, 64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(2, 64, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="dtype"):
        tritonblas.bmm(a, b)


def test_bmm_rejects_bad_out_shape():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    a = torch.randn(2, 64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(2, 64, 64, device="cuda", dtype=torch.float16)
    bad_out = torch.empty(2, 64, 32, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="out shape"):
        tritonblas.bmm(a, b, out=bad_out)


def test_bmm_rejects_bad_out_dtype():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    a = torch.randn(2, 64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(2, 64, 64, device="cuda", dtype=torch.float16)
    bad_out = torch.empty(2, 64, 64, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="out.dtype"):
        tritonblas.bmm(a, b, out=bad_out)


# ---------------------------------------------------------------------------
# Performance gate — bmm must be substantially faster than the host-side
# Python loop on the K-654 batched residual shapes. This asserts the
# *structural* improvement the bmm entrypoint exists to deliver
# (single-launch over BATCH * tiles vs O(BATCH) per-element launches);
# it does NOT assert parity with hipBLASLt — the absolute headroom there
# is tracked separately in the K-686 sweep CSVs and noted in the PR
# description's "Status vs success criterion" section.
#
# The lower bound (5x speedup vs the Python loop) is conservative: the
# K-654 measurements showed ~14x for 1024^3 b=8 bf16, and the in-PR
# targeted bench shows 7-50x across the four shapes. A regression below
# 5x indicates the persistent grid / chiplet / selector-cache
# machinery has been broken, which is exactly the kind of silent
# regression this gate is meant to catch.
# ---------------------------------------------------------------------------

# Honour the same env-var convention used by the rest of the repo's
# perf-bearing tests so CI can opt-out via TRITONBLAS_SKIP_PERF=1 on
# emulator / CPU-only nodes.
_SKIP_PERF = os.environ.get("TRITONBLAS_SKIP_PERF", "0") == "1"


def _time_min_us(fn, ev_pair, warmup=5, iters=20):
    s, e = ev_pair
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_us = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times_us.append(s.elapsed_time(e) * 1e3)
    return min(times_us)


@pytest.mark.skipif(_SKIP_PERF, reason="TRITONBLAS_SKIP_PERF=1")
@pytest.mark.parametrize(
    "batch,M,N,K,dtype,min_speedup",
    [
        # K-654 r1 / r2 (top-2 residuals). 5x is the floor below which
        # the persistent-grid implementation is considered broken.
        (4, 2048, 2048, 2048, torch.float16, 5.0),
        (8, 1024, 1024, 1024, torch.bfloat16, 5.0),
        (4, 1024, 1024, 1024, torch.float16, 5.0),
        (8, 512, 512, 512, torch.float16, 5.0),
    ],
)
def test_bmm_speedup_vs_host_loop(batch, M, N, K, dtype, min_speedup):
    """bmm must beat the host-side per-element loop by at least
    ``min_speedup`` on the K-654 residual shapes — the structural
    invariant that the bmm entrypoint exists to enforce."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    a = torch.randn(batch, M, K, device="cuda", dtype=dtype)
    b = torch.randn(batch, K, N, device="cuda", dtype=dtype)
    out_b = torch.empty(batch, M, N, device="cuda", dtype=dtype)
    ev_pair = (torch.cuda.Event(enable_timing=True),
               torch.cuda.Event(enable_timing=True))

    def bmm_call():
        tritonblas.bmm(a, b, out=out_b)

    def loop_call():
        for i in range(batch):
            tritonblas.matmul(a[i], b[i], out=out_b[i])

    t_bmm = _time_min_us(bmm_call, ev_pair)
    t_loop = _time_min_us(loop_call, ev_pair)
    speedup = t_loop / max(t_bmm, 1e-6)
    assert speedup >= min_speedup, (
        f"bmm speedup vs host loop on b={batch} {M}x{N}x{K} {dtype}: "
        f"loop={t_loop:.1f}us bmm={t_bmm:.1f}us speedup={speedup:.2f}x "
        f"< floor {min_speedup}x — persistent batched grid likely regressed."
    )
