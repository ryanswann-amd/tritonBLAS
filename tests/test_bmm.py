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
