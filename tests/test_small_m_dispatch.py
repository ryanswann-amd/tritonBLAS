"""
Correctness + dispatch-routing tests for the small-M persistent path
(``tritonblas.dispatch``).

Coverage
--------
1. ``small_m_block_override`` returns sane (LDS-budget-fit, EVEN_K-friendly)
   blocks for the small-M benchmarking cohort (M in {1,8,16,32,64},
   N,K in {1024,2048,4096,8192}).
2. ``should_use_small_m_path`` correctly gates on ``M`` and the
   thread-safe ``set_small_m_enabled`` API.
3. ``small_m_grid`` returns a grid that's clamped to ``num_cus``.
4. End-to-end correctness of ``tritonblas.matmul`` against an fp32
   reference on the small-M cohort, with a tolerance derived from the
   K-summation error bound for fp16/bf16.
5. EVEN_K=False edge case: K not divisible by the override's preferred
   BLOCK_K still produces correct output (the override walks BLOCK_K
   down to keep EVEN_K true; this test pins K to a value that exercises
   that walk).

Tolerance model (fp16 / bf16 GEMM with fp32 accumulator):
    |err_ij| <= K * eps_acc * max|A_i:| * max|B_:j|
For our normalized N(0,1) inputs and K up to 8192, max|A|, max|B| are
bounded by ~6 with high probability, and eps_acc for fp32 accum + fp16
load is ~1e-3.  We use ``rtol=1e-2``, ``atol=K * 5e-3`` per element on
the fp32 reference -- tight enough to catch a missed K-iter or wrong
accumulator init, loose enough not to fire on legitimate fp16 round-off.
"""
from __future__ import annotations

import os

import pytest
import torch

import tritonblas
from tritonblas.dispatch import (
    LDS_BUDGET_BYTES,
    NUM_PIPELINE_STAGES,
    SMALL_M_THRESHOLD,
    is_small_m_enabled,
    set_small_m_enabled,
    should_use_small_m_path,
    small_m_block_override,
    small_m_grid,
)


# ---- Cohort definition --------------------------------------------------------
SMALL_M_DIMS = [
    (M, N, K)
    for M in (1, 8, 16, 32, 64)
    for N in (1024, 2048, 4096, 8192)
    for K in (1024, 2048, 4096, 8192)
]

DTYPES = [torch.float16, torch.bfloat16]
NUM_CUS_MI300X = 304


# ---- Unit tests on the dispatch primitives ------------------------------------
@pytest.mark.parametrize("M, N, K", SMALL_M_DIMS)
def test_block_override_lds_budget(M, N, K):
    """The override must always produce blocks that fit in MI300X's 64 KB LDS
    under 2-stage software pipelining for fp16/bf16."""
    bm, bn, bk, gm, nw = small_m_block_override(M, N, K, NUM_CUS_MI300X)
    lds_bytes = NUM_PIPELINE_STAGES * (bm * bk + bk * bn) * 2
    assert lds_bytes <= LDS_BUDGET_BYTES, (
        f"M={M} N={N} K={K}: override (bm={bm} bn={bn} bk={bk}) needs "
        f"{lds_bytes} bytes of LDS, exceeds {LDS_BUDGET_BYTES} budget"
    )


@pytest.mark.parametrize("M, N, K", SMALL_M_DIMS)
def test_block_override_even_k(M, N, K):
    """BLOCK_K must divide K so EVEN_K stays true on the kernel fast path."""
    _, _, bk, _, _ = small_m_block_override(M, N, K, NUM_CUS_MI300X)
    assert K % bk == 0, f"M={M} N={N} K={K}: BLOCK_K={bk} does not divide K={K}"


@pytest.mark.parametrize("M, N, K", SMALL_M_DIMS)
def test_block_override_legal_mfma_tile(M, N, K):
    """BLOCK_M / BLOCK_N must be at least 16 (smallest legal MFMA tile on
    CDNA3) and BLOCK_M must be a multiple of 16."""
    bm, bn, _, _, _ = small_m_block_override(M, N, K, NUM_CUS_MI300X)
    assert bm >= 16 and bm % 16 == 0, f"BLOCK_M={bm} invalid for M={M}"
    assert bn >= 16, f"BLOCK_N={bn} below MFMA minimum for N={N}"


@pytest.mark.parametrize(
    "K_odd, expected_bk_max",
    # K values that are multiples of a power of 2 but not 256, exercising the
    # walk-down loop in small_m_block_override.
    [
        (1088, 64),   # 1088 = 64 * 17, not divisible by 128
        (2112, 64),   # 2112 = 64 * 33
        (4160, 64),   # 4160 = 64 * 65
        (1056, 32),   # 1056 = 32 * 33, not divisible by 64
    ],
)
def test_block_override_walks_bk_for_uneven_k(K_odd, expected_bk_max):
    """When K isn't a multiple of 256/128, BLOCK_K must walk down to a
    divisor so EVEN_K stays true on the fast path."""
    _, _, bk, _, _ = small_m_block_override(16, 4096, K_odd, NUM_CUS_MI300X)
    assert K_odd % bk == 0, f"K={K_odd}: BLOCK_K={bk} does not divide K"
    assert bk >= 16, f"K={K_odd}: BLOCK_K={bk} fell below MFMA minimum"
    assert bk <= expected_bk_max, (
        f"K={K_odd}: BLOCK_K={bk} exceeds the largest legal pow2 divisor "
        f"{expected_bk_max}"
    )


def test_should_use_small_m_threshold():
    """Gate triggers iff M <= SMALL_M_THRESHOLD."""
    assert should_use_small_m_path(SMALL_M_THRESHOLD, 4096, 4096, 16, 32, NUM_CUS_MI300X)
    assert should_use_small_m_path(1, 4096, 4096, 16, 32, NUM_CUS_MI300X)
    assert not should_use_small_m_path(SMALL_M_THRESHOLD + 1, 4096, 4096, 16, 32, NUM_CUS_MI300X)
    assert not should_use_small_m_path(8192, 4096, 4096, 16, 32, NUM_CUS_MI300X)


def test_set_small_m_enabled_thread_safe_api():
    """set_small_m_enabled flips the gate and is reversible.  This is the
    documented replacement for the env-var escape hatch."""
    saved = is_small_m_enabled()
    try:
        set_small_m_enabled(False)
        assert not is_small_m_enabled()
        assert not should_use_small_m_path(1, 4096, 4096, 16, 32, NUM_CUS_MI300X)
        set_small_m_enabled(True)
        assert is_small_m_enabled()
        assert should_use_small_m_path(1, 4096, 4096, 16, 32, NUM_CUS_MI300X)
    finally:
        set_small_m_enabled(saved)


def test_set_small_m_enabled_concurrent():
    """Hammer set_small_m_enabled from multiple threads; the lock should keep
    is_small_m_enabled() consistent (no torn reads / no exceptions).  The
    final state is whatever the last writer left."""
    import threading

    saved = is_small_m_enabled()
    try:
        errors = []

        def flip(target: bool, n: int) -> None:
            try:
                for _ in range(n):
                    set_small_m_enabled(target)
                    is_small_m_enabled()
            except Exception as e:  # pragma: no cover
                errors.append(e)

        ts = [
            threading.Thread(target=flip, args=(i % 2 == 0, 1000))
            for i in range(8)
        ]
        for t in ts: t.start()
        for t in ts: t.join()
        assert not errors, f"thread errors: {errors}"
        # Just assert the final state is some valid bool (the gate didn't
        # land in a weird state).
        assert isinstance(is_small_m_enabled(), bool)
    finally:
        set_small_m_enabled(saved)


@pytest.mark.parametrize("M, N", [(1, 1024), (16, 8192), (32, 4096), (64, 8192)])
def test_grid_clamped_to_num_cus(M, N):
    """``small_m_grid`` clamps the launch grid to ``num_cus`` and returns
    ``NUM_SMS == grid`` so the kernel's grid-stride loop is exact."""
    bm, bn, _, _, _ = small_m_block_override(M, N, 1024, NUM_CUS_MI300X)
    grid, num_sms = small_m_grid(M, N, bm, bn, NUM_CUS_MI300X)
    total_tiles = ((M + bm - 1) // bm) * ((N + bn - 1) // bn)
    assert grid == min(total_tiles, NUM_CUS_MI300X)
    assert grid >= 1
    assert num_sms == grid, "NUM_SMS must equal grid for the persistent loop stride"


# ---- End-to-end correctness on GPU --------------------------------------------
def _has_cuda() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _gemm_tolerance(K: int, dtype: torch.dtype) -> tuple:
    """fp16/bf16 GEMM tolerance against fp32 reference.
    The error bound for K accumulations of (fp16 * fp16) into fp32 is
    O(K * eps_fp32 * max|A| * max|B|).  For unit-variance Gaussian inputs
    max|A|, max|B| ~= 6, eps_fp32 ~= 1e-7, fp16 round-off ~= 1e-3, so the
    dominant term is K * 1e-3 * 6 * 6 ~= K * 4e-2.  We pick rtol=1e-2 (so
    larger entries are bounded by their own magnitude) and atol scaled by K.
    """
    if dtype == torch.bfloat16:
        # bf16 has ~3 fewer bits of mantissa → ~8x looser
        return (1e-2, max(1e-2, K * 8e-3))
    return (1e-2, max(1e-2, K * 1e-3))


# A representative slice of the cohort (kept small to keep CI runtime bounded;
# the full sweep is exercised by the benchmarking script).
E2E_DIMS = [
    (1, 1024, 1024),
    (1, 8192, 8192),
    (8, 2048, 4096),
    (16, 4096, 4096),
    (16, 1024, 8192),  # K-heavy, narrow N -- worst case for the path
    (32, 4096, 2048),
    (32, 8192, 8192),
    (64, 4096, 4096),
    (64, 8192, 8192),
]


@pytest.mark.skipif(not _has_cuda(), reason="requires GPU")
@pytest.mark.parametrize("M, N, K", E2E_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_small_m_matmul_correctness(M, N, K, dtype):
    """tritonblas.matmul on the small-M path matches an fp32 reference within
    the K-scaled GEMM error bound."""
    torch.manual_seed(42)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)

    saved = is_small_m_enabled()
    set_small_m_enabled(True)
    try:
        out = tritonblas.matmul(a, b)
    finally:
        set_small_m_enabled(saved)

    ref = (a.float() @ b.float()).to(dtype)
    rtol, atol = _gemm_tolerance(K, dtype)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not _has_cuda(), reason="requires GPU")
@pytest.mark.parametrize("M, N, K", E2E_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_small_m_disabled_matches_default(M, N, K, dtype):
    """With the small-M path disabled, tritonblas.matmul still produces a
    correct result (sanity check on the fall-through path)."""
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)

    saved = is_small_m_enabled()
    set_small_m_enabled(False)
    try:
        out = tritonblas.matmul(a, b)
    finally:
        set_small_m_enabled(saved)

    ref = (a.float() @ b.float()).to(dtype)
    rtol, atol = _gemm_tolerance(K, dtype)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not _has_cuda(), reason="requires GPU")
@pytest.mark.parametrize("M, N, K", [(1, 8192, 8192), (32, 4096, 4096)])
def test_small_m_path_matches_disabled_path(M, N, K):
    """The small-M and legacy paths produce results that match each other
    within fp16 tolerance.  Regression guard: a missed K-iter or mis-tile
    in the small-M path will diverge from the legacy path by far more than
    the legitimate per-path round-off difference."""
    torch.manual_seed(7)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    saved = is_small_m_enabled()
    try:
        set_small_m_enabled(True)
        out_small = tritonblas.matmul(a, b)
        set_small_m_enabled(False)
        out_legacy = tritonblas.matmul(a, b)
    finally:
        set_small_m_enabled(saved)

    # Both paths use fp32 accumulator and the same persistent kernel; the
    # only difference is tile order.  Bound the diff by ~2x the K-scaled
    # round-off (each path can drift by the per-path error bound).
    rtol, atol = _gemm_tolerance(K, torch.float16)
    torch.testing.assert_close(out_small, out_legacy, atol=2 * atol, rtol=2 * rtol)


@pytest.mark.skipif(not _has_cuda(), reason="requires GPU")
@pytest.mark.parametrize("M, N, K", [(16, 4096, 1500), (32, 4096, 3000)])
def test_small_m_uneven_k_correctness(M, N, K):
    """Edge case: K is NOT a multiple of the preferred BLOCK_K (256).  The
    override walks BLOCK_K down to a divisor; if that walk is buggy the
    EVEN_K=True fast path will silently drop the tail K iter and the result
    will be visibly wrong.  This test keeps the failure mode explicit."""
    torch.manual_seed(13)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    saved = is_small_m_enabled()
    set_small_m_enabled(True)
    try:
        out = tritonblas.matmul(a, b)
    finally:
        set_small_m_enabled(saved)

    ref = (a.float() @ b.float()).to(torch.float16)
    rtol, atol = _gemm_tolerance(K, torch.float16)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
