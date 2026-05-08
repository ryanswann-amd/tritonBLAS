"""Correctness + dispatch-routing tests for the small-M persistent and
split-K paths (``tritonblas.dispatch``).

Coverage
--------
1. ``small_m_block_override`` returns LDS-budget-fit, EVEN_K blocks for
   the full K-169 small-M cohort (happy path + invariants).
2. Edge case: BLOCK_K walks down for K not divisible by 256/128
   (EVEN_K=False guard).
3. Gate behaviour:
   * ``M`` threshold (positive + negative).
   * Env-var disable: ``TRITONBLAS_SMALL_M_DISABLE=1`` forces False.
   * M=64 + narrow-N negative case (preserves baseline performance).
4. Split-K gate: triggers on the structurally-CU-starved corner
   (M<=16, N<=2048, K>=8192) and rejects shapes outside it.
5. Grid clamp: ``small_m_grid`` returns ``min(total_tiles, num_cus)``.
6. End-to-end correctness on GPU (skipped without a GPU):
   * Both override paths match an fp32 reference at K-scaled tolerance.
   * The K=8192 narrow-N split-K cohort is covered explicitly.
   * Cross-path agreement: override and disabled paths agree within
     combined K-scaled tolerance.
   * Negative routing: ``addmm`` (bias) BYPASSES the override.

Tolerance model
---------------
For fp16 / bf16 GEMM with fp32 accumulator:

    |err_ij| <= K * eps_acc * max|A_i:| * max|B_:j|

For unit-variance Gaussian inputs max|A|, max|B| are ~6 with high
probability; eps_acc for an fp32 accumulator with fp16 loads is dominated
by the fp16 round-off term (~2**-10 ~= 1e-3).  So the dominant error term
is ~K * 6e-3 per element.  We use ``rtol=1e-2`` (so larger entries are
bounded by their own magnitude) and ``atol = K * 1e-3`` for fp16 (~6x
safety margin).  bf16 has ~3 fewer mantissa bits => ``atol = K * 8e-3``.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

# Workaround for a torch+ROCm hipBLASLt bug in older PyTorch ROCm builds
# (observed on torch 2.9.1+rocm6.3) where fp32 GEMM with M>=2 raises
# "HIP error: named symbol not found".  Setting these *before* importing
# torch forces rocBLAS, which has no such issue.  No effect on newer
# torch/ROCm builds where hipBLASLt works fine.
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
os.environ.setdefault("DISABLE_ADDMM_HIP_LT", "1")
os.environ.setdefault("ROCBLAS_USE_HIPBLASLT", "0")

import pytest
import torch

# Belt-and-braces: switch the preferred BLAS library at the torch level
# too, in case the env vars weren't read before torch imported its native
# bindings.
try:
    torch.backends.cuda.preferred_blas_library("cublas")
except Exception:
    pass

import tritonblas
from tritonblas.dispatch import (
    LDS_BUDGET_BYTES,
    MAX_SPLIT_K,
    MIN_BK_ITERS_PER_SHARD,
    NUM_PIPELINE_STAGES,
    SMALL_M_THRESHOLD,
    _split_k_factor,
    should_use_small_m_path,
    should_use_split_k_path,
    small_m_block_override,
    small_m_grid,
    split_k_block_override,
)
from tritonblas.kernels.split_k_gemm import split_k_matmul as _split_k_matmul_fn


SMALL_M_DIMS = [
    (M, N, K)
    for M in (1, 8, 16, 32, 64)
    for N in (1024, 2048, 4096, 8192)
    for K in (1024, 2048, 4096, 8192)
]

DTYPES = [torch.float16, torch.bfloat16]
NUM_CUS_MI300X = 304


@contextmanager
def _env(key, value):
    sentinel = object()
    saved = os.environ.get(key, sentinel)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    try:
        yield
    finally:
        if saved is sentinel:
            os.environ.pop(key, None)
        else:
            os.environ[key] = saved


@pytest.mark.parametrize("M, N, K", SMALL_M_DIMS)
def test_block_override_lds_budget(M, N, K):
    bm, bn, bk, gm, nw = small_m_block_override(M, N, K, NUM_CUS_MI300X)
    lds_bytes = NUM_PIPELINE_STAGES * (bm * bk + bk * bn) * 2
    assert lds_bytes <= LDS_BUDGET_BYTES


@pytest.mark.parametrize("M, N, K", SMALL_M_DIMS)
def test_block_override_even_k(M, N, K):
    _, _, bk, _, _ = small_m_block_override(M, N, K, NUM_CUS_MI300X)
    assert K % bk == 0


@pytest.mark.parametrize("M, N, K", SMALL_M_DIMS)
def test_block_override_legal_mfma_tile(M, N, K):
    bm, bn, _, _, _ = small_m_block_override(M, N, K, NUM_CUS_MI300X)
    assert bm >= 16 and bm % 16 == 0
    assert bn >= 16


@pytest.mark.parametrize(
    "K_odd, expected_bk_max",
    [(1088, 64), (2112, 64), (4160, 64), (1056, 32)],
)
def test_block_override_walks_bk_for_uneven_k(K_odd, expected_bk_max):
    """Negative case: a buggy walk leaving BK=128 would make EVEN_K a lie
    and silently drop the tail K iter."""
    _, _, bk, _, _ = small_m_block_override(16, 4096, K_odd, NUM_CUS_MI300X)
    assert K_odd % bk == 0
    assert 16 <= bk <= expected_bk_max


def test_should_use_small_m_threshold():
    assert should_use_small_m_path(SMALL_M_THRESHOLD, 4096, 4096, 16, 32, NUM_CUS_MI300X)
    assert should_use_small_m_path(1, 4096, 4096, 16, 32, NUM_CUS_MI300X)
    assert not should_use_small_m_path(SMALL_M_THRESHOLD + 1, 4096, 4096, 16, 32, NUM_CUS_MI300X)
    assert not should_use_small_m_path(8192, 4096, 4096, 16, 32, NUM_CUS_MI300X)


def test_should_use_small_m_env_disabled():
    """TRITONBLAS_SMALL_M_DISABLE=1 forces the gate to False (escape hatch
    for A/B benchmarking).  Negative case for the gate."""
    with _env("TRITONBLAS_SMALL_M_DISABLE", "1"):
        assert not should_use_small_m_path(1, 4096, 4096, 16, 32, NUM_CUS_MI300X)
    with _env("TRITONBLAS_SMALL_M_DISABLE", None):
        assert should_use_small_m_path(1, 4096, 4096, 16, 32, NUM_CUS_MI300X)


def test_should_use_small_m_env_false_value_is_enabled():
    for val in ("", "0", "false", "False"):
        with _env("TRITONBLAS_SMALL_M_DISABLE", val):
            assert should_use_small_m_path(1, 4096, 4096, 16, 32, NUM_CUS_MI300X)


def test_should_use_small_m_skips_m64_narrow_n():
    """M=64 + narrow N: baseline beats override -> gate must skip."""
    assert not should_use_small_m_path(64, 1024, 4096, 32, 32, NUM_CUS_MI300X)
    assert not should_use_small_m_path(64, 2048, 8192, 32, 32, NUM_CUS_MI300X)
    assert should_use_small_m_path(64, 4096, 4096, 32, 32, NUM_CUS_MI300X)
    assert should_use_small_m_path(64, 8192, 2048, 32, 64, NUM_CUS_MI300X)


def test_split_k_gate_positive_cu_starved():
    """Gate fires on the structurally CU-starved corner.

    Cases (each must satisfy: TILE_UNDERSUB_FACTOR=2 AND tile floor AND
    K >= MIN_K_FOR_SPLIT_K AND each shard has MIN_BK_ITERS+ iters):
    """
    # M=1 N=1024 K=8192: tiles=32, K=8192 floor=32 -- just at boundary.
    assert should_use_split_k_path(1, 1024, 8192, NUM_CUS_MI300X)
    # M=8 N=2048 K=8192: tiles=64, K=8192 floor=32 -- well above.
    assert should_use_split_k_path(8, 2048, 8192, NUM_CUS_MI300X)
    # M=16 N=1024 K=8192: tiles=32, K=8192 floor=32 -- at boundary.
    assert should_use_split_k_path(16, 1024, 8192, NUM_CUS_MI300X)
    # M=64 N=1024 K=8192 still fires (independent of small_m gate):
    # tiles = 2*32 = 64, K=8192 floor=32, tiles*2=128 <= 304.
    assert should_use_split_k_path(64, 1024, 8192, NUM_CUS_MI300X)
    # M=64 N=2048 K=4096 fires (SM gate excludes M=64 N<4096 but SK
    # gate is independent): tiles=2*64=128, K=4096 floor=128.
    assert should_use_split_k_path(64, 2048, 4096, NUM_CUS_MI300X)
    # K=4096 wide-N (N=4096): tiles=128 hits the K=4096 floor=128.
    assert should_use_split_k_path(8, 4096, 4096, NUM_CUS_MI300X)


def test_split_k_gate_negative():
    """Negative cases (each guard must independently reject):
      * M > SMALL_M_THRESHOLD: out of small-M scope entirely.
      * Tile count above num_cus / TILE_UNDERSUB_FACTOR (saturated).
      * Tile count below the per-K floor (overhead exceeds parallelism).
      * K below MIN_K_FOR_SPLIT_K (overhead dominates).
    """
    # M too large for the small-M scope.
    assert not should_use_split_k_path(SMALL_M_THRESHOLD + 1, 1024, 8192, NUM_CUS_MI300X)
    # M=16, N=8192: bm=16,bn=32,bk=256 -> tiles=1*256=256, tiles*2=512>304 -> off.
    assert not should_use_split_k_path(16, 8192, 8192, NUM_CUS_MI300X)
    # K below MIN_K_FOR_SPLIT_K (4096): split-K's per-tile bookkeeping
    # + atomic-store reduction overhead exceeds the parallelism win.
    assert not should_use_split_k_path(1, 1024, 1024, NUM_CUS_MI300X)
    assert not should_use_split_k_path(1, 1024, 2048, NUM_CUS_MI300X)
    # K=4096 narrow-N below the per-K tile floor (=128):
    # M=1 N=1024 K=4096 has tiles=32 < 128 -> off.
    assert not should_use_split_k_path(1, 1024, 4096, NUM_CUS_MI300X)
    # M=8 N=2048 K=4096: tiles=64 < 128 -> off (narrow-N at K=4096
    # regresses 0.05-0.27x with SK firing -- bench-driven veto).
    assert not should_use_split_k_path(8, 2048, 4096, NUM_CUS_MI300X)


def test_split_k_gate_env_disabled():
    """Env-var escape hatch: TRITONBLAS_SMALL_M_DISABLE=1 forces gate False
    even when the shape would normally fire."""
    with _env("TRITONBLAS_SMALL_M_DISABLE", "1"):
        assert not should_use_split_k_path(1, 1024, 8192, NUM_CUS_MI300X)
    # Sanity: re-enabled once the env var is unset.
    with _env("TRITONBLAS_SMALL_M_DISABLE", None):
        assert should_use_split_k_path(1, 1024, 8192, NUM_CUS_MI300X)


@pytest.mark.parametrize("M, N, K",
    [(1, 1024, 8192), (8, 2048, 8192), (16, 1024, 8192), (16, 2048, 8192)])
def test_split_k_override_invariants(M, N, K):
    """Invariants on the split-K override:
      * BLOCK_K divides K (EVEN_K stays true on the kernel hot path).
      * SPLIT_K is a power of 2 in [2, MAX_SPLIT_K].
      * Each K-shard has at least MIN_BK_ITERS_PER_SHARD BLOCK_K iters
        (no shard so small that the MFMA pipeline can't fill).
      * Launch grid (tiles * SPLIT_K) is bounded by num_cus -- no chip
        oversubscription.
      * LDS budget invariant inherited from small_m_block_override.
    """
    bm, bn, bk, gm, nw, sk = split_k_block_override(M, N, K, NUM_CUS_MI300X)
    assert K % bk == 0
    assert 2 <= sk <= MAX_SPLIT_K
    assert (sk & (sk - 1)) == 0, f"SPLIT_K={sk} not a power of 2"
    iters_per_shard = (K // bk) // sk
    assert iters_per_shard >= MIN_BK_ITERS_PER_SHARD
    tiles = ((M + bm - 1) // bm) * ((N + bn - 1) // bn)
    assert tiles * sk <= NUM_CUS_MI300X
    lds_bytes = NUM_PIPELINE_STAGES * (bm * bk + bk * bn) * 2
    assert lds_bytes <= LDS_BUDGET_BYTES


def test_split_k_factor_returns_one_when_subscribed():
    """Shapes where the persistent grid-stride is already enough -> factor
    drops to 1 (gate stays off)."""
    # M=64, N=8192: bm=32, bn=64 -> tiles = 2*128 = 256. tiles*2=512>304
    # -> persistent path saturates the chip, no SK win.
    assert _split_k_factor(64, 8192, 4096, NUM_CUS_MI300X) == 1
    assert _split_k_factor(64, 8192, 8192, NUM_CUS_MI300X) == 1
    # K=2048 < MIN_K_FOR_SPLIT_K (=4096) -> off regardless of tile count.
    assert _split_k_factor(1, 1024, 1024, NUM_CUS_MI300X) == 1
    assert _split_k_factor(1, 1024, 2048, NUM_CUS_MI300X) == 1
    # K=4096 narrow-N below the K=4096 tile floor (128):
    # M=8 N=2048 K=4096: tiles=64 < 128 -> off.
    assert _split_k_factor(8, 2048, 4096, NUM_CUS_MI300X) == 1
    # M=1 N=2048 K=4096: tiles=64 < 128 -> off.
    assert _split_k_factor(1, 2048, 4096, NUM_CUS_MI300X) == 1


@pytest.mark.parametrize("M, N", [(1, 1024), (16, 8192), (32, 4096), (64, 8192)])
def test_grid_clamped_to_num_cus(M, N):
    bm, bn, _, _, _ = small_m_block_override(M, N, 1024, NUM_CUS_MI300X)
    grid, num_sms = small_m_grid(M, N, bm, bn, NUM_CUS_MI300X)
    total_tiles = ((M + bm - 1) // bm) * ((N + bn - 1) // bn)
    assert grid == min(total_tiles, NUM_CUS_MI300X)
    assert num_sms == grid


def _has_cuda():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _gemm_tolerance(K, dtype):
    if dtype == torch.bfloat16:
        return (1e-2, max(1e-2, K * 8e-3))
    return (1e-2, max(1e-2, K * 1e-3))


E2E_DIMS = [
    (1, 1024, 1024), (1, 8192, 8192),
    (8, 2048, 4096),
    (16, 4096, 4096),
    (16, 1024, 8192),  # split-K
    (16, 2048, 8192),  # split-K
    (32, 4096, 2048), (32, 8192, 8192),
    (64, 4096, 4096), (64, 8192, 8192),
]

SPLIT_K_DIMS = [
    (1, 1024, 8192), (1, 2048, 8192),
    (8, 1024, 8192), (8, 2048, 8192),
    (16, 1024, 8192), (16, 2048, 8192),
]


@pytest.mark.skipif(not _has_cuda(), reason="requires GPU")
@pytest.mark.parametrize("M, N, K", E2E_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_small_m_matmul_correctness(M, N, K, dtype):
    torch.manual_seed(42)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    with _env("TRITONBLAS_SMALL_M_DISABLE", None):
        out = tritonblas.matmul(a, b)
    ref = (a.float() @ b.float()).to(dtype)
    rtol, atol = _gemm_tolerance(K, dtype)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not _has_cuda(), reason="requires GPU")
@pytest.mark.parametrize("M, N, K", SPLIT_K_DIMS)
def test_split_k_path_correctness(M, N, K):
    """The new split-K path must match the fp32 reference.  A correctness
    regression here would be silent in the persistent-only test cases
    since they don't exercise it."""
    assert should_use_split_k_path(M, N, K, NUM_CUS_MI300X)
    torch.manual_seed(11)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    with _env("TRITONBLAS_SMALL_M_DISABLE", None):
        out = tritonblas.matmul(a, b)
    ref = (a.float() @ b.float()).to(torch.float16)
    rtol, atol = _gemm_tolerance(K, torch.float16)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not _has_cuda(), reason="requires GPU")
@pytest.mark.parametrize("M, N, K", E2E_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_small_m_disabled_matches_default(M, N, K, dtype):
    """Sanity check: disabling the override still produces a correct result
    (fall-through to the Origami default path)."""
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    with _env("TRITONBLAS_SMALL_M_DISABLE", "1"):
        out = tritonblas.matmul(a, b)
    ref = (a.float() @ b.float()).to(dtype)
    rtol, atol = _gemm_tolerance(K, dtype)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not _has_cuda(), reason="requires GPU")
@pytest.mark.parametrize("M, N, K", [(1, 8192, 8192), (16, 1024, 8192), (32, 4096, 4096)])
def test_small_m_path_matches_disabled_path(M, N, K):
    """Cross-path regression guard: a missed K-iter or wrong tile in either
    path will diverge by far more than the legitimate per-path round-off
    difference.  Includes a K=8192 narrow-N split-K shape."""
    torch.manual_seed(7)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    with _env("TRITONBLAS_SMALL_M_DISABLE", None):
        out_override = tritonblas.matmul(a, b)
    with _env("TRITONBLAS_SMALL_M_DISABLE", "1"):
        out_legacy = tritonblas.matmul(a, b)
    rtol, atol = _gemm_tolerance(K, torch.float16)
    torch.testing.assert_close(out_override, out_legacy, atol=2 * atol, rtol=2 * rtol)


@pytest.mark.skipif(not _has_cuda(), reason="requires GPU")
@pytest.mark.parametrize("M, N, K", [(16, 4096, 1500), (32, 4096, 3000)])
def test_small_m_uneven_k_correctness(M, N, K):
    """K not a multiple of 256.  The override walks BLOCK_K down; if buggy
    the EVEN_K=True fast path silently drops the tail K iter."""
    torch.manual_seed(13)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    with _env("TRITONBLAS_SMALL_M_DISABLE", None):
        out = tritonblas.matmul(a, b)
    ref = (a.float() @ b.float()).to(torch.float16)
    rtol, atol = _gemm_tolerance(K, torch.float16)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not _has_cuda(), reason="requires GPU")
@pytest.mark.parametrize("M, N, K, SPLIT_K",
    [(1, 1024, 8192, 4),
     (8, 2048, 8192, 2),
     (16, 1024, 4096, 4),
     (1, 1024, 4096, 8)])
def test_split_k_kernel_direct(M, N, K, SPLIT_K):
    """Drive the split_k_matmul kernel directly (bypassing the dispatcher)
    so a bug in the kernel itself is caught even if the gate decides not
    to fire it.  Also exercises a few SPLIT_K values explicitly so the
    SPLIT_K computation in _split_k_factor doesn't mask kernel bugs by
    always picking the same factor.
    """
    torch.manual_seed(99)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    bm, bn, bk = 16, 32, 128
    while K % bk != 0 and bk > 16:
        bk //= 2
    _split_k_matmul_fn(a, b, c, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=SPLIT_K)
    ref = (a.float() @ b.float()).to(torch.float16)
    rtol, atol = _gemm_tolerance(K, torch.float16)
    torch.testing.assert_close(c, ref, atol=atol, rtol=rtol)


def test_split_k_factor_respects_min_iters_per_shard():
    """Negative case: when K is too small to give every shard
    MIN_BK_ITERS_PER_SHARD iters, _split_k_factor must fall back to a
    smaller SPLIT_K (or 1) rather than producing a shard that drops MFMA
    pipeline fill."""
    # Pick a CU-starved shape with a borderline K.
    # M=1 N=1024 K=1024: bm=16,bn=16,bk=128 (LDS-walked), tiles=64,
    # iters_total = 1024/128 = 8. max_sk_by_K = 8/4 = 2 -> sk=2 is OK.
    sk = _split_k_factor(1, 1024, 1024, NUM_CUS_MI300X)
    bm, bn, bk, _, _ = small_m_block_override(1, 1024, 1024, NUM_CUS_MI300X)
    iters_per_shard = (1024 // bk) // max(1, sk)
    assert iters_per_shard >= MIN_BK_ITERS_PER_SHARD or sk == 1


@pytest.mark.skipif(not _has_cuda(), reason="requires GPU")
def test_addmm_bias_does_not_take_override():
    """Negative routing: addmm (bias path) must NOT take the small-M
    override.  A buggy router that took it anyway would silently drop the
    bias because the override branch has no BIAS=True wiring."""
    M, N, K = 16, 1024, 4096
    torch.manual_seed(3)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    bias = torch.randn(N, device="cuda", dtype=torch.float16)
    out = tritonblas.addmm(bias, a, b)
    ref = ((a.float() @ b.float()) + bias.float()).to(torch.float16)
    rtol, atol = _gemm_tolerance(K, torch.float16)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
