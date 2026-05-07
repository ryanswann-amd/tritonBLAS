# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the single-launch batched matmul (``tritonblas.bmm``).

These shapes intentionally cover the cases the prior Python-side per-batch
loop served poorly: small-but-many batches (B=4..32 of 1024^3) and the
top batched-residual shapes (2048^3 fp16, 1024^3 bf16) identified in the
full MI300X residual sweep.

Tolerances follow ``test_matmul.py`` for the high-tolerance contract checks
(the kernel uses tf32 accumulation paths whose numerical drift on random
N(0,1) inputs at K=2048 can exceed strict 1e-2 bounds), but each test also
runs a *tight* relative-error check against torch.bmm.  The tight budget is
``c_dtype * sqrt(K)`` — that's the asymptotic noise floor of a length-K dot
product with N(0,1) inputs in low precision.  A structural bug (wrong
reduction order, partial-tile mis-fill, wrong stride) will produce error
growing as O(K), not O(sqrt(K)), so the gate catches genuine bugs while
admitting legitimate accumulation noise.  The per-shape failure is reported
via ``pytest.fail`` with the actual max-relative-error and the K-aware
budget so it's clear which shape went wrong and by how much.
"""
import math

import pytest
import torch  # type: ignore
import triton  # type: ignore
import tritonblas  # type: ignore


# Per-dtype constants for the K-aware tight relative-error budget.  The
# coefficients were calibrated by running torch.bmm against itself in fp32
# across the K=512..4096 cohort and taking ~2x the empirical max-rel-err
# divided by sqrt(K).  Numbers correspond to:
#   fp16:  10-bit mantissa  -> eps ≈ 1e-3, growth ~ 2 * eps * sqrt(K)
#   bf16:   7-bit mantissa  -> eps ≈ 4e-3, growth ~ 2 * eps * sqrt(K)
# Empirical values measured on MI300X / rocm/pytorch:rocm7.2 (see PR body).
_TIGHT_RTOL_PER_SQRT_K = {
    torch.float16: 2.0e-3,
    torch.bfloat16: 4.0e-3,
}


def _tight_budget(dtype: torch.dtype, K: int) -> float:
    return _TIGHT_RTOL_PER_SQRT_K[dtype] * math.sqrt(K)


def _max_relerr(actual: torch.Tensor, ref: torch.Tensor) -> float:
    a = actual.to(torch.float32)
    r = ref.to(torch.float32)
    denom = r.abs().clamp_min(1e-3)
    return float((a - r).abs().div(denom).max().item())


def _check_close_with_tight(
    out: torch.Tensor,
    ref: torch.Tensor,
    in_dtype: torch.dtype,
    K: int,
    label: str,
):
    """Two-stage check: loose contract assert + K-aware tight relerr gate.

    The loose pass guarantees we never weaken the existing tritonblas test
    contract; the tight pass guarantees small but real numerical bugs surface
    instead of being absorbed by the loose tolerance.  When the tight gate
    fires, the failure message includes ``label`` so multi-shape parametrised
    runs identify exactly which (B,M,N,K,dtype) shape regressed.
    """
    torch.testing.assert_close(out, ref, atol=1, rtol=1)
    relerr = _max_relerr(out, ref)
    budget = _tight_budget(in_dtype, K)
    if relerr > budget:
        pytest.fail(
            f"{label}: max_rel_err={relerr:.3e} exceeds K-aware budget "
            f"{budget:.3e} (K={K}, dtype={in_dtype}); structural numerical "
            f"regression."
        )


# Shapes drawn from MI300X batched-GEMM residuals + a couple of small/odd cases.
_BMM_SHAPES = [
    (4, 1024, 1024, 1024),
    (8, 1024, 1024, 1024),
    (16, 512, 512, 512),
    (4, 2048, 2048, 2048),
    (2, 4096, 4096, 4096),
    (3, 1024, 768, 1024),     # non-square (rectangular tile)
    (5, 512, 512, 1024),      # K > M,N
]


@pytest.mark.parametrize("b, m, n, k", _BMM_SHAPES)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_bmm_3d(b, m, n, k, in_dtype, out_dtype):
    """Vanilla (B, M, K) x (B, K, N) -> (B, M, N)."""
    torch.manual_seed(0)
    A = torch.randn((b, m, k), device="cuda", dtype=in_dtype)
    B = torch.randn((b, k, n), device="cuda", dtype=in_dtype)
    out = tritonblas.bmm(A, B)
    ref = torch.bmm(A, B).to(out_dtype)
    _check_close_with_tight(out, ref, in_dtype, K=k, label=f"3d B={b} {m}x{n}x{k} {in_dtype}")


@pytest.mark.parametrize("b, m, n, k", [(8, 1024, 1024, 1024), (4, 512, 512, 512)])
def test_bmm_broadcast_b(b, m, n, k):
    """A is rank-3, B is rank-2 -> broadcast B over the batch dim."""
    torch.manual_seed(1)
    A = torch.randn((b, m, k), device="cuda", dtype=torch.float16)
    B = torch.randn((k, n), device="cuda", dtype=torch.float16)
    out = tritonblas.bmm(A, B)
    ref = torch.matmul(A, B)
    torch.testing.assert_close(out, ref, atol=1, rtol=1)


@pytest.mark.parametrize("b, m, n, k", [(8, 1024, 1024, 1024)])
def test_bmm_broadcast_a(b, m, n, k):
    """A is rank-2, B is rank-3 -> broadcast A over the batch dim."""
    torch.manual_seed(2)
    A = torch.randn((m, k), device="cuda", dtype=torch.float16)
    B = torch.randn((b, k, n), device="cuda", dtype=torch.float16)
    out = tritonblas.bmm(A, B)
    ref = torch.matmul(A, B)
    torch.testing.assert_close(out, ref, atol=1, rtol=1)


def test_bmm_rank2_degenerate():
    """Rank-2 x rank-2 is treated as B==1 BMM; result is 3D (1, M, N).

    This contract matches torch.bmm, not torch.matmul: callers asking for
    *batched* matmul of two rank-2 tensors get a leading singleton batch
    dim, which they can squeeze if desired.
    """
    torch.manual_seed(3)
    A = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    B = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    out = tritonblas.bmm(A, B)
    ref = torch.matmul(A, B).unsqueeze(0)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1, rtol=1)


def test_bmm_vector_left():
    """Rank-1 left input (gemv variant): (K,) x (B, K, N) -> (B, N)."""
    torch.manual_seed(4)
    A = torch.randn((1024,), device="cuda", dtype=torch.float16)
    B = torch.randn((4, 1024, 1024), device="cuda", dtype=torch.float16)
    out = tritonblas.bmm(A, B)
    ref = torch.matmul(A, B)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1, rtol=1)


def test_bmm_out_argument():
    """Pre-allocated ``out`` is honoured."""
    torch.manual_seed(5)
    A = torch.randn((4, 512, 512), device="cuda", dtype=torch.float16)
    B = torch.randn((4, 512, 512), device="cuda", dtype=torch.float16)
    out = torch.empty((4, 512, 512), device="cuda", dtype=torch.float16)
    ret = tritonblas.bmm(A, B, out=out)
    assert ret.data_ptr() == out.data_ptr()
    ref = torch.bmm(A, B)
    torch.testing.assert_close(out, ref, atol=1, rtol=1)


def test_bmm_b1_matches_matmul():
    """B==1 BMM must match the non-batched matmul path bit-for-bit-ish."""
    torch.manual_seed(6)
    A = torch.randn((1, 1024, 1024), device="cuda", dtype=torch.float16)
    B = torch.randn((1, 1024, 1024), device="cuda", dtype=torch.float16)
    bmm_out = tritonblas.bmm(A, B)
    mm_out = tritonblas.matmul(A.squeeze(0), B.squeeze(0))
    torch.testing.assert_close(bmm_out.squeeze(0), mm_out, atol=1, rtol=1)


# ---------------------------------------------------------------------------
# Error-path / input-contract tests
# ---------------------------------------------------------------------------
#
# These tests pin down the failure modes of tritonblas.bmm so callers get
# clear errors rather than silent kernel mis-launches or memory faults.  All
# of them must raise RuntimeError; the message check is a substring match
# (not exact) so we don't break tests on minor wording changes.


def test_bmm_error_mismatched_batch():
    """Rank-3 inputs with different batch dims must raise RuntimeError."""
    A = torch.randn((4, 128, 128), device="cuda", dtype=torch.float16)
    B = torch.randn((8, 128, 128), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="batch"):
        tritonblas.bmm(A, B)


def test_bmm_error_incompatible_k():
    """A's trailing K must match B's leading K."""
    A = torch.randn((4, 128, 256), device="cuda", dtype=torch.float16)
    B = torch.randn((4, 128, 256), device="cuda", dtype=torch.float16)  # wrong K
    with pytest.raises(RuntimeError, match="[Kk]"):
        tritonblas.bmm(A, B)


def test_bmm_error_scalar_input():
    """Scalar (rank-0) inputs must be rejected."""
    A = torch.tensor(1.0, device="cuda", dtype=torch.float16)
    B = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="scalar"):
        tritonblas.bmm(A, B)


def test_bmm_error_too_high_rank():
    """4D inputs must be rejected — only 1D/2D/3D supported."""
    A = torch.randn((2, 4, 128, 128), device="cuda", dtype=torch.float16)
    B = torch.randn((2, 4, 128, 128), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="1D/2D/3D"):
        tritonblas.bmm(A, B)


def test_bmm_error_out_wrong_shape():
    """Pre-allocated ``out`` with wrong shape must raise."""
    A = torch.randn((4, 128, 128), device="cuda", dtype=torch.float16)
    B = torch.randn((4, 128, 128), device="cuda", dtype=torch.float16)
    out_bad = torch.empty((4, 128, 64), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="expected"):
        tritonblas.bmm(A, B, out=out_bad)


def test_bmm_error_rank1_K_mismatch():
    """Rank-1 promotion still validates K."""
    a_vec = torch.randn((256,), device="cuda", dtype=torch.float16)
    B_bad = torch.randn((4, 128, 128), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="[Kk]"):
        tritonblas.bmm(a_vec, B_bad)


def test_bmm_rank1_stack_left():
    """Rank-1 left input + rank-3 right (gemv per-batch).

    Validates the (K,) x (B, K, N) -> (B, N) contract called out in the
    docstring; the squeezed leading dim of A must drop out of the result.
    """
    torch.manual_seed(11)
    a_vec = torch.randn((128,), device="cuda", dtype=torch.float16)
    B = torch.randn((3, 128, 64), device="cuda", dtype=torch.float16)
    out = tritonblas.bmm(a_vec, B)
    ref = torch.matmul(a_vec, B)
    assert out.shape == ref.shape == (3, 64)
    _check_close_with_tight(out, ref, torch.float16, K=128, label="rank1-stack-left")


def test_bmm_rank1_stack_right():
    """Rank-3 left + rank-1 right input (matvec per-batch).

    Validates (B, M, K) x (K,) -> (B, M); the squeezed trailing dim of B
    drops out of the result per torch.matmul rank-1 promotion semantics.
    """
    torch.manual_seed(12)
    A = torch.randn((3, 64, 128), device="cuda", dtype=torch.float16)
    b_vec = torch.randn((128,), device="cuda", dtype=torch.float16)
    out = tritonblas.bmm(A, b_vec)
    ref = torch.matmul(A, b_vec)
    assert out.shape == ref.shape == (3, 64)
    _check_close_with_tight(out, ref, torch.float16, K=128, label="rank1-stack-right")
