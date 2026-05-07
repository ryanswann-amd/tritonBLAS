"""Tests for ``_is_compute_bound_bf16_nn_cohort`` shape-classifier.

Pure-Python tests; no GPU required. The predicate is currently not wired
into any matmul code path — see the comment above the definition in
``include/tritonblas/matmul.py`` — but is retained as a stable hook for
future tuning work, so we test it in isolation here so its semantics
cannot drift silently.
"""

import pytest
import torch

from tritonblas.matmul import _is_compute_bound_bf16_nn_cohort


def _row(M, K, dtype, device="cpu"):
    """Row-major M x K tensor: stride(1) == 1."""
    return torch.empty((M, K), dtype=dtype, device=device)


def _col(M, K, dtype, device="cpu"):
    """Column-major M x K tensor: stride(0) == 1."""
    return torch.empty((K, M), dtype=dtype, device=device).T


# --- Positive cases (gate must FIRE) ---------------------------------------

@pytest.mark.parametrize(
    "M,N,K",
    [
        (1024, 8192, 8192),
        (2048, 8192, 8192),
        (4096, 8192, 8192),
        (8192, 8192, 8192),
        (1024, 16384, 16384),
        # Boundary: equality on each threshold must admit.
        (1024, 8192, 8192),
    ],
)
def test_admits_inside_envelope_bf16_nn(M, N, K):
    a = _row(M, K, torch.bfloat16)
    b = _row(K, N, torch.bfloat16)
    assert _is_compute_bound_bf16_nn_cohort(M, N, K, a, b) is True


# --- Negative cases (gate must REFUSE) -------------------------------------

def test_refuses_below_M_threshold():
    M, N, K = 512, 8192, 8192
    a, b = _row(M, K, torch.bfloat16), _row(K, N, torch.bfloat16)
    assert _is_compute_bound_bf16_nn_cohort(M, N, K, a, b) is False


def test_refuses_below_N_threshold():
    M, N, K = 1024, 4096, 8192
    a, b = _row(M, K, torch.bfloat16), _row(K, N, torch.bfloat16)
    assert _is_compute_bound_bf16_nn_cohort(M, N, K, a, b) is False


def test_refuses_below_K_threshold():
    M, N, K = 1024, 8192, 4096
    a, b = _row(M, K, torch.bfloat16), _row(K, N, torch.bfloat16)
    assert _is_compute_bound_bf16_nn_cohort(M, N, K, a, b) is False


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_refuses_non_bf16(dtype):
    M, N, K = 1024, 8192, 8192
    a, b = _row(M, K, dtype), _row(K, N, dtype)
    assert _is_compute_bound_bf16_nn_cohort(M, N, K, a, b) is False


def test_refuses_NT_layout():
    """A row-major, B column-major (NT) -> b.stride(1) != 1."""
    M, N, K = 1024, 8192, 8192
    a, b = _row(M, K, torch.bfloat16), _col(K, N, torch.bfloat16)
    assert _is_compute_bound_bf16_nn_cohort(M, N, K, a, b) is False


def test_refuses_TN_layout():
    """A column-major, B row-major (TN) -> a.stride(1) != 1."""
    M, N, K = 1024, 8192, 8192
    a, b = _col(M, K, torch.bfloat16), _row(K, N, torch.bfloat16)
    assert _is_compute_bound_bf16_nn_cohort(M, N, K, a, b) is False


def test_refuses_mixed_dtype():
    M, N, K = 1024, 8192, 8192
    a = _row(M, K, torch.bfloat16)
    b = _row(K, N, torch.float16)
    assert _is_compute_bound_bf16_nn_cohort(M, N, K, a, b) is False


@pytest.mark.parametrize(
    "M,N,K,dtype",
    [
        # K-654 / K-612 cohorts where prior unconditional knob changes
        # regressed: predicate must NOT admit them.
        (4096, 4096, 4096, torch.bfloat16),  # square, below N/K threshold
        (8192, 8192, 4096, torch.bfloat16),  # large M/N, K too small
        (4096, 4096, 8192, torch.bfloat16),  # N too small
        (512, 1024, 512, torch.float16),     # small fp16
        (1024, 1024, 512, torch.bfloat16),   # tiny K
    ],
)
def test_protects_known_regression_cohorts(M, N, K, dtype):
    a, b = _row(M, K, dtype), _row(K, N, dtype)
    assert _is_compute_bound_bf16_nn_cohort(M, N, K, a, b) is False


# --- Lock-in: predicate must remain UNUSED until a new experiment wins -----


def test_predicate_is_not_wired_into_kernel_launch():
    """Guard against accidental re-activation of the empirically-refuted
    kpack=2 + num_warps=4 forcing inside the kernel launch path. The
    prior static-analysis prediction (the K-757 / K-873 audits) said this
    cohort would lift +12-25 pp MFMA-issue-rate; the empirical isolation
    on MI300X showed -1% to -29% wall-time REGRESSION at every cell of
    the 4-cell (kpack, num_warps) sweep. If a future investigator wants
    to wire the gate back in, this test should be updated AT THE SAME TIME
    as the activation, with a reference to the new isolation evidence.
    """
    import inspect

    import tritonblas.matmul as tb_matmul_mod

    src = inspect.getsource(tb_matmul_mod)
    activation_marker = "if _is_compute_bound_bf16_nn_cohort("
    n_call_sites = src.count(activation_marker)
    assert n_call_sites == 0, (
        f"_is_compute_bound_bf16_nn_cohort activation re-introduced "
        f"({n_call_sites} call site(s)) without updating this test. See the "
        f"module-level negative-result comment block in include/tritonblas/"
        f"matmul.py for the empirical refutation that caused the prior "
        f"activation to be removed."
    )
