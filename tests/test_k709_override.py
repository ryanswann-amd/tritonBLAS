"""K-709 tall-skinny FP16/BF16 BLOCK_K=128 override tests.

These tests guard the narrow shape-guarded override added in
include/tritonblas/matmul.py against two regression classes:

  1. Correctness on in-envelope shapes (the override must produce
     numerically equivalent results to torch matmul on the gated cohort).
  2. K-654-class leakage: out-of-envelope shapes must be bit-identical
     no-ops on the selector (no tile mutation when the predicate misses).

Run with:
    pytest tests/test_k709_override.py -v
"""

import os
from copy import deepcopy

import pytest
import torch

import tritonblas
from tritonblas.matmul import (
    _k709_apply_override,
    _k709_should_override,
)
from tritonblas.origami import OrigamiMatmulSelector


# Shapes inside the gate envelope (M>=2048, N<=32, K>=4096, fp16/bf16).
IN_ENVELOPE = [
    (2048, 32, 4096, torch.float16),
    (2048, 32, 8192, torch.bfloat16),
    (4096, 32, 4096, torch.float16),
    (8192, 32, 8192, torch.bfloat16),
]

# Shapes OUTSIDE the gate envelope (each fails exactly one predicate clause).
# These must remain bit-identical no-ops on the selector.
OUT_OF_ENVELOPE = [
    (2048, 64, 8192, torch.float16),     # N=64 above ceiling
    (2048, 32, 2048, torch.float16),     # K=2048 below floor
    (1024, 32, 8192, torch.float16),     # M=1024 below floor
    (2048, 32, 8192, torch.float32),     # dtype not in {fp16,bf16}
    (4096, 256, 1024, torch.bfloat16),   # totally out of cohort (K-654 region)
    (16384, 256, 1024, torch.float16),   # large-N + small-K corner
]


@pytest.mark.parametrize("M,N,K,dtype", IN_ENVELOPE)
def test_in_envelope_predicate_fires_and_is_correct(M, N, K, dtype):
    """In-envelope: predicate must fire AND result must match torch matmul."""
    # Predicate is an explicit pre-condition for the test to be meaningful.
    assert _k709_should_override(M, N, K, dtype, dtype, enable_streamk=False), (
        f"Predicate did not fire on in-envelope shape M={M},N={N},K={K},"
        f"dtype={dtype}; the test fixture is stale."
    )
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    out_tb = tritonblas.matmul(a, b)
    out_ref = torch.matmul(a, b)
    err = (out_tb.float() - out_ref.float()).abs().max().item()
    ref = out_ref.float().abs().max().item()
    rel = err / (ref + 1e-6)
    # FP16/BF16 GEMM tolerance: matches the rest of the suite (test_matmul.py).
    assert rel < 5e-2, (
        f"K-709 override produced incorrect result on M={M},N={N},K={K},"
        f"dtype={dtype}: rel_err={rel:.3e}, abs_err={err:.3e}, ref={ref:.3e}"
    )


@pytest.mark.parametrize("M,N,K,dtype", OUT_OF_ENVELOPE)
def test_out_of_envelope_selector_unchanged(M, N, K, dtype):
    """Out-of-envelope: predicate must NOT fire; selector mutation is skipped.

    This is the anti-leakage guarantee. We construct a selector twice:
    once we apply the override only if the predicate says we should
    (mirroring persistent_matmul_lt's call site); the other we leave
    untouched. The (block_m, block_n, block_k, num_stages) tuple must
    be identical.
    """
    fired = _k709_should_override(M, N, K, dtype, dtype, enable_streamk=False)
    assert not fired, (
        f"Predicate fired on out-of-envelope shape M={M},N={N},K={K},"
        f"dtype={dtype}; the gate has leaked."
    )

    dev = torch.device("cuda", torch.cuda.current_device())
    out_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32
    sel_a = OrigamiMatmulSelector(M, N, K, dtype, dtype, out_dtype, dev)
    sel_b = OrigamiMatmulSelector(M, N, K, dtype, dtype, out_dtype, dev)
    pre = (sel_b.block_m, sel_b.block_n, sel_b.block_k, sel_b._num_stages)
    # Mirror the call site: only apply override when predicate fires.
    if _k709_should_override(M, N, K, dtype, dtype, enable_streamk=False):
        _k709_apply_override(sel_b)
    post = (sel_b.block_m, sel_b.block_n, sel_b.block_k, sel_b._num_stages)
    assert pre == post, (
        f"Selector mutated on out-of-envelope shape M={M},N={N},K={K},"
        f"dtype={dtype}: pre={pre}, post={post}"
    )
    # And the two selectors must agree (sanity: the constructor is deterministic
    # for fixed inputs).
    assert (sel_a.block_m, sel_a.block_n, sel_a.block_k, sel_a._num_stages) == post


@pytest.mark.parametrize("M,N,K,dtype", IN_ENVELOPE)
def test_in_envelope_override_widens_block_k(M, N, K, dtype):
    """Spot-check that production override actually sets BLOCK_K=128 in-envelope."""
    dev = torch.device("cuda", torch.cuda.current_device())
    sel = OrigamiMatmulSelector(M, N, K, dtype, dtype, dtype, dev)
    bk_before = sel.block_k
    _k709_apply_override(sel)
    assert sel.block_k == 128, (
        f"production override should set BLOCK_K=128, got {sel.block_k} "
        f"(was {bk_before}) on M={M},N={N},K={K},dtype={dtype}"
    )


def test_kill_switch_disables_override(monkeypatch):
    """TRITONBLAS_DISABLE_K709=1 must short-circuit the predicate."""
    monkeypatch.setenv("TRITONBLAS_DISABLE_K709", "1")
    assert not _k709_should_override(
        2048, 32, 8192, torch.float16, torch.float16, enable_streamk=False
    )


def test_baseline_variant_disables_override(monkeypatch):
    """TRITONBLAS_K709_VARIANT=baseline must short-circuit the predicate."""
    monkeypatch.setenv("TRITONBLAS_K709_VARIANT", "baseline")
    assert not _k709_should_override(
        2048, 32, 8192, torch.float16, torch.float16, enable_streamk=False
    )


def test_streamk_disables_override():
    """enable_streamk=True must short-circuit the predicate (orthogonal path)."""
    assert not _k709_should_override(
        2048, 32, 8192, torch.float16, torch.float16, enable_streamk=True
    )
