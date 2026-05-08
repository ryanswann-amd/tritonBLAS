"""Unit tests for the K-923 narrow LDS-pressure mitigation predicate.

Boundary coverage for `_k923_lds_narrow_override`:

  - When TB_K923_ENABLE is unset / "0": ALWAYS a no-op, regardless of
    shape/dtype (default-off posture; opt-in only).
  - When TB_K923_ENABLE=1 AND the strict cohort A predicate
    (M==N==1024 AND K>=16384 AND dtype in {fp16, bf16}) matches:
    pins num_stages=2 and kpack=2, leaves BLK_K untouched.
  - When TB_K923_ENABLE=1 but the predicate does NOT match (off-shape
    or off-dtype): no-op.

This is a pure-Python predicate with no GPU dependency, so it runs in the
plain pytest collection without a CUDA/ROCm device.
"""

import os

import pytest
import torch

from tritonblas.matmul import _k923_lds_narrow_override


# (BLK_K_in, num_stages_in, kpack_in) the override receives upstream.
# Origami's pick on cohort A today is BK=256, NS=2; we pick non-overridden
# defaults below to make BEFORE/AFTER visibly distinct in the asserts.
_BLK_K_IN = 256
_NS_IN = 3      # deliberately not 2, to prove the override rewrites it
_KPACK_IN = 1   # deliberately not 2, to prove the override rewrites it


@pytest.fixture
def env_disabled(monkeypatch):
    """Default posture: TB_K923_ENABLE unset."""
    monkeypatch.delenv("TB_K923_ENABLE", raising=False)


@pytest.fixture
def env_enabled(monkeypatch):
    """Opt-in posture: TB_K923_ENABLE=1."""
    monkeypatch.setenv("TB_K923_ENABLE", "1")


# ---------------------------------------------------------------------------
# Default-off posture
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "M,N,K",
    [
        (1024, 1024, 16384),  # in-cohort
        (1024, 1024, 24576),  # in-cohort
        (1024, 1024, 32768),  # in-cohort
        (4096, 4096, 4096),   # off-cohort
        (1024, 1024, 8192),   # K too small
        (2048, 1024, 16384),  # M wrong
    ],
)
def test_default_is_noop_regardless_of_shape(env_disabled, M, N, K, dtype):
    """With TB_K923_ENABLE unset, the override MUST be a no-op for all
    shapes/dtypes — this is the post-K-923 default behaviour."""
    out = _k923_lds_narrow_override(M, N, K, dtype, _BLK_K_IN, _NS_IN, _KPACK_IN)
    assert out == (_BLK_K_IN, _NS_IN, _KPACK_IN)


def test_explicit_zero_is_noop(monkeypatch):
    """TB_K923_ENABLE=0 is also a no-op (explicit opt-out)."""
    monkeypatch.setenv("TB_K923_ENABLE", "0")
    out = _k923_lds_narrow_override(
        1024, 1024, 16384, torch.bfloat16, _BLK_K_IN, _NS_IN, _KPACK_IN
    )
    assert out == (_BLK_K_IN, _NS_IN, _KPACK_IN)


# ---------------------------------------------------------------------------
# Opt-in posture: predicate matches → override fires
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("K", [16384, 24576, 32768, 40960])
def test_in_cohort_pins_ns2_kpack2_when_enabled(env_enabled, K, dtype):
    """On the strict cohort A predicate with TB_K923_ENABLE=1:
    num_stages forced to 2, kpack forced to 2, BLK_K untouched."""
    out_blk_k, out_ns, out_kpack = _k923_lds_narrow_override(
        1024, 1024, K, dtype, _BLK_K_IN, _NS_IN, _KPACK_IN
    )
    assert out_blk_k == _BLK_K_IN, "BLK_K must NOT be rewritten by the override"
    assert out_ns == 2, f"num_stages must be pinned to 2, got {out_ns}"
    assert out_kpack == 2, f"kpack must be forced to 2, got {out_kpack}"


# ---------------------------------------------------------------------------
# Opt-in posture: predicate does NOT match → still a no-op
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "M,N,K,dtype,reason",
    [
        # Wrong dtype
        (1024, 1024, 16384, torch.float32, "fp32 not in {fp16, bf16}"),
        # K below 16384
        (1024, 1024, 15360, torch.bfloat16, "K<16384"),
        (1024, 1024, 8192, torch.float16, "K<16384"),
        # Wrong M
        (2048, 1024, 16384, torch.bfloat16, "M!=1024"),
        (512, 1024, 16384, torch.float16, "M!=1024"),
        # Wrong N
        (1024, 2048, 16384, torch.bfloat16, "N!=1024"),
        (1024, 512, 16384, torch.float16, "N!=1024"),
        # Both M and N wrong
        (4096, 4096, 16384, torch.bfloat16, "M!=N!=1024"),
    ],
)
def test_off_cohort_is_noop_even_when_enabled(
    env_enabled, M, N, K, dtype, reason
):
    """Even with TB_K923_ENABLE=1, off-cohort shapes/dtypes MUST be no-ops.
    OOT impact must remain structurally zero."""
    out = _k923_lds_narrow_override(M, N, K, dtype, _BLK_K_IN, _NS_IN, _KPACK_IN)
    assert out == (_BLK_K_IN, _NS_IN, _KPACK_IN), (
        f"override fired on off-cohort shape ({M},{N},{K},{dtype}): "
        f"reason expected no-op = {reason}"
    )


# ---------------------------------------------------------------------------
# Boundary at K==16384 exactly
# ---------------------------------------------------------------------------

def test_K_boundary_exact(env_enabled):
    """K==16384 (the inclusive lower bound) must trigger the override."""
    out = _k923_lds_narrow_override(
        1024, 1024, 16384, torch.bfloat16, _BLK_K_IN, _NS_IN, _KPACK_IN
    )
    assert out == (_BLK_K_IN, 2, 2)


def test_K_boundary_just_below(env_enabled):
    """K==16383 must NOT trigger the override (strict >= boundary)."""
    out = _k923_lds_narrow_override(
        1024, 1024, 16383, torch.bfloat16, _BLK_K_IN, _NS_IN, _KPACK_IN
    )
    assert out == (_BLK_K_IN, _NS_IN, _KPACK_IN)
