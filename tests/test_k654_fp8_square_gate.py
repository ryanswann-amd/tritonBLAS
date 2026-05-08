"""Predicate-narrowness tests for the K-654 FP8 e4m3fnuz medium-K square gate.

These tests are pure-Python and do NOT touch the GPU — they verify that the
shape predicate `_is_k654_fp8_e4m3_square_cohort` and its `_K654SelectorOverride`
wrapper fire ONLY on the intended (M=N=1024, K∈{256,512}, fp8_e4m3fnuz,
non-streamk) cohort and do not bleed onto neighbouring shapes / dtypes.

Narrowness is load-bearing: per the K-654 retry-sweep evidence (see
`output/sweep_kpack_mfma_*.csv`), wider tiles wired to M∈{2048,4096} crash
the production persistent path with an MLIR codegen assertion (same family
as K-605 / K-624 R3). A future refactor that broadens the predicate would
silently re-introduce that crash.
"""

import pytest
import torch

from tritonblas.matmul import (
    _K654SelectorOverride,
    _is_k654_fp8_e4m3_square_cohort,
    _maybe_apply_k654_gate,
)

# The override is meaningful only on architectures that expose
# torch.float8_e4m3fnuz (gfx940/941/942 — the AMD CDNA3/MI300 family).
_FNUZ = getattr(torch, "float8_e4m3fnuz", None)
requires_fnuz = pytest.mark.skipif(
    _FNUZ is None, reason="torch.float8_e4m3fnuz not available on this build"
)


# ---------------------------------------------------------------------------
# Predicate fires on the cohort
# ---------------------------------------------------------------------------

@requires_fnuz
@pytest.mark.parametrize("K", [256, 512])
def test_predicate_fires_on_cohort(K):
    assert _is_k654_fp8_e4m3_square_cohort(1024, 1024, K, _FNUZ, _FNUZ, False)


# ---------------------------------------------------------------------------
# Predicate does NOT fire — every negative axis matters
# ---------------------------------------------------------------------------

@requires_fnuz
@pytest.mark.parametrize(
    "M,N,K",
    [
        # Out-of-cohort M (codegen-bound; MUST NOT fire)
        (2048, 2048, 256),
        (2048, 2048, 512),
        (4096, 4096, 256),
        (4096, 4096, 512),
        (512, 512, 256),
        # Wrong K
        (1024, 1024, 128),
        (1024, 1024, 1024),
        # Non-square (would bleed onto SKINNY-N / wide-N cohorts)
        (1024, 2048, 256),
        (2048, 1024, 256),
        (1024, 512, 256),
        (1024, 64, 256),
    ],
)
def test_predicate_does_not_fire_off_cohort(M, N, K):
    assert not _is_k654_fp8_e4m3_square_cohort(M, N, K, _FNUZ, _FNUZ, False), (
        f"K-654 gate must NOT fire on shape {(M, N, K)} — would risk codegen "
        f"crash on M∈{{2048,4096}} or cohort-bleed regression."
    )


@requires_fnuz
@pytest.mark.parametrize(
    "a_dtype,b_dtype",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float8_e4m3fn) if hasattr(torch, "float8_e4m3fn") else (_FNUZ, torch.float16),
        (_FNUZ, torch.float16),  # mixed dtype
        (torch.float16, _FNUZ),
    ],
)
def test_predicate_requires_fp8_e4m3fnuz_on_both_operands(a_dtype, b_dtype):
    assert not _is_k654_fp8_e4m3_square_cohort(1024, 1024, 256, a_dtype, b_dtype, False)


@requires_fnuz
def test_predicate_skips_streamk_path():
    # StreamK has its own kernel pipeline; the K-654 tile pick targets the
    # persistent path and must not be applied to streamk.
    assert not _is_k654_fp8_e4m3_square_cohort(1024, 1024, 256, _FNUZ, _FNUZ, True)


# ---------------------------------------------------------------------------
# Wrapper publishes the documented tile and delegates everything else
# ---------------------------------------------------------------------------

class _FakeInner:
    """Stand-in for OrigamiMatmulSelector exposing a few non-overridden attrs."""
    group_m = 8
    num_sms = 8
    even_k = True
    sk_grid = 99
    COUNTERS_PER_XCD = 4

    def __init__(self):
        # An attribute that should NOT be shadowed by the wrapper
        self.opaque_value = "delegated"


def test_override_publishes_tile_constants():
    inner = _FakeInner()
    wrapped = _K654SelectorOverride(inner)
    # Hardcoded K-654 tile from the medium-K square sweep — these constants
    # are the contract this test protects.
    assert wrapped.block_m == 32
    assert wrapped.block_n == 64
    assert wrapped.block_k == 128
    assert wrapped.num_stages == 3
    assert wrapped.num_warps == 4
    assert wrapped.waves_per_eu == 2
    assert wrapped.matrix_instr_nonkdim == 16


def test_override_delegates_non_tile_attrs():
    inner = _FakeInner()
    wrapped = _K654SelectorOverride(inner)
    assert wrapped.group_m == 8
    assert wrapped.num_sms == 8
    assert wrapped.even_k is True
    assert wrapped.sk_grid == 99
    assert wrapped.COUNTERS_PER_XCD == 4
    assert wrapped.opaque_value == "delegated"


def test_override_delegation_raises_attributeerror_for_missing():
    wrapped = _K654SelectorOverride(_FakeInner())
    with pytest.raises(AttributeError):
        wrapped.does_not_exist


# ---------------------------------------------------------------------------
# _maybe_apply_k654_gate is a no-op outside the cohort
# ---------------------------------------------------------------------------

@requires_fnuz
def test_maybe_apply_returns_inner_off_cohort():
    inner = _FakeInner()
    out = _maybe_apply_k654_gate(inner, 2048, 2048, 256, _FNUZ, _FNUZ)
    assert out is inner, "must return original selector unchanged off-cohort"


@requires_fnuz
def test_maybe_apply_wraps_on_cohort():
    inner = _FakeInner()
    out = _maybe_apply_k654_gate(inner, 1024, 1024, 256, _FNUZ, _FNUZ)
    assert isinstance(out, _K654SelectorOverride)
    assert out._inner is inner
