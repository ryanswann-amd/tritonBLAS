# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Tests for the skinny-M Stream-K dispatch heuristic.

The heuristic auto-routes very-deep-K skinny-M shapes (M<=32, K>=16384,
N<=2048) through Stream-K and leaves every other shape on the persistent
path.  Calibration is empirical on MI300X (gfx942) with the current ROCm
+ Triton stack.  See ``_resolve_streamk`` in tritonblas.matmul for details.
"""
import pytest
import torch

from tritonblas.matmul import _resolve_streamk
import tritonblas


# -- Heuristic predicate ----------------------------------------------------

def test_heuristic_fires_inside_win_zone():
    # The two small-M cohort shapes where Stream-K beats persistent on
    # MI300X (1.115x and 1.138x respectively, measured on the current
    # gfx942 software stack).
    assert _resolve_streamk(16, 1024, 16384, enable_streamk=None) is True
    assert _resolve_streamk(16, 2048, 16384, enable_streamk=None) is True
    # Other shapes inside the predicate (M<=32, K>=16384, N<=2048):
    assert _resolve_streamk(8,  1024, 32768, enable_streamk=None) is True
    assert _resolve_streamk(32, 2048, 16384, enable_streamk=None) is True


def test_heuristic_skips_small_m_cohort_when_outside_win_zone():
    # Acceptance sweep (M=16, N in {1024..16384}, K in {1024..16384}).
    # The heuristic must stay OFF on every shape outside the (K>=16384 AND
    # N<=2048) win zone -- otherwise the regression band (sk/per = 0.59x-
    # 0.92x at K=16384 with N>=4096) would be triggered.
    for n in [1024, 2048, 4096, 8192, 16384]:
        for k in [1024, 2048, 4096, 8192]:  # K below the 16384 floor
            assert _resolve_streamk(16, n, k, enable_streamk=None) is False, \
                f"heuristic must not fire on M=16,N={n},K={k} (K below 16384)"
    for n in [4096, 8192, 16384]:  # N above the 2048 ceiling
        assert _resolve_streamk(16, n, 16384, enable_streamk=None) is False, \
            f"heuristic must not fire on M=16,N={n},K=16384 (regression band)"


def test_heuristic_skips_when_M_large():
    # M > 32 disables the heuristic regardless of N, K.
    assert _resolve_streamk(33, 2048, 16384, enable_streamk=None) is False
    assert _resolve_streamk(128, 1024, 32768, enable_streamk=None) is False
    assert _resolve_streamk(8192, 8192, 8192, enable_streamk=None) is False


def test_heuristic_skips_when_K_too_shallow():
    # Below K=16384 the persistent kernel's launch-overhead floor (~0.26ms
    # on MI300X) is not yet amortised by compute; Stream-K's reduction adds
    # cost without parallelism gain.
    assert _resolve_streamk(16, 1024, 16383, enable_streamk=None) is False
    assert _resolve_streamk(16, 1024, 8192, enable_streamk=None) is False
    assert _resolve_streamk(16, 1024, 64, enable_streamk=None) is False


def test_heuristic_skips_when_N_too_wide():
    # N > 2048 puts us in the empirical regression band.
    assert _resolve_streamk(16, 4096, 16384, enable_streamk=None) is False
    assert _resolve_streamk(16, 8192, 16384, enable_streamk=None) is False
    assert _resolve_streamk(16, 16384, 32768, enable_streamk=None) is False


def test_heuristic_skips_for_degenerate_shapes():
    assert _resolve_streamk(0, 8192, 8192, enable_streamk=None) is False
    assert _resolve_streamk(16, 0, 16384, enable_streamk=None) is False
    assert _resolve_streamk(16, 1024, 0, enable_streamk=None) is False
    assert _resolve_streamk(-1, 1024, 16384, enable_streamk=None) is False


def test_resolve_respects_explicit_user_choice():
    # Explicit False overrides auto-fire.
    assert _resolve_streamk(16, 1024, 16384, enable_streamk=False) is False
    # Explicit True overrides auto-skip (e.g. user benchmarking outside zone).
    assert _resolve_streamk(8192, 8192, 8192, enable_streamk=True) is True


# -- End-to-end correctness via the public matmul ---------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs ROCm/CUDA")
@pytest.mark.parametrize("M,N,K", [
    (16, 1024, 16384),  # heuristic auto-fires
    (16, 2048, 16384),  # heuristic auto-fires
    (16, 4096, 8192),   # heuristic skips (K too shallow)
    (16, 8192, 16384),  # heuristic skips (N too wide)
    (8192, 8192, 8192), # large cohort: heuristic stays off
])
def test_skinny_m_dispatch_matches_torch(M, N, K):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c_default = tritonblas.matmul(a, b)              # heuristic decides
    c_explicit_off = tritonblas.matmul(a, b, enable_streamk=False)
    c_ref = (a.float() @ b.float()).to(torch.float16)
    torch.testing.assert_close(c_default.float(), c_ref.float(), atol=0.5, rtol=0.05)
    torch.testing.assert_close(c_explicit_off.float(), c_ref.float(), atol=0.5, rtol=0.05)
