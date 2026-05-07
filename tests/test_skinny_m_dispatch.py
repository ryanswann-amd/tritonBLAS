# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Tests for the skinny-M Stream-K dispatch heuristic added in K-104 / S-002.

These exercise both the pure-Python heuristic predicate and the end-to-end
correctness of the matmul wrapper when the heuristic auto-enables Stream-K.
"""
import pytest
import torch

from tritonblas.matmul import (
    SKINNY_K_MIN_FOR_SPLIT,
    SKINNY_K_TO_N_RATIO,
    SKINNY_M_THRESHOLD,
    _resolve_streamk,
    _should_auto_streamk,
)
import tritonblas


# -- Heuristic predicate ----------------------------------------------------
#
# The thresholds are deliberately conservative; the heuristic only fires in
# the regime where empirical measurement (MI300X / gfx942 / ROCm 7.2) shows
# Stream-K can plausibly help: skinny M, deep K, K >> N, very small DP grid.

def test_heuristic_fires_for_deep_skinny_m():
    # M=16, N=1024, K=32768 -> grid = 8 tiles, K/N = 32, K=32768 -> True
    assert _should_auto_streamk(16, 1024, 32768, num_cus=304) is True
    # M=8, N=512, K=16384 -> grid = 4 tiles, K/N = 32 -> True
    assert _should_auto_streamk(8, 512, 16384, num_cus=304) is True


def test_heuristic_skips_for_typical_skinny_m_cohort():
    # The K-104 acceptance sweep is M=16 with K in {1k, 2k, 4k, 8k, 16k}.
    # On MI300X / gfx942 the persistent path is faster on every one of
    # these; the heuristic must NOT fire on them or it would regress.
    for n in [1024, 2048, 4096, 8192, 16384]:
        for k in [1024, 2048, 4096, 8192]:
            assert _should_auto_streamk(16, n, k, num_cus=304) is False, \
                f"heuristic must not fire on M=16,N={n},K={k} (regression risk)"


def test_heuristic_skips_when_M_large():
    assert _should_auto_streamk(SKINNY_M_THRESHOLD + 1, 8192, 8192, num_cus=304) is False
    assert _should_auto_streamk(128, 8192, 8192, num_cus=304) is False
    assert _should_auto_streamk(8192, 8192, 8192, num_cus=304) is False


def test_heuristic_skips_when_K_small():
    # Even with skinny M, K below the min-for-split threshold is not worth it
    assert _should_auto_streamk(16, 1024, SKINNY_K_MIN_FOR_SPLIT - 1, num_cus=304) is False
    assert _should_auto_streamk(16, 1024, 64, num_cus=304) is False


def test_heuristic_skips_when_K_to_N_ratio_too_small():
    # Skinny M, K big in absolute terms but K/N below the ratio threshold:
    # persistent's launch overhead dominates; Stream-K's reduction adds cost
    # without parallelism gain.  Empirically, K/N=4 still regressed
    # (M=16,N=4096,K=16384 measured 0.75x of persistent on MI300X), so the
    # ratio gate is K/N >= 8.
    assert _should_auto_streamk(16, 8192, 16384, num_cus=304) is False  # K/N = 2 < 8
    assert _should_auto_streamk(16, 16384, 32768, num_cus=304) is False  # K/N = 2 < 8
    assert _should_auto_streamk(16, 4096, 16384, num_cus=304) is False  # K/N = 4 < 8


def test_heuristic_skips_when_grid_already_fills_CUs():
    # Even at K=32768, a large N produces grid >= num_CUs/4 -> persistent OK
    # num_cus=304 -> need tiles_n < 76 i.e. N < 9728 to fire.
    assert _should_auto_streamk(16, 16384, 32768, num_cus=304) is False
    assert _should_auto_streamk(16, 32768, 32768, num_cus=304) is False


def test_heuristic_skips_for_degenerate_shapes():
    assert _should_auto_streamk(0, 8192, 8192) is False
    assert _should_auto_streamk(16, 0, 8192) is False
    assert _should_auto_streamk(16, 8192, 0) is False


def test_resolve_respects_explicit_user_choice():
    # User explicitly opted out -> heuristic must not override
    assert _resolve_streamk(16, 1024, 32768, enable_streamk=False, num_cus=304) is False
    # User explicitly opted in -> always True
    assert _resolve_streamk(8192, 8192, 8192, enable_streamk=True, num_cus=304) is True


def test_resolve_uses_heuristic_for_none():
    # None means "let the dispatcher choose"
    assert _resolve_streamk(16, 1024, 32768, enable_streamk=None, num_cus=304) is True
    assert _resolve_streamk(8192, 8192, 8192, enable_streamk=None, num_cus=304) is False
    # Typical skinny-M cohort shape -- heuristic stays off (no regression).
    assert _resolve_streamk(16, 8192, 8192, enable_streamk=None, num_cus=304) is False


# -- End-to-end correctness via the public matmul ---------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs ROCm/CUDA")
@pytest.mark.parametrize("M,N,K", [
    (16, 1024, 32768),  # heuristic auto-fires (deep K, skinny N)
    (16, 4096, 8192),   # heuristic skips (K/N=2 < 4)
    (16, 8192, 4096),   # heuristic skips (K < min-for-split)
    (32, 4096, 4096),   # heuristic skips
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
