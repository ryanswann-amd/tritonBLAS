# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the tall-skinny FP16/BF16 num_warps gate.

The override fires only on the empirically-validated cohort:
    dtype in {fp16, bf16} AND M in {2048, 4096} AND N <= 64 AND K >= 2048

Returns 4 inside the gate, 8 (the unmodified default) elsewhere. Out-of-cohort
shapes regress 8-34% with num_warps=4 (rocprofv2 + paired HIP-graph A/B on
MI300X / gfx942), so the gate must be strictly conjunctive on all four axes.
"""
import pytest
import torch

from tritonblas.matmul import _num_warps_for_tall_skinny_fp16_bf16 as f


# ---------------------------------------------------------------------------
# Positive cases (gate FIRES, returns 4)
# ---------------------------------------------------------------------------

POSITIVE_CASES = [
    # (M, N, K, dtype) — every empirically-validated worst-4 / neighbor-12 cell
    (2048, 32, 2048, torch.float16),
    (2048, 32, 2048, torch.bfloat16),
    (2048, 32, 4096, torch.float16),
    (2048, 32, 4096, torch.bfloat16),
    (2048, 32, 8192, torch.float16),
    (2048, 32, 8192, torch.bfloat16),
    (2048, 64, 4096, torch.float16),
    (2048, 64, 4096, torch.bfloat16),
    (4096, 32, 2048, torch.float16),
    (4096, 32, 2048, torch.bfloat16),
    (4096, 32, 4096, torch.float16),
    (4096, 32, 4096, torch.bfloat16),
    (4096, 32, 8192, torch.float16),
    (4096, 32, 8192, torch.bfloat16),
    (4096, 64, 4096, torch.float16),
    (4096, 64, 4096, torch.bfloat16),
]


@pytest.mark.parametrize("M,N,K,dtype", POSITIVE_CASES)
def test_gate_fires_in_cohort(M, N, K, dtype):
    assert f(M, N, K, dtype) == 4


# ---------------------------------------------------------------------------
# Negative cases (gate does NOT fire, returns 8 — bit-identical to upstream)
# ---------------------------------------------------------------------------

# (a) Wrong dtype — fp32 / fp8 / int8 / fp4 must keep num_warps=8
@pytest.mark.parametrize("dtype", [
    torch.float32,
    torch.int8,
    torch.int32,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
])
def test_gate_skipped_for_non_fp16_bf16_dtype(dtype):
    # Each of these tuples is an in-cohort (M, N, K) — the dtype clause must
    # keep us out of the gate regardless of geometry.
    for M, N, K in [(2048, 32, 8192), (4096, 64, 4096), (2048, 32, 2048)]:
        assert f(M, N, K, dtype) == 8


# (b) Wrong M — only M ∈ {2048, 4096} is gated. The out-of-cohort guard run
# showed M=8192 N=32 K=4096 regresses 11% with num_warps=4.
@pytest.mark.parametrize("M", [16, 64, 128, 512, 1024, 1535, 2047, 2049, 3072,
                               4095, 4097, 6144, 8192, 16384])
def test_gate_skipped_for_off_axis_M(M):
    for dtype in (torch.float16, torch.bfloat16):
        assert f(M, 32, 4096, dtype) == 8
        assert f(M, 64, 4096, dtype) == 8


# (c) Wrong N — N must be <= 64. Out-of-cohort guard showed N=128/256/512/1024
# regress 8-19% with num_warps=4. Boundary check at N=65, 128, 256, 512, 1024.
@pytest.mark.parametrize("N", [65, 96, 128, 256, 512, 1024, 2048, 4096])
def test_gate_skipped_for_large_N(N):
    for dtype in (torch.float16, torch.bfloat16):
        for M in (2048, 4096):
            assert f(M, N, 4096, dtype) == 8


# (d) Wrong K — K must be >= 2048. Below that the gain drops; the gate is
# conservative (keeps the cohort tight to the validated set).
@pytest.mark.parametrize("K", [1, 32, 64, 128, 256, 512, 1024, 2047])
def test_gate_skipped_for_small_K(K):
    for dtype in (torch.float16, torch.bfloat16):
        for M in (2048, 4096):
            for N in (32, 64):
                assert f(M, N, K, dtype) == 8


# (e) Boundary spot checks — every-axis-on-edge sweep
@pytest.mark.parametrize("M,N,K,dtype,expected", [
    (2048, 64, 2048, torch.float16, 4),     # all corners of the gate
    (4096, 32, 2048, torch.bfloat16, 4),
    (2048, 32, 100_000, torch.float16, 4),  # K very large still fires
    (2048, 1, 4096, torch.bfloat16, 4),     # N=1 still inside (<=64)
    (2048, 65, 2048, torch.float16, 8),     # N just above the boundary
    (2048, 32, 2047, torch.bfloat16, 8),    # K just below the boundary
    (2049, 32, 4096, torch.float16, 8),     # M not exactly 2048/4096
    (4097, 32, 4096, torch.bfloat16, 8),
    (3072, 32, 4096, torch.float16, 8),
])
def test_gate_boundary(M, N, K, dtype, expected):
    assert f(M, N, K, dtype) == expected


# (f) Square-shape leakage guard — must NOT fire on square shapes (these
# regress 11-34% with num_warps=4 per the K-710 outguard cohort).
@pytest.mark.parametrize("MN", [1024, 2048, 4096, 8192])
@pytest.mark.parametrize("K", [2048, 4096, 8192])
def test_gate_skipped_for_square(MN, K):
    for dtype in (torch.float16, torch.bfloat16):
        # MN=2048/4096 with N=MN > 64 -> excluded by N gate, ✓.
        # MN=1024/8192 -> excluded by M gate, ✓.
        assert f(MN, MN, K, dtype) == 8


# (g) Referential transparency — same inputs always return same value.
def test_gate_pure():
    for _ in range(3):
        assert f(2048, 32, 8192, torch.float16) == 4
        assert f(2048, 2048, 8192, torch.float16) == 8
