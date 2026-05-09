"""K-1793 regression tests for the N=192 K-COMPLEMENT hipBLASLt route-OUT gate.

The gate routes 18 verified-winning cells to torch.matmul (hipBLASLt):
    M in {2048, 4096, 8192} x N==192 x K in {4096, 8192, 16384} x dtype in {fp16, bf16}

These tests cover the gate predicate's structural invariants WITHOUT requiring
GPU execution (the predicate is a pure-Python membership test). A GPU
correctness/perf check is performed separately by the validation harness in
mc2-workspaces/K-1793/scripts/k1793_paired_n30.py.
"""
import itertools

import pytest
import torch

from tritonblas.matmul import (
    _K1793_GATE_DTYPES,
    _K1793_GATE_K,
    _K1793_GATE_M,
    _K1793_GATE_N,
    _k1793_route_to_hbl,
)


# ------ admit-set cardinality / structure ------

def test_k1793_admit_set_cardinality():
    """The K-1793 admit set covers exactly 18 cells (3 M x 1 N x 3 K x 2 dtype)."""
    cells = [
        (M, _K1793_GATE_N, K, dt)
        for M in _K1793_GATE_M
        for K in _K1793_GATE_K
        for dt in _K1793_GATE_DTYPES
    ]
    admitted = [c for c in cells if _k1793_route_to_hbl(*c)]
    assert len(admitted) == 18, (
        f"Expected 18 admitted cells, got {len(admitted)}: {admitted}"
    )


def test_k1793_admit_set_axes():
    """Pin the M / N / K / dtype axes against accidental drift."""
    assert _K1793_GATE_N == 192
    assert _K1793_GATE_M == frozenset({2048, 4096, 8192})
    assert _K1793_GATE_K == frozenset({4096, 8192, 16384})
    assert set(_K1793_GATE_DTYPES) == {torch.float16, torch.bfloat16}


# ------ exclusion: adjacent N values stay unrouted ------

@pytest.mark.parametrize("N", [16, 32, 48, 64, 80, 96, 128, 160, 224, 256, 512, 1024])
def test_k1793_adjacent_N_not_routed(N):
    """Adjacent N values (especially N=128 and N=256) must NOT trigger K-1793.

    Those are owned by upstream alias-stack slots (K-1685 P28 N=128, K-1538 P22
    N=256). K-1793's gate must be N==192 only — widening it would shadow.
    """
    for M, K, dt in itertools.product(_K1793_GATE_M, _K1793_GATE_K, _K1793_GATE_DTYPES):
        assert not _k1793_route_to_hbl(M, N, K, dt), (
            f"K-1793 gate accidentally routed N={N} (must stay unrouted)"
        )


# ------ exclusion: out-of-band M / K / dtype stay unrouted ------

@pytest.mark.parametrize("M", [512, 1024, 1536, 3072, 6144, 16384])
def test_k1793_out_of_band_M_not_routed(M):
    """Only M in {2048, 4096, 8192} should be routed."""
    for K in _K1793_GATE_K:
        for dt in _K1793_GATE_DTYPES:
            assert not _k1793_route_to_hbl(M, _K1793_GATE_N, K, dt)


@pytest.mark.parametrize("K", [256, 512, 1024, 2048, 3072, 12288, 32768, 65536])
def test_k1793_out_of_band_K_not_routed(K):
    """Only K in {4096, 8192, 16384} should be routed."""
    for M in _K1793_GATE_M:
        for dt in _K1793_GATE_DTYPES:
            assert not _k1793_route_to_hbl(M, _K1793_GATE_N, K, dt)


@pytest.mark.parametrize("dt", [torch.float32, torch.float64, torch.int8])
def test_k1793_out_of_band_dtype_not_routed(dt):
    """Only fp16 / bf16 are routed; fp32 / fp64 / int8 stay on tritonblas path."""
    for M in _K1793_GATE_M:
        for K in _K1793_GATE_K:
            assert not _k1793_route_to_hbl(M, _K1793_GATE_N, K, dt)


# ------ membership matrix: every admitted cell flips True ------

@pytest.mark.parametrize("M", sorted(_K1793_GATE_M))
@pytest.mark.parametrize("K", sorted(_K1793_GATE_K))
@pytest.mark.parametrize("dt", list(_K1793_GATE_DTYPES))
def test_k1793_admitted_cells_route(M, K, dt):
    """Every (M, N=192, K, dtype) in the admit set must route to HBL."""
    assert _k1793_route_to_hbl(M, _K1793_GATE_N, K, dt)


# ------ disjointness from K-1764 (M=N=4096 mid-K square cohort) ------

def test_k1793_disjoint_from_K1764_admit_set():
    """K-1793 is N=192-only; K-1764 is N==4096. No (M, N, K, dt) cell can be in both."""
    for M, K, dt in itertools.product(_K1793_GATE_M, _K1793_GATE_K, _K1793_GATE_DTYPES):
        # K-1764 only fires when N==4096; K-1793's N is 192 -> trivially disjoint.
        assert _K1793_GATE_N != 4096
