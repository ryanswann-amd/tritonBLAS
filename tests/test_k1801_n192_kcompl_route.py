"""K-1801 regression tests for the N=192 K-COMPLEMENT hipBLASLt route-OUT gate.

Productionizes K-1782's paired n=30 hot-cache HIP-graph audit: the gate routes
18 verified-winning cells to torch.matmul (hipBLASLt):

    M in {2048, 4096, 8192} x N == 192 x K in {4096, 8192, 16384}
    x dtype in {fp16, bf16}

Pure-Python predicate-structure tests; GPU correctness/perf is verified
separately by mc2-workspaces/K-1801/scripts/k1801_paired_n30.py.
"""
import itertools

import pytest
import torch

from tritonblas.matmul import (
    _K1801_GATE_DTYPES,
    _K1801_GATE_K,
    _K1801_GATE_M,
    _K1801_GATE_N,
    _k1801_route_to_hbl,
)


def test_k1801_admit_set_cardinality():
    """The K-1801 admit set covers exactly 18 cells (3 M x 1 N x 3 K x 2 dtype)."""
    cells = [
        (M, _K1801_GATE_N, K, dt)
        for M in _K1801_GATE_M
        for K in _K1801_GATE_K
        for dt in _K1801_GATE_DTYPES
    ]
    admitted = [c for c in cells if _k1801_route_to_hbl(*c)]
    assert len(admitted) == 18, (
        f"Expected 18 admitted cells, got {len(admitted)}: {admitted}"
    )


def test_k1801_admit_set_axes():
    """Pin the M / N / K / dtype axes against accidental drift."""
    assert _K1801_GATE_N == 192
    assert _K1801_GATE_M == frozenset({2048, 4096, 8192})
    assert _K1801_GATE_K == frozenset({4096, 8192, 16384})
    assert set(_K1801_GATE_DTYPES) == {torch.float16, torch.bfloat16}


@pytest.mark.parametrize("N", [16, 32, 48, 64, 80, 96, 128, 160, 224, 256, 384, 512, 1024, 4096])
def test_k1801_adjacent_N_not_routed(N):
    """Adjacent N values (especially N=128 and N=256) must NOT trigger K-1801.

    Those are owned by upstream alias-stack slots (K-1685 P28 N=128, K-1538 P22
    N=256). K-1801's gate must be N==192 only - widening would shadow them.
    """
    for M, K, dt in itertools.product(_K1801_GATE_M, _K1801_GATE_K, _K1801_GATE_DTYPES):
        assert not _k1801_route_to_hbl(M, N, K, dt), (
            f"K-1801 gate accidentally routed N={N} (must stay unrouted)"
        )


@pytest.mark.parametrize("M", [512, 1024, 1536, 3072, 6144, 16384])
def test_k1801_out_of_band_M_not_routed(M):
    """Only M in {2048, 4096, 8192} should be routed."""
    for K in _K1801_GATE_K:
        for dt in _K1801_GATE_DTYPES:
            assert not _k1801_route_to_hbl(M, _K1801_GATE_N, K, dt)


@pytest.mark.parametrize("K", [256, 512, 1024, 2048, 3072, 12288, 32768, 65536])
def test_k1801_out_of_band_K_not_routed(K):
    """Only K in {4096, 8192, 16384} should be routed."""
    for M in _K1801_GATE_M:
        for dt in _K1801_GATE_DTYPES:
            assert not _k1801_route_to_hbl(M, _K1801_GATE_N, K, dt)


@pytest.mark.parametrize("dt", [torch.float32, torch.float64, torch.int8])
def test_k1801_out_of_band_dtype_not_routed(dt):
    """Only fp16 / bf16 are routed; fp32 / fp64 / int8 stay on the tritonblas path."""
    for M in _K1801_GATE_M:
        for K in _K1801_GATE_K:
            assert not _k1801_route_to_hbl(M, _K1801_GATE_N, K, dt)


@pytest.mark.parametrize("M", sorted(_K1801_GATE_M))
@pytest.mark.parametrize("K", sorted(_K1801_GATE_K))
@pytest.mark.parametrize("dt", list(_K1801_GATE_DTYPES))
def test_k1801_admitted_cells_route(M, K, dt):
    """Every (M, N=192, K, dtype) in the admit set must route to HBL."""
    assert _k1801_route_to_hbl(M, _K1801_GATE_N, K, dt)


def test_k1801_disjoint_from_K1764_admit_set():
    """K-1801 is N=192 only; K-1764 is the M=N=4096 cohort.

    The two admit sets are trivially disjoint by N-axis projection.
    """
    assert _K1801_GATE_N != 4096  # would shadow K-1764's M=N=4096 cohort
    assert _K1801_GATE_N != 8192  # would shadow K-1749's M=N=8192 cohort
    assert _K1801_GATE_N != 128   # would shadow K-1685 P28's N=128 alias slot
    assert _K1801_GATE_N != 256   # would shadow K-1538 P22's N=256 alias slot
