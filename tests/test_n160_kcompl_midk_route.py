"""
Tests for N=160 K-COMPLEMENT mid-K route-OUT gate.

Pins the admit set, the dtype/N projections, the stream-K carve-out,
and the disjointness from adjacent N rungs.
"""
import pytest
import torch

from tritonblas.matmul import (
    _N160_KCOMPL_MIDK_ROUTE_HBL,
    _route_n160_kcompl_midk_to_hbl,
)


_EXPECTED_M = (2048, 4096, 8192)
_EXPECTED_K = (4096, 8192, 16384)
_EXPECTED_DTYPES = (torch.float16, torch.bfloat16)


def test_admit_set_cardinality():
    assert len(_N160_KCOMPL_MIDK_ROUTE_HBL) == 18


def test_admit_set_full_dense():
    expected = {
        (M, 160, K, dt)
        for M in _EXPECTED_M
        for K in _EXPECTED_K
        for dt in _EXPECTED_DTYPES
    }
    assert _N160_KCOMPL_MIDK_ROUTE_HBL == expected


def test_admit_set_n_axis_only_160():
    assert {n for (_, n, _, _) in _N160_KCOMPL_MIDK_ROUTE_HBL} == {160}


def test_admit_set_m_axis():
    assert {m for (m, _, _, _) in _N160_KCOMPL_MIDK_ROUTE_HBL} == set(_EXPECTED_M)


def test_admit_set_k_axis():
    assert {k for (_, _, k, _) in _N160_KCOMPL_MIDK_ROUTE_HBL} == set(_EXPECTED_K)


def test_admit_set_dtype_axis():
    assert {dt for (_, _, _, dt) in _N160_KCOMPL_MIDK_ROUTE_HBL} == set(_EXPECTED_DTYPES)


@pytest.mark.parametrize("M", _EXPECTED_M)
@pytest.mark.parametrize("K", _EXPECTED_K)
@pytest.mark.parametrize("dtype", _EXPECTED_DTYPES)
def test_predicate_admits_routed_cells(M, K, dtype):
    a = torch.empty(M, K, dtype=dtype, device="meta")
    b = torch.empty(K, 160, dtype=dtype, device="meta")
    assert _route_n160_kcompl_midk_to_hbl(a, b, enable_streamk=False) is True


@pytest.mark.parametrize("M", _EXPECTED_M)
@pytest.mark.parametrize("K", _EXPECTED_K)
@pytest.mark.parametrize("dtype", _EXPECTED_DTYPES)
def test_predicate_streamk_carveout(M, K, dtype):
    """Caller-explicit enable_streamk=True must bypass the gate."""
    a = torch.empty(M, K, dtype=dtype, device="meta")
    b = torch.empty(K, 160, dtype=dtype, device="meta")
    assert _route_n160_kcompl_midk_to_hbl(a, b, enable_streamk=True) is False


@pytest.mark.parametrize("N_neighbor", [128, 192, 224])
def test_predicate_rejects_adjacent_n(N_neighbor):
    """Adjacent-N firewall: N!=160 must NOT be routed by this gate."""
    a = torch.empty(4096, 8192, dtype=torch.float16, device="meta")
    b = torch.empty(8192, N_neighbor, dtype=torch.float16, device="meta")
    assert _route_n160_kcompl_midk_to_hbl(a, b, enable_streamk=False) is False


def test_predicate_rejects_unverified_K():
    """K outside the verified mid-K subset must NOT route."""
    for K in (1024, 2048, 32768, 65536):
        a = torch.empty(4096, K, dtype=torch.float16, device="meta")
        b = torch.empty(K, 160, dtype=torch.float16, device="meta")
        assert _route_n160_kcompl_midk_to_hbl(a, b, enable_streamk=False) is False


def test_predicate_rejects_unverified_M():
    """M outside the verified subset must NOT route."""
    for M in (1024, 1536, 16384):
        a = torch.empty(M, 8192, dtype=torch.float16, device="meta")
        b = torch.empty(8192, 160, dtype=torch.float16, device="meta")
        assert _route_n160_kcompl_midk_to_hbl(a, b, enable_streamk=False) is False


def test_predicate_rejects_unverified_dtype():
    """Dtype outside verified set must NOT route."""
    for dtype in (torch.float32,):
        a = torch.empty(4096, 8192, dtype=dtype, device="meta")
        b = torch.empty(8192, 160, dtype=dtype, device="meta")
        assert _route_n160_kcompl_midk_to_hbl(a, b, enable_streamk=False) is False
