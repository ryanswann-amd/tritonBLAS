"""K-1938 / K-1926 — route-table verification for the P41 N=608
wave-misaligned skinny-N K-COMPLEMENT alias-stack slot.

Pure unit tests: do not require a GPU.  Verifies that

  1. The verified-winner frozenset has exactly the 15 cells produced by
     the paired n=30 HIP-graph hot-cache bench (gate hbl/tb<=0.85,
     paired-t p<0.05).
  2. ``route_to_hbl`` returns True for every winner and False for the
     three documented non-winners.
  3. ``route_to_hbl`` returns False for every shape outside the cohort
     (different N, mismatched dtypes, scalars, etc.) — i.e. the slot
     does not over-fire.
"""
import pytest
import torch

from tritonblas._route_predicate import (
    _P41_SKINNY_N608_KCOMPL_VERIFIED_WIN,
    _p41_skinny_n608_kcompl_route_to_hbl,
    route_to_hbl,
)


WINNERS = [
    (2048, 608, 4096, torch.bfloat16),
    (2048, 608, 8192, torch.bfloat16),
    (2048, 608, 16384, torch.bfloat16),
    (4096, 608, 16384, torch.bfloat16),
    (8192, 608, 4096, torch.bfloat16),
    (8192, 608, 8192, torch.bfloat16),
    (8192, 608, 16384, torch.bfloat16),
    (2048, 608, 4096, torch.float16),
    (2048, 608, 8192, torch.float16),
    (2048, 608, 16384, torch.float16),
    (4096, 608, 8192, torch.float16),
    (4096, 608, 16384, torch.float16),
    (8192, 608, 4096, torch.float16),
    (8192, 608, 8192, torch.float16),
    (8192, 608, 16384, torch.float16),
]

# Cohort cells that did NOT meet the gate (hbl/tb > 0.85).
NON_WINNERS = [
    (4096, 608, 4096, torch.bfloat16),
    (4096, 608, 8192, torch.bfloat16),
    (4096, 608, 4096, torch.float16),
]


def test_frozenset_size():
    assert len(_P41_SKINNY_N608_KCOMPL_VERIFIED_WIN) == 15


@pytest.mark.parametrize("M,N,K,dt", WINNERS)
def test_winners_route_to_hbl(M, N, K, dt):
    assert _p41_skinny_n608_kcompl_route_to_hbl(M, N, K, dt, dt) is True
    assert route_to_hbl(M, N, K, dt, dt) is True


@pytest.mark.parametrize("M,N,K,dt", NON_WINNERS)
def test_non_winners_do_not_route(M, N, K, dt):
    assert _p41_skinny_n608_kcompl_route_to_hbl(M, N, K, dt, dt) is False
    assert route_to_hbl(M, N, K, dt, dt) is False


@pytest.mark.parametrize("N", [512, 544, 576, 640, 672, 1024])
def test_other_n_not_routed(N):
    """Sanity: shapes outside N=608 must not be claimed by the P41 slot."""
    assert _p41_skinny_n608_kcompl_route_to_hbl(2048, N, 4096, torch.bfloat16,
                                                torch.bfloat16) is False


def test_mixed_dtype_not_routed():
    """A/B dtype mismatch must short-circuit to False."""
    assert _p41_skinny_n608_kcompl_route_to_hbl(2048, 608, 4096, torch.bfloat16,
                                                torch.float16) is False
