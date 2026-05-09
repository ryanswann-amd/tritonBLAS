"""K-1538 P22 — regression-pin tests for the 14th-position alias-stack
frozenset _K1538_P22_SKINNY_N512_KCOMPL_30.

Mirrors the K-1493 P20 alias-stack test pattern (admit + sibling-N
firewall + dispatch integration + disable-env short-circuit).
"""
import os

import pytest
import torch

from tritonblas._route_predicate import (
    _K1538_P22_SKINNY_N512_KCOMPL_30,
    _k1538_p22_skinny_n512_alias_routeout,
    k971_route_decision,
)


# All 30 N=512 K-COMPLEMENT cells (M ∈ {2048,4096,8192} × K ∈
# {2048,4096,8192,16384,32768} × {bf16,fp16}).
_ADMIT_30 = [
    (2048,  512,  2048, torch.bfloat16), (2048,  512,  2048, torch.float16),
    (2048,  512,  4096, torch.bfloat16), (2048,  512,  4096, torch.float16),
    (2048,  512,  8192, torch.bfloat16), (2048,  512,  8192, torch.float16),
    (2048,  512, 16384, torch.bfloat16), (2048,  512, 16384, torch.float16),
    (2048,  512, 32768, torch.bfloat16), (2048,  512, 32768, torch.float16),
    (4096,  512,  2048, torch.bfloat16), (4096,  512,  2048, torch.float16),
    (4096,  512,  4096, torch.bfloat16), (4096,  512,  4096, torch.float16),
    (4096,  512,  8192, torch.bfloat16), (4096,  512,  8192, torch.float16),
    (4096,  512, 16384, torch.bfloat16), (4096,  512, 16384, torch.float16),
    (4096,  512, 32768, torch.bfloat16), (4096,  512, 32768, torch.float16),
    (8192,  512,  2048, torch.bfloat16), (8192,  512,  2048, torch.float16),
    (8192,  512,  4096, torch.bfloat16), (8192,  512,  4096, torch.float16),
    (8192,  512,  8192, torch.bfloat16), (8192,  512,  8192, torch.float16),
    (8192,  512, 16384, torch.bfloat16), (8192,  512, 16384, torch.float16),
    (8192,  512, 32768, torch.bfloat16), (8192,  512, 32768, torch.float16),
]

# Sibling-N rejection sentinels (cells JUST off the N=512 column — must NOT
# match the P22 frozenset).
_SIBLING_N_REJECTS = [
    (2048,  256, 4096, torch.bfloat16),   # P13 N=256
    (2048, 1024, 4096, torch.bfloat16),   # P16 N=1024
    (2048,  128, 4096, torch.bfloat16),   # P13 N=128
    (2048,16384, 4096, torch.bfloat16),   # P19 N=16384
    (4096,  511, 8192, torch.float16),    # off-by-one on N
    (4096,  513, 8192, torch.float16),    # off-by-one on N
    (1024,  512, 8192, torch.bfloat16),   # M not in {2048,4096,8192}
]


@pytest.mark.parametrize("M,N,K,dt", _ADMIT_30)
def test_admit_30(M, N, K, dt):
    assert _k1538_p22_skinny_n512_alias_routeout(M, N, K, dt), \
        f"({M},{N},{K},{dt}) must be in P22 frozenset"


@pytest.mark.parametrize("M,N,K,dt", _SIBLING_N_REJECTS)
def test_sibling_n_firewall(M, N, K, dt):
    assert not _k1538_p22_skinny_n512_alias_routeout(M, N, K, dt), \
        f"({M},{N},{K},{dt}) must NOT be in P22 frozenset (sibling-N firewall)"


def test_cardinality():
    assert len(_K1538_P22_SKINNY_N512_KCOMPL_30) == 30


def test_dispatch_integration_routes_all_30_to_hbl():
    """All 30 N=512 cells route OUT under the full chain (caught by P15+P17
    upstream; P22 is unreachable but documents the envelope)."""
    for M, N, K, dt in _ADMIT_30:
        decision = k971_route_decision(M, N, K, dt, dt, False, False)
        assert decision is True, \
            f"({M},{N},{K},{dt}) expected route-OUT under full chain"


def test_disable_env_set_short_circuit():
    """`disable_env_set=True` overrides everything and returns False."""
    for M, N, K, dt in _ADMIT_30[:5]:
        decision = k971_route_decision(M, N, K, dt, dt, False, False,
                                       disable_env_set=True)
        assert decision is False
