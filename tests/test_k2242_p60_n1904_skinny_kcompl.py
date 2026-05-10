"""K-2242 P60 invariant pytest — N=1904 K-COMPLEMENT alias-stack 17th rung.

Pinned invariants (R-1532 / R-1720 / R-1775 — src holds data, tests hold invariants):
1. Cardinality: frozenset has exactly 8 entries.
2. Residue: all entries have N=1904, mod 64 == 48, mod 128 == 112.
3. Dtype set: {torch.bfloat16, torch.float16}.
4. Full grid membership: M ∈ {4096, 8192} × K ∈ {8192, 16384}.
5. Pairwise disjointness vs every prior K-COMPLEMENT alias-stack slot.
6. +64 step from K-2223 P59 N=1840 (predecessor rung).
7. Adjacency firewalls: N=1840 (predecessor) and N=1968 (next +64) MUST NOT appear in P60 set.
"""
import pytest

from tritonblas._route_predicate import (
    _K2242_P60_SKINNY_N1904_KCOMPL_ALIASSTACK_8,
    _K2223_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_8,
    _K2203_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_8,
    _K2183_P57_SKINNY_N1712_KCOMPL_ALIASSTACK_8,
    _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8,
    _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18,
    _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18,
    _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
)

S = _K2242_P60_SKINNY_N1904_KCOMPL_ALIASSTACK_8


def test_cardinality():
    assert len(S) == 8


def test_residue_n_1904_and_mod_64_eq_48_mod_128_eq_112():
    for (M, N, K, dt) in S:
        assert N == 1904
        assert N % 64 == 48
        assert N % 128 == 112


def test_dtype_set():
    assert {dt for (_, _, _, dt) in S} == {"torch.bfloat16", "torch.float16"}


def test_full_grid_membership():
    assert {M for (M, _, _, _) in S} == {4096, 8192}
    assert {K for (_, _, K, _) in S} == {8192, 16384}
    for M in (4096, 8192):
        for K in (8192, 16384):
            for dt in ("torch.bfloat16", "torch.float16"):
                assert (M, 1904, K, dt) in S


@pytest.mark.parametrize("prior,name", [
    (_K2223_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_8, "K2223_P59_N1840"),
    (_K2203_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_8, "K2203_P58_N1776"),
    (_K2183_P57_SKINNY_N1712_KCOMPL_ALIASSTACK_8, "K2183_P57_N1712"),
    (_K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8, "K2169_P56_N1648"),
    (_K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18, "K2158_P55_N1584"),
    (_K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18, "K2127_P54_N1456"),
    (_K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18, "K1922_P40_N544"),
])
def test_pairwise_disjoint_vs_prior_slot(prior, name):
    inter = S & prior
    assert not inter, f"P60 N=1904 overlaps {name}: {inter}"


def test_plus64_step_from_k2223_p59_n1840():
    p59_ns = {N for (_, N, _, _) in _K2223_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_8}
    p60_ns = {N for (_, N, _, _) in S}
    assert p59_ns == {1840}
    assert p60_ns == {1904}
    assert (1904 - 1840) == 64


def test_adjacency_firewalls():
    for (M, N, K, dt) in S:
        assert N != 1840
        assert N != 1968
