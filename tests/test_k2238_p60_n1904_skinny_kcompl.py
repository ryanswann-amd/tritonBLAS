"""K-2238 (S-002) — invariants for P60 N=1904 K-COMPLEMENT alias-stack.

Per minimalist src/tests split (R-1532 / R-1720 / R-1775): the
`_route_predicate.py` module holds the *data* (frozenset literal); this
file holds the *invariants* — cardinality, residue class, dtype set,
disjointness with prior off-by-48 rungs, the 8-cell membership
expansion, and adjacency firewalls (the cells immediately neighbouring
N=1904 in the K-2223 N=1840 admit set must remain un-routed).

The frozenset itself is a single-N slice (N=1904 only) over the
task-specified verification cohort M in {4096, 8192} x K in {8192, 16384}
x {bf16, fp16}, i.e. 2 * 2 * 2 = 8 cells (smaller than the canonical
18-cell envelope used by K-2127 / K-2158 — extension to the full grid
is deferred to a follow-up rung-stitch task).

17th-rung extension of the off-by-48 contiguous ladder branched off
fix/K-2223@cb603bf (which itself branches off fix/K-2203@aa9ccc4 ->
fix/K-2183@9375e7a -> fix/K-2169@161e017 -> fix/K-2158@beab03e ->
fix/K-2127@9c4f984 -> fix/K-1922@0024a71 — matches K-2091 / K-2092 /
K-2107 / K-2127 / K-2158 / K-2169 / K-2183 / K-2203 / K-2223 lineage).
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K2238_P60_SKINNY_N1904_KCOMPL_ALIASSTACK_8,
)

EXPECTED_M = (4096, 8192)
EXPECTED_K = (8192, 16384)
EXPECTED_DT = ("torch.bfloat16", "torch.float16")
EXPECTED_N = 1904
EXPECTED_CARDINALITY = 8


def test_cardinality_is_8():
    assert len(_K2238_P60_SKINNY_N1904_KCOMPL_ALIASSTACK_8) == EXPECTED_CARDINALITY


def test_n_is_residue_48_mod_64():
    # Off-by-48 wave-misaligned membership invariant
    for (M, N, K, dt) in _K2238_P60_SKINNY_N1904_KCOMPL_ALIASSTACK_8:
        assert N == EXPECTED_N
        assert N % 64 == 48
        # mod 128 == 112 is the K-2169 N=1648 / K-2203 N=1776 sub-class —
        # admit identically per K-2150
        # R-K2150.MOD-64-IS-BINDING-MOD-128-IS-NOT-FOR-RESIDUE-48-FAMILY.
        assert N % 128 == 112


def test_full_grid_membership():
    expected = {
        (M, EXPECTED_N, K, dt)
        for M in EXPECTED_M
        for K in EXPECTED_K
        for dt in EXPECTED_DT
    }
    assert _K2238_P60_SKINNY_N1904_KCOMPL_ALIASSTACK_8 == expected


def test_dtype_set_is_bf16_fp16_only():
    dts = {dt for (_, _, _, dt) in _K2238_P60_SKINNY_N1904_KCOMPL_ALIASSTACK_8}
    assert dts == set(EXPECTED_DT)


def test_disjoint_with_prior_off_by_48_rungs():
    # K-2238 is a single-N slice at N=1904; every prior off-by-48 rung is
    # at a strictly different N (544..1840) so disjointness must hold
    # trivially.
    from tritonblas._route_predicate import (
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18,
        _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18,
        _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8,
        _K2183_P57_SKINNY_N1712_KCOMPL_ALIASSTACK_8,
        _K2203_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_8,
        _K2223_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_8,
    )
    assert _K2238_P60_SKINNY_N1904_KCOMPL_ALIASSTACK_8.isdisjoint(
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18
    )
    assert _K2238_P60_SKINNY_N1904_KCOMPL_ALIASSTACK_8.isdisjoint(
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18
    )
    assert _K2238_P60_SKINNY_N1904_KCOMPL_ALIASSTACK_8.isdisjoint(
        _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18
    )
    assert _K2238_P60_SKINNY_N1904_KCOMPL_ALIASSTACK_8.isdisjoint(
        _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8
    )
    assert _K2238_P60_SKINNY_N1904_KCOMPL_ALIASSTACK_8.isdisjoint(
        _K2183_P57_SKINNY_N1712_KCOMPL_ALIASSTACK_8
    )
    assert _K2238_P60_SKINNY_N1904_KCOMPL_ALIASSTACK_8.isdisjoint(
        _K2203_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_8
    )
    assert _K2238_P60_SKINNY_N1904_KCOMPL_ALIASSTACK_8.isdisjoint(
        _K2223_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_8
    )


def test_n1904_extends_off_by_48_ladder_above_n1840():
    # Sanity: N=1904 is the next admitted residue-48 step above K-2223 P59
    # N=1840 in this rung sequence (+64 step on the residue-48 lattice).
    assert (EXPECTED_N - 1840) % 64 == 0
    assert (EXPECTED_N - 1840) == 64
    assert EXPECTED_N > 1840


def test_adjacency_firewalls_n1904_neighbors():
    # The cells immediately neighbouring N=1904 on the residue-48 lattice
    # (N=1840 is the K-2223 admit immediately below; N=1968 is the next
    # un-audited rung above) and the M=2048 row at N=1904 must remain
    # OUTSIDE the K-2238 frozenset — preserve the per-N audit handles per
    # the K-1175 stacked-predicate convention.
    forbidden = [
        # M=2048 row at N=1904 (excluded from the 8-cell verification cohort)
        (2048, 1904, 8192, "torch.bfloat16"),
        (2048, 1904, 8192, "torch.float16"),
        (2048, 1904, 16384, "torch.bfloat16"),
        (2048, 1904, 16384, "torch.float16"),
        # K=4096 column at N=1904 (excluded from the 8-cell verification cohort)
        (4096, 1904, 4096, "torch.bfloat16"),
        (8192, 1904, 4096, "torch.float16"),
        # N=1840 (K-2223 admit, lives in its own frozenset, not this one)
        (4096, 1840, 8192, "torch.bfloat16"),
        # N=1968 (next un-audited residue-48 rung above 1904)
        (4096, 1968, 8192, "torch.bfloat16"),
    ]
    for cell in forbidden:
        assert cell not in _K2238_P60_SKINNY_N1904_KCOMPL_ALIASSTACK_8, (
            f"K-2238 frozenset must NOT contain firewall cell {cell}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
