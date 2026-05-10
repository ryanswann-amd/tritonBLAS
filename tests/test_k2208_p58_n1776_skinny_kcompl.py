"""K-2208 (S-002) — invariants for P58 N=1776 K-COMPLEMENT alias-stack.

Per minimalist src/tests split (R-1532 / R-1720 / R-1775): the
`_route_predicate.py` module holds the *data* (frozenset literal); this
file holds the *invariants* — cardinality, residue class, dtype set,
disjointness with prior off-by-48 rungs, the 8-cell membership
expansion, and adjacency firewalls (the cells immediately neighbouring
N=1776 in the K-2183 14-cell admit set must remain un-routed).

The frozenset itself is a single-N slice (N=1776 only) over the
task-specified verification cohort M in {4096, 8192} x K in {8192, 16384}
x {bf16, fp16}, i.e. 2 * 2 * 2 = 8 cells (smaller than the canonical
18-cell envelope used by K-2127 / K-2158 — extension to the full grid
is deferred to a follow-up rung-stitch task).

15th-rung extension of the off-by-48 contiguous ladder branched off
fix/K-2183@9375e7a (which itself branches off fix/K-2169@161e017 ->
fix/K-2158@beab03e -> fix/K-2127@9c4f984 -> fix/K-1922@0024a71 —
matches K-2091 / K-2092 / K-2107 / K-2127 / K-2158 / K-2169 / K-2183
lineage).
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K2208_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_8,
)

EXPECTED_M = (4096, 8192)
EXPECTED_K = (8192, 16384)
EXPECTED_DT = ("torch.bfloat16", "torch.float16")
EXPECTED_N = 1776
EXPECTED_CARDINALITY = 8


def test_cardinality_is_8():
    assert len(_K2208_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_8) == EXPECTED_CARDINALITY


def test_n_is_residue_48_mod_64():
    # Off-by-48 wave-misaligned membership invariant
    for (M, N, K, dt) in _K2208_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_8:
        assert N == EXPECTED_N
        assert N % 64 == 48
        # mod 128 == 112 is the K-2055/K-2060/K-2063/K-2066 / K-2071/
        # K-2085/K-2091 / K-2107 / K-2127 / K-2169 sub-class — admit
        # identically per K-2150 R-K2150.MOD-64-IS-BINDING.
        assert N % 128 == 112


def test_full_grid_membership():
    expected = {
        (M, EXPECTED_N, K, dt)
        for M in EXPECTED_M
        for K in EXPECTED_K
        for dt in EXPECTED_DT
    }
    assert _K2208_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_8 == expected


def test_dtype_set_is_bf16_fp16_only():
    dts = {dt for (_, _, _, dt) in _K2208_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_8}
    assert dts == set(EXPECTED_DT)


def test_disjoint_with_prior_off_by_48_rungs():
    # K-2208 is a single-N slice at N=1776; every prior off-by-48 rung is
    # at a strictly different N (544..1712) so disjointness must hold
    # trivially.
    from tritonblas._route_predicate import (
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18,
        _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18,
        _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8,
        _K2183_P57_SKINNY_N1712_KCOMPL_ALIASSTACK_8,
    )
    assert _K2208_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_8.isdisjoint(
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18
    )
    assert _K2208_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_8.isdisjoint(
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18
    )
    assert _K2208_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_8.isdisjoint(
        _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18
    )
    assert _K2208_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_8.isdisjoint(
        _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8
    )
    assert _K2208_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_8.isdisjoint(
        _K2183_P57_SKINNY_N1712_KCOMPL_ALIASSTACK_8
    )


def test_n1776_extends_off_by_48_ladder_above_n1712():
    # Sanity: N=1776 is the next admitted residue-48 step above K-2183 P57
    # N=1712 in this rung sequence (+64 step on the residue-48 lattice).
    assert (EXPECTED_N - 1712) % 64 == 0
    assert (EXPECTED_N - 1712) == 64
    assert EXPECTED_N > 1712


def test_adjacency_firewalls_n1776_neighbors():
    # The cells immediately neighbouring N=1776 on the residue-48 lattice
    # (N=1712 is the K-2183 admit immediately below; N=1840 is the next
    # un-audited rung above) and the M=2048 row at N=1776 must remain
    # OUTSIDE the K-2208 frozenset — preserve the per-N audit handles per
    # the K-1175 stacked-predicate convention.
    forbidden = [
        # M=2048 row at N=1776 (excluded from the 8-cell verification cohort)
        (2048, 1776, 8192, "torch.bfloat16"),
        (2048, 1776, 8192, "torch.float16"),
        (2048, 1776, 16384, "torch.bfloat16"),
        (2048, 1776, 16384, "torch.float16"),
        # K=4096 column at N=1776 (excluded from the 8-cell verification cohort)
        (4096, 1776, 4096, "torch.bfloat16"),
        (8192, 1776, 4096, "torch.float16"),
        # N=1712 (K-2183 admit, lives in its own frozenset, not this one)
        (4096, 1712, 8192, "torch.bfloat16"),
        # N=1840 (next un-audited residue-48 rung above 1776)
        (4096, 1840, 8192, "torch.bfloat16"),
    ]
    for cell in forbidden:
        assert cell not in _K2208_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_8, (
            f"K-2208 frozenset must NOT contain firewall cell {cell}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
