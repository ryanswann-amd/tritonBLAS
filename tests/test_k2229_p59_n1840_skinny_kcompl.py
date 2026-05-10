"""K-2229 (S-002) — invariants for P59 N=1840 K-COMPLEMENT alias-stack.

Per minimalist src/tests split (R-1532 / R-1720 / R-1775): the
`_route_predicate.py` module holds the *data* (frozenset literal); this
file holds the *invariants* — cardinality, residue class, dtype set,
disjointness with prior off-by-48 rungs, the 18-cell membership
expansion, and adjacency firewalls (the cells immediately neighbouring
N=1840 in the K-2214 admit set must remain un-routed).

The frozenset itself is a single-N slice (N=1840 only) over the full
canonical 18-cell envelope: M in {2048,4096,8192} x K in {4096,8192,
16384} x {bf16, fp16} -- the K-2209 verification cohort.

16th-rung extension of the off-by-48 contiguous ladder branched off
fix/K-2214@163c3f8 (which itself branches off fix/K-2183@9375e7a ->
fix/K-2169@161e017 -> fix/K-2158@beab03e -> fix/K-2127@9c4f984 ->
fix/K-1922@0024a71 -- matches K-2091 / K-2092 / K-2107 / K-2127 /
K-2158 / K-2169 / K-2183 / K-2214 lineage).
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K2229_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_18,
)

EXPECTED_M = (2048, 4096, 8192)
EXPECTED_K = (4096, 8192, 16384)
EXPECTED_DTYPES = ("torch.bfloat16", "torch.float16")
EXPECTED_N = 1840
EXPECTED_CARDINALITY = 18


def test_cardinality_is_18():
    assert len(_K2229_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_18) == EXPECTED_CARDINALITY


def test_n_is_residue_48_mod_64():
    # Off-by-48 wave-misaligned membership invariant
    for (M, N, K, dt) in _K2229_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_18:
        assert N == EXPECTED_N
        assert N % 64 == 48
        # mod 128 == 48 returns to the K-2127 P54 / K-2169 P56 sub-class
        # -- admit identically per K-2150 R-K2150.MOD-64-IS-BINDING.
        assert N % 128 == 48


def test_full_grid_membership():
    expected = {
        (M, EXPECTED_N, K, dt)
        for M in EXPECTED_M
        for K in EXPECTED_K
        for dt in EXPECTED_DTYPES
    }
    assert _K2229_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_18 == expected


def test_dtype_set_is_bf16_fp16_only():
    dts = {dt for (_, _, _, dt) in _K2229_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_18}
    assert dts == {"torch.bfloat16", "torch.float16"}


def test_full_3x3_grid_per_dtype():
    for dt in EXPECTED_DTYPES:
        cells = {
            (M, K) for (M, _, K, d) in _K2229_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_18
            if d == dt
        }
        expected = {(M, K) for M in EXPECTED_M for K in EXPECTED_K}
        assert cells == expected, f"Missing 3x3 grid for dtype {dt}"


def test_disjoint_with_prior_off_by_48_rungs():
    # K-2229 is a single-N slice at N=1840; every prior off-by-48 rung is
    # at a strictly different N (544..1776) so disjointness must hold
    # trivially.
    from tritonblas._route_predicate import (
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18,
        _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18,
        _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8,
        _K2183_P57_SKINNY_N1712_KCOMPL_ALIASSTACK_8,
        _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10,
    )
    assert _K2229_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_18.isdisjoint(
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18
    )
    assert _K2229_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_18.isdisjoint(
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18
    )
    assert _K2229_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_18.isdisjoint(
        _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18
    )
    assert _K2229_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_18.isdisjoint(
        _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8
    )
    assert _K2229_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_18.isdisjoint(
        _K2183_P57_SKINNY_N1712_KCOMPL_ALIASSTACK_8
    )
    assert _K2229_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_18.isdisjoint(
        _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10
    )


def test_n1840_extends_off_by_48_ladder_above_n1776():
    # Sanity: N=1840 is the next admitted residue-48 step above K-2214 P58
    # N=1776 in this rung sequence (+64 step on the residue-48 lattice).
    assert (EXPECTED_N - 1776) % 64 == 0
    assert (EXPECTED_N - 1776) == 64
    assert EXPECTED_N > 1776


def test_adjacency_firewalls_n1840_neighbors():
    # The cells immediately neighbouring N=1840 on the residue-48 lattice
    # (N=1776 is the K-2214 admit immediately below; N=1904 is the next
    # un-audited rung above) and off-residue N must remain OUTSIDE the
    # K-2229 frozenset -- preserve the per-N audit handles per the K-1175
    # stacked-predicate convention.
    forbidden = [
        # N=1776 (K-2214 admit, lives in its own frozenset, not this one)
        (4096, 1776, 8192, "torch.bfloat16"),
        # N=1904 (next un-audited residue-48 rung above 1840)
        (4096, 1904, 8192, "torch.bfloat16"),
        # Off-residue N (1856 is wave-aligned mod 64 == 0, not residue-48)
        (4096, 1856, 8192, "torch.bfloat16"),
        # Off-residue N (1824 is residue-32, not residue-48)
        (4096, 1824, 8192, "torch.bfloat16"),
        # Wrong dtype (fp32) at admit cell
        (4096, 1840, 8192, "torch.float32"),
    ]
    for cell in forbidden:
        assert cell not in _K2229_P59_SKINNY_N1840_KCOMPL_ALIASSTACK_18, (
            f"K-2229 frozenset must NOT contain firewall cell {cell}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
