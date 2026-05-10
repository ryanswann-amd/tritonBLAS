"""K-2214 (S-002) — invariants for P58 N=1776 K-COMPLEMENT alias-stack.

Per minimalist src/tests split (R-1532 / R-1720 / R-1775): the
`_route_predicate.py` module holds the *data* (frozenset literal); this
file holds the *invariants* — cardinality, residue class, dtype set,
disjointness with prior off-by-48 rungs, the 10-cell membership
expansion, and adjacency firewalls (the cells immediately neighbouring
N=1776 in the K-2183 14-cell admit set must remain un-routed).

The frozenset itself is a single-N slice (N=1776 only) over the
task-specified verification cohort: 9 bf16 cells (M in {2048,4096,8192}
x K in {4096,8192,16384}) + 1 fp16 spot-check at (M=4096, K=8192) =
10 cells (smaller than the canonical 18-cell envelope used by K-2127 /
K-2158 — extension to the full grid is deferred to a follow-up
rung-stitch task).

15th-rung extension of the off-by-48 contiguous ladder branched off
fix/K-2183@9375e7a (which itself branches off fix/K-2169@161e017 ->
fix/K-2158@beab03e -> fix/K-2127@9c4f984 -> fix/K-1922@0024a71 —
matches K-2091 / K-2092 / K-2107 / K-2127 / K-2158 / K-2169 / K-2183
lineage).
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10,
)

EXPECTED_BF16_M = (2048, 4096, 8192)
EXPECTED_BF16_K = (4096, 8192, 16384)
EXPECTED_FP16_SPOT = (4096, 1776, 8192, "torch.float16")
EXPECTED_N = 1776
EXPECTED_CARDINALITY = 10


def test_cardinality_is_10():
    assert len(_K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10) == EXPECTED_CARDINALITY


def test_n_is_residue_48_mod_64():
    # Off-by-48 wave-misaligned membership invariant
    for (M, N, K, dt) in _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10:
        assert N == EXPECTED_N
        assert N % 64 == 48
        # mod 128 == 112 is the K-2055-2066 / K-2071-2091 / K-2169 sub-class
        # — admit identically per K-2150 R-K2150.MOD-64-IS-BINDING.
        assert N % 128 == 112


def test_full_grid_membership():
    expected_bf16 = {
        (M, EXPECTED_N, K, "torch.bfloat16")
        for M in EXPECTED_BF16_M
        for K in EXPECTED_BF16_K
    }
    expected = expected_bf16 | {EXPECTED_FP16_SPOT}
    assert _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10 == expected


def test_dtype_set_is_bf16_fp16_only():
    dts = {dt for (_, _, _, dt) in _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10}
    assert dts == {"torch.bfloat16", "torch.float16"}


def test_bf16_cells_are_full_3x3_grid():
    bf16_cells = {
        (M, K) for (M, _, K, dt) in _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10
        if dt == "torch.bfloat16"
    }
    expected_bf16 = {(M, K) for M in EXPECTED_BF16_M for K in EXPECTED_BF16_K}
    assert bf16_cells == expected_bf16


def test_fp16_spot_is_single_cell():
    fp16_cells = [
        cell for cell in _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10
        if cell[3] == "torch.float16"
    ]
    assert len(fp16_cells) == 1
    assert fp16_cells[0] == EXPECTED_FP16_SPOT


def test_disjoint_with_prior_off_by_48_rungs():
    # K-2214 is a single-N slice at N=1776; every prior off-by-48 rung is
    # at a strictly different N (544..1712) so disjointness must hold
    # trivially.
    from tritonblas._route_predicate import (
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18,
        _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18,
        _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8,
        _K2183_P57_SKINNY_N1712_KCOMPL_ALIASSTACK_8,
    )
    assert _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10.isdisjoint(
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18
    )
    assert _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10.isdisjoint(
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18
    )
    assert _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10.isdisjoint(
        _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18
    )
    assert _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10.isdisjoint(
        _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8
    )
    assert _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10.isdisjoint(
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
    # un-audited rung above) and the off-cohort fp16 cells at N=1776 must
    # remain OUTSIDE the K-2214 frozenset — preserve the per-N audit
    # handles per the K-1175 stacked-predicate convention.
    forbidden = [
        # fp16 at non-spot M/K cells (only fp16 admit is M=4096, K=8192)
        (2048, 1776, 4096, "torch.float16"),
        (2048, 1776, 8192, "torch.float16"),
        (2048, 1776, 16384, "torch.float16"),
        (4096, 1776, 4096, "torch.float16"),
        (4096, 1776, 16384, "torch.float16"),
        (8192, 1776, 4096, "torch.float16"),
        (8192, 1776, 8192, "torch.float16"),
        (8192, 1776, 16384, "torch.float16"),
        # N=1712 (K-2183 admit, lives in its own frozenset, not this one)
        (4096, 1712, 8192, "torch.bfloat16"),
        # N=1840 (next un-audited residue-48 rung above 1776)
        (4096, 1840, 8192, "torch.bfloat16"),
        # Off-residue N (1792 is wave-aligned, not residue-48)
        (4096, 1792, 8192, "torch.bfloat16"),
    ]
    for cell in forbidden:
        assert cell not in _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10, (
            f"K-2214 frozenset must NOT contain firewall cell {cell}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
