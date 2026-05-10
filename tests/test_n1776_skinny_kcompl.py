"""Invariants for the skinny N=1776 K-COMPLEMENT alias-stack rung.

Per the minimalist src/tests split: the `_route_predicate.py` module
holds the *data* (frozenset literal); this file holds the *invariants* —
cardinality, residue class, dtype set, disjointness with prior off-by-48
rungs, the 8-cell membership expansion, and adjacency firewalls (the
cells immediately neighbouring N=1776 in the prior 14-cell admit set
must remain un-routed).

The frozenset itself is a single-N slice (N=1776 only) over the
verification cohort M in {4096, 8192} x K in {8192, 16384} x {bf16,
fp16}, i.e. 2 * 2 * 2 = 8 cells (smaller than the canonical 18-cell
envelope used by the N=1456 / N=1584 rungs — extension to the full grid
is deferred to a follow-up rung-stitch task).

Next-rung extension of the off-by-48 contiguous ladder.
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _SKINNY_N1776_KCOMPL_ALIASSTACK,
)

EXPECTED_M = (4096, 8192)
EXPECTED_K = (8192, 16384)
EXPECTED_DT = ("torch.bfloat16", "torch.float16")
EXPECTED_N = 1776
EXPECTED_CARDINALITY = 8


def test_cardinality_is_8():
    assert len(_SKINNY_N1776_KCOMPL_ALIASSTACK) == EXPECTED_CARDINALITY


def test_n_is_residue_48_mod_64():
    # Off-by-48 wave-misaligned membership invariant
    for (M, N, K, dt) in _SKINNY_N1776_KCOMPL_ALIASSTACK:
        assert N == EXPECTED_N
        assert N % 64 == 48
        # mod 128 == 112 sub-class admits identically per the
        # residue-48-family rule (only mod 64 == 48 binds).
        assert N % 128 == 112


def test_full_grid_membership():
    expected = {
        (M, EXPECTED_N, K, dt)
        for M in EXPECTED_M
        for K in EXPECTED_K
        for dt in EXPECTED_DT
    }
    assert _SKINNY_N1776_KCOMPL_ALIASSTACK == expected


def test_dtype_set_is_bf16_fp16_only():
    dts = {dt for (_, _, _, dt) in _SKINNY_N1776_KCOMPL_ALIASSTACK}
    assert dts == set(EXPECTED_DT)


def test_disjoint_with_prior_off_by_48_rungs():
    # This frozenset is a single-N slice at N=1776; every prior off-by-48
    # rung is at a strictly different N (544..1712) so disjointness must
    # hold trivially.
    from tritonblas._route_predicate import (
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18,
        _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18,
        _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8,
        _K2183_P57_SKINNY_N1712_KCOMPL_ALIASSTACK_8,
    )
    assert _SKINNY_N1776_KCOMPL_ALIASSTACK.isdisjoint(
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18
    )
    assert _SKINNY_N1776_KCOMPL_ALIASSTACK.isdisjoint(
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18
    )
    assert _SKINNY_N1776_KCOMPL_ALIASSTACK.isdisjoint(
        _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18
    )
    assert _SKINNY_N1776_KCOMPL_ALIASSTACK.isdisjoint(
        _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8
    )
    assert _SKINNY_N1776_KCOMPL_ALIASSTACK.isdisjoint(
        _K2183_P57_SKINNY_N1712_KCOMPL_ALIASSTACK_8
    )


def test_n1776_extends_off_by_48_ladder_above_n1712():
    # Sanity: N=1776 is the next admitted residue-48 step above N=1712 in
    # this rung sequence (+64 step on the residue-48 lattice).
    assert (EXPECTED_N - 1712) % 64 == 0
    assert (EXPECTED_N - 1712) == 64
    assert EXPECTED_N > 1712


def test_adjacency_firewalls_n1776_neighbors():
    # The cells immediately neighbouring N=1776 on the residue-48 lattice
    # (N=1712 is the prior admit immediately below; N=1840 is the next
    # un-audited rung above) and the M=2048 row at N=1776 must remain
    # OUTSIDE the new frozenset — preserve the per-N audit handles per the
    # stacked-predicate convention.
    forbidden = [
        # M=2048 row at N=1776 (excluded from the 8-cell verification cohort)
        (2048, 1776, 8192, "torch.bfloat16"),
        (2048, 1776, 8192, "torch.float16"),
        (2048, 1776, 16384, "torch.bfloat16"),
        (2048, 1776, 16384, "torch.float16"),
        # K=4096 column at N=1776 (excluded from the 8-cell verification cohort)
        (4096, 1776, 4096, "torch.bfloat16"),
        (8192, 1776, 4096, "torch.float16"),
        # N=1712 (prior admit, lives in its own frozenset, not this one)
        (4096, 1712, 8192, "torch.bfloat16"),
        # N=1840 (next un-audited residue-48 rung above 1776)
        (4096, 1840, 8192, "torch.bfloat16"),
    ]
    for cell in forbidden:
        assert cell not in _SKINNY_N1776_KCOMPL_ALIASSTACK, (
            f"N=1776 frozenset must NOT contain firewall cell {cell}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
