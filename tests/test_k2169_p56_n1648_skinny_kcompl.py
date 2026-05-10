"""K-2169 (S-002) — invariants for P56 N=1648 K-COMPLEMENT alias-stack.

Per minimalist src/tests split (R-1532 / R-1720 / R-1775): the
`_route_predicate.py` module holds the *data* (frozenset literal); this
file holds the *invariants* — cardinality, residue class, dtype set,
disjointness with prior off-by-48 rungs, and the 8-cell membership
expansion.

The frozenset itself is a single-N slice (N=1648 only) over the
task-specified verification cohort M in {4096, 8192} x K in {8192, 16384}
x {bf16, fp16}, i.e. 2 * 2 * 2 = 8 cells (smaller than the canonical
18-cell envelope used by K-2127 / K-2158 — extension to the full grid
is deferred to a follow-up rung-stitch task).

13th-rung extension of the off-by-48 contiguous ladder branched off
fix/K-2158@beab03e (which itself branches off fix/K-2127@9c4f984 →
fix/K-1922@0024a71 — matches K-2091 / K-2092 / K-2107 / K-2127 / K-2158
lineage).
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8,
)

EXPECTED_M = (4096, 8192)
EXPECTED_K = (8192, 16384)
EXPECTED_DT = ("torch.bfloat16", "torch.float16")
EXPECTED_N = 1648
EXPECTED_CARDINALITY = 8


def test_cardinality_is_8():
    assert len(_K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8) == EXPECTED_CARDINALITY


def test_n_is_residue_48_mod_64():
    # Off-by-48 wave-misaligned membership invariant
    for (M, N, K, dt) in _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8:
        assert N == EXPECTED_N
        assert N % 64 == 48
        # mod 128 == 112 is the K-2055 / K-2071 / K-2127 sub-class — admit
        # identically per K-2150 R-K2150.MOD-64-IS-BINDING.
        assert N % 128 == 112


def test_full_grid_membership():
    expected = {
        (M, EXPECTED_N, K, dt)
        for M in EXPECTED_M
        for K in EXPECTED_K
        for dt in EXPECTED_DT
    }
    assert _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8 == expected


def test_dtype_set_is_bf16_fp16_only():
    dts = {dt for (_, _, _, dt) in _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8}
    assert dts == set(EXPECTED_DT)


def test_disjoint_with_prior_off_by_48_rungs():
    # K-2169 is a single-N slice at N=1648; every prior off-by-48 rung is at
    # a strictly different N (544..1584) so disjointness must hold trivially.
    from tritonblas._route_predicate import (
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18,
        _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18,
    )
    assert _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8.isdisjoint(
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18
    )
    assert _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8.isdisjoint(
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18
    )
    assert _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8.isdisjoint(
        _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18
    )


def test_n1648_extends_off_by_48_ladder_above_n1584():
    # Sanity: N=1648 is the next admitted residue-48 step above K-2158 P55
    # N=1584 in this rung sequence (+64 step on the residue-48 lattice).
    assert (EXPECTED_N - 1584) % 64 == 0
    assert (EXPECTED_N - 1584) == 64
    assert EXPECTED_N > 1584


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
