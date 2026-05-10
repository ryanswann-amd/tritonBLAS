"""K-2139 (S-002) — invariants for P54 N=1520 K-COMPLEMENT alias-stack.

Per minimalist src/tests split (R-1532 / R-1720 / R-1775): the
`_route_predicate.py` module holds the *data* (frozenset literal); this
file holds the *invariants* — cardinality, residue class, dtype set,
disjointness with prior off-by-48 rungs, and the 18-cell membership
expansion.

The frozenset itself is a single-N slice (N=1520 only) over the canonical
M ∈ {2048, 4096, 8192} × K ∈ {4096, 8192, 16384} × {bf16, fp16} envelope,
i.e. 3 × 3 × 2 = 18 cells.

12th-rung extension of the off-by-48 contiguous ladder branched off
fix/K-1922@0024a71 (matches K-2091 / K-2092 / K-2107 / K-2127 lineage).
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K2139_P54_SKINNY_N1520_KCOMPL_ALIASSTACK_18,
)

EXPECTED_M = (2048, 4096, 8192)
EXPECTED_K = (4096, 8192, 16384)
EXPECTED_DT = ("torch.bfloat16", "torch.float16")
EXPECTED_N = 1520
EXPECTED_CARDINALITY = 18


def test_cardinality_is_18():
    assert len(_K2139_P54_SKINNY_N1520_KCOMPL_ALIASSTACK_18) == EXPECTED_CARDINALITY


def test_n_is_residue_48_mod_64():
    # Off-by-48 wave-misaligned membership invariant
    for (M, N, K, dt) in _K2139_P54_SKINNY_N1520_KCOMPL_ALIASSTACK_18:
        assert N == EXPECTED_N
        assert N % 64 == 48


def test_full_grid_membership():
    expected = {
        (M, EXPECTED_N, K, dt)
        for M in EXPECTED_M
        for K in EXPECTED_K
        for dt in EXPECTED_DT
    }
    assert _K2139_P54_SKINNY_N1520_KCOMPL_ALIASSTACK_18 == expected


def test_dtype_set_is_bf16_fp16_only():
    dts = {dt for (_, _, _, dt) in _K2139_P54_SKINNY_N1520_KCOMPL_ALIASSTACK_18}
    assert dts == set(EXPECTED_DT)


def test_disjoint_with_prior_off_by_48_rungs():
    # K-2139 is a single-N slice at N=1520; every prior off-by-48 rung is at
    # a strictly different N (816..1456) so disjointness must hold trivially.
    from tritonblas._route_predicate import (
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
    )
    assert _K2139_P54_SKINNY_N1520_KCOMPL_ALIASSTACK_18.isdisjoint(
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18
    )


def test_n1520_is_next_plus_64_above_n1456():
    # Sanity: N=1520 is the next contiguous +64 step beyond K-2127 P54 N=1456
    assert EXPECTED_N - 64 == 1456


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
