"""K-2152 (S-002) — invariants for P56 N=1584 K-COMPLEMENT alias-stack.

Per minimalist src/tests split (R-1532 / R-1720 / R-1775): the
`_route_predicate.py` module holds the *data* (frozenset literal); this
file holds the *invariants* — cardinality, residue class, dtype set,
disjointness with prior off-by-48 rungs, and the 18-cell membership
expansion.

The frozenset itself is a single-N slice (N=1584 only) over the canonical
M ∈ {2048, 4096, 8192} × K ∈ {4096, 8192, 16384} × {bf16, fp16} envelope,
i.e. 3 × 3 × 2 = 18 cells.

12th-rung extension of the off-by-48 contiguous ladder branched off the
K-2136 (P55 N=1520) baseline — direct +64 step beyond N=1520 (lineage:
K-1978 P45 N=816 → K-1990/K-1994 N=880 → K-2010/K-2014/K-2024 P46 N=944
→ K-2031/K-2036/K-2041 P47 N=1008 → K-2043/K-2047/K-2052 P48 N=1072 →
K-2055/K-2060/K-2063 N=1136 → K-2071..K-2085 P50 N=1200 → K-2071/K-2085/
K-2091 P51 N=1264 → K-2097/K-2101/K-2106 P52 N=1328 → K-2107 P53 N=1392
→ K-2118/K-2124/K-2130 P54 N=1456 → K-2136 P55 N=1520 → K-2152 P56 N=1584).

N=1584 satisfies (N % 64 == 48) AND (N % 128 == 48) — the mod-128=48
sub-class shared with N=1456 / N=1328 / N=1200 (vs the mod-128=112
sub-class containing N=1520 / N=1392 / N=1264).  Crosses the K-1908
≥12-rung closed-form refactor actionability threshold.
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K2152_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18,
)

EXPECTED_M = (2048, 4096, 8192)
EXPECTED_K = (4096, 8192, 16384)
EXPECTED_DT = ("torch.bfloat16", "torch.float16")
EXPECTED_N = 1584
EXPECTED_CARDINALITY = 18


def test_cardinality_is_18():
    assert len(_K2152_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18) == EXPECTED_CARDINALITY


def test_n_is_residue_48_mod_64():
    # Off-by-48 wave-misaligned membership invariant
    for (M, N, K, dt) in _K2152_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18:
        assert N == EXPECTED_N
        assert N % 64 == 48
        # mod-128 = 48 sibling-class to N=1456 / N=1328 / N=1200 (NOT the
        # 112 class containing N=1520 / N=1392 / N=1264)
        assert N % 128 == 48


def test_n1584_is_24p75_fractional_waves():
    # 1584 / 64 == 24.75 — same 0.75 fractional-wave class as every prior
    # rung in the off-by-48 ladder (K-1978..K-2136).
    assert EXPECTED_N / 64 == 24.75


def test_full_grid_membership():
    expected = {
        (M, EXPECTED_N, K, dt)
        for M in EXPECTED_M
        for K in EXPECTED_K
        for dt in EXPECTED_DT
    }
    assert _K2152_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18 == expected


def test_dtype_set_is_bf16_fp16_only():
    dts = {dt for (_, _, _, dt) in _K2152_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18}
    assert dts == set(EXPECTED_DT)


def test_disjoint_with_prior_off_by_48_rungs():
    # K-2152 is a single-N slice at N=1584; every prior off-by-48 rung is at
    # a strictly different N (544/1520) so disjointness must hold trivially.
    from tritonblas._route_predicate import (
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
        _K2136_P55_SKINNY_N1520_KCOMPL_ALIASSTACK_18,
    )
    assert _K2152_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18.isdisjoint(
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18
    )
    assert _K2152_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18.isdisjoint(
        _K2136_P55_SKINNY_N1520_KCOMPL_ALIASSTACK_18
    )


def test_n1584_is_next_plus_64_above_n1520():
    # Sanity: N=1584 is the next contiguous +64 step beyond K-2136 P55 N=1520
    assert EXPECTED_N - 64 == 1520


def test_n1584_block_n_128_fractional_tile_count():
    # BLOCK_N=128 → 1584 / 128 = 12.375 fractional tiles per N-row — the
    # 0.375 sub-class shared with N=1456 (11.375), N=1328 (10.375), N=1200
    # (9.375); distinct from the 0.875 sub-class containing N=1520 /
    # N=1392 / N=1264.
    assert EXPECTED_N / 128 == 12.375


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
