"""K-2175 (S-002) — invariants for P57 N=1648 K-COMPLEMENT alias-stack.

Per minimalist src/tests split (R-1532 / R-1720 / R-1775): the
`_route_predicate.py` module holds the *data* (frozenset literal); this
file holds the *invariants* — cardinality, residue class, dtype set,
disjointness with prior off-by-48 rungs (incl. the K-1908 closed-form
admit set), and the 18-cell membership expansion.

The frozenset itself is a single-N slice (N=1648 only) over the canonical
M ∈ {2048, 4096, 8192} × K ∈ {4096, 8192, 16384} × {bf16, fp16} envelope,
i.e. 3 × 3 × 2 = 18 cells.

13th-rung extension of the off-by-48 contiguous ladder branched off the
K-2152 (P56 N=1584) baseline (which itself landed the K-1908 closed-form
refactor).  Direct +64 step beyond N=1584 (lineage:
K-1978 P45 N=816 → K-1990/K-1994 N=880 → K-2010/K-2014/K-2024 P46 N=944
→ K-2031/K-2036/K-2041 P47 N=1008 → K-2043/K-2047/K-2052 P48 N=1072 →
K-2055/K-2060/K-2063 N=1136 → K-2071..K-2085 P50 N=1200 → K-2071/K-2085/
K-2091 P51 N=1264 → K-2097/K-2101/K-2106 P52 N=1328 → K-2107 P53 N=1392
→ K-2118/K-2124/K-2130 P54 N=1456 → K-2136 P55 N=1520 → K-2152 P56 N=1584
→ K-2175 P57 N=1648).

N=1648 satisfies (N % 64 == 48) AND (N % 128 == 112) — the mod-128=112
sub-class shared with N=1520 / N=1392 / N=1264 (vs the mod-128=48
sub-class containing N=1584 / N=1456 / N=1328 / N=1200).  N=1648 is
NOT a member of the K-1908 closed-form admit set ({1520, 1584}); the
two routes are disjoint by construction.
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K2175_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18,
    _K1908_OFFBY48_ADMIT_N,
)

EXPECTED_M = (2048, 4096, 8192)
EXPECTED_K = (4096, 8192, 16384)
EXPECTED_DT = ("torch.bfloat16", "torch.float16")
EXPECTED_N = 1648
EXPECTED_CARDINALITY = 18


def test_cardinality_is_18():
    assert len(_K2175_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18) == EXPECTED_CARDINALITY


def test_n_is_residue_48_mod_64():
    # Off-by-48 wave-misaligned membership invariant
    for (M, N, K, dt) in _K2175_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18:
        assert N == EXPECTED_N
        assert N % 64 == 48
        # mod-128 = 112 sibling-class to N=1520 / N=1392 / N=1264 (NOT the
        # 48 class containing N=1584 / N=1456 / N=1328 / N=1200)
        assert N % 128 == 112


def test_n1648_is_25p75_fractional_waves():
    # 1648 / 64 == 25.75 — same 0.75 fractional-wave class as every prior
    # rung in the off-by-48 ladder (K-1978..K-2152).
    assert EXPECTED_N / 64 == 25.75


def test_full_grid_membership():
    expected = {
        (M, EXPECTED_N, K, dt)
        for M in EXPECTED_M
        for K in EXPECTED_K
        for dt in EXPECTED_DT
    }
    assert _K2175_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18 == expected


def test_dtype_set_is_bf16_fp16_only():
    dts = {dt for (_, _, _, dt) in _K2175_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18}
    assert dts == set(EXPECTED_DT)


def test_disjoint_with_k1908_closed_form_admit_set():
    # K-2175 N=1648 must NOT collide with the K-1908 closed-form admit set
    # ({1520, 1584}); otherwise the dispatcher would double-route.
    assert EXPECTED_N not in _K1908_OFFBY48_ADMIT_N


def test_disjoint_with_prior_off_by_48_rungs():
    # K-2175 is a single-N slice at N=1648; every prior off-by-48 rung is at
    # a strictly different N (544 / 1520 / 1584) so disjointness must hold.
    from tritonblas._route_predicate import (
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
        _K2136_P55_SKINNY_N1520_KCOMPL_ALIASSTACK_18,
        _K2152_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18,
    )
    assert _K2175_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18.isdisjoint(
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18
    )
    assert _K2175_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18.isdisjoint(
        _K2136_P55_SKINNY_N1520_KCOMPL_ALIASSTACK_18
    )
    assert _K2175_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18.isdisjoint(
        _K2152_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18
    )


def test_n1648_is_next_plus_64_above_n1584():
    # Sanity: N=1648 is the next contiguous +64 step beyond K-2152 P56 N=1584
    assert EXPECTED_N - 64 == 1584


def test_n1648_block_n_128_fractional_tile_count():
    # BLOCK_N=128 → 1648 / 128 = 12.875 fractional tiles per N-row — the
    # 0.875 sub-class shared with N=1520 (11.875), N=1392 (10.875), N=1264
    # (9.875); distinct from the 0.375 sub-class containing N=1584 /
    # N=1456 / N=1328 / N=1200.
    assert EXPECTED_N / 128 == 12.875


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
