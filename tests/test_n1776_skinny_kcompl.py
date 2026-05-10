"""Invariants for the skinny N=1776 K-COMPLEMENT alias-stack rung.

Per the minimalist src/tests split: the `_route_predicate.py` module
holds the *data* (frozenset literal); this file holds the *invariants* —
cardinality, residue class, dtype set, disjointness with prior off-by-48
rungs, the 3-cell admit-gate-clearing membership expansion, the
per-cell admit-gate clearance assertions (each admitted cell measured
≥1.5× uplift on the paired n=30 HIP-graph hot-cache 3-engine bench),
and adjacency firewalls (the cells immediately neighbouring the
admitted set on the (M,K,dt) grid at N=1776 — including the 5 verified
cells that did NOT clear the per-cell ≥1.5× gate — must remain
un-routed).

The frozenset itself is a single-N slice (N=1776 only) restricted to
the 3 cells that cleared the per-cell ≥1.5× admit gate on the paired
n=30 HIP-graph hot-cache 3-engine bench:

  - (M=4096, N=1776, K=8192,  bf16)  — 1.585×
  - (M=4096, N=1776, K=8192,  fp16)  — 1.553×
  - (M=8192, N=1776, K=16384, bf16)  — 1.500×

Geomean over the admitted 3 cells: ≈1.546×, clearing the ≥1.5×
admit-gate cohort threshold.

Next-rung extension of the off-by-48 contiguous ladder.
"""
from __future__ import annotations

import math

import pytest

from tritonblas._route_predicate import (
    _SKINNY_N1776_KCOMPL_ALIASSTACK,
)

EXPECTED_N = 1776
EXPECTED_CARDINALITY = 3
ADMIT_GATE = 1.5

# Per-cell measured uplift TB-B/TB-A from the paired n=30 HIP-graph
# hot-cache 3-engine bench on MI300X gfx942.  Only cells with uplift
# ≥ ADMIT_GATE are admitted to the frozenset.
PER_CELL_UPLIFT = {
    (4096, 1776, 8192,  "torch.bfloat16"): 1.585,
    (4096, 1776, 8192,  "torch.float16"):  1.553,
    (8192, 1776, 16384, "torch.bfloat16"): 1.500,
}


def test_cardinality_is_3():
    assert len(_SKINNY_N1776_KCOMPL_ALIASSTACK) == EXPECTED_CARDINALITY


def test_n_is_residue_48_mod_64():
    # Off-by-48 wave-misaligned membership invariant
    for (M, N, K, dt) in _SKINNY_N1776_KCOMPL_ALIASSTACK:
        assert N == EXPECTED_N
        assert N % 64 == 48
        # mod 128 == 112 sub-class admits identically per the
        # residue-48-family rule (only mod 64 == 48 binds).
        assert N % 128 == 112


def test_admitted_set_matches_gate_clearing_cells():
    # The admitted frozenset is exactly the set of cells that cleared
    # the per-cell ≥1.5× uplift admit gate on the verification bench.
    expected = set(PER_CELL_UPLIFT.keys())
    assert _SKINNY_N1776_KCOMPL_ALIASSTACK == expected


def test_each_admitted_cell_clears_per_cell_admit_gate():
    # Per-cell guarantee: every admitted cell measured ≥1.5× uplift.
    for cell in _SKINNY_N1776_KCOMPL_ALIASSTACK:
        assert PER_CELL_UPLIFT[cell] >= ADMIT_GATE, (
            f"admitted cell {cell} has uplift "
            f"{PER_CELL_UPLIFT[cell]} < gate {ADMIT_GATE}"
        )


def test_admitted_set_geomean_clears_admit_gate():
    # Cohort guarantee on the admitted subset: geomean uplift over the
    # admitted cells must clear the ≥1.5× admit gate.
    uplifts = [PER_CELL_UPLIFT[cell] for cell in _SKINNY_N1776_KCOMPL_ALIASSTACK]
    log_mean = sum(math.log(u) for u in uplifts) / len(uplifts)
    geomean = math.exp(log_mean)
    assert geomean >= ADMIT_GATE, (
        f"admitted-cohort geomean {geomean:.4f} < gate {ADMIT_GATE}"
    )


def test_dtype_set_is_bf16_and_fp16():
    # Both bf16 and fp16 appear in the admitted set (bf16 in 2 cells,
    # fp16 in 1).  No third dtype creeps in.
    dts = {dt for (_, _, _, dt) in _SKINNY_N1776_KCOMPL_ALIASSTACK}
    assert dts == {"torch.bfloat16", "torch.float16"}


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


def test_held_below_gate_cells_not_admitted():
    # The 5 cells in the original 8-cell verification cohort whose
    # measured per-cell uplift fell below the ≥1.5× admit gate must NOT
    # be in the frozenset.  These cells stay on the upstream
    # `persistent_matmul` route until a stronger residue-48 signal is
    # measured for them.
    held_below_gate = [
        # M=4096, K=16384 row — uplift 1.167× (bf16) / 1.194× (fp16)
        (4096, 1776, 16384, "torch.bfloat16"),
        (4096, 1776, 16384, "torch.float16"),
        # M=8192, K=8192 row  — uplift 1.240× (bf16) / 1.036× (fp16)
        (8192, 1776, 8192,  "torch.bfloat16"),
        (8192, 1776, 8192,  "torch.float16"),
        # M=8192, K=16384, fp16 — uplift 1.458× (bf16 cleared at 1.500×)
        (8192, 1776, 16384, "torch.float16"),
    ]
    for cell in held_below_gate:
        assert cell not in _SKINNY_N1776_KCOMPL_ALIASSTACK, (
            f"N=1776 frozenset must NOT contain held-below-gate cell {cell}"
        )


def test_adjacency_firewalls_n1776_neighbors():
    # The cells immediately neighbouring the admitted N=1776 set on the
    # residue-48 lattice (N=1712 prior admit; N=1840 next un-audited rung
    # above) and the M=2048 row / K=4096 column at N=1776 must remain
    # OUTSIDE the new frozenset — preserve the per-N audit handles per
    # the stacked-predicate convention.
    forbidden = [
        # M=2048 row at N=1776 (excluded from the verification cohort)
        (2048, 1776, 8192, "torch.bfloat16"),
        (2048, 1776, 8192, "torch.float16"),
        (2048, 1776, 16384, "torch.bfloat16"),
        (2048, 1776, 16384, "torch.float16"),
        # K=4096 column at N=1776 (excluded from the verification cohort)
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
