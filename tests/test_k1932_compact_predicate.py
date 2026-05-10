"""K-1932 (S-002) — closed-form compact predicate equivalence tests.

Locks the K-1908/K-1918 closed-form predicate
``_k1932_p32_p39_compact_predicate`` (a 3-axis Cartesian-product membership
test plus a 6-cell carve-out exclusion) against the literal two-frozenset
union it claims to be equivalent to:

    _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120  (120 cells, P32-P38)
        ∪
    _P39_SKINNY_N448_KCOMPL_VERIFIED_WIN_18    ( 18 cells, P39 N=448)

    = 138-cell P32-P39 admit set.

The closed-form predicate is RETAINED in source as a structural identity
invariant (drift firewall) only — it is NOT on the dispatcher hot path
because the live MI300X paired n=30 hot-cache micro-bench falsified the
K-1908/K-1918 conjectured dispatcher-overhead reduction:

    predicate_only_micro:  A=190.8 ns/call vs B=248.2 ns/call
                           ⇒ B is ~30% slower (0.769× speedup, NOT a win)

(See the K-1932 RCA comment block in ``include/tritonblas/matmul.py`` and
the artifacts under ``output/``.)  These tests therefore exist to:

  1. Pin cell-by-cell BIT-EQUIVALENCE between the closed form and the
     literal frozenset union over the FULL 144-cell M×N×K×dt containing
     rectangle (M ∈ {2048, 4096, 8192}, N ∈ {96, 160, 224, 288, 320, 352,
     384, 448}, K ∈ {4096, 8192, 16384}, dt ∈ {bf16, fp16};
     3×8×3×2 = 144) — the same domain the import-time guard
     ``_k1932_predicate_equiv_guard`` walks.  CI fires here if either side
     ever drifts.
  2. Pin the admit cardinality at exactly 138 on BOTH sides.
  3. Lock a representative slice of out-of-distribution (OOD) NEGATIVE
     cells (cells the predicate must NOT admit) so a future regression
     that silently widens the closed form fires in CI rather than only at
     module import on a dev box.

Per the minimalist split convention (R-1532 / R-1720 / R-1775): the
literal cell roster lives in ``_route_predicate.py`` (single source of
truth); this file holds the structural invariants.
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120,
    _P39_SKINNY_N448_KCOMPL_VERIFIED_WIN_18,
    _K1932_P32_P39_COMPACT_M,
    _K1932_P32_P39_COMPACT_N,
    _K1932_P32_P39_COMPACT_K,
    _K1932_P32_P39_COMPACT_CARVE6,
    _k1932_p32_p39_compact_predicate,
)


# Containing rectangle the import-time guard walks.  Kept literal here (not
# re-derived from the predicate's frozensets) so a silent expansion of the
# predicate's positive axes cannot tautologically hide a divergence.
_M_AXIS = (2048, 4096, 8192)
_N_AXIS = (96, 160, 224, 288, 320, 352, 384, 448)
_K_AXIS = (4096, 8192, 16384)
_DT_AXIS = ("torch.bfloat16", "torch.float16")

_UNION_138 = _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120 | _P39_SKINNY_N448_KCOMPL_VERIFIED_WIN_18

_FULL_RECTANGLE = tuple(
    (M, N, K, dt)
    for M in _M_AXIS
    for N in _N_AXIS
    for K in _K_AXIS
    for dt in _DT_AXIS
)


def test_full_rectangle_size():
    """The containing rectangle must be exactly 144 cells (3*8*3*2)."""
    assert len(_FULL_RECTANGLE) == 144
    # And contain no duplicates.
    assert len(set(_FULL_RECTANGLE)) == 144


def test_legacy_union_cardinality_138():
    """Pin the legacy two-frozenset union at exactly 138 cells (120 + 18)."""
    assert len(_K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120) == 120
    assert len(_P39_SKINNY_N448_KCOMPL_VERIFIED_WIN_18) == 18
    # P32-P38 (N ∈ {96,...,384}) is disjoint from P39 (N=448) by sibling-N
    # firewall, so the union is exactly 120 + 18 = 138.
    assert _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120.isdisjoint(
        _P39_SKINNY_N448_KCOMPL_VERIFIED_WIN_18
    )
    assert len(_UNION_138) == 138


def test_compact_predicate_cardinality_138():
    """Pin the closed-form admit count at exactly 138 over the 144-cell rectangle."""
    n_admit = sum(
        1 for cell in _FULL_RECTANGLE
        if _k1932_p32_p39_compact_predicate(*cell)
    )
    assert n_admit == 138


def test_compact_predicate_axes_match_rectangle():
    """The predicate's positive axis sets must exactly match the rectangle axes
    the equivalence claim is made over.  Catches a silent axis expansion that
    would tautologically pass the bit-equivalence sweep."""
    assert _K1932_P32_P39_COMPACT_M == frozenset(_M_AXIS)
    assert _K1932_P32_P39_COMPACT_N == frozenset(_N_AXIS)
    assert _K1932_P32_P39_COMPACT_K == frozenset(_K_AXIS)


def test_carveout_exactly_six_cells_all_inside_rectangle():
    """The 6-cell negative carveout is exactly 6 cells, all of which lie inside
    the positive Cartesian product (otherwise the carveout is dead code)."""
    assert len(_K1932_P32_P39_COMPACT_CARVE6) == 6
    for cell in _K1932_P32_P39_COMPACT_CARVE6:
        M, N, K, dt = cell
        assert M in _K1932_P32_P39_COMPACT_M
        assert N in _K1932_P32_P39_COMPACT_N
        assert K in _K1932_P32_P39_COMPACT_K
        assert dt in _DT_AXIS
        # And every carveout cell must NOT be in the union (it's an
        # exclusion, so the union must already exclude it).
        assert cell not in _UNION_138


def test_bit_equivalence_144_cells():
    """Cell-by-cell bit-equivalence: for every cell in the 144-cell containing
    rectangle, the closed-form predicate must return the SAME verdict as
    membership in the literal frozenset union.  This is the load-bearing
    contract — any future drift between the K-1908/K-1918 closed form and
    the K-1881 + K-1887 alias-stack literals fires here.
    """
    mismatches = []
    for cell in _FULL_RECTANGLE:
        legacy = cell in _UNION_138
        compact = _k1932_p32_p39_compact_predicate(*cell)
        if legacy != compact:
            mismatches.append((cell, legacy, compact))
    assert mismatches == [], (
        f"K-1932 compact predicate diverges from legacy frozenset union on "
        f"{len(mismatches)} of 144 cells (first 5): {mismatches[:5]}"
    )


# ---------------------------------------------------------------------------
# Out-of-distribution (OOD) NEGATIVE control cells.  Every cell here MUST
# return False from the closed-form predicate (and MUST NOT be in the
# legacy union).  These cells are deliberately drawn from neighborhoods
# that would catch a silent widening of the predicate:
#
#   * Wrong-N axis (N=128/192/256/512/768/1024 — not in {96,...,448}).
#   * Wrong-M axis (M=1024/16384 — not in {2048,4096,8192}).
#   * Wrong-K axis (K=2048/32768 — not in {4096,8192,16384}).
#   * Wrong dtype (torch.float32 — not in {bf16, fp16}).
#   * The 6 carveout cells themselves (must remain excluded).
# ---------------------------------------------------------------------------
_OOD_NEGATIVE_CELLS = [
    # Wrong N (inside the M/K/dt rectangle, but N ∉ {96,...,448})
    (2048, 128, 4096, "torch.bfloat16"),
    (4096, 192, 8192, "torch.float16"),
    (8192, 256, 16384, "torch.bfloat16"),
    (4096, 512, 8192, "torch.bfloat16"),
    (2048, 768, 4096, "torch.float16"),
    (8192, 1024, 16384, "torch.bfloat16"),
    # Wrong M (M ∉ {2048, 4096, 8192})
    (1024, 96, 4096, "torch.bfloat16"),
    (1024, 448, 8192, "torch.float16"),
    (16384, 224, 16384, "torch.bfloat16"),
    # Wrong K (K ∉ {4096, 8192, 16384})
    (2048, 96, 2048, "torch.bfloat16"),
    (4096, 288, 32768, "torch.float16"),
    (8192, 448, 2048, "torch.bfloat16"),
    # Wrong dtype (dt ∉ {bf16, fp16})
    (2048, 96, 4096, "torch.float32"),
    (4096, 384, 8192, "torch.float32"),
    (8192, 448, 16384, "torch.float32"),
    # Carveouts — must remain excluded by the closed form
    (2048, 320, 4096, "torch.bfloat16"),
    (2048, 352, 4096, "torch.bfloat16"),
    (2048, 384, 8192, "torch.bfloat16"),
    (2048, 384, 8192, "torch.float16"),
    (4096, 384, 8192, "torch.bfloat16"),
    (4096, 384, 8192, "torch.float16"),
]


@pytest.mark.parametrize("cell", _OOD_NEGATIVE_CELLS)
def test_ood_cells_not_admitted(cell):
    """Each OOD/carveout cell must be rejected by BOTH the closed form AND
    the legacy frozenset union (mutually-reinforcing negative case)."""
    M, N, K, dt = cell
    assert _k1932_p32_p39_compact_predicate(M, N, K, dt) is False, (
        f"closed-form predicate unexpectedly admits OOD cell {cell}"
    )
    assert cell not in _UNION_138, (
        f"legacy frozenset union unexpectedly admits OOD cell {cell}"
    )


def test_predicate_returns_bool():
    """The closed-form predicate must return a strict bool (not truthy int /
    None / etc.) so its verdict can be combined safely with the dispatcher's
    short-circuit `if ... return True` chain."""
    # One known-positive and one known-negative.
    assert _k1932_p32_p39_compact_predicate(2048, 96, 4096, "torch.bfloat16") is True
    assert _k1932_p32_p39_compact_predicate(2048, 320, 4096, "torch.bfloat16") is False
    assert _k1932_p32_p39_compact_predicate(1024, 128, 2048, "torch.float32") is False
