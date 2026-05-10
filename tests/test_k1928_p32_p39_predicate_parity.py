"""K-1928 (S-002) — bit-equivalence parity test for the closed-form predicate
that collapses the P32–P39 alias-stack cascade.

This test is the FORMAL ORACLE for the K-1908/K-1918 predicate substitution.
Asserts that ``_K1928_p32_p39_skinny_kcompl_predicate`` returns the same
routing decision as the prior two-frozenset cascade

    (M,N,K,dt) in _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120
 or (M,N,K,dt) in _P39_SKINNY_N448_KCOMPL_VERIFIED_WIN_18

over the full 144-cell ``MSET × NSET × KSET × dtype`` deployment grid AND
over a broader 540-cell dispatcher universe (negative class).  Required:
0 false positives, 0 false negatives across all 540 cells.

If this test ever fails, EITHER the predicate has drifted OR a P-slot has
silently grown the underlying frozensets — both must be re-litigated before
landing.

Provenance:
- K-1908 derived the depth-2 closed-form (S1) bit-equivalent to the
  K-1881 P32–P38 frozenset (120 cells, 6 carve-outs).
- K-1918 ran an exhaustive 177,417-predicate depth-3 sweep and proved
  the literal N-set form is the unique 95/90 axis-aligned conjunctive
  predicate over the full 138-cell P32–P39 union.
- K-1928 (this test) validates the closed form against the live oracle
  on c42/MI300X and gates dispatch substitution.
"""
from itertools import product

import pytest

from tritonblas._route_predicate import (
    _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120,
    _P39_SKINNY_N448_KCOMPL_VERIFIED_WIN_18,
    _K1928_P32_P39_NSET,
    _K1928_P32_P39_KSET,
    _K1928_P32_P39_MSET,
    _K1928_P32_P39_CARVE6,
    _K1928_p32_p39_skinny_kcompl_predicate,
)


# Live oracle = literal union of the two productionised frozensets.
ORACLE_138 = (
    _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120
    | _P39_SKINNY_N448_KCOMPL_VERIFIED_WIN_18
)
DTYPES = ("torch.bfloat16", "torch.float16")


def test_oracle_cardinality_138():
    """Sanity: the live oracle is exactly 138 cells (120 + 18)."""
    assert len(_K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120) == 120
    assert len(_P39_SKINNY_N448_KCOMPL_VERIFIED_WIN_18) == 18
    assert len(ORACLE_138) == 138, (
        f"Expected 138-cell oracle, got {len(ORACLE_138)} (cascade overlap?)"
    )


def test_predicate_set_constants_are_correct():
    """The N/K/M sets cover P32–P39 N rungs and the canonical K/M grids."""
    assert _K1928_P32_P39_NSET == frozenset(
        {96, 160, 224, 288, 320, 352, 384, 448}
    )
    assert _K1928_P32_P39_KSET == frozenset({4096, 8192, 16384})
    assert _K1928_P32_P39_MSET == frozenset({2048, 4096, 8192})
    assert len(_K1928_P32_P39_CARVE6) == 6


def test_predicate_bit_equivalent_to_oracle_full_grid():
    """Predicate ≡ ORACLE_138 over the 144-cell MSET×NSET×KSET×dtype grid.

    This is the canonical bit-equivalence assertion: every predicate-positive
    cell must be in the oracle, and every oracle-positive cell must be
    predicate-positive."""
    grid = list(product(
        _K1928_P32_P39_MSET, _K1928_P32_P39_NSET,
        _K1928_P32_P39_KSET, DTYPES,
    ))
    assert len(grid) == 8 * 3 * 3 * 2 == 144

    pred_positive = {(M, N, K, d) for (M, N, K, d) in grid
                     if _K1928_p32_p39_skinny_kcompl_predicate(M, N, K, d)}
    oracle_positive = {c for c in grid if c in ORACLE_138}

    missing = oracle_positive - pred_positive    # FN
    extra   = pred_positive - oracle_positive    # FP
    assert not missing, f"Predicate misses {len(missing)} oracle cells: {sorted(missing)[:5]}"
    assert not extra,   f"Predicate over-fires on {len(extra)} cells: {sorted(extra)[:5]}"
    assert len(pred_positive) == 138


def test_predicate_negative_on_broader_universe():
    """Predicate must not fire on the broader dispatcher universe (negatives).

    Universe ⊇ MSET × N_GRID × K_GRID × dtype where N_GRID covers the
    full 32–512 step-32 axis plus N=768/N=1536 (P30's domain).  All cells
    NOT in ORACLE_138 must be predicate-negative."""
    n_grid = tuple(range(32, 513, 32)) + (768, 1536)
    k_grid = (2048, 4096, 8192, 16384, 32768)
    universe = set()
    for M, N, K, d in product(_K1928_P32_P39_MSET, n_grid, k_grid, DTYPES):
        universe.add((M, N, K, d))

    fp = []
    fn = []
    for cell in universe:
        oracle_says = cell in ORACLE_138
        pred_says   = _K1928_p32_p39_skinny_kcompl_predicate(*cell)
        if pred_says and not oracle_says:
            fp.append(cell)
        elif oracle_says and not pred_says:
            fn.append(cell)
    assert not fp, f"{len(fp)} false positives on broader universe: {fp[:5]}"
    assert not fn, f"{len(fn)} false negatives on broader universe: {fn[:5]}"


def test_predicate_carve6_cells_route_negative():
    """The 6 CARVE6 cells must NOT route via this predicate (P30/upstream owns)."""
    for cell in _K1928_P32_P39_CARVE6:
        assert not _K1928_p32_p39_skinny_kcompl_predicate(*cell), (
            f"CARVE6 cell {cell} unexpectedly routed via K-1928 predicate"
        )
        # And critically, the live oracle must agree:
        assert cell not in ORACLE_138, (
            f"CARVE6 cell {cell} is in live oracle — provenance drift"
        )


def test_dispatcher_uses_predicate_not_cascade():
    """Verify the dispatcher path (`_k971_route_to_hbl`) uses the predicate."""
    import importlib
    _m = importlib.import_module("tritonblas.matmul")
    co_names = _m._k971_route_to_hbl.__code__.co_names
    assert "_K1928_p32_p39_skinny_kcompl_predicate" in co_names, (
        "_k971_route_to_hbl no longer references the K-1928 predicate"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
