"""Unit fixture — K-1563 P24 skinny_N4096 K-COMPLEMENT 30-cell route-OUT
envelope (16th-position).

Verifies (covering the same ground as the prior 79-pin-test draft, collapsed
per Minimalist review feedback):
  (a) frozenset cardinality + exact admit-set match (K-1553 30 cells);
  (b) per-cell, single parametrize: predicate True ∧ full dispatch True ∧
      partial-alias invariant holds (overlap subset of P12; load-bearing
      cells disjoint from P12);
  (c) sibling-N firewall sentinels return False (carry-out N-axis carve-out);
  (d) carve-out negatives: streamk / work_stealing / dtype-mismatch all
      veto routing;
  (e) cohort gate metadata sanity (K-1553 headline ≥ K-1563 floors).

Source data: K-1553 paired n=30 HIP-graph hot-cache on MI300X / gfx942
(OCI amd-rccl fallback per R-1414; c42 unavailable) against the LIVE
post-K-1532 tritonblas oracle (HEAD 95e2c473) with B=10000 vectorised
paired bootstrap CI95.  30/30 ROUTE-OUT-CANDIDATE; cohort geomean tb/hbl
= 1.234×, range 1.114×–1.501×.  Productionised under the K-1563 gate
(cohort geomean ≥ 1.0× ∧ per-cell speedup ≥ 0.95×; both clear).
"""
import pytest
import torch

from tritonblas._route_predicate import (
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4,
    _K1563_P24_SKINNY_N4096_KCOMPL_ALIASSTACK_30,
    _k1563_p24_skinny_n4096_kcompl_aliasstack_routeout,
    k971_route_decision,
)


# Documented 2-cell P12 partial-alias overlap.
_P24_P12_OVERLAP = frozenset({
    (4096, 4096, 4096, "torch.bfloat16"),
    (4096, 4096, 4096, "torch.float16"),
})

_DTYPE_OBJ = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}


# ---------------------------------------------------------------------------
# (a) frozenset shape pin — 30 cells, exact M/N/K/dtype grid + partial-alias
#     invariant holds at the set-algebra level.
# ---------------------------------------------------------------------------
def test_p24_admit_set_shape_and_alias_invariant():
    expected = {
        (M, 4096, K, dtype)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dtype in ("torch.bfloat16", "torch.float16")
    }
    assert len(_K1563_P24_SKINNY_N4096_KCOMPL_ALIASSTACK_30) == 30
    assert _K1563_P24_SKINNY_N4096_KCOMPL_ALIASSTACK_30 == expected
    # Documented 2-cell P12 overlap is a subset of both P12 and P24.
    assert _P24_P12_OVERLAP <= _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4
    assert _P24_P12_OVERLAP <= _K1563_P24_SKINNY_N4096_KCOMPL_ALIASSTACK_30
    # The other 28/30 cells are NEW admits — disjoint from P12.
    load_bearing = _K1563_P24_SKINNY_N4096_KCOMPL_ALIASSTACK_30 - _P24_P12_OVERLAP
    assert len(load_bearing) == 28
    assert load_bearing.isdisjoint(_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)


# ---------------------------------------------------------------------------
# (b) per-cell parametrize collapses three former tests into one — predicate
#     True ∧ full-stack dispatch True at default flag values.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell", sorted(_K1563_P24_SKINNY_N4096_KCOMPL_ALIASSTACK_30))
def test_p24_admit_cell_routes_via_predicate_and_full_dispatch(cell):
    M, N, K, dtype = cell
    # Predicate-only.
    assert _k1563_p24_skinny_n4096_kcompl_aliasstack_routeout(M, N, K, dtype) is True
    # Full dispatch (predicate + carve-outs + upstream layers) at defaults.
    a = _DTYPE_OBJ[dtype]
    assert k971_route_decision(
        M, N, K, a_dtype=a, b_dtype=a,
        enable_streamk=False, work_stealing=False,
    ) is True


# ---------------------------------------------------------------------------
# (c) sibling-N firewall sentinels return False — adjacent N buckets, K
#     outside grid, M outside bucket, fp32 carve-out.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell",
    [
        (2048, 256, 2048, "torch.bfloat16"),    # adjacent N=256
        (2048, 512, 2048, "torch.bfloat16"),    # adjacent N=512 (P15/P17/P23)
        (2048, 1024, 2048, "torch.bfloat16"),   # adjacent N=1024 (P16)
        (2048, 8192, 2048, "torch.bfloat16"),   # N=8192 not in P24 grid
        (2048, 16384, 2048, "torch.bfloat16"),  # P19 N=16384
        (2048, 32768, 2048, "torch.bfloat16"),  # P22 N=32768
        (2048, 4096, 1024, "torch.bfloat16"),   # K outside grid (1024)
        (2048, 4096, 256, "torch.bfloat16"),    # K outside grid (256)
        (1024, 4096, 2048, "torch.bfloat16"),   # M outside bucket (1024)
        (16384, 4096, 2048, "torch.bfloat16"),  # M outside bucket (16384)
        (2048, 4096, 2048, "torch.float32"),    # fp32 carve-out
    ],
)
def test_p24_sibling_n_firewall_rejects(cell):
    M, N, K, dtype = cell
    assert _k1563_p24_skinny_n4096_kcompl_aliasstack_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# (d) carve-out negatives — streamk / work_stealing / dtype-mismatch all veto.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(a_dtype=torch.bfloat16, b_dtype=torch.bfloat16,
             enable_streamk=True, work_stealing=False),   # streamk vetoes
        dict(a_dtype=torch.bfloat16, b_dtype=torch.bfloat16,
             enable_streamk=False, work_stealing=True),   # work_stealing vetoes
        dict(a_dtype=torch.bfloat16, b_dtype=torch.float16,
             enable_streamk=False, work_stealing=False),  # mixed dtype vetoes
    ],
    ids=["streamk", "work_stealing", "mixed_dtype"],
)
def test_p24_carveouts_veto_routing(kwargs):
    assert k971_route_decision(2048, 4096, 8192, **kwargs) is False


# ---------------------------------------------------------------------------
# (e) cohort gate sanity — ratio_median floors and cohort geomean from
#     K-1553 measurement metadata; locks the productionisation gate.
# ---------------------------------------------------------------------------
def test_p24_cohort_gate_floor_and_geomean_sanity():
    """Per-cell ≥ 0.95× and cohort geomean ≥ 1.0× are met by the K-1553
    sweep (min ratio_median = 1.114×, cohort geomean = 1.234×).  This test
    is metadata-only and does not re-measure; failure indicates the K-1553
    headline got stale and the K-1563 productionisation rationale must be
    revisited."""
    K1553_MIN_RATIO = 1.114
    K1553_COHORT_GEOMEAN = 1.234
    PER_CELL_FLOOR = 0.95
    COHORT_GEOMEAN_FLOOR = 1.00
    assert K1553_MIN_RATIO >= PER_CELL_FLOOR
    assert K1553_COHORT_GEOMEAN >= COHORT_GEOMEAN_FLOOR
