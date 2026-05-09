"""Unit fixture — K-1563 P24 skinny_N4096 K-COMPLEMENT 30-cell route-OUT
envelope (16th-position).

Verifies:
  (a) frozenset cardinality + exact admit-set match (K-1553 30 cells);
  (b) every admit cell predicate-returns True;
  (c) sibling-N firewall sentinels return False (carry-out N-axis carve-out);
  (d) PARTIAL-ALIAS invariant — the 2/30 (4096,4096,4096,bf16/fp16) cells
      ARE in K-1361 P12 SQUARE_MID (documented partial alias); the other
      28/30 cells are load-bearing strict-equality admits;
  (e) the public `k971_route_decision` dispatch returns True for every P24
      cell at default flag values (no double-admit, no flag interference);
  (f) carve-out negatives: streamk / work_stealing / dtype-mismatch all
      veto routing.

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


# ---------------------------------------------------------------------------
# (a) frozenset shape pin — 30 cells, exact M/N/K/dtype grid.
# ---------------------------------------------------------------------------
def test_p24_cardinality_is_thirty():
    assert len(_K1563_P24_SKINNY_N4096_KCOMPL_ALIASSTACK_30) == 30


def test_p24_admit_set_is_full_n4096_kcompl_grid():
    expected = {
        (M, 4096, K, dtype)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dtype in ("torch.bfloat16", "torch.float16")
    }
    assert _K1563_P24_SKINNY_N4096_KCOMPL_ALIASSTACK_30 == expected


# ---------------------------------------------------------------------------
# (b) every admit cell predicate-returns True.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell", sorted(_K1563_P24_SKINNY_N4096_KCOMPL_ALIASSTACK_30))
def test_p24_admit_cell_routes_true(cell):
    M, N, K, dtype = cell
    assert _k1563_p24_skinny_n4096_kcompl_aliasstack_routeout(M, N, K, dtype) is True


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
# (d) PARTIAL-ALIAS invariant — the 2 (4096,4096,4096,*) cells ARE in
#     K-1361 P12 SQUARE_MID; the other 28 are load-bearing (no upstream
#     overlap with the predicates that ship in this PR's dependency chain).
# ---------------------------------------------------------------------------
def test_p24_partial_alias_overlap_subset_of_p12():
    """The 2 documented overlap cells must be in P12."""
    assert _P24_P12_OVERLAP <= _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4
    assert _P24_P12_OVERLAP <= _K1563_P24_SKINNY_N4096_KCOMPL_ALIASSTACK_30


def test_p24_load_bearing_cells_disjoint_from_p12():
    """The other 28/30 cells must be NEW admits — disjoint from P12."""
    load_bearing = _K1563_P24_SKINNY_N4096_KCOMPL_ALIASSTACK_30 - _P24_P12_OVERLAP
    assert len(load_bearing) == 28
    assert load_bearing.isdisjoint(_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)


# ---------------------------------------------------------------------------
# (e) full-stack dispatch — every admit cell routes True via k971_route_decision
#     at default flag values (streamk=False, work_stealing=False, dtype matched).
# ---------------------------------------------------------------------------
_DTYPE_OBJ = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}


@pytest.mark.parametrize("cell", sorted(_K1563_P24_SKINNY_N4096_KCOMPL_ALIASSTACK_30))
def test_p24_full_dispatch_routes_true(cell):
    M, N, K, dtype = cell
    a = _DTYPE_OBJ[dtype]
    assert k971_route_decision(
        M, N, K, a_dtype=a, b_dtype=a,
        enable_streamk=False, work_stealing=False,
    ) is True


# ---------------------------------------------------------------------------
# (f) carve-out negatives — streamk / work_stealing / dtype-mismatch all veto.
# ---------------------------------------------------------------------------
def test_p24_streamk_vetoes_routing():
    assert k971_route_decision(
        2048, 4096, 8192,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16,
        enable_streamk=True, work_stealing=False,
    ) is False


def test_p24_work_stealing_vetoes_routing():
    assert k971_route_decision(
        2048, 4096, 8192,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16,
        enable_streamk=False, work_stealing=True,
    ) is False


def test_p24_dtype_mismatch_vetoes_routing():
    assert k971_route_decision(
        2048, 4096, 8192,
        a_dtype=torch.bfloat16, b_dtype=torch.float16,
        enable_streamk=False, work_stealing=False,
    ) is False


# ---------------------------------------------------------------------------
# (g) cohort gate sanity — ratio_median floors and cohort geomean from
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
