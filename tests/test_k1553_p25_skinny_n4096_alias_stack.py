"""Unit fixture — K-1553 P25 skinny_N4096 K-COMPLEMENT 30-cell alias-stack
route-OUT envelope (17th-position).

Verifies:
  (a) frozenset cardinality + exact admit-set match (K-1553 30 cells);
  (b) every admit cell predicate-returns True;
  (c) sibling-N firewall sentinels return False (carry-out N-axis carve-out);
  (d) ALIAS-STACK invariant — every P25 cell is also routed by P24 ⨄ P12
      (the slot is documentation/insurance, not a new admit);
  (e) PARENT-MEASUREMENT-EQUIVALENCE — P25 frozenset is exactly equal to
      K-1566 P24 frozenset (P25 is the K-1553-named alias of P24 over the
      same 30-cell admit set);
  (f) the public `k971_route_decision` dispatch returns True for every P25
      cell at default flag values (no double-admit, no flag interference);
  (g) carve-out negatives: streamk / work_stealing / dtype-mismatch all
      veto routing.

Source data: K-1553 paired n=30 HIP-graph hot-cache on MI300X / gfx942
(combined with K-1559 60-cell mid-band N ∈ {4096, 8192} confirmation),
TRITONBLAS_DISABLE_K971=1, B=10000 vectorised paired bootstrap against
the live post-K-1532 routing oracle (HEAD 95e2c47); 30/30 admit at strict
ratio_median ≥ 1.05 ∧ p(<1.05) < 0.01 gate; cohort geomean tb/hbl =
1.234×, range 1.114×-1.501×, 0 regressions.
"""
import pytest
import torch

from tritonblas._route_predicate import (
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4,
    _K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30,
    _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30,
    _k1553_p25_skinny_n4096_kcompl_aliasstack_routeout,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# (a) frozenset shape pin — 30 cells, exact M/N/K/dtype grid.
# ---------------------------------------------------------------------------
def test_p25_cardinality_is_thirty():
    assert len(_K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30) == 30


def test_p25_admit_set_is_full_n4096_kcompl_grid():
    expected = {
        (M, 4096, K, dtype)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dtype in ("torch.bfloat16", "torch.float16")
    }
    assert _K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30 == expected


# ---------------------------------------------------------------------------
# (b) every admit cell predicate-returns True.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30)
)
def test_p25_admit_cell_routes_true(cell):
    M, N, K, dtype = cell
    assert _k1553_p25_skinny_n4096_kcompl_aliasstack_routeout(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# (c) sibling-N firewall sentinels return False — adjacent N buckets, K
#     outside grid, M outside bucket, fp32 carve-out.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell",
    [
        (2048,   512, 2048, "torch.bfloat16"),  # P15/P17/P23 N=512
        (2048,  1024, 2048, "torch.bfloat16"),  # P16 N=1024
        (2048,  2048, 2048, "torch.bfloat16"),  # K971/P12 N=2048
        (2048,  8192, 2048, "torch.bfloat16"),  # adjacent N=8192
        (2048, 16384, 2048, "torch.bfloat16"),  # P19 N=16384
        (2048, 32768, 2048, "torch.bfloat16"),  # P22 N=32768
        (2048,  4096, 1024, "torch.bfloat16"),  # K outside grid (1024)
        (2048,  4096, 6144, "torch.bfloat16"),  # K outside grid (6144)
        (1024,  4096, 2048, "torch.bfloat16"),  # M outside bucket (1024)
        (16384, 4096, 2048, "torch.bfloat16"),  # M outside bucket (16384)
        (2048,  4096, 2048, "torch.float32"),   # fp32 carve-out
    ],
)
def test_p25_sibling_n_firewall_rejects(cell):
    M, N, K, dtype = cell
    assert _k1553_p25_skinny_n4096_kcompl_aliasstack_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# (d) ALIAS-STACK invariant — every P25 cell is also routed by P24 ⨄ P12.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30)
)
def test_p25_alias_stack_covered_by_p24_or_p12(cell):
    in_p24 = cell in _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30
    in_p12 = cell in _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4
    assert in_p24 or in_p12, (
        f"P25 alias-stack cell {cell} not covered by P24 ⨄ P12; "
        "alias invariant violated.")


def test_p25_alias_overlap_with_p12_is_exactly_two_diagonal_cells():
    """K-1295 P12 covers (4096,4096,4096,{bf16,fp16}) on the diagonal; the
    P25 envelope shares exactly those 2 cells with P12 (and the remaining 28
    are sibling-N-firewall disjoint with P12)."""
    overlap = (
        _K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30
        & _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4
    )
    assert overlap == frozenset({
        (4096, 4096, 4096, "torch.bfloat16"),
        (4096, 4096, 4096, "torch.float16"),
    })


# ---------------------------------------------------------------------------
# (e) PARENT-MEASUREMENT-EQUIVALENCE — P25 frozenset == K-1566 P24 frozenset.
# ---------------------------------------------------------------------------
def test_p25_equals_p24_frozenset():
    """P25 is the K-1553-named alias of K-1566 P24 over the same 30-cell
    admit set; frozensets must be exactly equal.  Deviation indicates
    either (i) P24 contracted, or (ii) P25 was authored against a different
    K-1553 sub-set than the K-1566 productionization."""
    assert (
        _K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30
        == _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30
    )


# ---------------------------------------------------------------------------
# (f) full-stack dispatch — every admit cell routes True via k971_route_decision
#     at default flag values (streamk=False, work_stealing=False, dtype matched).
# ---------------------------------------------------------------------------
_DTYPE_OBJ = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}


@pytest.mark.parametrize(
    "cell", sorted(_K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30)
)
def test_p25_full_dispatch_routes_true(cell):
    M, N, K, dtype = cell
    a = _DTYPE_OBJ[dtype]
    assert k971_route_decision(
        M, N, K, a_dtype=a, b_dtype=a,
        enable_streamk=False, work_stealing=False,
    ) is True


# ---------------------------------------------------------------------------
# (g) carve-out negatives — streamk / work_stealing / dtype-mismatch all veto.
# ---------------------------------------------------------------------------
def test_p25_streamk_vetoes_routing():
    assert k971_route_decision(
        2048, 4096, 8192,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16,
        enable_streamk=True, work_stealing=False,
    ) is False


def test_p25_work_stealing_vetoes_routing():
    assert k971_route_decision(
        2048, 4096, 8192,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16,
        enable_streamk=False, work_stealing=True,
    ) is False


def test_p25_dtype_mismatch_vetoes_routing():
    assert k971_route_decision(
        2048, 4096, 8192,
        a_dtype=torch.bfloat16, b_dtype=torch.float16,
        enable_streamk=False, work_stealing=False,
    ) is False
