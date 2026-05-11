"""Unit fixture — K-1552 P23 skinny_N512 K-COMPLEMENT 30-cell alias-stack
route-OUT envelope (15th-position).

Verifies:
  (a) frozenset cardinality + exact admit-set match (K-1534 30 cells);
  (b) every admit cell predicate-returns True;
  (c) sibling-N firewall sentinels return False (carry-out N-axis carve-out);
  (d) ALIAS-STACK invariant — every P23 cell is also routed by P15 ⨄ P17 ⨄ P5
      (the slot is documentation/insurance, not a new admit);
  (e) the public `k971_route_decision` dispatch returns True for every P23
      cell at default flag values (no double-admit, no flag interference);
  (f) carve-out negatives: streamk / work_stealing / dtype-mismatch all
      veto routing.

Source data: K-1534 paired n=30 HIP-graph hot-cache on MI300X / gfx942
on MI300X / gfx942, TRITONBLAS_DISABLE_K971=1, B=10000 vectorised
paired bootstrap; 30/30 admit at strict ratio_median ≥ 1.05 ∧
p(<1.05) < 0.01 gate; cohort geomean tb/hbl = 1.454×, range 1.093×–2.111×.
"""
import pytest
import torch

from tritonblas._route_predicate import (
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT,
    _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17,
    _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30,
    _k1552_p23_skinny_n512_kcompl_aliasstack_routeout,
    R_K979_P5_route_to_hbl,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# (a) frozenset shape pin — 30 cells, exact M/N/K/dtype grid.
# ---------------------------------------------------------------------------
def test_p23_cardinality_is_thirty():
    assert len(_K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30) == 30


def test_p23_admit_set_is_full_n512_kcompl_grid():
    expected = {
        (M, 512, K, dtype)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dtype in ("torch.bfloat16", "torch.float16")
    }
    assert _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30 == expected


# ---------------------------------------------------------------------------
# (b) every admit cell predicate-returns True.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell", sorted(_K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30))
def test_p23_admit_cell_routes_true(cell):
    M, N, K, dtype = cell
    assert _k1552_p23_skinny_n512_kcompl_aliasstack_routeout(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# (c) sibling-N firewall sentinels return False — adjacent N buckets, K
#     outside grid, M outside bucket, fp32 carve-out.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell",
    [
        (2048, 256, 2048, "torch.bfloat16"),    # adjacent N=256
        (2048, 1024, 2048, "torch.bfloat16"),   # adjacent N=1024
        (2048, 16384, 2048, "torch.bfloat16"),  # P19 N=16384
        (2048, 32768, 2048, "torch.bfloat16"),  # P22 N=32768
        (2048, 512, 1024, "torch.bfloat16"),    # K outside grid (1024)
        (1024, 512, 2048, "torch.bfloat16"),    # M outside bucket (1024)
        (2048, 512, 2048, "torch.float32"),     # fp32 carve-out
    ],
)
def test_p23_sibling_n_firewall_rejects(cell):
    M, N, K, dtype = cell
    assert _k1552_p23_skinny_n512_kcompl_aliasstack_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# (d) ALIAS-STACK invariant — every P23 cell is also routed by P15 ⨄ P17 ⨄ P5.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell", sorted(_K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30))
def test_p23_alias_stack_covered_by_p15_p17_or_p5(cell):
    M, N, K, dtype = cell
    in_p15 = cell in _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT
    in_p17 = cell in _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17
    in_p5 = R_K979_P5_route_to_hbl(M, N, K, dtype)
    assert in_p15 or in_p17 or in_p5, (
        f"P23 alias-stack cell {cell} not covered by P15 ⨄ P17 ⨄ P5; "
        "alias invariant violated.")


# ---------------------------------------------------------------------------
# (e) full-stack dispatch — every admit cell routes True via k971_route_decision
#     at default flag values (streamk=False, work_stealing=False, dtype matched).
# ---------------------------------------------------------------------------
_DTYPE_OBJ = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}


@pytest.mark.parametrize("cell", sorted(_K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30))
def test_p23_full_dispatch_routes_true(cell):
    M, N, K, dtype = cell
    a = _DTYPE_OBJ[dtype]
    assert k971_route_decision(
        M, N, K, a_dtype=a, b_dtype=a,
        enable_streamk=False, work_stealing=False,
    ) is True


# ---------------------------------------------------------------------------
# (f) carve-out negatives — streamk / work_stealing / dtype-mismatch all veto.
# ---------------------------------------------------------------------------
def test_p23_streamk_vetoes_routing():
    assert k971_route_decision(
        2048, 512, 8192,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16,
        enable_streamk=True, work_stealing=False,
    ) is False


def test_p23_work_stealing_vetoes_routing():
    assert k971_route_decision(
        2048, 512, 8192,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16,
        enable_streamk=False, work_stealing=True,
    ) is False


def test_p23_dtype_mismatch_vetoes_routing():
    assert k971_route_decision(
        2048, 512, 8192,
        a_dtype=torch.bfloat16, b_dtype=torch.float16,
        enable_streamk=False, work_stealing=False,
    ) is False
