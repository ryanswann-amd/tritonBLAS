"""Unit fixture — K-1611 P26 skinny_N2048 K-COMPLEMENT 30-cell alias-stack
route-OUT envelope (18th-position).

Verifies:
  (a) frozenset cardinality + exact admit-set match (K-1611 30 cells over
      the M ∈ {2048, 4096, 8192} × N=2048 × K ∈ {2048, 4096, 8192, 16384,
      32768} × {bf16, fp16} envelope);
  (b) every admit cell predicate-returns True;
  (c) sibling-N firewall sentinels return False (off-N rows, off-grid
      M / K, dtype carve-out);
  (d) ALIAS-STACK invariant (mixed-coverage variant) — each of the 8 INERT
      cells on the M=2048 row IS routed by some upstream layer
      (K971_ROUTE_TABLE ⨄ R_K979_P5 ⨄ K-1295 P12), so the slot is alias
      documentation for those 8 cells; the remaining 22 cells on
      M ∈ {4096, 8192} are NEW route-OUT (not required to be covered
      upstream);
  (e) the public `k971_route_decision` dispatch returns True for every P26
      cell at default flag values (no double-admit, no flag interference);
  (f) carve-out negatives: streamk / work_stealing / dtype-mismatch all
      veto routing.

Source data: K-1611 paired n=30 HIP-graph hot-cache on MI300X / gfx942
(50 replays/iter, alternating tb-first / hbl-first, B=10000 vectorised
paired bootstrap CI95 of the MEDIAN ratio per R-1472 #3) against the LIVE
post-K-1592 18-frozenset routing oracle (HEAD `db93d92`); 21/30 strict
admits at ratio_median ≥ 1.05 ∧ ci95_lo > 1.00 gate, 8/30 inert (M=2048
row alias of upstream), 1/30 deferred parity at (8192,2048,8192,fp16)
r=1.011× (HBL still faster, included for envelope completeness); 0
regressions; admit-set cohort geomean tb/hbl = 1.297× (range
1.116×-1.625×); all-30 cohort geomean = 1.200×.
"""
import pytest
import torch

from tritonblas._route_predicate import (
    K971_ROUTE_TABLE,
    R_K979_P5_route_to_hbl,
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4,
    _K1611_P26_INERT_ALIAS_CELLS_8,
    _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30,
    _k1611_p26_skinny_n2048_kcompl_aliasstack_routeout,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# (a) frozenset shape pin — 30 cells, exact M/N/K/dtype grid.
# ---------------------------------------------------------------------------
def test_p26_cardinality_is_thirty():
    assert len(_K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30) == 30


def test_p26_admit_set_is_full_n2048_kcompl_grid():
    expected = {
        (M, 2048, K, dtype)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dtype in ("torch.bfloat16", "torch.float16")
    }
    assert _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30 == expected


def test_p26_inert_alias_subset_cardinality_is_eight():
    assert len(_K1611_P26_INERT_ALIAS_CELLS_8) == 8
    assert _K1611_P26_INERT_ALIAS_CELLS_8.issubset(
        _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30)


def test_p26_inert_alias_subset_is_exactly_the_m2048_row():
    expected = {
        (2048, 2048,  2048, "torch.bfloat16"),
        (2048, 2048,  2048, "torch.float16"),
        (2048, 2048,  4096, "torch.bfloat16"),
        (2048, 2048,  8192, "torch.bfloat16"),
        (2048, 2048, 16384, "torch.bfloat16"),
        (2048, 2048, 16384, "torch.float16"),
        (2048, 2048, 32768, "torch.bfloat16"),
        (2048, 2048, 32768, "torch.float16"),
    }
    assert _K1611_P26_INERT_ALIAS_CELLS_8 == expected


# ---------------------------------------------------------------------------
# (b) every admit cell predicate-returns True.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30)
)
def test_p26_admit_cell_routes_true(cell):
    M, N, K, dtype = cell
    assert _k1611_p26_skinny_n2048_kcompl_aliasstack_routeout(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# (c) sibling-N firewall sentinels return False — adjacent N buckets, K
#     outside grid, M outside bucket, fp32 carve-out.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell",
    [
        # adjacent N buckets — every prior K-COMPLEMENT layer
        (2048,   128, 2048, "torch.bfloat16"),  # P13/P8 N=128
        (2048,   256, 2048, "torch.bfloat16"),  # P13/P21 N=256
        (2048,   512, 2048, "torch.bfloat16"),  # P15/P17/P23 N=512
        (2048,  1024, 2048, "torch.bfloat16"),  # P16/P26-K1592 N=1024
        (2048,  4096, 2048, "torch.bfloat16"),  # P24/P25 N=4096
        (2048,  8192, 2048, "torch.bfloat16"),  # adjacent N=8192
        (2048, 16384, 2048, "torch.bfloat16"),  # P19 N=16384
        (2048, 32768, 2048, "torch.bfloat16"),  # P22 N=32768
        # off-grid K (envelope K ∈ {2048, 4096, 8192, 16384, 32768})
        (2048,  2048,  1024, "torch.bfloat16"),
        (2048,  2048,  3072, "torch.bfloat16"),
        (2048,  2048,  6144, "torch.bfloat16"),
        (4096,  2048, 12288, "torch.bfloat16"),
        (8192,  2048, 65536, "torch.bfloat16"),
        # off-grid M (envelope M ∈ {2048, 4096, 8192})
        (1024,  2048, 2048, "torch.bfloat16"),
        (3072,  2048, 2048, "torch.bfloat16"),
        (6144,  2048, 2048, "torch.bfloat16"),
        (16384, 2048, 2048, "torch.bfloat16"),
        # dtype carve-out — only bf16 / fp16 in envelope
        (2048,  2048, 2048, "torch.float32"),
        (4096,  2048, 4096, "torch.float8_e4m3fnuz"),
    ],
)
def test_p26_sibling_n_firewall_rejects(cell):
    M, N, K, dtype = cell
    assert _k1611_p26_skinny_n2048_kcompl_aliasstack_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# (d) ALIAS-STACK invariant (mixed coverage) — every cell in the 8-cell
#     INERT subset MUST be routed by some upstream layer
#     (K971_ROUTE_TABLE ⨄ R_K979_P5 ⨄ K-1295 P12).  The 22 NEW cells on
#     M ∈ {4096, 8192} are load-bearing here and are NOT required to be
#     covered upstream.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1611_P26_INERT_ALIAS_CELLS_8)
)
def test_p26_inert_cell_covered_by_upstream(cell):
    M, N, K, dtype = cell
    in_k971 = (M, N, K, dtype) in K971_ROUTE_TABLE
    in_p5 = R_K979_P5_route_to_hbl(M, N, K, dtype)
    in_p12 = (M, N, K, dtype) in _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4
    assert in_k971 or in_p5 or in_p12, (
        f"K-1611 P26 inert-alias cell {cell} not covered by "
        "K971_ROUTE_TABLE ⨄ R_K979_P5 ⨄ K-1295 P12; alias invariant "
        "violated.")


def test_p26_new_route_out_subset_is_disjoint_from_inert_alias_subset():
    """The 22 NEW route-OUT cells (load-bearing here) and the 8 INERT alias
    cells (alias of upstream) partition the 30-cell envelope."""
    new_route_out = (
        _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30
        - _K1611_P26_INERT_ALIAS_CELLS_8
    )
    assert len(new_route_out) == 22
    assert new_route_out.isdisjoint(_K1611_P26_INERT_ALIAS_CELLS_8)
    # All NEW cells live on M ∈ {4096, 8192}
    for cell in new_route_out:
        assert cell[0] in (4096, 8192)


# ---------------------------------------------------------------------------
# (e) public `k971_route_decision` dispatch returns True for every P26 cell
#     at default flag values — verifies no double-admit / flag interference
#     and that the chain wiring at the 18th-position is reachable.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30)
)
def test_p26_chain_dispatch_routes_true(cell):
    M, N, K, dtype = cell
    routed = k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False, disable_env_set=False,
    )
    assert routed is True, (
        f"K-1611 P26 cell {cell} should route to hipBLASLt via the chain "
        "(either upstream short-circuit for the 8 inert cells or the new "
        "18th-position membership check for the 22 NEW cells).")


# ---------------------------------------------------------------------------
# (f) carve-outs — streamk / work_stealing / dtype mismatch all veto routing.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell",
    sorted({
        (4096, 2048, 4096, "torch.bfloat16"),  # max-speedup admit cell
        (8192, 2048, 32768, "torch.float16"),  # min-speedup admit cell
    }),
)
def test_p26_streamk_vetoes_routing(cell):
    M, N, K, dtype = cell
    assert k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=True, work_stealing=False, disable_env_set=False,
    ) is False


@pytest.mark.parametrize(
    "cell",
    sorted({
        (4096, 2048, 4096, "torch.bfloat16"),
        (8192, 2048, 32768, "torch.float16"),
    }),
)
def test_p26_work_stealing_vetoes_routing(cell):
    M, N, K, dtype = cell
    assert k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=True, disable_env_set=False,
    ) is False


def test_p26_dtype_mismatch_vetoes_routing():
    # bf16 lhs vs fp16 rhs — dtype-pair mismatch always vetoes per
    # _k971_route_to_hbl Clause-1 (`str(a_dtype) != str(b_dtype)`).
    assert k971_route_decision(
        4096, 2048, 4096, "torch.bfloat16", "torch.float16",
        enable_streamk=False, work_stealing=False, disable_env_set=False,
    ) is False
