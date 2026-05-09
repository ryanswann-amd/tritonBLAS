"""Unit fixture — K-1567 P24 skinny_N8192 K-COMPLEMENT 30-cell route-OUT
envelope (17th-position).

Verifies (collapsed per The Minimalist's K-1586 review):
  (a) frozenset cardinality + exact admit-set match (K-1567 30 cells);
  (b) single parametrized test asserting every admit cell predicate-routes
      True (one parametrized case over the admit set, not 30 hand-listed
      pins);
  (c) sibling-N firewall sentinels return False (carry-out N-axis carve-out);
  (d) sibling-N firewall vs every prior K-COMPLEMENT frozenset is FULLY
      disjoint — every cell is a load-bearing NEW route-OUT, no upstream
      layer covers any N=8192 cell;
  (e) the public `k971_route_decision` dispatch returns True for every
      P24 N=8192 cell at default flag values (no double-admit, no flag
      interference);
  (f) carve-out negatives: streamk / work_stealing / dtype-mismatch all
      veto routing.

Source data: K-1567 paired n=30 HIP-graph hot-cache on MI300X / gfx942,
TRITONBLAS_DISABLE_K971=1 against the live post-K-1552 routing oracle;
B=10000 vectorised paired bootstrap; 30/30 admit at strict
ratio_median ≥ 1.05 ∧ p(<1.05) < 0.01 gate; cohort geomean tb/hbl ≈ 1.220×,
range ≈ 1.085×–1.402× (archival K-1567 measurement; see PR description).
Validates R-1478 #1 N-axis attenuation chain anchor at the previously-empty
N=8192 rung (full chain 1.451 → 1.234 → 1.220 → 1.174 → 1.118).
"""
import pytest
import torch

from tritonblas._route_predicate import (
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4,
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT,
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29,
    _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17,
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30,
    _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT,
    _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30,
    _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30,
    _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30,
    _K1567_P24_SKINNY_N8192_KCOMPL_ROUTEOUT_30,
    K971_ROUTE_TABLE,
    _k1567_p24_skinny_n8192_kcompl_routeout,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# (a) frozenset shape pin — 30 cells, exact M/N/K/dtype grid.
# ---------------------------------------------------------------------------
def test_p24_n8192_cardinality_is_thirty():
    assert len(_K1567_P24_SKINNY_N8192_KCOMPL_ROUTEOUT_30) == 30


def test_p24_n8192_admit_set_is_full_n8192_kcompl_grid():
    expected = {
        (M, 8192, K, dtype)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dtype in ("torch.bfloat16", "torch.float16")
    }
    assert _K1567_P24_SKINNY_N8192_KCOMPL_ROUTEOUT_30 == expected


# ---------------------------------------------------------------------------
# (b) every admit cell predicate-routes True AND full-stack dispatch routes
#     True — collapsed into a single parametrized test over the admit set
#     per The Minimalist's K-1586 review (was 30 + 30 hand-listed cases).
# ---------------------------------------------------------------------------
_DTYPE_OBJ = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}


@pytest.mark.parametrize(
    "cell",
    sorted(_K1567_P24_SKINNY_N8192_KCOMPL_ROUTEOUT_30),
)
def test_p24_n8192_admit_cell_routes_true(cell):
    """Every cell in the admit set must route True via both the leaf
    predicate AND the public k971_route_decision dispatch (default flags)."""
    M, N, K, dtype = cell
    # leaf predicate
    assert _k1567_p24_skinny_n8192_kcompl_routeout(M, N, K, dtype) is True
    # full-stack dispatch with matched dtype, no streamk / work_stealing
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
        (2048,   128, 2048, "torch.bfloat16"),  # P13 N=128
        (2048,   256, 2048, "torch.bfloat16"),  # P13/P21 N=256
        (2048,   512, 2048, "torch.bfloat16"),  # P15/P17/P23 N=512
        (2048,  1024, 2048, "torch.bfloat16"),  # P16 N=1024
        (2048,  4096, 2048, "torch.bfloat16"),  # P24 N=4096 (K-1566)
        (2048, 16384, 2048, "torch.bfloat16"),  # P19 N=16384
        (2048, 32768, 2048, "torch.bfloat16"),  # P22 N=32768
        (2048,  8192, 1024, "torch.bfloat16"),  # K outside grid (1024)
        (2048,  8192, 1536, "torch.bfloat16"),  # K outside grid (1536)
        (1024,  8192, 2048, "torch.bfloat16"),  # M outside bucket (1024)
        (3072,  8192, 2048, "torch.bfloat16"),  # M outside bucket (3072)
        (2048,  8192, 2048, "torch.float32"),   # fp32 carve-out
    ],
)
def test_p24_n8192_sibling_n_firewall_rejects(cell):
    M, N, K, dtype = cell
    assert _k1567_p24_skinny_n8192_kcompl_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# (d) sibling-N firewall vs every prior K-COMPLEMENT frozenset is FULLY
#     disjoint — every cell is a load-bearing NEW route-OUT.  Collapsed
#     into a single test that computes the union of all intersections in
#     one pass (was 12 parametrized cases + a separate union test).
# ---------------------------------------------------------------------------
def test_p24_n8192_sibling_n_firewall_is_fully_disjoint():
    """K-1567 N=8192 envelope has NO upstream coverage; every cell is a
    load-bearing NEW route-OUT.  Computes the union of intersections vs
    all prior K-COMPLEMENT frozensets in one pass and asserts EMPTY."""
    siblings = (
        ("K971_ROUTE_TABLE",  K971_ROUTE_TABLE),
        ("P12_square_mid",    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4),
        ("P13_N128",          _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18),
        ("P13_N256",          _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12),
        ("P15_N512",          _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT),
        ("P16_N1024",         _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29),
        ("P17_N512_BASE",     _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17),
        ("P19_N16384",        _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30),
        ("P21_N256_Kmid",     _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT),
        ("P22_N32768",        _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30),
        ("P23_N512_alias",    _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30),
        ("P24_N4096",         _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30),
    )
    overlaps = {
        name: sorted(_K1567_P24_SKINNY_N8192_KCOMPL_ROUTEOUT_30 & sibling)
        for name, sibling in siblings
        if _K1567_P24_SKINNY_N8192_KCOMPL_ROUTEOUT_30 & sibling
    }
    assert overlaps == {}, (
        f"K-1567 N=8192 envelope overlaps prior frozensets: {overlaps} — "
        "every prior K-COMPLEMENT predicate uses N ∈ {128, 256, 512, 1024, "
        "4096, 16384, 32768} and P8 / K971 cap at N ≤ 2048, so the N=8192 "
        "column must be FULLY disjoint by construction.  A non-empty "
        "overlap converts K-1567 from a load-bearing NEW route-OUT into a "
        "partial alias and requires a re-audit.")


# ---------------------------------------------------------------------------
# (e) carve-out negatives — streamk / work_stealing / dtype-mismatch all veto.
# ---------------------------------------------------------------------------
def test_p24_n8192_streamk_vetoes_routing():
    assert k971_route_decision(
        2048, 8192, 8192,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16,
        enable_streamk=True, work_stealing=False,
    ) is False


def test_p24_n8192_work_stealing_vetoes_routing():
    assert k971_route_decision(
        2048, 8192, 8192,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16,
        enable_streamk=False, work_stealing=True,
    ) is False


def test_p24_n8192_dtype_mismatch_vetoes_routing():
    assert k971_route_decision(
        2048, 8192, 8192,
        a_dtype=torch.bfloat16, b_dtype=torch.float16,
        enable_streamk=False, work_stealing=False,
    ) is False


# ---------------------------------------------------------------------------
# (f) Coverage check — closes the K-COMPLEMENT N-ladder mid-band gap.
# ---------------------------------------------------------------------------
def test_p24_n8192_completes_kcompl_n_ladder_midband():
    """K-COMPLEMENT N-ladder coverage post-K-1567: every cell in the
    M ∈ {2048,4096,8192} × N=8192 × K ∈ {2048,4096,8192,16384,32768} ×
    {bf16, fp16} grid should now be route-OUT — closing the previously-
    empty mid-band rung between P24 N=4096 (K-1566) and P19 N=16384
    (K-1478)."""
    for M in (2048, 4096, 8192):
        for K in (2048, 4096, 8192, 16384, 32768):
            for dtype in (torch.bfloat16, torch.float16):
                assert k971_route_decision(
                    M, 8192, K, a_dtype=dtype, b_dtype=dtype,
                    enable_streamk=False, work_stealing=False,
                ) is True, (
                    f"K-COMPLEMENT N=8192 mid-band cell "
                    f"({M},8192,{K},{dtype}) failed to route-OUT post-"
                    "K-1567; N-ladder coverage gap not closed.")
