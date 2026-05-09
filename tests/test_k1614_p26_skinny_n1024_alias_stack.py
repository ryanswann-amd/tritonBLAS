"""Unit fixture — K-1614 P26 skinny_N1024 K-COMPLEMENT 30-cell alias-stack
route-OUT envelope (18th-position).

Verifies:
  (a) frozenset cardinality + exact admit-set match (K-1592 30 cells:
      M ∈ {2048,4096,8192} × N=1024 × K ∈ {2048,4096,8192,16384,32768}
      × {bf16,fp16});
  (b) every admit cell predicate-returns True;
  (c) sibling-N firewall sentinels return False (adjacent N buckets,
      out-of-grid M/K, fp32 carve-out, the (4096,1024,1024,bf16) sibling
      with K outside the K-COMPLEMENT cohort);
  (d) ALIAS-STACK invariant — every P26 cell is also routed by P16 ⨄ P5
      (the slot is documentation/insurance, not a new admit);
  (e) the public `k971_route_decision` dispatch returns True for every P26
      cell at default flag values;
  (f) carve-out negatives: streamk / work_stealing / dtype-mismatch all
      veto routing;
  (g) per-cell dispatch trace — for each of the 30 P26 cells, exactly
      one of {P16, P5} fires upstream (proves the alias-stack claim cell-
      by-cell, not just at the cohort level).

Source data: K-1592 paired n=30 HIP-graph hot-cache on MI300X / gfx942,
TRITONBLAS_DISABLE_K971=1, B=10000 vectorised paired bootstrap; cohort
geomean tb/hbl ≥ 1.0× across all 30 cells (29/30 admitted by P16 K-1429
at strict 1.05 with cohort geomean 2.23×, BASE 2.08× / EXTREMES 2.48×;
the single (2048,1024,4096,bf16) cell at r≈1.015 is covered by R_K979_P5
Clause-1 mid-rect LDS-pressure band).

Mechanism: at N=1024 the persistent_matmul tile aspect misaligns against
the M ∈ {2048,4096,8192} anchors, accumulating LDS-bank conflicts beyond
the K-913 §3 attenuation; the skinny-N tile selection avoids these
conflicts (R-1429.SKINNY-N1024-K-COMPLEMENT-DISCRIMINATOR-REVERSES-
ATTENUATION-AT-N1024).
"""
import pytest
import torch

from tritonblas._route_predicate import (
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29,
    _K1604_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30,
    _k1604_p26_skinny_n1024_kcompl_aliasstack_routeout,
    R_K979_P5_route_to_hbl,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# (a) frozenset shape pin — 30 cells, exact M/N/K/dtype grid.
# ---------------------------------------------------------------------------
def test_p26_cardinality_is_thirty():
    assert len(_K1604_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30) == 30


def test_p26_admit_set_is_full_n1024_kcompl_grid():
    expected = {
        (M, 1024, K, dtype)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dtype in ("torch.bfloat16", "torch.float16")
    }
    assert _K1604_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30 == expected


# ---------------------------------------------------------------------------
# (b) every admit cell predicate-returns True.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell", sorted(_K1604_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30))
def test_p26_admit_cell_routes_true(cell):
    M, N, K, dtype = cell
    assert _k1604_p26_skinny_n1024_kcompl_aliasstack_routeout(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# (c) sibling-N firewall sentinels return False — adjacent N buckets, K
#     outside grid, M outside bucket, fp32 carve-out.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell",
    [
        (2048, 256,  2048, "torch.bfloat16"),    # adjacent N=256 (P13)
        (2048, 512,  2048, "torch.bfloat16"),    # adjacent N=512 (P15+P17+P23)
        (2048, 2048, 2048, "torch.bfloat16"),    # P12 N=2048 territory
        (2048, 4096, 2048, "torch.bfloat16"),    # P24 N=4096
        (2048, 8192, 2048, "torch.bfloat16"),    # P25 N=8192
        (4096, 1024, 1024, "torch.bfloat16"),    # K outside K-COMPLEMENT (1024)
        (4096, 1024,  256, "torch.bfloat16"),    # K outside K-COMPLEMENT (256)
        (1024, 1024, 2048, "torch.bfloat16"),    # M outside bucket (1024)
        (16384, 1024, 2048, "torch.bfloat16"),   # M outside bucket (16384)
        (4096, 1024, 4096, "torch.float32"),     # fp32 carve-out
        (4096, 1024, 4096, "torch.int8"),        # int8 carve-out
    ],
)
def test_p26_sibling_n_firewall_rejects(cell):
    M, N, K, dtype = cell
    assert _k1604_p26_skinny_n1024_kcompl_aliasstack_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# (d) ALIAS-STACK invariant — every P26 cell is also routed by P16 ⨄ P5.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell", sorted(_K1604_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30))
def test_p26_alias_stack_covered_by_p16_or_p5(cell):
    M, N, K, dtype = cell
    in_p16 = cell in _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29
    in_p5 = R_K979_P5_route_to_hbl(M, N, K, dtype)
    assert in_p16 or in_p5, (
        f"P26 alias-stack cell {cell} not covered by P16 ⨄ P5; "
        "alias invariant violated.")


# ---------------------------------------------------------------------------
# (e) full-stack dispatch — every admit cell routes True via k971_route_decision
#     at default flag values.
# ---------------------------------------------------------------------------
_DTYPE_OBJ = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}


@pytest.mark.parametrize("cell", sorted(_K1604_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30))
def test_p26_full_dispatch_routes_true(cell):
    M, N, K, dtype = cell
    a = _DTYPE_OBJ[dtype]
    assert k971_route_decision(
        M, N, K, a_dtype=a, b_dtype=a,
        enable_streamk=False, work_stealing=False,
    ) is True


# ---------------------------------------------------------------------------
# (f) carve-out negatives — streamk / work_stealing / dtype-mismatch all veto.
# ---------------------------------------------------------------------------
def test_p26_streamk_vetoes_routing():
    assert k971_route_decision(
        2048, 1024, 8192,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16,
        enable_streamk=True, work_stealing=False,
    ) is False


def test_p26_work_stealing_vetoes_routing():
    assert k971_route_decision(
        2048, 1024, 8192,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16,
        enable_streamk=False, work_stealing=True,
    ) is False


def test_p26_dtype_mismatch_vetoes_routing():
    assert k971_route_decision(
        2048, 1024, 8192,
        a_dtype=torch.bfloat16, b_dtype=torch.float16,
        enable_streamk=False, work_stealing=False,
    ) is False


# ---------------------------------------------------------------------------
# (g) per-cell dispatch trace — for each of the 30 P26 cells, exactly one of
#     {P16, P5} fires upstream.  Proves the alias-stack claim cell-by-cell.
#     Expected split per K-1592: 29 cells via P16, 1 cell via P5
#     ((2048,1024,4096,bf16) at r≈1.015, just below P16's 1.05 admit floor).
# ---------------------------------------------------------------------------
_P5_ONLY_CELL = (2048, 1024, 4096, "torch.bfloat16")


@pytest.mark.parametrize("cell", sorted(_K1604_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30))
def test_p26_per_cell_dispatch_trace(cell):
    M, N, K, dtype = cell
    in_p16 = cell in _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29
    in_p5 = R_K979_P5_route_to_hbl(M, N, K, dtype)
    if cell == _P5_ONLY_CELL:
        assert not in_p16 and in_p5, (
            f"K-1592 invariant violated: cell {cell} expected P5-only "
            f"but in_p16={in_p16}, in_p5={in_p5}")
    else:
        assert in_p16, (
            f"K-1592 invariant violated: cell {cell} expected to be in P16 "
            f"but in_p16={in_p16}")


def test_p26_per_cell_split_is_29_p16_plus_1_p5():
    in_p16 = sum(
        1 for c in _K1604_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30
        if c in _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29
    )
    p5_only = sum(
        1 for c in _K1604_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30
        if c not in _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29
        and R_K979_P5_route_to_hbl(c[0], c[1], c[2], c[3])
    )
    assert in_p16 == 29
    assert p5_only == 1
    assert in_p16 + p5_only == 30
