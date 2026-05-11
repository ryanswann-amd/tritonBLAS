"""Pin tests for the P17 ``skinny_N512`` K-COMPLEMENT BASE 17-cell
strict-equality route-OUT frozenset (11th-position).

Coverage:

* Frozenset shape (size + exact admit-set match against the bucket rule).
* Predicate behaviour on every admit cell (17/17 must return True with
  realistic production tile dispatch — bf16 / fp16, BLOCK_K is set by the
  autotune flow at 128 / 256 for these shapes).
* Boundary controls — N just above/below 512 cells must NOT route via P17.
* Disjointness firewalls vs the prior 10-predicate stack (P8 / K971 /
  P12 / P13(N=128) / P13(N=256) / P15 / P16); these were previously
  asserted at module load — moved here per minimalist convention.
* P15 ⨄ P17 = 29-cell N=512 K-COMPLEMENT closure (one P5-pre-routed cell
  is the 30th member of the bucket and is pinned separately).
* **P5-precedence pin**: (2048, 512, 4096, bf16) is intentionally NOT in
  the P17 frozenset because R_K979_P5 routes it OUT at chain pos 4 — this
  test pins that precedence so a future P5 change does not silently leave
  the cell unmeasured.
* Full-stack ``k971_route_decision`` integration: every P17 admit cell
  routes True through the 11-deep dispatch chain.
"""
from __future__ import annotations

import pytest
import torch

from tritonblas._route_predicate import (
    _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17,
    _k1437_p17_skinny_n512_kcompl_base_routeout,
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29,
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4,
    K971_ROUTE_TABLE,
    _P8_MFMA_ISSUE_STALL_ROUTEOUT,
    R_K979_P5_route_to_hbl,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# Frozenset shape pin (regression — must match the K-1409 paired n=30 BASE
# admit set, minus the single (2048,512,4096,bf16) cell which is
# P5-pre-routed at chain pos 4 < pos 11).
# ---------------------------------------------------------------------------

EXPECTED_BASE_ADMIT_17 = frozenset({
    # M=2048 row × K ∈ {4096, 8192, 16384} × {bf16, fp16}; (2048,512,4096,bf16)
    # excluded — P5-pre-routed (test_p5_routes_excluded_cell pins that).
    (2048, 512,  4096, "torch.float16"),
    (2048, 512,  8192, "torch.bfloat16"),
    (2048, 512,  8192, "torch.float16"),
    (2048, 512, 16384, "torch.bfloat16"),
    (2048, 512, 16384, "torch.float16"),
    # M=4096 row × K ∈ {4096, 8192, 16384} × {bf16, fp16}
    (4096, 512,  4096, "torch.bfloat16"),
    (4096, 512,  4096, "torch.float16"),
    (4096, 512,  8192, "torch.bfloat16"),
    (4096, 512,  8192, "torch.float16"),
    (4096, 512, 16384, "torch.bfloat16"),
    (4096, 512, 16384, "torch.float16"),
    # M=8192 row × K ∈ {4096, 8192, 16384} × {bf16, fp16}
    (8192, 512,  4096, "torch.bfloat16"),
    (8192, 512,  4096, "torch.float16"),
    (8192, 512,  8192, "torch.bfloat16"),
    (8192, 512,  8192, "torch.float16"),
    (8192, 512, 16384, "torch.bfloat16"),
    (8192, 512, 16384, "torch.float16"),
})


def test_frozenset_size_is_exactly_17():
    assert len(_K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17) == 17


def test_frozenset_matches_expected_admit_set():
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17 == EXPECTED_BASE_ADMIT_17


def test_every_admit_cell_in_base_sweep_grid():
    for M, N, K, dtype_str in _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17:
        assert M in (2048, 4096, 8192)
        assert N == 512
        assert K in (4096, 8192, 16384)
        assert dtype_str in ("torch.bfloat16", "torch.float16")


# ---------------------------------------------------------------------------
# Predicate behavior — admit cells (17 / 17 must return True).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", sorted(EXPECTED_BASE_ADMIT_17))
def test_predicate_returns_true_for_admit_cell(M, N, K, dtype_str):
    dtype = torch.bfloat16 if dtype_str == "torch.bfloat16" else torch.float16
    assert _k1437_p17_skinny_n512_kcompl_base_routeout(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# Boundary controls — must NOT route via P17 (other predicates may catch
# them — these checks are P17-local).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", [
    (2048,  512, 2048, "torch.bfloat16"),    # K=2048 — P15 EXTREMES territory
    (2048,  512, 32768, "torch.bfloat16"),   # K=32768 — P15 EXTREMES territory
    (2048,  256, 8192, "torch.bfloat16"),    # adjacent N=256 — P13 N=256 territory
    (2048, 1024, 8192, "torch.bfloat16"),    # adjacent N=1024 — P16 N=1024 territory
    (1024,  512, 8192, "torch.bfloat16"),    # M=1024 — out of bucket
    (2048,  512, 8192, "torch.float32"),     # dtype not in {bf16,fp16}
    (2048,  512, 1024, "torch.bfloat16"),    # K=1024 — out of bucket
])
def test_predicate_returns_false_outside_bucket(M, N, K, dtype_str):
    dtype = (torch.bfloat16 if dtype_str == "torch.bfloat16"
             else torch.float16 if dtype_str == "torch.float16"
             else torch.float32)
    assert _k1437_p17_skinny_n512_kcompl_base_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# P5-precedence pin — (2048, 512, 4096, bf16) is the one bucket cell
# excluded from the P17 frozenset because R_K979_P5 routes it at chain
# position 4 (long before P17 at position 11).  This test locks that
# precedence so a future P5 narrowing cannot silently leave this cell
# unrouted.
# ---------------------------------------------------------------------------

P5_OVERLAP_CELL = (2048, 512, 4096, torch.bfloat16)


def test_p5_overlap_cell_is_not_in_p17_frozenset():
    M, N, K, dtype = P5_OVERLAP_CELL
    assert (M, N, K, str(dtype)) not in _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17


def test_p5_routes_excluded_cell_at_chain_pos_4():
    """The (2048, 512, 4096, bf16) bucket cell must route to hipBLASLt via
    R_K979_P5 (chain position 4); P17 (position 11) is structurally
    unreachable for this cell.  A future P5 change that drops this cell
    must be paired with adding the cell back to P17.
    """
    M, N, K, dtype = P5_OVERLAP_CELL
    assert R_K979_P5_route_to_hbl(M, N, K, dtype) is True


def test_full_dispatch_routes_p5_overlap_cell_to_hbl():
    """Belt-and-braces: even though the cell is not in P17, the full
    11-deep dispatch chain must still route it OUT to hipBLASLt (via P5).
    """
    M, N, K, dtype = P5_OVERLAP_CELL
    assert k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False,
    ) is True


# ---------------------------------------------------------------------------
# Disjointness firewalls vs the prior 10-predicate stack (sibling-N + N=512
# BASE ⨄ EXTREMES partition).  These were previously asserted at module
# load; moved into the test suite per minimalist convention.
# ---------------------------------------------------------------------------

def test_disjoint_from_p8_mfma_envelope():
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17.isdisjoint(
        _P8_MFMA_ISSUE_STALL_ROUTEOUT)


def test_disjoint_from_k971_route_table():
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17.isdisjoint(K971_ROUTE_TABLE)


def test_disjoint_from_k1361_p12_square_mid():
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17.isdisjoint(
        _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)


def test_disjoint_from_k1367_p13_skinny_n128():
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17.isdisjoint(
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)


def test_disjoint_from_k1397_p13_skinny_n256():
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17.isdisjoint(
        _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12)


def test_disjoint_from_k1409_p15_skinny_n512_extremes():
    """BASE K ∈ {4096, 8192, 16384} ⨄ EXTREMES K ∈ {2048, 32768}; the
    union is the 29 admitted cells of the 30-cell N=512 K-COMPL grid
    (one cell — the P5-overlap — routes via P5 instead)."""
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17.isdisjoint(
        _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT)


def test_disjoint_from_k1429_p16_skinny_n1024():
    assert _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17.isdisjoint(
        _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29)


# ---------------------------------------------------------------------------
# N=512 K-COMPL closure pin: P15 EXTREMES ⨄ P17 BASE = 29 cells of the
# 30-cell bucket grid.  The remaining cell is (2048,512,4096,bf16) which
# is pinned by test_full_dispatch_routes_p5_overlap_cell_to_hbl.
# ---------------------------------------------------------------------------

def test_n512_kcompl_p15_union_p17_covers_29_of_30_bucket_cells():
    union = (
        _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17
        | _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT
    )
    expected_full_bucket = frozenset(
        (M, 512, K, dt)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    # P15 may itself omit cells per its own admit gate; only assert P17 ⊆
    # bucket and that the union contains everything except the single
    # P5-overlap cell.
    missing = expected_full_bucket - union
    assert missing.issubset({(2048, 512, 4096, "torch.bfloat16")})


# ---------------------------------------------------------------------------
# Full-stack k971_route_decision integration — the 11-deep dispatch chain
# must route every P17 admit cell to hipBLASLt with realistic production
# tile dispatch flags (streamk=False, work_stealing=False, matched dtypes).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", sorted(EXPECTED_BASE_ADMIT_17))
def test_full_dispatch_chain_routes_admit_cell_to_hbl(M, N, K, dtype_str):
    dtype = torch.bfloat16 if dtype_str == "torch.bfloat16" else torch.float16
    assert k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False,
    ) is True
