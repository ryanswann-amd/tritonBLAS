"""K-1531 (S-002): pin tests for the P21 ``skinny_N2048`` K-COMPLEMENT
29-cell route-OUT frozenset (14th-position).

Pin coverage (analogous to K-1429 P16 / K-1437 P17 productionisation tests):

* 29 admit cells × 1 dispatch trace each (positive admit verification)
* 1 deferred parity cell — verifies (8192, 2048, 8192, fp16) is NOT in the
  frozenset (TB-faster at ratio_med=0.997, CI95=[0.994, 0.998]; route-OUT
  would regress and is excluded to preserve the zero-regressions invariant)
* 6 boundary controls — N just above/below 2048, M out of bucket, dtype
  out of bucket, K out of bucket
* Frozenset-shape pin (29 cells exactly, anchor enumeration)
* Disjointness pins vs every prior K-COMPLEMENT frozenset (sibling-N firewall:
  K-1367 P13 N=128, K-1397 P13 N=256, K-1409 P15 N=512, K-1429 P16 N=1024,
  K-1437 P17 N=512 BASE, K-1478 P19 N=16384, K-1493 P20 N=16384)
* Phase 3 post-merge contract: the 21 "newly-routed" cells (M ∈ {4096, 8192}
  rows + 2 M=2048 fp16 K∈{4096,8192} cells) must be routed to hipBLASLt by
  ``k971_route_decision`` — this is the regression test that catches the
  reviewer's "+198 LOC dead code" failure mode (frozenset present but
  unreachable from the dispatch chain).
"""
from __future__ import annotations

import pytest
import torch

from tritonblas._route_predicate import (
    _K1518_P21_SKINNY_N2048_KCOMPL_ROUTEOUT_N,
    _k1531_p21_skinny_n2048_routeout,
    # K-1493 P20 frozenset is exported under the K-1478 P20 alias-stack name.
    _K1478_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_N,
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30,
    _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17,
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29,
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# Frozenset shape pins (regression — must match K-1518 paired n=30 admit set
# re-validated against the LIVE post-K-1493 oracle in K-1531 Phase 1)
# ---------------------------------------------------------------------------

EXPECTED_ADMIT_29 = frozenset({
    # M=2048 row × K ∈ {2048, 4096, 8192, 16384, 32768} × {bf16, fp16}
    # (10/10 — 8 are pre-routed by K-905/K-971/K-1295/K-1335; included
    # for envelope completeness mirroring K-1493 P20 alias-stack pattern)
    (2048, 2048,  2048, "torch.bfloat16"),
    (2048, 2048,  2048, "torch.float16"),
    (2048, 2048,  4096, "torch.bfloat16"),
    (2048, 2048,  4096, "torch.float16"),
    (2048, 2048,  8192, "torch.bfloat16"),
    (2048, 2048,  8192, "torch.float16"),
    (2048, 2048, 16384, "torch.bfloat16"),
    (2048, 2048, 16384, "torch.float16"),
    (2048, 2048, 32768, "torch.bfloat16"),
    (2048, 2048, 32768, "torch.float16"),
    # M=4096 row × K ∈ {2048, 4096, 8192, 16384, 32768} × {bf16, fp16} (10/10)
    (4096, 2048,  2048, "torch.bfloat16"),
    (4096, 2048,  2048, "torch.float16"),
    (4096, 2048,  4096, "torch.bfloat16"),
    (4096, 2048,  4096, "torch.float16"),
    (4096, 2048,  8192, "torch.bfloat16"),
    (4096, 2048,  8192, "torch.float16"),
    (4096, 2048, 16384, "torch.bfloat16"),
    (4096, 2048, 16384, "torch.float16"),
    (4096, 2048, 32768, "torch.bfloat16"),
    (4096, 2048, 32768, "torch.float16"),
    # M=8192 row × K ∈ {2048, 4096, 8192, 16384, 32768} × {bf16, fp16} (9/10)
    # NOTE: (8192, 2048, 8192, "torch.float16") DEFERRED — TB-faster parity
    (8192, 2048,  2048, "torch.bfloat16"),
    (8192, 2048,  2048, "torch.float16"),
    (8192, 2048,  4096, "torch.bfloat16"),
    (8192, 2048,  4096, "torch.float16"),
    (8192, 2048,  8192, "torch.bfloat16"),
    (8192, 2048, 16384, "torch.bfloat16"),
    (8192, 2048, 16384, "torch.float16"),
    (8192, 2048, 32768, "torch.bfloat16"),
    (8192, 2048, 32768, "torch.float16"),
})

# Cell deferred from the frozenset — TB-faster, route-OUT would regress.
DEFERRED_PARITY_CELL = (8192, 2048, 8192, "torch.float16")

# The 21 cells that K-1531 Phase 1 measurement found "newly-routed":
# live=False AND admit at K-1472 loose gate (ratio_med >= 1.05 AND CI95-lo > 1.00)
# — i.e. genuinely-additive coverage that this PR claims to deliver.
# This is the contract the reviewer wanted asserted: if the predicate ever
# stops routing one of these cells, this test fails loudly.
NEWLY_ROUTED_21 = frozenset({
    (2048, 2048,  4096, "torch.float16"),     # r=1.296
    (2048, 2048,  8192, "torch.float16"),     # r=1.190
    (4096, 2048,  2048, "torch.bfloat16"),    # r=1.655 (cohort MAX)
    (4096, 2048,  2048, "torch.float16"),     # r=1.636
    (4096, 2048,  4096, "torch.bfloat16"),    # r=1.607
    (4096, 2048,  4096, "torch.float16"),     # r=1.564
    (4096, 2048,  8192, "torch.bfloat16"),    # r=1.543
    (4096, 2048,  8192, "torch.float16"),     # r=1.505
    (4096, 2048, 16384, "torch.bfloat16"),    # r=1.173
    (4096, 2048, 16384, "torch.float16"),     # r=1.164
    (4096, 2048, 32768, "torch.bfloat16"),    # r=1.256
    (4096, 2048, 32768, "torch.float16"),     # r=1.270
    (8192, 2048,  2048, "torch.bfloat16"),    # r=1.186
    (8192, 2048,  2048, "torch.float16"),     # r=1.173
    (8192, 2048,  4096, "torch.bfloat16"),    # r=1.167
    (8192, 2048,  4096, "torch.float16"),     # r=1.143
    (8192, 2048,  8192, "torch.bfloat16"),    # r=1.149
    (8192, 2048, 16384, "torch.bfloat16"),    # r=1.151
    (8192, 2048, 16384, "torch.float16"),     # r=1.123
    (8192, 2048, 32768, "torch.bfloat16"),    # r=1.126
    (8192, 2048, 32768, "torch.float16"),     # r=1.109
})


def _dt(s):
    return torch.bfloat16 if s == "torch.bfloat16" else torch.float16


# ---------------------------------------------------------------------------
# Frozenset shape regressions
# ---------------------------------------------------------------------------

def test_frozenset_size_is_exactly_29():
    assert len(_K1518_P21_SKINNY_N2048_KCOMPL_ROUTEOUT_N) == 29


def test_frozenset_matches_expected_admit_set():
    assert _K1518_P21_SKINNY_N2048_KCOMPL_ROUTEOUT_N == EXPECTED_ADMIT_29


def test_deferred_parity_cell_is_excluded():
    assert DEFERRED_PARITY_CELL not in _K1518_P21_SKINNY_N2048_KCOMPL_ROUTEOUT_N


def test_newly_routed_21_subset_of_frozenset():
    assert NEWLY_ROUTED_21.issubset(_K1518_P21_SKINNY_N2048_KCOMPL_ROUTEOUT_N)


# ---------------------------------------------------------------------------
# Predicate behaviour — admit cells return True
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", sorted(EXPECTED_ADMIT_29))
def test_predicate_returns_true_for_every_admit_cell(M, N, K, dtype_str):
    assert _k1531_p21_skinny_n2048_routeout(M, N, K, _dt(dtype_str)) is True


def test_predicate_returns_false_for_deferred_parity_cell():
    M, N, K, dtype_str = DEFERRED_PARITY_CELL
    assert _k1531_p21_skinny_n2048_routeout(M, N, K, _dt(dtype_str)) is False


# ---------------------------------------------------------------------------
# Boundary controls — out-of-bucket cells must NOT be routed by P21.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", [
    (4096, 1024, 4096, "torch.bfloat16"),    # adjacent N=1024 — covered by P16
    (4096, 4096, 4096, "torch.bfloat16"),    # adjacent N=4096 — out of bucket
    (1024, 2048, 4096, "torch.bfloat16"),    # M=1024 — out of bucket
    (4096, 2048, 1024, "torch.bfloat16"),    # K=1024 — out of bucket
    (4096, 2048, 65536, "torch.bfloat16"),   # K=65536 — out of bucket
    (4096, 2048,  4096, "torch.float32"),    # dtype not in {bf16, fp16}
])
def test_predicate_returns_false_outside_bucket(M, N, K, dtype_str):
    if dtype_str == "torch.float32":
        dtype = torch.float32
    else:
        dtype = _dt(dtype_str)
    assert _k1531_p21_skinny_n2048_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# Sibling-N disjointness firewalls (every prior K-COMPLEMENT frozenset
# must key on a different N from P21's N=2048).
# ---------------------------------------------------------------------------

def test_disjoint_from_k1367_p13_skinny_n128():
    assert _K1518_P21_SKINNY_N2048_KCOMPL_ROUTEOUT_N.isdisjoint(
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)


def test_disjoint_from_k1397_p13_skinny_n256():
    assert _K1518_P21_SKINNY_N2048_KCOMPL_ROUTEOUT_N.isdisjoint(
        _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12)


def test_disjoint_from_k1409_p15_skinny_n512():
    assert _K1518_P21_SKINNY_N2048_KCOMPL_ROUTEOUT_N.isdisjoint(
        _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT)


def test_disjoint_from_k1429_p16_skinny_n1024():
    assert _K1518_P21_SKINNY_N2048_KCOMPL_ROUTEOUT_N.isdisjoint(
        _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29)


def test_disjoint_from_k1437_p17_skinny_n512_base():
    assert _K1518_P21_SKINNY_N2048_KCOMPL_ROUTEOUT_N.isdisjoint(
        _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17)


def test_disjoint_from_k1478_p19_skinny_n16384():
    assert _K1518_P21_SKINNY_N2048_KCOMPL_ROUTEOUT_N.isdisjoint(
        _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30)


def test_disjoint_from_k1493_p20_skinny_n16384():
    assert _K1518_P21_SKINNY_N2048_KCOMPL_ROUTEOUT_N.isdisjoint(
        _K1478_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_N)


# ---------------------------------------------------------------------------
# Cohort scoping — every admit cell must satisfy the K-1518 sweep grid.
# ---------------------------------------------------------------------------

def test_every_admit_cell_in_sweep_grid():
    for M, N, K, dtype_str in _K1518_P21_SKINNY_N2048_KCOMPL_ROUTEOUT_N:
        assert M in (2048, 4096, 8192)
        assert N == 2048
        assert K in (2048, 4096, 8192, 16384, 32768)
        assert dtype_str in ("torch.bfloat16", "torch.float16")


# ---------------------------------------------------------------------------
# End-to-end dispatch contract — k971_route_decision must route every
# "newly routed" cell to hipBLASLt.  This is the test the reviewer wanted:
# if this PR's predicate is dead code (frozenset present but never reached
# from the dispatch chain), at least one of these assertions will fail.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str", sorted(NEWLY_ROUTED_21))
def test_k971_routes_newly_routed_cells_to_hbl(M, N, K, dtype_str):
    dtype = _dt(dtype_str)
    routed = k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False,
        work_stealing=False,
        disable_env_set=False,
    )
    assert routed is True, (
        f"K-1531 P21 contract violated: ({M},{N},{K},{dtype_str}) "
        f"is in the productionised admit set but k971_route_decision "
        f"refused to route it to hipBLASLt — predicate is unreachable "
        f"or has been clobbered."
    )


def test_k971_does_not_route_deferred_parity_cell_via_p21():
    """The (8192, 2048, 8192, fp16) parity cell is excluded from P21.  An
    earlier predicate may still route it (if so, that's not our problem),
    but P21 itself must NOT claim it — verified at the predicate level."""
    M, N, K, dtype_str = DEFERRED_PARITY_CELL
    dtype = _dt(dtype_str)
    assert _k1531_p21_skinny_n2048_routeout(M, N, K, dtype) is False
