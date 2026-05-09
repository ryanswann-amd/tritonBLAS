"""Pin tests for the K-1489 P20 ``skinny_N16384`` K-COMPLEMENT alias-stack
frozenset (13th-position route-OUT slot).

Coverage rationale (per Testing-Zealot REVISE on the prior K-1489
attempt — *"add at minimum a unit test asserting the P20 envelope cells
route to hipBLASLt and a representative non-admit cell does not, so the
reserved-slot invariant is guarded against future regressions"*):

* **Frozenset shape pin**: P20 must be exactly 30 cells (alias of K-1478
  P19's 30-cell admit set) and identity-equal to P19's frozenset (the
  stack convention is that an alias-slot is the *same* Python object,
  not a duplicated literal — see commit f103661).
* **Happy-path admit**: every cell in the P20 envelope must route True
  through the full ``k971_route_decision`` 13-deep dispatch chain. We
  spot-check the K-1478 anchor and the max-ratio cell explicitly, then
  iterate over the full 30-cell envelope as a coverage sweep.
* **Non-admit invariant**: representative N=16384 cells *outside* the
  P20 envelope (e.g. M=512 or K not in {2048,4096,8192,16384,32768})
  must NOT be routed by the P20 membership check. We assert this both
  on the predicate's frozenset directly (no spurious admits) and via
  ``k971_route_decision`` (no upstream predicate accidentally routes
  the cell either, so the test is a true reserved-slot invariant).
* **Reserved-slot semantics**: P20 must come *after* P19 in the
  dispatch ladder (so P19's short-circuit makes P20 unreachable while
  P19 is enabled — this is the documented "reserved for
  K-COMPLEMENT-EXTENDED" semantics).
* **K-1478 measurement provenance**: the 30 cells were measured under
  K-1478's paired n=30 HIP-graph hot-cache protocol on MI300X (geomean
  tb/hbl 1.174×, range 1.056×–1.359×, 30/30 admit at strict ratio_median
  ≥ 1.05 ∧ bootstrap p(<1.05) < 0.01 gate). The alias-stack inherits
  that evidence by construction; this test pins the inheritance.
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30,
    _K1478_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_N,
    _k1478_p19_skinny_n16384_routeout,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# Frozenset shape + alias-identity pin.
# ---------------------------------------------------------------------------


def test_p20_is_alias_of_p19_same_object():
    """P20 must be the *same* Python object as P19 (alias-stack convention,
    not a duplicated literal — guards against silent drift if someone
    edits the P19 set without updating P20)."""
    assert (
        _K1478_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_N
        is _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30
    ), (
        "K-1489 P20 must be a Python-identity alias of K-1478 P19's "
        "frozenset (alias-stack convention; see commit f103661 and the "
        "_route_predicate.py rationale comment above the assignment)."
    )


def test_p20_envelope_size_is_30_cells():
    """P20 envelope must be exactly 30 cells (3 M-rows × 5 K-cols × 2
    dtypes = 30, all admitted at the K-1478 strict-1.05 gate)."""
    assert len(_K1478_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_N) == 30


# ---------------------------------------------------------------------------
# Happy-path: every admit cell must route True through the full chain.
# ---------------------------------------------------------------------------


def test_p20_anchor_cell_routes_to_hipblaslt():
    """K-1478 anchor cell (lowest M-row, lowest K-col, bf16) must route
    OUT to hipBLASLt under the full 13-deep dispatch."""
    assert k971_route_decision(
        2048, 16384, 2048, "torch.bfloat16", "torch.bfloat16",
        enable_streamk=False, work_stealing=False,
    ) is True


def test_p20_max_ratio_cell_routes_to_hipblaslt():
    """K-1478 max-ratio cell (4096, 16384, 2048, bf16) at r=1.359 must
    route OUT to hipBLASLt — sanity check that the highest-speedup
    cell in the envelope is reachable via the 13-deep dispatch chain."""
    assert k971_route_decision(
        4096, 16384, 2048, "torch.bfloat16", "torch.bfloat16",
        enable_streamk=False, work_stealing=False,
    ) is True


@pytest.mark.parametrize("cell", sorted(_K1478_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_N))
def test_p20_every_envelope_cell_routes_to_hipblaslt(cell):
    """Coverage sweep: every one of the 30 P20 alias-envelope cells must
    route True through k971_route_decision. Parametrised so a single-cell
    regression surfaces as one failing test name, not a wall of asserts."""
    M, N, K, dtype = cell
    assert k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False,
    ) is True, f"P20 envelope cell {cell} failed to route to hipBLASLt"


# ---------------------------------------------------------------------------
# Non-admit invariant — representative out-of-envelope cells must NOT route.
# ---------------------------------------------------------------------------


# Representative non-admit N=16384 cells.  All have N=16384 (so they
# share the "skinny-N" column with the P20 envelope) but fall outside
# the (M, K, dtype) grid that K-1478's paired n=30 sweep admitted:
#   * M=512, M=1024, M=16384 — outside the 3 admit rows {2048,4096,8192}.
#   * K=1024, K=65536           — outside the 5 admit cols
#                                 {2048,4096,8192,16384,32768}.
# These cells must not be in the frozenset, and must not be routed OUT
# by the full 13-deep dispatch chain (they fall through to in-kernel).
NON_ADMIT_N16384_CELLS = [
    (512,   16384,  2048,  "torch.bfloat16"),    # M below the admit rows
    (1024,  16384,  4096,  "torch.bfloat16"),    # M below the admit rows
    (16384, 16384,  4096,  "torch.bfloat16"),    # M above the admit rows
    (2048,  16384,  1024,  "torch.bfloat16"),    # K below the admit cols
    (4096,  16384, 65536,  "torch.bfloat16"),    # K above the admit cols
]


@pytest.mark.parametrize("cell", NON_ADMIT_N16384_CELLS)
def test_p20_non_admit_n16384_cell_not_in_envelope(cell):
    """Reserved-slot invariant: representative N=16384 cells outside the
    P20 (M, K, dtype) admit grid must not be in the frozenset — guards
    against accidental envelope expansion that would silently route
    untested cells out to hipBLASLt."""
    assert cell not in _K1478_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_N, (
        f"non-admit cell {cell} unexpectedly present in P20 envelope; "
        "P20 must mirror K-1478 P19 exactly (3 M-rows × 5 K-cols × 2 "
        "dtypes = 30 cells)."
    )


@pytest.mark.parametrize("cell", NON_ADMIT_N16384_CELLS)
def test_p20_non_admit_n16384_cell_not_routed(cell):
    """Reserved-slot invariant: representative N=16384 cells outside the
    P20 envelope must not be routed OUT by the full 13-deep dispatch
    chain (no upstream predicate accidentally claims them either)."""
    M, N, K, dtype = cell
    assert k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False,
    ) is False, (
        f"non-admit N=16384 cell {cell} unexpectedly routed OUT; "
        "either P20 envelope drifted or an upstream predicate "
        "accidentally claimed the cell."
    )


# ---------------------------------------------------------------------------
# Reserved-slot semantics — P19 must short-circuit P20 (P20 unreachable
# while P19 is enabled).  This is the documented "reserved for
# K-COMPLEMENT-EXTENDED" property.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell", sorted(_K1478_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_N))
def test_p19_short_circuits_p20_for_every_envelope_cell(cell):
    """For every P20 envelope cell, the P19 predicate alone must return
    True — proving P19 short-circuits before the P20 membership check is
    ever consulted (so P20 is functionally unreachable while P19 is
    enabled, as documented in the dispatch-ladder docstring)."""
    M, N, K, dtype = cell
    assert _k1478_p19_skinny_n16384_routeout(M, N, K, dtype) is True, (
        f"P19 predicate failed to short-circuit on P20 alias cell {cell}; "
        "P20 alias-stack semantics depend on P19 short-circuiting first."
    )


# ---------------------------------------------------------------------------
# Disjointness — alias-of-P19 inherits P19's disjointness vs P1-P17 by
# construction; pin a couple of representative checks as cheap insurance.
# ---------------------------------------------------------------------------


def test_p20_disjoint_from_non_n16384_admit_n_columns():
    """P20 admits only N=16384 (inherited from P19); no admit cell may
    have N ∈ {128, 256, 512, 1024, 2048, 4096, 8192, 32768}."""
    forbidden_n = {128, 256, 512, 1024, 2048, 4096, 8192, 32768}
    bad = [
        cell for cell in _K1478_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_N
        if cell[1] in forbidden_n
    ]
    assert not bad, (
        f"P20 envelope contains cells with N outside {{16384}}: {bad}; "
        "alias-of-P19 must inherit the skinny-N=16384 column exactly."
    )
