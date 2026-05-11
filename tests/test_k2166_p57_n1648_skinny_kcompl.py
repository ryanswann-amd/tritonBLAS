"""K-2166 P57 — N=1648 K-COMPLEMENT alias-stack invariants and routing.

Pins the structural invariants that justify admission into the off-by-48
ladder (cardinality, N-axis, residue-48 family invariant), plus behavioural
routing assertions that exercise `_k971_route_to_hbl` end-to-end at the
in-set cell and the two adjacent-rung spillover-firewall control cells:

  * N=1584 — the prior +64 rung (K-2150 admit envelope).  Must STILL route
    to HBL via the K-2150 frozenset (no-regression guarantee on the
    immediate prior rung — the +2-LOC additive dispatcher diff must not
    perturb K-2150 routing in either direction).
  * N=1712 — the next +64 step beyond N=1648; held OUT of the K-2166 admit
    envelope (future ticket).  Must NOT route to HBL via the K-2166
    frozenset (no scope creep from the +2-LOC additive dispatcher diff).
"""
from __future__ import annotations

from tritonblas._route_predicate import (
    _K2150_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18 as FZ_PRIOR,
    _K2166_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18 as FZ,
)
from tritonblas.matmul import _k971_route_to_hbl


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_every_entry_has_n1648():
    assert all(N == 1648 for (_, N, _, _) in FZ)


def test_off_by_48_family_invariant():
    """Every entry must satisfy the off-by-48 ladder invariant N % 64 == 48.

    This is the structural property that justifies admission into the
    K-COMPLEMENT alias-stack — same residue class as every prior rung
    (K-2106 N=1328, K-2111 N=1392, K-2127 N=1456, K-2136 N=1520,
    K-2150 N=1584). 1648 % 64 == 48 (12.875 BLOCK_N=128 tiles per N-row,
    0.125 fractional-tile residue exploited by HBL on this family)."""
    for (M, N, K, dt) in FZ:
        assert N % 64 == 48, f"entry ({M},{N},{K},{dt}) violates off-by-48 invariant"


def test_dispatcher_routes_in_set_cell_to_hbl():
    """The canonical (M=4096, N=1648, K=8192, bf16) cell must route to HBL."""
    assert _k971_route_to_hbl(
        4096, 1648, 8192, "torch.bfloat16", "torch.bfloat16",
        enable_streamk=False, work_stealing=False,
    ) is True


def test_dispatcher_does_not_route_n1712_next_rung_control_to_hbl():
    """The next +64 step beyond this rung (N=1712) is held out of the K-2166
    admit envelope (future ticket) — must NOT route to HBL via the K-2166
    frozenset.  This guards against silent scope creep from the +2-LOC
    additive dispatcher diff."""
    assert _k971_route_to_hbl(
        4096, 1712, 8192, "torch.bfloat16", "torch.bfloat16",
        enable_streamk=False, work_stealing=False,
    ) is False


def test_dispatcher_still_routes_n1584_prior_rung_to_hbl_no_regression():
    """The immediate prior +64 rung (N=1584, K-2150 admit envelope) must
    continue to route to HBL via the K-2150 frozenset.  This guards
    against silent regression on the neighbouring rung — the K-2166
    additive +2-LOC dispatcher diff must NOT perturb K-2150 routing."""
    assert (4096, 1584, 8192, "torch.bfloat16") in FZ_PRIOR
    assert _k971_route_to_hbl(
        4096, 1584, 8192, "torch.bfloat16", "torch.bfloat16",
        enable_streamk=False, work_stealing=False,
    ) is True


def test_k2166_and_k2150_envelopes_are_disjoint():
    """The two adjacent rungs must share no cells — disjoint admit envelopes
    are the structural firewall that lets the +2-LOC additive dispatcher
    diff land safely with no risk of double-routing or spillover."""
    assert FZ.isdisjoint(FZ_PRIOR)
