"""P38 (S-002) — 29th-slot N=416 K-COMPLEMENT verified-winner subset tests.

K-1880 productionises the K-1873 verified-winner subset of the N=416 wave-
misaligned skinny-N K-COMPLEMENT cohort: 18 cells (M ∈ {2048, 4096, 8192}
× N=416 × K ∈ {4096, 8192, 16384} × {bf16, fp16}) MINUS the single upstream-
aliased cell (2048, 416, 4096, "torch.bfloat16") already routed by the
K-1003 R-K979 P5 Clause-1 mid-rect non-square predicate.  17 admit cells.

Mechanism: K-913 §3 LDS-bank-conflict on N=416 mod 128 = 32 narrow third
tile (R-1811 wave-misalignment).  Mirrors the K-1810 P32 / K-1817 P33 /
K-1831 P34 / K-1837 P35 / K-1850 P36 / K-1866 P37 wire-up pattern.

Per the K-1880 Minimalist round, this file pins only three structural
invariants (R-1532 / R-1720 — the same invariants are enforced by
construction at every prior P-slot; everything else is redundant with the
literal frozenset):

  1. Membership sanity — cardinality ==17 and the envelope is exactly the
     N=416 K-COMPLEMENT grid minus the P5-aliased cell.
  2. P5-alias exclusion — the (2048,416,4096,bf16) cell is NOT in P38.
  3. Dispatcher-admit smoke — every P38 cell admits via the live
     `_k971_route_to_hbl` dispatcher.
"""
from __future__ import annotations

from tritonblas._route_predicate import _P38_SKINNY_N416_KCOMPL_VERIFIED_WIN_17


FZ = _P38_SKINNY_N416_KCOMPL_VERIFIED_WIN_17

# The single cell already routed by K-1003 R-K979 P5 Clause-1 (mid-rect
# non-square: minMN=416 ∈ [256, 2304], maxMN=2048 ∈ [1792, 3072], K=4096
# ∈ [1240, 8064]).  Excluded from P38 per R-K1825.CHECK-ALIAS-STACK-
# COVERAGE-MAP-FIRST.
_UPSTREAM_ALIASED_EXCLUSION = frozenset({
    (2048, 416, 4096, "torch.bfloat16"),
})


def test_membership_sanity():
    """Cardinality ==17 and envelope == full N=416 K-COMPLEMENT grid minus
    the single P5-aliased bf16 cell.  Pinned to detect any silent
    contraction (winner cells leak back to TB) or expansion (re-includes
    the P5-aliased cell, causing duplicate routing)."""
    full = frozenset(
        (M, 416, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert len(full) == 18
    assert FZ == full - _UPSTREAM_ALIASED_EXCLUSION
    assert len(FZ) == 17


def test_p5_alias_excluded():
    """The (2048,416,4096,bf16) cell is already routed by P5 Clause-1
    (measured ratio_TB/HBL = 1.002, p = 0.394 paired-t n=30).  P38 MUST
    NOT include it — duplicate routing would violate R-K1825."""
    assert FZ & _UPSTREAM_ALIASED_EXCLUSION == set()


def test_dispatcher_admits_all_p38_cells():
    """Smoke check: every cell in P38 must admit via the live
    `_k971_route_to_hbl` dispatcher — confirms the K-1880 consolidated
    canonical-set probe routes the N=416 admit cells.  Per R-Testing-Zealot
    (K-1880 RETRY) this is also asserted parametrised cell-by-cell over the
    full 51-cell K-1880 canonical roster in
    `test_k1880_p36_p38_consolidation.py::test_canonical_cell_routes_through_k971_dispatcher`,
    so a future drop of any single P38 cell from the route surfaces as a
    named pytest failure (not just a smoke failure here)."""
    from tritonblas.matmul import _k971_route_to_hbl
    for (M, N, K, dt) in FZ:
        assert _k971_route_to_hbl(M, N, K, dt, dt, False, False), (
            f"P38 cell ({M}, {N}, {K}, {dt}) failed to admit via _k971_route_to_hbl"
        )


def test_p5_aliased_cell_still_routes_via_upstream_path():
    """Per R-Testing-Zealot (K-1880 RETRY): the (2048,416,4096,bf16) cell
    is excluded from P38 per R-K1825 because P5 Clause-1 already routes it
    upstream.  Pin that the exclusion did NOT drop dispatch coverage —
    the cell still routes True via `_k971_route_to_hbl` via the P5 path.
    If a future P5 change loses this cell, it would silently fall back to
    TB-native (regression); this test surfaces that as a named failure."""
    from tritonblas.matmul import _k971_route_to_hbl
    M, N, K, dt = 2048, 416, 4096, "torch.bfloat16"
    assert _k971_route_to_hbl(M, N, K, dt, dt, False, False), (
        f"P5-aliased cell ({M},{N},{K},{dt}) lost upstream coverage — "
        f"P38 excluded it expecting P5 Clause-1 to route it, but "
        f"_k971_route_to_hbl returns False"
    )
