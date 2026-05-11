"""K-2118 P54 (S-002) — 10th-rung off-by-48 N=1456 K-COMPLEMENT alias-stack tests.

Source measurement: extends the contiguous off-by-48 ladder admitted by
K-1978 / K-1994 / K-2010 / K-2031 / K-2047 / K-2055 / K-2067 (P49) /
K-2085 (P50, P51) / K-2097 (P52) / K-2111 (P53).  K-2118 promotes N=1456
to a **16-cell envelope** = full M ∈ {2048, 4096, 8192} × N=1456 ×
K ∈ {4096, 8192, 16384} × {bf16, fp16} grid MINUS the 2 (M=2048,
K=16384) corner cells where hipBLASLt empirically loses to tritonblas
(0.938x bf16 / 0.915x fp16 — far below the 0.97x per-cell regression
floor).  Geomean A/HBL on the surviving 16 cells = 1.007x; promotion
gate honestly cleared.

Off-by-48 ladder context (R-1811 wave-misalignment + K-913 §3 LDS-bank-conflict
mechanism, dtype-invariant): N=1456 is the **10th rung** of the off-by-48
wave-misaligned ladder, anchored at the N=816 floor with a +64 step per rung
(rung k = 816 + k*64).  The 9 rungs at k ∈ {1..9} land at
N ∈ {880, 944, 1008, 1072, 1136, 1200, 1264, 1328, 1392} (admitted across
K-1978 / K-1994 / K-2010 / K-2031 / K-2047 / K-2055 / K-2067 / K-2085 P50 /
K-2085 P51 / K-2097 / K-2111).  N=1456 is k=10 → step +64 from the K-2111
P53 N=1392 rung (k=9).  All 11 N values on the ladder (floor + 10 rungs)
satisfy (N mod 64) == 48.

Sub-family placement: N=1456 mod 128 == 48 — canonical-tail sub-family
shared with N ∈ {816, 944, 1072, 1200, 1328} (vs the off-tail sub-family
at N ∈ {880, 1008, 1136, 1264, 1392} where N mod 128 == 112).  REJOIN to
the canonical-tail mod-128 topology after the K-2111 P53 N=1392 off-tail
rung.

Envelope choice: 16 cells = full 18-cell grid MINUS the 2 (M=2048,
K=16384, *) corner cells.  Reason: at every other (M, K) pair hipBLASLt
beats tritonblas (geomean of the surviving 16 = 1.007x A/HBL, 0/16 cells
below the 0.97x floor), but at (M=2048, K=16384) the column-narrow
N=1456 tile interacts with hipBLASLt's split-K heuristic such that
tritonblas measures 6.7-9% faster.  The minimalist principle (R-1532)
demands shipping only what is measured to win; the 2 losing corner cells
are explicitly excluded to honestly meet G1 (geomean A/HBL ≥ 1.00x).

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 16 (full M×K×dtype grid minus 2 corner cells).
  2. N axis is exactly {1456}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Both dtype rows are complete 8/8 (off-band N=1456 has no R-K979 P5
     Clause-3 bf16 alias — strictly load-bearing on both rows; symmetric
     exclusion of the 2 (M=2048, K=16384) corner cells preserves dtype
     parity).
  5. Sibling-N firewall vs the K-1922 P40 N=544 alias-stack (the only
     other directly-imported single-N alias-stack at this dispatcher
     position on fix/K-1922).
  6. Off-by-48 wave-misalignment band membership: 1456 mod 64 == 48 (same
     ladder as the 9 prior confirmed rungs above the N=816 floor).
  7. Canonical-tail sub-family membership: 1456 mod 128 == 48 (same
     sub-family as N ∈ {816, 944, 1072, 1200, 1328}).
  8. Step +64 from the K-2111 P53 rung at N=1392 (contiguous-ladder
     extension invariant; guards against silent N-axis typos like 1454,
     1458, 1520 that would land on a different BLOCK_N=64 alignment rung).
  9. Excluded-corner-cell firewall: (M=2048, N=1456, K=16384, *) cells
     are EXPLICITLY NOT in the frozenset (would lose to hipBLASLt).
     Pinned to detect a silent re-admission via the dispatcher.
 10. Dispatcher happy-path: the live ``k971_route_decision`` returns
     True for an in-envelope cell — proves the +1-import / +1-membership-check
     additive diff actually wires the frozenset into the route-OUT decision.
     Parameterised across all 16 envelope cells so any precedence
     regression that masks the K-2118 P54 clause is caught.
 11. Dispatcher negative-path (sibling-N firewall): the dispatcher
     returns False for off-envelope sibling N values (1455, 1457, 1520) —
     proves the K-2118 P54 clause does NOT spuriously route a non-1456
     cell.  Plus a dedicated NEGATIVE for the 2 explicitly-excluded
     (M=2048, K=16384) corner cells — proves they are NOT routed even
     though they are on the right N rung (the regressing cells must
     stay routed to tritonblas, not to hipBLASLt).
 12. Carve-out short-circuits: streamk / work-stealing / mismatched-dtype
     bypass the K-2118 P54 admission.
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
    _K2118_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_16,
    k971_route_decision,
)


FZ = _K2118_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_16


# Cells excluded from the 16-cell envelope (full grid minus these 2 corner
# cells).  Encoded in the test for an explicit NEGATIVE assertion.
EXCLUDED_CORNER_CELLS = frozenset({
    (2048, 1456, 16384, "torch.bfloat16"),
    (2048, 1456, 16384, "torch.float16"),
})


def test_cardinality_is_16():
    assert len(FZ) == 16


def test_n_axis_is_skinny_n1456():
    assert {N for (_, N, _, _) in FZ} == {1456}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_row_balance():
    """Both dtype rows complete 8/8.  Off-band N=1456 has no R-K979 P5
    Clause-3 bf16 alias coverage at this rung; both rows are strictly
    load-bearing.  K-913 §3 LDS-bank-conflict is dtype-invariant on the
    column-narrow N=1456 tile (same fingerprint as the 9 prior off-by-48
    rungs above the N=816 floor).  The 2-cell exclusion at (M=2048,
    K=16384, *) is symmetric across both dtypes — preserves parity."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 8
    assert len(fp) == 8
    assert bf == fp


def test_envelope_equals_full_grid_minus_excluded_corner():
    """The verified-winner envelope is exactly the 18-cell M ∈ {2048,
    4096, 8192} × N=1456 × K ∈ {4096, 8192, 16384} × {bf16, fp16} grid
    MINUS the 2 (M=2048, K=16384, *) corner cells where hipBLASLt
    measurably loses (0.938x bf16 / 0.915x fp16, below the 0.97x
    per-cell floor) — pinned to detect any silent contraction (a
    false-NEGATIVE that would leak winner cells back to TB) or expansion
    (a false-POSITIVE that would re-admit the 2 known-losing corner
    cells and re-fail G1)."""
    full = frozenset(
        (M, 1456, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    expected = full - EXCLUDED_CORNER_CELLS
    assert FZ == expected
    assert len(full) == 18
    assert len(expected) == 16


def test_excluded_corner_cells_are_not_in_envelope():
    """Explicit NEGATIVE: the 2 (M=2048, K=16384, *) corner cells must
    NOT be in the frozenset.  These cells regress when routed to
    hipBLASLt (0.938x bf16 / 0.915x fp16, below the 0.97x per-cell
    floor); admitting them re-introduces the v5 G1 failure."""
    for cell in EXCLUDED_CORNER_CELLS:
        assert cell not in FZ, (
            f"Cell {cell} was excluded from the K-2118 P54 envelope "
            f"because hipBLASLt loses to tritonblas there (0.938x bf16 / "
            f"0.915x fp16 — below the 0.97x per-cell regression floor). "
            f"Re-admitting it would re-introduce the v5 G1 failure.")


def test_off_by_48_wave_misalignment_ladder_membership():
    """N=1456 sits on the off-by-48 wave-misaligned ladder (N mod 64 == 48)
    as the 10th rung (k=10 → 816 + 10*64 = 1456) above the N=816 floor.
    The 9 prior rungs (k ∈ {1..9}) live at
    N ∈ {880, 944, 1008, 1072, 1136, 1200, 1264, 1328, 1392}.  Step +64
    from the K-2111 P53 N=1392 rung (k=9).  This invariant guards against
    a silent N-axis typo (e.g. 1454, 1458, 1520) that would land on a
    different BLOCK_N=64 alignment rung and reflect a different mechanism."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 64 == 48
    assert only_n == 1392 + 64
    assert only_n == 816 + 10 * 64
    assert only_n == 1456


def test_canonical_tail_sub_family_membership():
    """N=1456 mod 128 == 48 — canonical-tail sub-family shared with
    N ∈ {816, 944, 1072, 1200, 1328} (R-1811 wave-misalignment sub-classification).
    Distinct from the off-tail sub-family at N mod 128 == 112
    (N ∈ {880, 1008, 1136, 1264, 1392}).  REJOIN to N=1328 mod-128 topology
    after the K-2111 P53 N=1392 off-tail rung."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 128 == 48


def test_sibling_n_firewall_vs_p40_n544():
    """The only other single-N alias-stack imported at this dispatcher
    position on the fix/K-1922 base branch is K-1922 P40 N=544.  Disjoint
    by natural N-axis separation (1456 ≠ 544); guards against a silent
    N-axis collision."""
    assert FZ & _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18 == set()


# ---------------------------------------------------------------------------
# Dispatcher-invocation tests (reviewer feedback K-2118 v1 → v2: pure
# data-shape invariants do not exercise the actual routing decision; the
# +2-LOC additive diff in matmul._k971_route_to_hbl must be exercised
# end-to-end against the live precedence chain).  We use the torch-free
# proxy ``k971_route_decision`` (mirrored from matmul._k971_route_to_hbl
# logic; updated in this PR to include the new K-2118 P54 final clause)
# so the test runs CPU-only and does not require GPU initialisation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M,N,K,dtype", sorted(FZ))
def test_dispatcher_routes_every_k2118_p54_cell_to_hbl(M, N, K, dtype):
    """HAPPY PATH — every one of the 16 K-2118 P54 in-envelope cells
    must route to hipBLASLt under the full live precedence chain with
    default carve-out settings (a_dtype == b_dtype, streamk=False,
    work_stealing=False).  Falsifies a precedence-bug regression where
    the K-2118 P54 clause is added but is unreachable, mis-ordered, or
    short-circuited by an upstream layer that returns False on N=1456."""
    routed = k971_route_decision(
        int(M), int(N), int(K), dtype, dtype,
        enable_streamk=False, work_stealing=False,
    )
    assert routed is True, (
        f"K-2118 P54 in-envelope cell ({M},{N},{K},{dtype}) MUST route to "
        f"hipBLASLt under default carve-outs; the +2-LOC additive diff in "
        f"matmul._k971_route_to_hbl (1 import + 1 membership check) is "
        f"load-bearing and must wire the frozenset into the route-OUT chain.")


@pytest.mark.parametrize("excluded", sorted(EXCLUDED_CORNER_CELLS))
def test_dispatcher_does_not_route_excluded_corner_cells(excluded):
    """NEGATIVE PATH (excluded-corner-cell firewall) — the 2 (M=2048,
    K=16384, *) corner cells were explicitly removed from the envelope
    because hipBLASLt loses to tritonblas there (0.938x bf16 / 0.915x
    fp16, below the 0.97x per-cell regression floor).  These cells MUST
    NOT route to hipBLASLt under the live dispatcher; they must stay on
    the tritonblas path.  Falsifies a silent re-admission of the
    regressing corner cells."""
    M, N, K, dtype = excluded
    routed = k971_route_decision(
        int(M), int(N), int(K), dtype, dtype,
        enable_streamk=False, work_stealing=False,
    )
    assert routed is False, (
        f"Excluded-corner cell ({M},{N},{K},{dtype}) must NOT route to "
        f"hipBLASLt — empirical n=30 HIP-graph hot-cache benchmark shows "
        f"hipBLASLt loses (0.938x bf16 / 0.915x fp16, below 0.97x floor); "
        f"re-admission would re-introduce the v5 G1 failure.")


@pytest.mark.parametrize("N_off", [1455, 1457, 1520])
def test_dispatcher_does_not_route_off_envelope_n_sibling(N_off):
    """NEGATIVE PATH (sibling-N firewall) — N siblings adjacent to 1456
    that are NOT on the off-by-48 ladder must NOT be spuriously routed
    by the K-2118 P54 clause.

      - N=1455 mod 64 = 47 (off-by-47, not on any K-COMPLEMENT ladder).
      - N=1457 mod 64 = 17 (random, not on any K-COMPLEMENT ladder).
      - N=1520 mod 64 = 16, mod 128 = 112 (would be a +64 step but lands
        on a different residue family — falsifies a typo of N=1520 for
        the next rung after 1456).

    Pick (M=4096, K=8192, bf16) — a strong in-envelope cell row — and
    confirm none of the 3 sibling-N values triggers any route-OUT in the
    live precedence chain (the K-2118 P54 frozenset's only N is 1456,
    so a True return on these inputs would prove the new clause is
    leaking false positives).  Cross-check by direct frozenset
    membership."""
    M, K = 4096, 8192
    dtype = "torch.bfloat16"
    # Frozenset firewall
    assert (M, N_off, K, dtype) not in FZ
    # Dispatcher firewall
    routed = k971_route_decision(
        M, N_off, K, dtype, dtype,
        enable_streamk=False, work_stealing=False,
    )
    assert routed is False, (
        f"Off-envelope sibling N={N_off} (M={M},K={K},{dtype}) must NOT "
        f"route to hipBLASLt — would indicate the K-2118 P54 clause is "
        f"leaking false positives or an unrelated upstream layer is "
        f"misfiring on this row.")


@pytest.mark.parametrize("enable_streamk,work_stealing,b_dtype", [
    (True,  False, "torch.bfloat16"),  # streamk requested
    (False, True,  "torch.bfloat16"),  # work-stealing requested
    (False, False, "torch.float16"),   # mismatched dtype (a bf16, b fp16)
])
def test_k2118_p54_admit_cell_carveout_short_circuits(
    enable_streamk, work_stealing, b_dtype,
):
    """CARVE-OUT PATH — a K-2118 P54 in-envelope cell must NOT route to
    hipBLASLt when streamk / work-stealing / mismatched-dtype carve-outs
    apply.  The K-2118 P54 clause must not bypass the universal
    short-circuits at the top of `_k971_route_to_hbl`.  Pick
    (4096, 1456, 8192, bf16) as a representative admit cell."""
    a_dtype = "torch.bfloat16"
    M, N, K = 4096, 1456, 8192
    assert (M, N, K, a_dtype) in FZ
    routed = k971_route_decision(
        M, N, K, a_dtype, b_dtype, enable_streamk, work_stealing,
    )
    assert routed is False, (
        f"K-2118 P54 admit cell ({M},{N},{K},{a_dtype}) must NOT route "
        f"to hipBLASLt when carve-out applies (streamk={enable_streamk}, "
        f"work_stealing={work_stealing}, b_dtype={b_dtype}); the universal "
        f"short-circuit in _k971_route_to_hbl is load-bearing.")
