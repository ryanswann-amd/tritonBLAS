"""P34 (S-002) — 25th-slot N=128 wave-aligned mid-K verified-winner subset tests.

Source measurement: K-1828 paired n=30 HIP-graph hot-cache capture/replay
on MI300X / gfx942 (OCI fallback amd-rccl partition per INFRA-0048) across
the 18-cell N=128 mid-K sub-cohort = M ∈ {2048, 4096, 8192} × N=128 ×
K ∈ {4096, 8192, 16384} × {bf16, fp16}.  TB-native (persistent_matmul_lt,
dispatcher bypassed) vs hipBLASLt (torch.matmul with hipblaslt backend)
recorded:

  * 18/18 cells admit at the strict ≥1.05 winner gate (cohort gmean
    tb/hbl = 1.680×, range 1.155×–2.268×, 0 regressions).
  * Adjacency-band drift sentinel (6 cells, N=64 + N=160 at K=8192) shows
    HBL/oracle gmean = 0.995× with 6/6 cells ≥ 0.95 — no regression.
  * Post-patch live oracle vs HBL on the 18 P34 cells: gmean HBL/oracle
    = 1.000×, 18/18 routed to HBL, 18/18 ≥ 0.95 parity.

All 18 cells are intentional ALIAS-OVERLAP entries — they already fire via
upstream slots in the dispatch chain:

  * R-K979 P5 Clause-3 (5th slot, bf16-only): catches all 9 bf16 cells via
    minMN ≤ 192 ∧ K ≥ 2048.
  * K-1367 P13 N=128 (7th slot): catches all 18 cells (M ∈ {2048,4096,8192}
    × N=128 × K ∈ {4096,8192,16384} × {bf16, fp16}) by strict equality.
  * K-1673 P28 N=128 alias-stack (19th slot): full alias-overlap with P13
    on the same 18 cells (the K-1673 30-cell envelope contains them).

P34 is therefore the K-1810/K-1817-style verified-winner AUDIT HANDLE for
the wave-aligned N=128 rung — it mirrors the P32 (N=160) and P33 (N=224)
frozenset shape exactly so the K-1794-style cohort drift sentinel can pin
a single named handle for N=128 alongside its sibling-N rungs.  Per the
K-1810 alias-overlap discipline (where 9 bf16 cells overlapped P5 by
design), full alias-overlap is acceptable when the slot is shipped as a
verified-winner audit handle rather than a load-bearing route-OUT.

Mechanism (K-913 §3 / R-K1673 dtype-invariance): BLOCK_N=128 packs N=128
into a single wave-aligned K-block column but the K-913 §3 LDS-bank-
conflict fingerprint (SQ_LDS_BANK_CONFLICT/inst = 1.45–2.13 cyc/inst per
K-1673 RCA, vs HBL = 0.000) still dominates persistent_matmul; tile-
reshape cannot trade for atomic-reduction.  hipBLASLt's split-K kernel
selection clears the band, dtype-invariant per R-K1673 (the bank-
arbitration topology lives below the dtype lane mux).  Same mechanism
already productionized at K-1673 P28 (N=128 K-COMPLEMENT alias-stack),
K-1810 P32 (N=160), K-1775 P31 (N=256), and K-1817 P33 (N=224); now
applied to the wave-aligned N=128 mid-K rung as the 25th-slot audit
handle.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775 / R-1817):

  1. Cardinality is exactly 18.
  2. N axis is exactly {128}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Per-shape dtype-mirror (bf16 ↔ fp16 admit sets coincide).
  5. Sibling-N firewall vs every prior verified-winner alias-stack at a
     different N (P31 N=256, P32 N=160, P33 N=224).
  6. P28-N128 ALIAS-CONTAINMENT: P34 ⊆ P28 (the K-1673 30-cell N=128
     K-COMPLEMENT alias-stack already covers all 18 P34 cells; P34 is
     the wave-aligned mid-K subset of P28).
  7. P13-N128 ALIAS-EQUALITY: P34 == K-1367 P13 N=128 18-cell envelope
     (P34 is bit-identical to P13 by construction).
  8. Envelope is exactly the full 18-cell M ∈ {2048,4096,8192} × N=128
     × K ∈ {4096,8192,16384} × {bf16,fp16} grid.
"""
from __future__ import annotations

from tritonblas._route_predicate import (
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29,
    _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
    _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18,
    _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18,
    _P34_SKINNY_N128_KMID_VERIFIED_WIN_18,
)


FZ = _P34_SKINNY_N128_KMID_VERIFIED_WIN_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n128():
    assert {N for (_, N, _, _) in FZ} == {128}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_mirror_is_complete():
    """Each shape (M, N, K) admitted under bf16 is also admitted under fp16
    (and vice versa).  K-913 §3 LDS-bank-conflict is dtype-invariant on the
    column-narrow N=128 tile (same mechanism as K-1673 P28 at N=128 K-extremes,
    K-1810 P32 at N=160, K-1817 P33 at N=224)."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert bf == fp
    assert len(bf) == 9


def test_sibling_n_firewall_vs_p31_n256():
    assert FZ & _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28 == set()
    assert FZ & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == set()


def test_sibling_n_firewall_vs_p32_n160():
    assert FZ & _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18 == set()


def test_sibling_n_firewall_vs_p33_n224():
    """K-1828 P34 (N=128) must be N-axis disjoint from K-1817 P33 (N=224)
    and K-1810 P32 (N=160) — the three slots cover different N-rungs of
    the contiguous K-COMPLEMENT N-ladder verified-winner alias-stack."""
    assert FZ & _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    """N=128 rung must be N-disjoint from the N=64 K-COMPLEMENT rung
    (K-1700 P29) — confirms the sibling-N firewall at the N=64 / N=128
    boundary."""
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """K-1711 P30 covers N ∈ {384, 768, 1536}; P34 (N=128) must be
    N-disjoint."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_p28_alias_containment():
    """K-1673 P28 N=128 K-COMPLEMENT alias-stack (30 cells: K ∈ {2048,
    4096, 8192, 16384, 32768}) STRICTLY CONTAINS the K-1828 P34 mid-K
    subset (18 cells: K ∈ {4096, 8192, 16384}).  P34 is the wave-aligned
    mid-K verified-winner subset of P28; this containment is the alias-
    stack discipline that lets P34 ship as the K-1810/K-1817-style audit
    handle without being load-bearing."""
    assert FZ <= _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30
    assert len(FZ) == 18
    assert len(_K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30) == 30
    # The 12 P28 cells NOT in P34 are exactly the K ∈ {2048, 32768} extremes:
    p28_minus_p34 = _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 - FZ
    assert {K for (_, _, K, _) in p28_minus_p34} == {2048, 32768}
    assert len(p28_minus_p34) == 12


def test_p13_n128_alias_equality():
    """K-1367 P13 N=128 18-cell K-COMPLEMENT envelope is BIT-IDENTICAL to
    K-1828 P34 by construction (same M ∈ {2048,4096,8192} × N=128 ×
    K ∈ {4096,8192,16384} × {bf16,fp16} grid).  P34 is the verified-winner
    audit handle for the same 18 cells; P13 fires at the 7th slot well
    before P34's 25th slot, so the P34 membership check on those 18 cells
    is unreachable while P13 remains enabled — full alias documentation
    per the K-1810 P32 alias-overlap discipline (where 9 bf16 cells
    overlapped R-K979 P5 by design)."""
    assert FZ == _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18


def test_envelope_equals_full_n128_kmid_grid():
    """The verified-winner envelope is exactly the full 18-cell M ∈ {2048,
    4096,8192} × N=128 × K ∈ {4096,8192,16384} × {bf16,fp16} grid — pinned
    to detect any silent contraction or expansion of the K-1828 sub-cohort
    envelope.  Per K-1828 paired n=30 sweep, all 18 cells admit at strict
    ≥1.05 gate (gmean tb/hbl = 1.680×); both dtype rows are wave-aligned
    mid-K winners, mirroring K-1810 P32 and K-1817 P33 frozenset shape."""
    full = frozenset(
        (M, 128, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18
