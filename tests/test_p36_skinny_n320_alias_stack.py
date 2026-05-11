"""P36 (S-002) — 27th-slot N=320 K-COMPLEMENT verified-winner subset tests.

Source measurement: K-1843 paired n=30 HIP-graph hot-cache + 3-pass rocprofv2
PMC sweep (LDS / VALU·MFMA / VMEM·L2; 108 cell-engine-pass datapoints) on
MI300X / gfx942 (OCI MI300X fallback per INFRA-0048) across the 36-cell
N ∈ {320, 352} sub-cohort = M ∈ {2048, 4096, 8192} × N ∈ {320, 352} ×
K ∈ {4096, 8192, 16384} × {bf16, fp16}.  K-1850 productionises the N=320
half (18 cells minus 1 already-routed cell = 17 admit cells).

The N=320 slice recorded:

  * bf16 N=320 (8/9 admit): 8 cells gate-pass at the strict ≥1.05 ∧ p<0.05
    floor; 1 cell — (2048, 320, 4096, "torch.bfloat16") — landed at
    ratio_TB/HBL = 1.0001, p = 0.293 inside the 0.95× parity band because
    it is already routed by an upstream alias-stack slot (TB and HBL paths
    converge on the same kernel).  Per
    R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST that cell is excluded from
    P36 to avoid duplicate routing.
  * fp16 N=320 (9/9 admit): all 9 cells gate-pass; no upstream alias
    coverage at the off-by-64 wave-misaligned N=320 rung.  All 9 are
    load-bearing route-OUT entries (closes the dtype-mirror gap with the
    8 bf16 cells above the (2048, 320, 4096) exclusion).

Per-N geomean TB/HBL = 1.545× (range 1.326×–1.911×, 17/17 admit at the
strict ≥1.05 ∧ p<0.05 gate after the upstream-alias exclusion).

Mechanism (K-913 §3 / R-K1673 dtype-invariance + R-1811 wave-misalignment):
BLOCK_N=128 packs N=320 into wave-misaligned K-block columns (off-by-64 N
rung above N=256, two-and-a-half BLOCK_N tiles per N-row).  K-1843 PMC
delta ranking confirms LDS_DOMINANT in 36/36 cells of the N ∈ {320, 352}
cohort with lds_wait_ratio_TB/HBL spanning 1.96×–25.27× (median ≈ 8.5×),
MFMA per-wave ratio everywhere ≤ 0.96× (TB does *less* MFMA per wave; not
the limiter), VMEM per-wave ratio mostly ≤ 1.0 (rules out memory-bandwidth
as the gap mechanism).  Same SCHEDULER_LDS A4 failure mode as the K-1681 /
K-1710 / K-1781 / K-1812 / K-1824 / K-1832 wave-misaligned skinny-N class.
persistent_matmul cannot trade tile reshape for atomic-reduction;
hipBLASLt's split-K kernel selection clears the band by ~55% on average.
Same fingerprint productionised at K-1673 P28 (N=128), K-1700 P29 (N=64),
K-1748 P30 (N ∈ {384, 768, 1536}), K-1775 P31 (N=256), K-1810 P32 (N=160),
K-1817 P33 (N=224), K-1831 P34 (N=96), K-1837 P35 (N=288); now applied to
the off-by-64 wave-misaligned N=320 rung between the N=256 P31 cliff and
the N=384 P30 rung.

Note: unlike K-1817 P33 (N=224), K-1831 P34 (N=96), K-1837 P35 (N=288) —
which had NO upstream alias overlap — K-1850 P36 (N=320) has ONE
upstream-routed cell (2048, 320, 4096, bf16); both dtype rows are still
load-bearing because (a) the fp16 row is fully load-bearing 9/9, and (b)
the bf16 row contributes 8 load-bearing cells outside the upstream alias.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 17.
  2. N axis is exactly {320}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. The single upstream-aliased cell (2048, 320, 4096, bf16) is excluded.
  5. fp16 dtype-row is complete (9/9); bf16 dtype-row has 8 cells (9 minus
     the upstream-aliased exclusion).
  6. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, P32 N=160, P33 N=224, P34 N=96, P35 N=288, and the
     K-1367/K-1397 P13 N ∈ {128, 256} envelopes).
  7. Envelope is exactly the 18-cell M ∈ {2048,4096,8192} × N=320
     × K ∈ {4096,8192,16384} × {bf16,fp16} grid MINUS the single
     (2048, 320, 4096, bf16) upstream alias.
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
    _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18,
    _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18,
    _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17,
)


FZ = _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17

# The single cell in the K-1843 N=320 sub-cohort that is already routed by an
# upstream alias-stack slot, hence excluded from P36 per R-K1825.CHECK-ALIAS-
# STACK-COVERAGE-MAP-FIRST.  Pinned here so that any silent re-admission would
# fail the envelope test below.
_UPSTREAM_ALIASED_EXCLUSION = frozenset({
    (2048, 320, 4096, "torch.bfloat16"),
})


def test_cardinality_is_17():
    assert len(FZ) == 17


def test_n_axis_is_skinny_n320():
    assert {N for (_, N, _, _) in FZ} == {320}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_upstream_aliased_cell_is_excluded():
    """The (2048, 320, 4096, bf16) cell measured ratio_TB/HBL=1.0001 / p=0.293
    in K-1843's paired-n30 sweep on the LIVE oracle (it is already routed by
    an upstream alias-stack slot).  P36 MUST NOT include it — duplicate
    routing would violate R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST."""
    assert FZ & _UPSTREAM_ALIASED_EXCLUSION == set()


def test_dtype_row_balance():
    """fp16 row is complete (9/9 of the K-1843 N=320 sub-cohort).  bf16 row
    contributes 8 cells (9 minus the (2048, 320, 4096) upstream-aliased
    exclusion).  K-913 §3 LDS-bank-conflict is dtype-invariant on the
    column-narrow N=320 tile (same mechanism as K-1673 P28 at N=128, K-1810
    P32 at N=160, K-1775 P31 at N=256, K-1817 P33 at N=224, K-1831 P34 at
    N=96, K-1837 P35 at N=288); the 8/9 vs 9/9 asymmetry is purely an
    upstream-coverage artefact, not a mechanism asymmetry."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 8
    assert len(fp) == 9
    # The fp16 admits are a strict superset of the bf16 admits — bf16's
    # missing shape (2048, 320, 4096) is exactly the upstream-aliased one.
    assert bf <= fp
    assert fp - bf == {(2048, 320, 4096)}


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """P30 covers N ∈ {384, 768, 1536}; P36 covers N=320 — disjoint by
    natural N-axis separation.  P36 sits BETWEEN the N=256 P31 cliff and
    the N=384 P30 rung (off-by-64 rung above N=256)."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p31_n256():
    assert FZ & _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28 == set()
    assert FZ & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == set()


def test_sibling_n_firewall_vs_p32_n160():
    assert FZ & _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18 == set()


def test_sibling_n_firewall_vs_p33_n224():
    assert FZ & _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18 == set()


def test_sibling_n_firewall_vs_p34_n96():
    assert FZ & _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18 == set()


def test_sibling_n_firewall_vs_p35_n288():
    """K-1850 P36 (N=320) must be N-axis disjoint from K-1837 P35 (N=288).
    Pairwise-disjointness invariant per K-1175 — the two slots cover
    adjacent off-by-32 (N=288) and off-by-64 (N=320) wave-misaligned rungs
    above the N=256 P31 cliff.  Same K-1843 cohort PMC fingerprint
    (LDS_DOMINANT 36/36) but disjoint admit envelopes by construction."""
    assert FZ & _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18 == set()


def test_envelope_equals_n320_kcompl_grid_minus_upstream_alias():
    """The verified-winner envelope is exactly the 18-cell M ∈ {2048,
    4096,8192} × N=320 × K ∈ {4096,8192,16384} × {bf16,fp16} grid MINUS
    the single (2048, 320, 4096, bf16) upstream-aliased cell — pinned to
    detect any silent contraction (a false-NEGATIVE that would leak winner
    cells back to TB) or expansion (a false-POSITIVE that would re-include
    the upstream-aliased cell) of the K-1843 sub-cohort envelope.  Per
    K-1843 the N=320 column was 17/18 verified-winner with 1/18 inside
    the 0.95× parity band (already-routed); all 17 are load-bearing
    route-OUT entries."""
    full = frozenset(
        (M, 320, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full - _UPSTREAM_ALIASED_EXCLUSION
    assert len(full) == 18
    assert len(FZ) == 17
