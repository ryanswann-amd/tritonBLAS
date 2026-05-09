"""P37 (S-002) — 28th-slot N=352 K-COMPLEMENT verified-winner subset tests.

Source measurement: K-1853 paired n=30 HIP-graph hot-cache + 3-pass rocprofv2
PMC sweep (LDS / VALU·MFMA / VMEM·L2; 108 cell-engine-pass datapoints) on
MI300X / gfx942 (OCI MI300X fallback per INFRA-0048) across the 18-cell
N=352 sub-cohort = M ∈ {2048, 4096, 8192} × N=352 × K ∈ {4096, 8192, 16384}
× {bf16, fp16}.  K-1868 productionises the N=352 half (18 cells, no upstream
alias overlap → all 18 are net-new admit cells), completing the second half
of the K-1843 36-cell N ∈ {320, 352} sub-cohort that K-1850 P36 productionised
the first (N=320) half of.

The N=352 slice recorded:

  * bf16 N=352 (9/9 admit): all 9 cells gate-pass at the strict ≥1.05 ∧
    p<0.05 floor; no upstream alias coverage at the off-by-96 wave-misaligned
    N=352 rung (R-K1825 audit confirms N=352 is N-axis disjoint from every
    prior K-COMPLEMENT slot's N-axis projection: P5, P13, P21, P28-P36).
  * fp16 N=352 (9/9 admit): all 9 cells gate-pass; same R-K1825 result.
    Both dtype rows are fully load-bearing — closes the wave-misaligned
    N=352 dtype-mirror.

Per-N geomean TB/HBL = 1.243× (range 1.066×–1.481×; per-cell paired-CI lo
spans 1.0589–1.4699 and is strictly > 1.05 in all 18 cells, satisfying the
non-overlapping-CI requirement).

Mechanism (K-913 §3 / R-K1673 dtype-invariance + R-1811 wave-misalignment):
BLOCK_N=128 packs N=352 into wave-misaligned K-block columns (off-by-96 N
rung above N=256: 352 mod 128 = 96 — two full BLOCK_N tiles plus a 96-wide
remainder per N-row).  K-1853 PMC delta ranking confirms LDS_DOMINANT in
18/18 cells with SQ_LDS_BANK_CONFLICT TB/HBL median 336× (per-cell range
76×–1920×) and SQ_WAIT_INST_LDS median 10.3×, while SQ_INSTS_MFMA TB/HBL
= 0.99× (identical arithmetic work — TB just stalls ~2× longer waiting on
LDS, with TB MFMA-busy fraction ~9.4% vs HBL ~17.7%).  Same SCHEDULER_LDS
A4 failure mode as the K-1681 / K-1710 / K-1781 / K-1812 / K-1824 / K-1832 /
K-1843 wave-misaligned skinny-N class.  N=352 is pathologically worse than
N=320 for tritonblas's persistent_matmul LDS swizzle: SQ_LDS_BANK_CONFLICT
shows TB/HBL ratios in ALL 18 cells (vs only 6/18 for the K-1846 N=320
column).  hipBLASLt's split-K kernel selection clears the band by ~24% on
average (geomean 1.243×).
Same fingerprint productionised at K-1673 P28 (N=128), K-1700 P29 (N=64),
K-1748 P30 (N ∈ {384, 768, 1536}), K-1775 P31 (N=256), K-1810 P32 (N=160),
K-1817 P33 (N=224), K-1831 P34 (N=96), K-1837 P35 (N=288), K-1850 P36
(N=320); now applied to the off-by-96 wave-misaligned N=352 rung — the
last un-promoted skinny-N rung between the N=256 P31 cliff and the N=384
P30 rung, completing the wave-misaligned skinny-N N ∈ {96, 160, 224, 288,
320, 352} ladder.

Note: unlike K-1850 P36 (N=320), which had ONE upstream-routed cell that
required exclusion, K-1868 P37 (N=352) has NO upstream alias overlap — both
dtype rows contribute the full 9 cells (mirrors K-1817 P33 N=224, K-1831
P34 N=96, K-1837 P35 N=288 alias-overlap-absent discipline).  Cardinality
is therefore the full 18 (not 17).

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18.
  2. N axis is exactly {352}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. NO upstream alias-stack slot overlaps P37 (N=352 is N-axis disjoint
     from every prior K-COMPLEMENT slot — P5, P13, P21, P28-P36).
  5. Both dtype rows are fully load-bearing: bf16 9/9, fp16 9/9 (no
     dtype-row asymmetry, unlike P36's 8/9 bf16 + 9/9 fp16 where one cell
     was upstream-aliased).
  6. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, P32 N=160, P33 N=224, P34 N=96, P35 N=288, P36 N=320, and
     the K-1367/K-1397 P13 N ∈ {128, 256} envelopes).
  7. Envelope is exactly the 18-cell M ∈ {2048,4096,8192} × N=352
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
    _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18,
    _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18,
    _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17,
    _P37_SKINNY_N352_KCOMPL_VERIFIED_WIN_18,
)


FZ = _P37_SKINNY_N352_KCOMPL_VERIFIED_WIN_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n352():
    assert {N for (_, N, _, _) in FZ} == {352}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_no_upstream_alias_overlap():
    """Per R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST audit, no upstream
    alias-stack slot covers any N=352 cell — P37 is the first slot to admit
    the N=352 rung.  Unlike K-1850 P36 (N=320), where (2048, 320, 4096, bf16)
    was upstream-aliased and excluded, P37 has zero upstream-overlap and
    therefore the full 18-cell envelope is admit-eligible."""
    upstream_slots = (
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
    for slot in upstream_slots:
        assert FZ & slot == set(), (
            "P37 must not overlap any upstream K-COMPLEMENT alias-stack slot"
        )


def test_dtype_row_balance_full_9_9():
    """Both dtype rows are fully load-bearing: bf16 9/9 + fp16 9/9.  No
    dtype-row asymmetry (unlike P36's 8/9 + 9/9 which was an upstream-
    coverage artefact, not a mechanism asymmetry).  K-913 §3 LDS-bank-
    conflict is dtype-invariant on the column-narrow N=352 tile, and there
    is no upstream cell to subtract from either row."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    # The bf16 and fp16 admit shape-projections are identical (full-grid).
    assert bf == fp


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """P30 covers N ∈ {384, 768, 1536}; P37 covers N=352 — disjoint by
    natural N-axis separation.  P37 sits BETWEEN the K-1850 P36 N=320 rung
    and the K-1748 P30 N=384 rung (off-by-96 rung above N=256, last
    un-promoted skinny-N rung in the wave-misaligned ladder)."""
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
    assert FZ & _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18 == set()


def test_sibling_n_firewall_vs_p36_n320():
    """K-1868 P37 (N=352) must be N-axis disjoint from K-1850 P36 (N=320).
    Pairwise-disjointness invariant per K-1175 — the two slots cover
    adjacent off-by-64 (N=320) and off-by-96 (N=352) wave-misaligned rungs
    above the N=256 P31 cliff.  Both halves of the K-1843 36-cell N ∈
    {320, 352} sub-cohort have the same K-1843 PMC fingerprint
    (LDS_DOMINANT 36/36) but disjoint admit envelopes by construction."""
    assert FZ & _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17 == set()


def test_envelope_equals_n352_kcompl_full_grid():
    """The verified-winner envelope is exactly the full 18-cell M ∈ {2048,
    4096,8192} × N=352 × K ∈ {4096,8192,16384} × {bf16,fp16} grid — pinned
    to detect any silent contraction (a false-NEGATIVE that would leak
    winner cells back to TB) or expansion (a false-POSITIVE that would
    extend P37 outside the K-1853 measured cohort) of the K-1853 sub-cohort
    envelope.  Per K-1853 the N=352 column was 18/18 verified-winner with
    paired-CI lo strictly > 1.05 in every cell (geomean TB/HBL = 1.243×);
    all 18 are load-bearing route-OUT entries (no upstream alias subtractions
    needed)."""
    full = frozenset(
        (M, 352, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18
    assert len(FZ) == 18
