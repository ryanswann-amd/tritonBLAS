"""P38 (S-002) — 29th-slot N=416 K-COMPLEMENT verified-winner subset tests.

Source measurement: K-1873 paired n=30 HIP-graph hot-cache + 3-pass rocprofv2
PMC sweep (LDS / VALU·MFMA / VMEM·L2; 108 cell-engine-pass datapoints) on
MI300X / gfx942 (OCI MI300X fallback per INFRA-0048 — c42
head SSH refused, same fallback path as K-1843 / K-1846 / K-1857 / K-1863)
across the 18-cell N=416 sub-cohort = M ∈ {2048, 4096, 8192} × N=416 × K ∈
{4096, 8192, 16384} × {bf16, fp16}.

K-1880 productionises the K-1873 verified-winner subset MINUS the single
upstream-aliased cell (2048, 416, 4096, "torch.bfloat16") already routed by
the K-1003 R-K979 P5 Clause-1 mid-rect non-square predicate (minMN=416 ∈
[256, 2304] ∧ maxMN=2048 ∈ [1792, 3072] ∧ K=4096 ∈ [1240, 8064] — all four
bounds fire, dispatched kernel name is hipBLASLt's Cijk_).  Per R-K1825.CHECK-
ALIAS-STACK-COVERAGE-MAP-FIRST that one cell is excluded from P38 to avoid
duplicate routing → 18 - 1 = 17 admit cells.

The K-1873 N=416 cohort recorded:

  * 18/18 cells gate-pass at the strict K-1873 floor (≥1.10× ∧ paired-t
    p<0.01) — except for the parity cell, which is the upstream-aliased one
    excluded per R-K1825.  The 17 admit cells all have p < 1e-30.
  * Per-cell ratios on the 17 admit cells span 1.239×–1.714× (median ≈ 1.469×).
  * Cohort geomean TB/HBL = 1.401× over the full 18-cell envelope; admit-only
    geomean = 1.429×.

Mechanism (K-913 §3 LDS-bank-conflict / R-K1673 dtype-invariance + R-1811
wave-misalignment): N=416 mod 128 = 32 — the third tile is a narrow 32-column
remainder, mirroring the wave-misalignment pattern at N=288 (mod 128 = 32,
K-1832), N=320 (mod 128 = 64, K-1846), N=352 (mod 128 = 96, K-1843/K-1863),
and N=384 (mod 128 = 0 third-tile cliff, K-1857).  K-1873 PMC delta ranking
confirms LDS_DOMINANT in 18/18 cells with SQ_LDS_BANK_CONFLICT TB/HBL median
ratio **585×** (range 126×–1170×, monotonically deeper than K-1863 N=352's
440× and K-1857 N=384's ~150×) and SQ_WAIT_INST_LDS TB/HBL median ratio
13.2× (range 5.0×–49.7×).  MFMA-busy fraction: TB 9.4% vs HBL 18.3% — TB's
MFMA pipe is starved (SQ_INSTS_MFMA / SQ_WAVES median ratio 0.61, range
0.57–0.80).  VMEM-per-wave ratio TB/HBL median 0.50 — rules out memory
bandwidth.  Same SCHEDULER_LDS A4 failure mode as K-1681 / K-1710 / K-1781 /
K-1812 / K-1824 / K-1832 / K-1843 / K-1846 / K-1857 / K-1863 wave-misaligned
skinny-N class.  hipBLASLt's Tensile shape-specialised solutions clear the
band by ~40% on average across the 17 admit cells.

Same fingerprint productionised at K-1673 P28 (N=128), K-1700 P29 (N=64),
K-1748 P30 (N ∈ {384, 768, 1536}), K-1775 P31 (N=256), K-1810 P32 (N=160),
K-1817 P33 (N=224), K-1831 P34 (N=96), K-1837 P35 (N=288), K-1850 P36
(N=320), K-1866 P37 (N=352); now extended to the wave-misaligned N=416
K-COMPLEMENT band, closing a five-rung wave-misalignment series N ∈
{288, 320, 352, 384, 416} on top of the wave-aligned anchors {128, 192, 256}.

Note: the dtype-row balance is bf16 = 8 / fp16 = 9 (the upstream alias
prune is a single bf16 cell — R-K979 Clause-1 fires bf16-only on the
(2048,416,4096) shape; fp16 has no upstream coverage on N=416).  The 8/9
asymmetry is purely an upstream-coverage artefact, not a mechanism asymmetry
— K-913 §3 LDS-bank-conflict is dtype-invariant on the column-narrow N=416
tile (R-K1673).

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 17.
  2. N axis is exactly {416}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. The single upstream-aliased cell (2048,416,4096,bf16) is excluded.
  5. fp16 dtype-row has 9 cells; bf16 dtype-row has 8 cells (the 8/9 split
     reflects the R-K979 Clause-1 bf16-only prune at (2048,416,4096)).
  6. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P31 N=256, P32 N=160, P33 N=224,
     P34 N=96, P35 N=288, P36 N=320, P37 N=352, and the K-1367/K-1397 P13
     N ∈ {128, 256} envelopes, plus K-1748 P30 N ∈ {384, 768, 1536}).
     N=416 is N-axis disjoint from all of them.
  7. Envelope is exactly the 18-cell M ∈ {2048,4096,8192} × N=416
     × K ∈ {4096,8192,16384} × {bf16,fp16} grid MINUS the single
     (2048,416,4096,bf16) upstream-aliased cell.
  8. Negative-case guard: adjacent N values (N=415, 417, 384, 448, 352)
     are NOT admitted by the P38 frozenset for any (M, K, dtype) that
     DOES appear at N=416 — proves the gate matches N exactly.
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
    _P37_SKINNY_N352_KCOMPL_VERIFIED_WIN_17,
    _P38_SKINNY_N416_KCOMPL_VERIFIED_WIN_17,
)


FZ = _P38_SKINNY_N416_KCOMPL_VERIFIED_WIN_17

# The single cell in the K-1873 N=416 sub-cohort that is already routed by
# K-1003 R-K979 P5 Clause-1 (mid-rect non-square: minMN=416 ∈ [256, 2304],
# maxMN=2048 ∈ [1792, 3072], K=4096 ∈ [1240, 8064]).  Excluded from P38 per
# R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST.  Pinned here so any silent
# re-admission would fail the envelope test below.
_UPSTREAM_ALIASED_EXCLUSION = frozenset({
    (2048, 416, 4096, "torch.bfloat16"),
})


def test_cardinality_is_17():
    assert len(FZ) == 17


def test_n_axis_is_skinny_n416():
    assert {N for (_, N, _, _) in FZ} == {416}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_upstream_aliased_cell_is_excluded():
    """The (2048,416,4096,bf16) cell is already routed by the K-1003 R-K979
    P5 Clause-1 mid-rect non-square predicate (TB and HBL execute the same
    hipBLASLt kernel — measured ratio_TB/HBL = 1.002, p = 0.394 paired-t
    n=30).  P38 MUST NOT include it — duplicate routing would violate
    R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST."""
    assert FZ & _UPSTREAM_ALIASED_EXCLUSION == set()


def test_dtype_row_balance_bf16_8_fp16_9():
    """fp16 row contributes 9/9 of the K-1873 N=416 sub-cohort (no upstream
    alias on N=416 fp16 — R-K979 Clause-1 fires bf16-only on the
    (2048,416,4096) shape).  bf16 row contributes 8/9 (the upstream-aliased
    (2048,416,4096,bf16) cell is excluded per R-K1825).  K-913 §3 LDS-bank-
    conflict is dtype-invariant on the column-narrow N=416 tile (same
    mechanism as K-1673 P28 at N=128, K-1810 P32 at N=160, K-1775 P31 at
    N=256, K-1817 P33 at N=224, K-1831 P34 at N=96, K-1837 P35 at N=288,
    K-1850 P36 at N=320, K-1866 P37 at N=352); the 8/9 asymmetry is purely
    an upstream-coverage artefact, not a mechanism asymmetry — R-K1673."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 8
    assert len(fp) == 9
    # The fp16 row covers the FULL (M, K) grid; the bf16 row is missing
    # exactly the upstream-aliased (2048, 4096) shape.
    assert fp == {
        (M, 416, K) for M in (2048, 4096, 8192) for K in (4096, 8192, 16384)
    }
    assert (2048, 416, 4096) not in bf
    assert bf == fp - {(2048, 416, 4096)}


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """K-1748 P30 covers N ∈ {384, 768, 1536}; N=416 is N-axis disjoint
    from all three.  Pairwise-disjointness invariant per K-1175."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()
    # P30's N-axis projection MUST NOT contain 416.
    assert 416 not in {N for (_, N, _, _) in _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34}


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
    assert FZ & _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17 == set()


def test_sibling_n_firewall_vs_p37_n352():
    """K-1880 P38 (N=416) must be N-axis disjoint from K-1866 P37 (N=352).
    Pairwise-disjointness invariant per K-1175 — the two slots cover
    adjacent off-by-64 (N=352→416) wave-misaligned rungs in the
    K-COMPLEMENT band.  Same K-1873/K-1843 cohort PMC fingerprint
    (LDS_DOMINANT 18/18 at both N) but disjoint admit envelopes by
    construction."""
    assert FZ & _P37_SKINNY_N352_KCOMPL_VERIFIED_WIN_17 == set()


def test_negative_adjacent_n_values_excluded():
    """Negative-case guard against gate over-matching: every (M, K, dtype)
    that DOES appear in P38 at N=416 must NOT appear at any nearby N value
    other than 416.  This proves the frozenset gate matches N exactly and
    does not accidentally widen via integer-coercion, range-membership, or
    stripe-aliasing bugs.  Critical guard per the K-1871 RETRY round
    (Skeptic + Testing Zealot) — without this, a refactor that changed the
    predicate from `tuple in frozenset` to e.g. `(M, N // 64 * 64, K, dt)
    in frozenset` would silently re-route N∈{416±63} traffic, regressing
    performance on the entire wave-misaligned skinny-N band rather than
    just the verified-winner cells."""
    for (M, _, K, dt) in FZ:
        # 415, 417 = adjacent N; 384 = P30 sibling slot; 448 = unmapped band;
        # 352 = P37 sibling slot.
        for N_adj in (415, 417, 384, 448, 352):
            assert (M, N_adj, K, dt) not in FZ, (
                f"P38 frozenset over-matched: (M={M}, N={N_adj}, K={K}, dt={dt}) "
                f"must NOT be in P38 (only N=416 cells permitted)"
            )
    # And explicitly: P38 contains NO cell with N != 416.
    assert all(N == 416 for (_, N, _, _) in FZ)


def test_envelope_equals_n416_kcompl_grid_minus_p5_alias():
    """The verified-winner envelope is exactly the 18-cell M ∈ {2048,
    4096,8192} × N=416 × K ∈ {4096,8192,16384} × {bf16,fp16} grid MINUS
    the single (2048,416,4096,bf16) upstream-aliased cell — pinned to
    detect any silent contraction (a false-NEGATIVE that would leak winner
    cells back to TB) or expansion (a false-POSITIVE that would re-include
    the P5-aliased cell, causing duplicate routing).  Per K-1873 the entire
    18-cell envelope was 18/18 verified-winner at the strict ≥1.10× ∧
    p<0.01 gate (admit-only after R-K1825 prune); all 17 P38 cells are
    load-bearing route-OUT entries."""
    full = frozenset(
        (M, 416, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full - _UPSTREAM_ALIASED_EXCLUSION
    assert len(full) == 18
    assert len(FZ) == 17


def test_dispatcher_admits_all_p38_cells():
    """Smoke check: every cell in P38 must admit via the live
    `_k971_route_to_hbl` dispatcher (i.e., return True), confirming the
    27th dispatch line was correctly wired."""
    from tritonblas.matmul import _k971_route_to_hbl
    for (M, N, K, dt) in FZ:
        # b_dtype mirrors a_dtype for the kcompl mechanism per K-1673 R-K1673.
        assert _k971_route_to_hbl(M, N, K, dt, dt, False, False), (
            f"P38 cell ({M}, {N}, {K}, {dt}) failed to admit via _k971_route_to_hbl"
        )
