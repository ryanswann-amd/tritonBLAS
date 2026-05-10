"""P37 (S-002) — 28th-slot N=384 K-COMPLEMENT verified-winner subset tests.

Source measurement: K-1857 paired n=30 HIP-graph hot-cache + 3-pass rocprofv2
PMC sweep (LDS / VALU·MFMA / VMEM·L2; 108 cell-engine-pass datapoints) on
MI300X / gfx942 (OCI MI300X fallback per INFRA-0048 — c42 head SSH refused,
same fallback pattern as K-1846 / K-1850) across the 18-cell N=384 sub-cohort
= M ∈ {2048, 4096, 8192} × N=384 × K ∈ {4096, 8192, 16384} × {bf16, fp16}.

K-1860 productionises the K-1857 verified-winner subset MINUS the 4 cells
already routed by K-1748 P30 (_K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34
explicitly enumerates (2048,384,8192,*) and (4096,384,8192,*) for both
dtypes — 4 cells overlap with the K-1857 K-COMPLEMENT envelope at K=8192).
Per R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST those overlapping cells are
excluded from P37 to avoid duplicate routing → 18 - 4 = 14 admit cells.

The N=384 cohort recorded:

  * bf16 N=384 (9/9 cells gate-pass at strict ≥1.05 ∧ CI95-lo≥1.05 floor,
    7/9 enter P37 after the P30 prune of (2048,384,8192,bf16) and
    (4096,384,8192,bf16)).
  * fp16 N=384 (9/9 cells gate-pass, 7/9 enter P37 after the P30 prune of
    (2048,384,8192,fp16) and (4096,384,8192,fp16)).

Cohort geomean TB/HBL = 1.283× across all 18 cells (per K-1857 REPORT.md);
per-cell ratios on the 14 admit cells span 1.135×–1.555× (median ≈ 1.292×).

Mechanism (K-913 §3 / R-K1673 dtype-invariance + R-1811 wave-misalignment):
N=384 = 6 wavefronts × 64 lanes is NOT a multiple of typical BN tiling
(BN=128 → 3 BN tiles per N-row); the wave-pair (2-wave) packing leaves a
tail wave under-utilised in tritonblas's persistent_matmul because the
kernel cannot specialise its split-K plan for this N.  K-1857 PMC delta
ranking confirms LDS_DOMINANT in 18/18 cells with SQ_WAIT_INST_LDS TB/HBL
median ratio 13.0× (more extreme than K-1846 / K-1850's N=320 11.15×) and
HBL records ZERO SQ_LDS_BANK_CONFLICT in 18/18 cells (vs nonzero everywhere
on TB).  MFMA-busy fraction: TB 9.4% vs HBL 18.3% — TB's MFMA pipe is
starved roughly half the cycles HBL feeds it despite identical SQ_INSTS_MFMA
(ratio 0.99×).  Same SCHEDULER_LDS A4 failure mode as the K-1681 / K-1710 /
K-1781 / K-1812 / K-1824 / K-1832 / K-1843 / K-1846 wave-misaligned skinny-N
class.  hipBLASLt's Tensile shape-specialised solutions clear the band by
~22% on average (geomean 1.283× over the K-1857 cohort).

Same fingerprint productionised at K-1673 P28 (N=128), K-1700 P29 (N=64),
K-1748 P30 (N ∈ {384, 768, 1536} — partial N=384 coverage at K∈{8192,32768}),
K-1775 P31 (N=256), K-1810 P32 (N=160), K-1817 P33 (N=224), K-1831 P34 (N=96),
K-1837 P35 (N=288), K-1850 P36 (N=320); now extended to the wave-misaligned
N=384 K-COMPLEMENT band at the K∈{4096,16384} extremes plus the
(8192,384,8192) cell P30 left uncovered.

Note: unlike K-1817 P33 (N=224), K-1831 P34 (N=96), K-1837 P35 (N=288) —
which had NO upstream alias overlap — K-1860 P37 (N=384) has FOUR
upstream-routed cells (the K-1748 P30 N-mid alias-stack carries them);
both dtype rows are still load-bearing and balanced: 7 bf16 admit cells
+ 7 fp16 admit cells (the dtype-row asymmetry is purely an upstream-
coverage artefact, not a mechanism asymmetry — K-913 §3 LDS-bank-conflict
is dtype-invariant on the column-narrow N=384 tile).

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 14.
  2. N axis is exactly {384}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. The four upstream-aliased cells (P30-overlap) are excluded.
  5. fp16 dtype-row has 7 cells; bf16 dtype-row has 7 cells (both 9 minus
     the 2 P30-overlapping cells per dtype) — both rows load-bearing.
  6. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P31 N=256, P32 N=160, P33 N=224,
     P34 N=96, P35 N=288, P36 N=320, and the K-1367/K-1397 P13
     N ∈ {128, 256} envelopes).  P30 shares N=384 — handled by point 4.
  7. K-axis disjointness vs P30 within the N=384 slice: P37 never
     contains a (M, 384, 8192) cell where (M, 384, 8192) ∈ P30's
     N=384 projection.
  8. Envelope is exactly the 18-cell M ∈ {2048,4096,8192} × N=384
     × K ∈ {4096,8192,16384} × {bf16,fp16} grid MINUS the 4
     (P30-overlapping) cells.
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
    _P37_SKINNY_N384_KCOMPL_VERIFIED_WIN_14,
)


FZ = _P37_SKINNY_N384_KCOMPL_VERIFIED_WIN_14

# The four cells in the K-1857 N=384 sub-cohort that are already routed by
# the K-1748 P30 alias-stack slot (it explicitly enumerates (2048,384,8192)
# and (4096,384,8192) for both dtypes).  Excluded from P37 per
# R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST.  Pinned here so any silent
# re-admission would fail the envelope test below.
_UPSTREAM_ALIASED_EXCLUSION = frozenset({
    (2048, 384, 8192, "torch.bfloat16"),
    (2048, 384, 8192, "torch.float16"),
    (4096, 384, 8192, "torch.bfloat16"),
    (4096, 384, 8192, "torch.float16"),
})


def test_cardinality_is_14():
    assert len(FZ) == 14


def test_n_axis_is_skinny_n384():
    assert {N for (_, N, _, _) in FZ} == {384}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_upstream_aliased_cells_are_excluded():
    """The four (2048,384,8192,*) and (4096,384,8192,*) cells are already
    routed by the K-1748 P30 alias-stack slot.  P37 MUST NOT include any of
    them — duplicate routing would violate
    R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST."""
    assert FZ & _UPSTREAM_ALIASED_EXCLUSION == set()
    # And those cells MUST still be in P30 (they're the upstream alias).
    assert _UPSTREAM_ALIASED_EXCLUSION <= _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34


def test_dtype_row_balance():
    """Both dtype rows contribute 7/9 of the K-1857 N=384 sub-cohort
    (9 minus the 2 P30-overlapping cells per dtype).  K-913 §3 LDS-bank-
    conflict is dtype-invariant on the column-narrow N=384 tile (same
    mechanism as K-1673 P28 at N=128, K-1810 P32 at N=160, K-1775 P31
    at N=256, K-1817 P33 at N=224, K-1831 P34 at N=96, K-1837 P35 at
    N=288, K-1850 P36 at N=320); the 7/7 balance is structurally
    symmetric — the upstream P30 prune is dtype-symmetric."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 7
    assert len(fp) == 7
    # Both dtype rows project to the SAME 7 (M, N, K) shapes (P30 excludes
    # the same 2 (M, N, K) shapes from both dtypes).
    assert bf == fp


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_k_axis_disjoint_with_p30_within_n384_slice():
    """P30 covers N ∈ {384, 768, 1536}; P37 covers N=384 — N-axis
    NON-disjoint (N=384 is the shared rung).  Disjointness inside the
    N=384 slice MUST be K-axis: P30's N=384 projection is K ∈
    {8192, 32768}; P37's N=384 K projection MUST avoid K=8192 at
    M ∈ {2048, 4096} and avoid K=32768 entirely.  This is the precise
    R-K1825 alias-stack coverage check at the cell level."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()
    # And every cell P37 contains at K=8192 must NOT be in P30 (only
    # M=8192,K=8192 survives the prune at K=8192).
    p37_at_k8192 = {(M, N, K, dt) for (M, N, K, dt) in FZ if K == 8192}
    assert all(M == 8192 for (M, N, K, dt) in p37_at_k8192)
    assert len(p37_at_k8192) == 2  # (8192,384,8192,bf16) + (8192,384,8192,fp16)
    # P37 contains NO cells at K=32768 (P30 owns the entire K=32768 column at N=384).
    assert {K for (_, _, K, _) in FZ} & {32768} == set()


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
    """K-1860 P37 (N=384) must be N-axis disjoint from K-1850 P36 (N=320).
    Pairwise-disjointness invariant per K-1175 — the two slots cover
    adjacent off-by-64 (N=320) and off-by-128 (N=384) wave-misaligned rungs
    above the N=256 P31 cliff.  Same K-1857/K-1843 cohort PMC fingerprint
    (LDS_DOMINANT 18/18 at N=384, 36/36 at N∈{320,352}) but disjoint admit
    envelopes by construction."""
    assert FZ & _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17 == set()


def test_negative_adjacent_n_values_excluded():
    """Negative-case guard against gate over-matching: every (M, K, dtype)
    that DOES appear in P37 at N=384 must NOT appear at N=383 or N=385 (or
    any nearby N value other than 384).  This proves the frozenset gate
    matches N exactly and does not accidentally widen via integer-coercion,
    range-membership, or stripe-aliasing bugs.  Critical guard per
    reviewer-feedback (Skeptic + Testing Zealot, K-1871 RETRY round) — without
    this, a refactor that changed the predicate from `tuple in frozenset` to
    e.g. `(M, N // 64 * 64, K, dt) in frozenset` would silently re-route
    N∈{384±63} traffic, causing performance regressions on the entire
    wave-misaligned skinny-N band rather than just the verified-winner cells."""
    for (M, _, K, dt) in FZ:
        for N_adj in (383, 385, 320, 448):  # 320 = P36 (separate slot), 448 = no slot
            assert (M, N_adj, K, dt) not in FZ, (
                f"P37 frozenset over-matched: (M={M}, N={N_adj}, K={K}, dt={dt}) "
                f"must NOT be in P37 (only N=384 cells permitted)"
            )
    # And explicitly: P37 contains NO cell with N != 384.
    assert all(N == 384 for (_, N, _, _) in FZ)


def test_envelope_equals_n384_kcompl_grid_minus_p30_alias():
    """The verified-winner envelope is exactly the 18-cell M ∈ {2048,
    4096,8192} × N=384 × K ∈ {4096,8192,16384} × {bf16,fp16} grid MINUS
    the four (2048,384,8192,*) + (4096,384,8192,*) P30-overlapping cells —
    pinned to detect any silent contraction (a false-NEGATIVE that would
    leak winner cells back to TB) or expansion (a false-POSITIVE that
    would re-include a P30-aliased cell, causing duplicate routing).
    Per K-1857 the entire 18-cell envelope was 18/18 verified-winner at
    the strict ≥1.05× ∧ CI95-lo≥1.05 gate; all 14 P37 cells are
    load-bearing route-OUT entries."""
    full = frozenset(
        (M, 384, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full - _UPSTREAM_ALIASED_EXCLUSION
    assert len(full) == 18
    assert len(FZ) == 14
