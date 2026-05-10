"""K-2130 P54 (S-002) — 10th-rung N=1456 K-COMPLEMENT alias-stack tests.

Source measurement: K-2130 paired n=30 HIP-graph hot-cache MI300X / gfx942
4-engine bench (TB BEFORE / TB AFTER / hipBLASLt / rocBLAS) over the 18-cell
sub-cohort M ∈ {2048, 4096, 8192} × N=1456 × K ∈ {4096, 8192, 16384} ×
{bf16, fp16}, branched off the LIVE oracle base
ryanswann-amd/tritonBLAS:fix/K-1922@0024a71.

Result: 17/18 cells gate-pass at the strict admit gate (AFTER ≤ 1.0526× HBL
∧ no individual cell < 0.95× HBL).  The single non-admit corner (M=2048,
K=16384, bf16) measured AFTER/HBL = 1.1054× — co-confirmed by rocBLAS at
1.1115× HBL on the SAME cell (so it is an LT-confound on the host hipBLASLt
path, not a TB-AFTER regression) — but is excluded from the routed set per
the explicit reviewer mandate to either re-bench with higher n to disprove
variance or shrink the frozenset to 17 cells; the latter path is taken
here so the strict admit gates pass honestly without a hand-wave.

N=1456 is the 10th contiguous rung of the off-by-48 wave-misaligned ladder
(816/880/944/1008/1072/1136/1200/1264/1328/1392 → 1456) — same residue
class (N mod 64 == 48) as every prior rung; mod-128 lands on 48 (the 6th
mod-128=48 member of this ladder, returning to the P45/P46/P48/P50/P52
mod-128 sub-class).

Mechanism (K-913 §3 / R-K1673 dtype-invariance + R-1811 wave-misalignment):
BLOCK_N=128 packs N=1456 into 11.375 BLOCK_N tiles per N-row (vs N=1392's
10.875 / N=1328's 10.375 / N=1264's 9.875 — the fractional-tile pattern is
the same SCHEDULER_LDS A4 failure mode as every prior off-by-48 rung).
Per K-2061 PMC delta-attribution, residue-48 carries a stable LDS bank-
conflict differential (~6×) vs residue-32 (~2×) and wave-aligned (~1×) at
(M=4096, K=8192, bf16); per K-2107 F-K2107.OFF-BY-48-LDS-BC-6X-VS-WAVE-
ALIGNED-3X-DIFFERENTIAL-OVER-OFF-BY-32, the K-2031/K-2041 plateau-decay
forecast is empirically falsified for 3 consecutive rungs and N=1456 sits
squarely inside the climbing regime.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 17 (full-grid 18 minus the explicitly-excluded
     LT-confound corner (M=2048, K=16384, bf16)).
  2. N axis is exactly {1456}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Dtype-row balance is 8/9 (bf16 has 8 cells, fp16 has 9), reflecting
     the single excluded bf16 corner; both rows remain strictly load-bearing.
  5. The excluded cell is exactly (M=2048, N=1456, K=16384, bf16) — pinned
     by an explicit non-membership assertion so a silent re-admit (e.g. via
     a careless edit re-adding the dropped tuple to the literal frozenset)
     is caught at test time.
  6. Positive-membership critical-path assertions on a representative
     spread of admitted cells (one bf16 + one fp16 per M-axis value,
     spanning the K-axis) so a silent contraction of the literal tuple
     list is caught from the inside, not just the outside.
  7. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, the consolidated P32–P38 N ∈ {96, 160, 224, 288, 320, 352,
     384}, K-1367/K-1397 P13 N ∈ {128, 256}, and K-1922 P40 N=544).
  8. Sibling-N firewall vs the adjacent rungs of THIS off-by-48 ladder
     (N=1392 prior K-2107 rung, N=1520 next +64 step) — both share the
     `N mod 64 == 48` residue class as N=1456, so a band-keyed typo would
     not be caught by mod-64 / N-axis assertions alone.
  9. Off-by-48 wave-misalignment band membership: 1456 mod 64 == 48
     (same residue class as the 9 prior rungs of this contiguous
     ladder — distinct N-axis rung, same mechanism).
"""
from __future__ import annotations

from tritonblas._route_predicate import (
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29,
    _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34,
    _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120,
    _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
    _K2130_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_17,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
)


FZ = _K2130_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_17
EXCLUDED_CELL = (2048, 1456, 16384, "torch.bfloat16")


def test_cardinality_is_17():
    assert len(FZ) == 17


def test_n_axis_is_skinny_n1456():
    assert {N for (_, N, _, _) in FZ} == {1456}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_row_balance_8_over_9():
    """Dtype-row balance reflects the single excluded bf16 corner: bf16 has
    8 cells (3M × 3K - 1 excluded), fp16 has the full 9 cells (3M × 3K).
    Pin the asymmetry explicitly so a future edit that drops the wrong cell
    or expands back to 9/9 without re-validating the gate is caught."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 8
    assert len(fp) == 9
    # The fp16 row covers the full M×K grid; bf16 row is missing exactly
    # the one excluded (M, K) corner.
    assert fp - bf == {(2048, 1456, 16384)}
    assert bf - fp == set()


def test_excluded_cell_is_explicit_lt_confound_corner():
    """The single excluded cell is exactly (M=2048, N=1456, K=16384, bf16)
    — the LT-confound corner (rationale lives in REPORT.md, not the
    dispatcher hot path).  This invariant catches a silent re-admit (e.g.
    via a careless edit that re-adds the dropped tuple to the literal
    frozenset)."""
    assert EXCLUDED_CELL not in FZ


def test_positive_membership_admitted_cells():
    """Positive critical-path assertion: every admitted cell of the 17-cell
    envelope must literally be in FZ.  Pin a few representative cells (one
    per M-axis value, both dtypes, spanning the K-axis) so a future edit
    that silently drops a cell from the literal tuple list — for example,
    deleting the wrong row by accident — is caught at test time rather
    than as a silent route-OUT regression on production traffic."""
    # Spot-check coverage: one bf16 + one fp16 per M, spanning K-axis.
    admitted = [
        (2048, 1456,  4096, "torch.bfloat16"),
        (2048, 1456,  8192, "torch.float16"),
        (4096, 1456,  8192, "torch.bfloat16"),
        (4096, 1456, 16384, "torch.float16"),
        (8192, 1456,  4096, "torch.bfloat16"),
        (8192, 1456, 16384, "torch.float16"),
        # The (M=2048, K=16384, fp16) cell is the dtype-twin of the
        # excluded bf16 corner — admitted (fp16 is not LT-confounded
        # at this corner) and explicitly pinned so the dtype-row
        # asymmetry stays load-bearing in BOTH directions.
        (2048, 1456, 16384, "torch.float16"),
    ]
    for cell in admitted:
        assert cell in FZ, f"admitted cell {cell} missing from FZ"


def test_sibling_n_firewall_vs_n1392_and_n1520():
    """Sibling-N firewall on the off-by-48 ladder itself: the immediately
    adjacent rungs N=1392 (the prior K-2107 P53 rung) and N=1520 (the next
    +64 step) must NOT appear in the K-2130 envelope.  Both share the same
    `N mod 64 == 48` residue class as N=1456, so a band-keyed typo (e.g.
    accidentally substituting one band-mate N-literal for another) would
    not be caught by mod-64 / N-axis-set assertions alone — only an
    explicit non-membership assertion against the adjacent rungs catches
    it.  This locks the 17-cell envelope from BOTH directions: positive
    membership (admitted cells in) above, and band-mate N-firewall (sibling
    N out) here."""
    sibling_ns = (1392, 1520)
    for sibling_n in sibling_ns:
        assert sibling_n % 64 == 48, (
            f"sibling N={sibling_n} must be in the off-by-48 band for this "
            f"firewall test to be load-bearing")
        for M in (2048, 4096, 8192):
            for K in (4096, 8192, 16384):
                for dt in ("torch.bfloat16", "torch.float16"):
                    assert (M, sibling_n, K, dt) not in FZ, (
                        f"sibling-N cell {(M, sibling_n, K, dt)} leaked "
                        f"into the K-2130 N=1456 envelope")


def test_envelope_equals_full_n1456_kcompl_grid_minus_excluded():
    """The verified-winner envelope is exactly the 18-cell M ∈ {2048,
    4096,8192} × N=1456 × K ∈ {4096,8192,16384} × {bf16,fp16} grid MINUS
    the single LT-confound corner — pinned to detect any silent contraction
    (a false-NEGATIVE that would leak winner cells back to TB) or expansion
    (a false-POSITIVE that would re-admit the LT-confound corner) of the
    K-2130 verification envelope."""
    full = frozenset(
        (M, 1456, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    expected = full - {EXCLUDED_CELL}
    assert FZ == expected
    assert len(expected) == 17
    # Symmetric-difference is exactly the excluded cell.
    assert (full ^ FZ) == {EXCLUDED_CELL}


def test_off_by_48_wave_misalignment_band_membership():
    """N=1456 sits in the off-by-48 wave-misaligned band (N mod 64 == 48),
    the same residue class as every prior rung of this contiguous ladder
    (P45 N=816, N=880, P46 N=944, P47 N=1008, P48 N=1072, N=1136, P50
    N=1200, P51 N=1264, P52 N=1328, P53 N=1392).  This invariant guards
    against a silent N-axis typo (e.g. 1454, 1458) that would land on a
    different BLOCK_N=128 alignment rung and thus reflect a different
    mechanism.  Note: mod-128 == 48 (the 6th mod-128=48 member of the
    ladder — alternates 48/80/112 across the ladder rungs), and mod-64
    == 48 is the binding mechanism invariant per K-2061 PMC delta-
    attribution."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 64 == 48
    assert only_n % 128 == 48
    assert only_n == 1456


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """P30 covers N ∈ {384, 768, 1536}; P54 covers N=1456 — disjoint by
    natural N-axis separation (P54 sits between the P30 N=768 and N=1536
    rungs, just below the N=1536 anchor by 80 N-units)."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p31_n256():
    assert FZ & _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28 == set()
    assert FZ & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == set()


def test_sibling_n_firewall_vs_consolidated_p32_p38():
    """K-1881 consolidated P32–P38 covers N ∈ {96, 160, 224, 288, 320,
    352, 384}; K-2130 P54 covers N=1456 — disjoint by natural N-axis
    separation."""
    assert FZ & _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120 == set()


def test_sibling_n_firewall_vs_p40_n544():
    """K-1922 P40 covers N=544 (off-by-32 band); K-2130 P54 covers N=1456
    (off-by-48 band) — disjoint by both N-axis and residue class."""
    assert FZ & _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18 == set()
