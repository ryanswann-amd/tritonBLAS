"""K-2107 P53 (S-002) — 9th-rung N=1392 K-COMPLEMENT alias-stack tests.

Source measurement: K-2107 paired n=30 HIP-graph hot-cache MI300X / gfx942
4-engine bench (TB BEFORE / TB AFTER / hipBLASLt / rocBLAS) over the 18-cell
sub-cohort M ∈ {2048, 4096, 8192} × N=1392 × K ∈ {4096, 8192, 16384} ×
{bf16, fp16}, branched off the LIVE oracle base
ryanswann-amd/tritonBLAS:fix/K-1922@0024a71.

Result: 18/18 cells gate-pass at the relaxed ≥1.05 ∧ p<0.05 floor; admit
gate (cohort AFTER ≥ 1.0× HBL ∧ no individual cell < 0.95× HBL) PASS.
N=1392 is the 9th contiguous rung of the off-by-48 wave-misaligned ladder
(816/880/944/1008/1072/1136/1200/1264/1328 → 1392) — same residue class
(N mod 64 == 48) as every prior rung; mod-128 lands on 80 (a new mod-128
sub-class within the same wave-misaligned family).

Mechanism (K-913 §3 / R-K1673 dtype-invariance + R-1811 wave-misalignment):
BLOCK_N=128 packs N=1392 into 10.875 BLOCK_N tiles per N-row (vs N=1328's
10.375 / N=1264's 9.875 / N=1200's 9.375 — the fractional-tile pattern is
the same SCHEDULER_LDS A4 failure mode as every prior off-by-48 rung).
Per K-2061 PMC delta-attribution, residue-48 carries a stable LDS bank-
conflict differential (~6×) vs residue-32 (~2×) and wave-aligned (~1×) at
(M=4096, K=8192, bf16); per K-2091 F-K2091.LADDER-CLIMB-NOT-DECAY, the
K-2031/K-2041 plateau-decay forecast is empirically falsified for 2
consecutive rungs and N=1392 sits squarely inside the climbing regime.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18 (full grid; no upstream-aliased exclusions).
  2. N axis is exactly {1392}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Both dtype rows are complete 9/9 (off-band N=1392 has no R-K979 P5
     Clause-3 bf16 alias — strictly load-bearing on both rows).
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, the consolidated P32–P38 N ∈ {96, 160, 224, 288, 320, 352,
     384}, K-1367/K-1397 P13 N ∈ {128, 256}, and K-1922 P40 N=544).
  6. Off-by-48 wave-misalignment band membership: 1392 mod 64 == 48
     (same residue class as the 8 prior rungs of this contiguous
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
    _K2107_P53_SKINNY_N1392_KCOMPL_ALIASSTACK_18,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
)


FZ = _K2107_P53_SKINNY_N1392_KCOMPL_ALIASSTACK_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n1392():
    assert {N for (_, N, _, _) in FZ} == {1392}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_row_balance():
    """Both dtype rows complete 9/9.  Off-band N=1392 has no R-K979 P5
    Clause-3 bf16 alias coverage at this rung; both rows are strictly
    load-bearing.  K-913 §3 LDS-bank-conflict is dtype-invariant on the
    column-narrow N=1392 tile (same fingerprint as P51 N=1264, P50 N=1200,
    P48 N=1072 — every prior off-by-48 rung shows balanced bf16/fp16
    admit per K-2055 / K-2071 / K-2085 / K-2091)."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    assert bf == fp


def test_envelope_equals_full_n1392_kcompl_grid():
    """The verified-winner envelope is exactly the 18-cell M ∈ {2048,
    4096,8192} × N=1392 × K ∈ {4096,8192,16384} × {bf16,fp16} grid (no
    upstream-aliased exclusions at this off-band rung) — pinned to detect
    any silent contraction (a false-NEGATIVE that would leak winner cells
    back to TB) or expansion (a false-POSITIVE that would leak non-winner
    cells out to HBL) of the K-2107 verification envelope."""
    full = frozenset(
        (M, 1392, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18


def test_off_by_48_wave_misalignment_band_membership():
    """N=1392 sits in the off-by-48 wave-misaligned band (N mod 64 == 48),
    the same residue class as every prior rung of this contiguous ladder
    (P45 N=816, N=880, P46 N=944, P47 N=1008, P48 N=1072, N=1136, P50
    N=1200, P51 N=1264, P52 N=1328).  This invariant guards against a
    silent N-axis typo (e.g. 1390, 1394) that would land on a different
    BLOCK_N=128 alignment rung and thus reflect a different mechanism.
    Note: mod-128 == 80 (a new mod-128 sub-class — alternates 48/80/112
    across the ladder rungs), but mod-64 == 48 is the binding mechanism
    invariant per K-2061 PMC delta-attribution."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 64 == 48
    assert only_n == 1392


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """P30 covers N ∈ {384, 768, 1536}; P53 covers N=1392 — disjoint by
    natural N-axis separation (P53 sits between the P30 N=768 and N=1536
    rungs, well above all of them by N-axis projection)."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p31_n256():
    assert FZ & _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28 == set()
    assert FZ & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == set()


def test_sibling_n_firewall_vs_consolidated_p32_p38():
    """K-1881 consolidated P32–P38 covers N ∈ {96, 160, 224, 288, 320,
    352, 384}; K-2107 P53 covers N=1392 — disjoint by natural N-axis
    separation."""
    assert FZ & _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120 == set()


def test_sibling_n_firewall_vs_p40_n544():
    """K-1922 P40 covers N=544 (off-by-32 band); K-2107 P53 covers N=1392
    (off-by-48 band) — disjoint by both N-axis and residue class."""
    assert FZ & _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18 == set()
