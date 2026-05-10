"""K-2163 P57 (S-002) — 13th-rung N=1648 K-COMPLEMENT alias-stack tests.

Source measurement: K-2163 paired n=80 HIP-graph hot-cache MI300X / gfx942
3-engine bench (TB-BEFORE / TB-AFTER / hipBLASLt) over the 18-cell
sub-cohort M ∈ {2048, 4096, 8192} × N=1648 × K ∈ {4096, 8192, 16384} ×
{bf16, fp16}.

Result: 18/18 cells gate-pass at the 0.95× HBL parity floor with cohort
geomean TB-AFTER/HBL ≈ 1.00×.  N=1648 sits inside the off-by-48 wave-
misaligned ladder climbing/plateau regime (per K-2091/K-2107/K-2124/
K-2136/K-2150 F-K2091.LADDER-CLIMB-NOT-DECAY): 13th contiguous +64-step
rung above the K-1978 P45 N=816 base.  The previously-forecast plateau-
decay knee at N=1300–1500 was empirically falsified for 6 consecutive
rungs (P51→P52→P53→P54→P55→P56) and continues at P57.

Mechanism (R-1811 wave-misalignment + K-913 §3 LDS-bank-conflict):
BLOCK_N=128 packs N=1648 into 12.875 BLOCK_N tiles per N-row (vs N=1584's
12.375 / N=1520's 11.875 / N=1456's 11.375 — same SCHEDULER_LDS A4
fractional-tile failure mode as every prior off-by-48 rung).  1648/wave64
= 25.75 waves per N-row, 0.75 fractional-wave misalignment identical to
the entire family.  Per K-2061 PMC delta-attribution, residue-48 carries
a stable LDS bank-conflict differential (~6×) vs residue-32 (~2×) and
wave-aligned (~1×) at (M=4096, K=8192, bf16).  N=1648 % 128 == 112,
joining the K-1990/K-2031/K-2055/K-2091/K-2107/K-2136 mod-128=112
sub-class (7th mod-128=112 step in the ladder).

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18 (full grid; no upstream-aliased exclusions).
  2. N axis is exactly {1648}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Both dtype rows are complete 9/9 (off-by-48 N=1648 has no R-K979 P5
     Clause-3 bf16 alias — strictly load-bearing on both rows).
  5. Sibling-N firewall vs the 12 prior off-by-48 ladder rungs and every
     other prior K-COMPLEMENT alias-stack at a different N.
  6. Off-by-48 wave-misalignment band membership: 1648 mod 64 == 48
     (same off-by-48 family as P45 N=816, P46 N=944, ..., P56 N=1584 —
     13th contiguous rung).
  7. mod-128 residue-class membership: 1648 mod 128 == 112 (same
     mod-128=112 sub-class as K-1990 N=880, K-2031 N=1008, K-2055
     N=1136, K-2091 N=1264, K-2107 N=1392, K-2136 N=1520).
  8. Disjointness with the immediate 12th-rung predecessor (K-2150 P56
     N=1584) — distinct N-axis rungs on the same off-by-48 family.
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
    _K2150_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18,
    _K2163_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
)


FZ = _K2163_P57_SKINNY_N1648_KCOMPL_ALIASSTACK_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n1648():
    assert {N for (_, N, _, _) in FZ} == {1648}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_row_balance():
    """Both dtype rows complete 9/9.  Off-by-48 N=1648 has no R-K979 P5
    Clause-3 bf16 alias coverage at this rung; both rows are strictly
    load-bearing.  K-913 §3 LDS-bank-conflict is dtype-invariant on the
    column-narrow N=1648 tile (same fingerprint as all 12 prior
    off-by-48 rungs P45 N=816 → P56 N=1584)."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    assert bf == fp


def test_envelope_equals_full_n1648_kcompl_grid():
    """The verified-winner envelope is exactly the 18-cell M ∈ {2048,
    4096,8192} × N=1648 × K ∈ {4096,8192,16384} × {bf16,fp16} grid (no
    upstream-aliased exclusions at this off-by-48 rung) — pinned to
    detect any silent contraction (a false-NEGATIVE that would leak
    winner cells back to TB) or expansion (a false-POSITIVE that would
    leak non-winner cells out to HBL) of the K-2163 verification
    envelope."""
    full = frozenset(
        (M, 1648, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18


def test_off_by_48_wave_misalignment_band_membership():
    """N=1648 sits in the off-by-48 wave-misaligned band (N mod 64 == 48),
    the same band as every prior rung in the K-1978 → K-2150 ladder
    (P45 N=816, P46 N=944, ..., P56 N=1584).  This invariant guards
    against a silent N-axis typo (e.g. 1640, 1656) that would land on a
    different wave-alignment rung and reflect a different mechanism."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 64 == 48
    assert only_n == 1648


def test_mod_128_residue_class_membership():
    """N=1648 mod 128 == 112, joining the K-1990 N=880 / K-2031 N=1008 /
    K-2055 N=1136 / K-2091 N=1264 / K-2107 N=1392 / K-2136 N=1520
    mod-128=112 sub-class.  This invariant pins the BLOCK_N=128
    fractional-tile fingerprint (12.875 tiles per N-row) — distinct from
    the mod-128=48 sub-class (P45 N=816, P46 N=944, P49 N=1136*, P52
    N=1328, P56 N=1584) which lands on .375 fractional tiles."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 128 == 112


def test_thirteenth_rung_contiguous_plus_64_above_p56():
    """N=1648 is exactly +64 above K-2150 P56 N=1584 (13th contiguous
    rung above the K-1978 P45 N=816 base).  The +64 step preserves the
    N % 64 == 48 invariant.  This invariant guards against any future
    N-axis typo that would break the contiguous ladder spacing."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n == 1584 + 64
    assert (only_n - 816) % 64 == 0
    assert ((only_n - 816) // 64) == 13  # 13th rung above P45 N=816


def test_sibling_n_firewall_vs_p56_n1584():
    """K-2150 P56 covers N=1584; K-2163 P57 covers N=1648 — disjoint by
    natural +64 N-axis separation along the off-by-48 ladder."""
    assert FZ & _K2150_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18 == set()


def test_sibling_n_firewall_vs_p40_n544():
    """K-1922 P40 covers N=544; K-2163 P57 covers N=1648 — disjoint by
    natural N-axis separation (different wave-misalignment band: P40 is
    off-by-32, P57 is off-by-48)."""
    assert FZ & _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18 == set()


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """P30 covers N ∈ {384, 768, 1536}; P57 covers N=1648 — disjoint by
    natural N-axis separation."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p31_n256():
    assert FZ & _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28 == set()
    assert FZ & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == set()


def test_sibling_n_firewall_vs_consolidated_p32_p38():
    """K-1881 consolidated P32–P38 covers N ∈ {96, 160, 224, 288, 320,
    352, 384}; K-2163 P57 covers N=1648 — disjoint by natural N-axis
    separation (different wave-misalignment band entirely)."""
    assert FZ & _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120 == set()
