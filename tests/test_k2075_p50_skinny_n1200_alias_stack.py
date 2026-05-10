"""K-2075 P50 (S-002) — 40th-slot N=1200 K-COMPLEMENT verified-winner alias-stack tests.

Source measurement: K-2075 paired n=30 HIP-graph hot-cache MI300X / gfx942
(vs the live post-K-2060 oracle, HEAD = 80f3629) over the 18-cell sub-cohort
M ∈ {2048, 4096, 8192} × N=1200 × K ∈ {4096, 8192, 16384} × {bf16, fp16}.

K-2075 promotes N=1200 as the 7th candidate wave-misaligned rung in the
off-by-48 (mod 64) residue family above:
  - K-1978 P45 N=816  (mod 128 = 48)
  - K-1994 P45 N=880  (mod 128 = 48)
  - K-2014/2020/2024 P46 N=944 (mod 128 = 48)
  - K-2031 P47 N=1008 (mod 128 = 112 — alternate tail)
  - K-2047 P48 N=1072 (mod 128 = 48 — rejoin to canonical tail)
  - K-2060 P49 N=1136 (mod 128 = 112 — alternate-tail branch)
  - K-2075 P50 N=1200 (this rung; mod 128 = 48 — rejoin to canonical tail)

(1200 = 1136 + 64 = next 64-stride rung in the off-by-48 class.)

Mechanism (K-913 §3 / R-K1673 dtype-invariance + R-1811 wave-misalignment):
BLOCK_N=128 packs N=1200 into 9.375 tiles per N-row (1200 mod 128 = 48
= 3/8-tile tail).  1200 mod 64 = 48 places it in the off-by-48 wave-
misaligned class — the SIMD lane mismatch that has driven every prior rung's
admission.  K-2075 N=1200 shares the (mod 128 = 48) BLOCK_N tail with the
canonical K-1978/K-1994/K-2014/K-2047 sub-family (rejoins after the
K-2031/K-2060 mod-128 = 112 alternate-tail intermezzo).  Per K-1908, the
existing S1-form `(N % 64 != 0) ∧ (N <= 384) ∧ (K >= 4096)` predicate does
NOT cover N=1200 (N=1200 > 384 — the predicate's N-ceiling cuts off below
this rung); explicit alias-stack promotion is required.

K-2061 forward-looking PMC delta-attribution flagged N=1200 as the
residue-48 family ceiling (autotuner tile-key transition VGPR=76 → VGPR=128
at N=1200).  K-2075 converts that prospective prediction into a live
admit/reject decision via the standard K-2047/K-2052/K-2060 18-cell
benchmark protocol.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18 (full grid; no upstream-aliased exclusions).
  2. N axis is exactly {1200}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Both dtype rows are complete 9/9 (off-by-48 N=1200 has no R-K979 P5
     Clause-3 bf16 alias — strictly load-bearing on both rows).
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, the consolidated P32–P38 N ∈ {96, 160, 224, 288, 320, 352,
     384}, P40 N=544, P48 N=1072, P49 N=1136).
  6. Off-by-48 wave-misalignment band membership: 1200 mod 64 == 48
     (same wave class as K-1978/K-1994/K-2014/K-2031/K-2047/K-2060 rungs);
     1200 mod 128 == 48 (canonical tail topology — rejoins K-2047 N=1072).
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
    _K2047_P48_SKINNY_N1072_KCOMPL_ALIASSTACK_18,
    _K2060_P49_SKINNY_N1136_KCOMPL_ALIASSTACK_18,
    _K2075_P50_SKINNY_N1200_KCOMPL_ALIASSTACK_18,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
)


FZ = _K2075_P50_SKINNY_N1200_KCOMPL_ALIASSTACK_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n1200():
    assert {N for (_, N, _, _) in FZ} == {1200}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_row_balance():
    """Both dtype rows complete 9/9.  Off-by-48 N=1200 has no R-K979 P5
    Clause-3 bf16 alias coverage at this rung; both rows are strictly
    load-bearing.  K-913 §3 LDS-bank-conflict is dtype-invariant on the
    column-narrow N=1200 tile (same fingerprint as every prior off-by-48
    rung K-1978..K-2060)."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    assert bf == fp


def test_envelope_equals_full_n1200_kcompl_grid():
    """The verified-winner envelope is exactly the 18-cell M ∈ {2048,
    4096,8192} × N=1200 × K ∈ {4096,8192,16384} × {bf16,fp16} grid (no
    upstream-aliased exclusions at this off-by-48 rung) — pinned to
    detect any silent contraction or expansion of the K-2075 verification
    envelope."""
    full = frozenset(
        (M, 1200, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18


def test_off_by_48_wave_misalignment_band_membership():
    """N=1200 sits in the off-by-48 wave-misaligned band (N mod 64 == 48),
    the same wave class as K-1978 P45 N=816, K-1994 P45 N=880, K-2014
    P46 N=944, K-2031 P47 N=1008, K-2047 P48 N=1072, and K-2060 P49 N=1136.
    N=1200 mod 128 == 48 — rejoin to the canonical tail topology shared
    with K-1978/K-1994/K-2014/K-2047 (vs the K-2031/K-2060 mod-128 = 112
    alternate-tail branch).  This invariant guards against a silent N-axis
    typo (e.g. 1192, 1208, 1216) that would land on a different SIMD wave
    alignment and thus reflect a different mechanism."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 64 == 48
    assert only_n % 128 == 48
    assert only_n == 1200


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """P30 covers N ∈ {384, 768, 1536}; P50 covers N=1200 — disjoint by
    natural N-axis separation (P50 sits between the P30 N=768 and N=1536
    rungs)."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p31_n256():
    assert FZ & _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28 == set()
    assert FZ & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == set()


def test_sibling_n_firewall_vs_consolidated_p32_p38():
    """K-1881 consolidated P32–P38 covers N ∈ {96, 160, 224, 288, 320,
    352, 384}; K-2075 P50 covers N=1200 — disjoint by natural N-axis
    separation."""
    assert FZ & _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120 == set()


def test_sibling_n_firewall_vs_p40_n544():
    """K-1922 P40 covers N=544; K-2075 P50 covers N=1200 — disjoint by
    natural N-axis separation."""
    assert FZ & _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18 == set()


def test_sibling_n_firewall_vs_p48_n1072():
    """K-2047 P48 covers N=1072; K-2075 P50 covers N=1200 — disjoint by
    natural N-axis separation (1200 = 1072 + 128, two rungs up in
    off-by-48 family)."""
    assert FZ & _K2047_P48_SKINNY_N1072_KCOMPL_ALIASSTACK_18 == set()


def test_sibling_n_firewall_vs_p49_n1136():
    """K-2060 P49 covers N=1136; K-2075 P50 covers N=1200 — disjoint by
    natural N-axis separation (1200 = 1136 + 64, immediate next rung in
    off-by-48 family).  This firewall is strictly load-bearing because
    P49 is the immediate predecessor in the off-by-48 ladder."""
    assert FZ & _K2060_P49_SKINNY_N1136_KCOMPL_ALIASSTACK_18 == set()
