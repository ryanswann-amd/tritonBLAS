"""K-2091 P51 (S-002) -- 41st-slot N=1264 K-COMPLEMENT alias-stack tests.

Source measurement: K-2091 paired n=30 HIP-graph hot-cache MI300X / gfx942
(vs the live post-K-1922 oracle, HEAD 0024a71) over the 18-cell sub-cohort
M in {2048, 4096, 8192} x N=1264 x K in {4096, 8192, 16384} x {bf16, fp16}.

Result: 8th-rung admit in the off-by-48 wave-misaligned residue family
above K-1978 P45 N=816, K-1990/K-1994 P45 N=880, K-2014/2020/2024 P46
N=944, K-2031/K-2041 P47 N=1008, K-2047 P48 N=1072, K-2055/K-2063 P49
N=1136, and K-2077 P50 N=1200.

Mechanism (K-913 §3 / R-K1673 dtype-invariance + R-1811 wave-misalignment):
1264 mod 64 = 48 -- wave-MISALIGNED at the 64-lane SIMD; off-by-48 modular
class.  1264 mod 128 = 112 -- BLOCK_N=128 packs into 9 full tiles + 1 BN=112
tail = 9/9.875-tile tail.  REJOIN to K-2055/K-2063 P49 N=1136 mod-128 = 112
tail topology, continuing the strict 48/112/48/112/48/112/48/112 mod-128
alternation across the residue-48 ladder (a direct mechanical consequence
of stride-64 sampling of a mod-128 axis).  Empirically magnitude-orthogonal
per K-2031/K-2063/K-2077: the binding constraint is the (mod 64 = 48)
wave-lane mismatch, not the BLOCK_N=128 packing remainder.

K-2071 already collected 16-cell BEFORE data for N=1264 (TB/HBL = 1.832x
geomean, 16/16 strict admit) -- K-2091 is the natural single-rung
promotion to bring it into the live dispatcher.

Per the K-1908 compact-predicate analysis, the existing S1-form
`(N % 64 != 0) AND (N <= 384) AND (K >= 4096)` predicate does NOT cover
N=1264 (N=1264 > 384).  Explicit alias-stack promotion is required until
S1 is extended above N=384 in a separate task.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants -- R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18 (full grid; no upstream-aliased exclusions).
  2. N axis is exactly {1264}.
  3. M in {2048, 4096, 8192}; K in {4096, 8192, 16384};
     dtype in {torch.bfloat16, torch.float16}.
  4. Both dtype rows are complete 9/9 (off-by-48 N=1264 has no R-K979 P5
     Clause-3 bf16 alias -- strictly load-bearing on both rows).
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N.
  6. N=1264 mod 64 == 48 (wave-misalignment band membership).
  7. N=1264 mod 128 == 112 (REJOIN to N=1136 tail topology).
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
    _K2091_P51_SKINNY_N1264_KCOMPL_ALIASSTACK_18,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
)


FZ = _K2091_P51_SKINNY_N1264_KCOMPL_ALIASSTACK_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n1264():
    assert {N for (_, N, _, _) in FZ} == {1264}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_row_balance():
    """Both dtype rows complete 9/9.  Off-by-48 N=1264 has no R-K979 P5
    Clause-3 bf16 alias coverage at this rung; both rows are strictly
    load-bearing.  K-913 sec.3 LDS-bank-conflict is dtype-invariant on the
    column-narrow N=1264 tile."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    assert bf == fp


def test_envelope_equals_full_n1264_kcompl_grid():
    """The verified-winner envelope is exactly the 18-cell M in {2048,
    4096,8192} x N=1264 x K in {4096,8192,16384} x {bf16,fp16} grid (no
    upstream-aliased exclusions at this off-by-48 rung) -- pinned to detect
    any silent contraction or expansion of the K-2091 verification envelope."""
    full = frozenset(
        (M, 1264, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18


def test_off_by_48_wave_misalignment_band_membership():
    """N=1264 sits in the off-by-48 wave-misaligned band (N mod 64 == 48),
    the same band as P45 N=816 / N=880, P46 N=944, P47 N=1008, P48 N=1072,
    P49 N=1136, P50 N=1200.  This invariant guards against a silent N-axis
    typo (e.g. 1260, 1268) that would land on a different SIMD-alignment
    rung and thus reflect a different mechanism."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 64 == 48
    assert only_n == 1264


def test_mod_128_rejoin_to_n1136_topology():
    """N=1264 mod 128 == 112 -- REJOIN to K-2055/K-2063 P49 N=1136
    mod-128 = 112 tail topology, continuing the strict
    48/112/48/112/48/112/48/112 alternation across the residue-48 ladder."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 128 == 112


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """P30 covers N in {384, 768, 1536}; P51 covers N=1264 -- disjoint by
    natural N-axis separation."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p31_n256():
    assert FZ & _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28 == set()
    assert FZ & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == set()


def test_sibling_n_firewall_vs_consolidated_p32_p38():
    """K-1881 consolidated P32-P38 covers N in {96, 160, 224, 288, 320,
    352, 384}; K-2091 P51 covers N=1264 -- disjoint by natural N-axis
    separation."""
    assert FZ & _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120 == set()


def test_sibling_n_firewall_vs_p40_n544():
    """K-1922 P40 covers N=544; K-2091 P51 covers N=1264 -- disjoint."""
    assert FZ & _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18 == set()
