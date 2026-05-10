"""K-2055 P48 (S-002) — 38th-slot N=1136 K-COMPLEMENT verified-winner alias-stack tests.

Source measurement: K-2055 paired n=30 HIP-graph hot-cache MI300X / gfx942
(vs the live post-K-1922 oracle, HEAD = 0024a71) over the 18-cell sub-cohort
M ∈ {2048, 4096, 8192} × N=1136 × K ∈ {4096, 8192, 16384} × {bf16, fp16}.

K-2055 is the 6th confirmed wave-misaligned rung in the off-by-48 (mod 64)
residue family above K-1978 P45 N=816, K-1990/K-1994 P45 N=880, K-2010/K-2028
P46 N=944, K-2031/K-2036 P47 N=1008, and the implicit N=1072 5th rung
(1136 = 1072 + 64 = next 64-stride rung in the off-by-48 class).

Mechanism (K-913 §3 / R-K1673 dtype-invariance + R-1811 wave-misalignment):
BLOCK_N=128 packs N=1136 into 8.875 BLOCK_N tiles per N-row (1136 mod 128
= 112 = 8/9-tile tail).  1136 mod 64 = 48 places it in the off-by-48
wave-misaligned class — the same SIMD-lane mismatch as N=816, N=880, N=944,
N=1008, N=1072.  N=1136 has the same `mod 128 = 112` residue as N=880 and
N=1008 (a mod-128 sub-family of the off-by-48 class), distinct from N=816
and N=944 (mod 128 = 48).  Per K-2031 F-MOD-64-VS-MOD-128-MECHANISM-
SEPARATION-CONFIRMED, the (mod 64) wave-lane mismatch is the binding
constraint, not the BLOCK_N=128 packing remainder.

Per the K-1908 compact-predicate analysis, the existing S1-form
`(N % 64 != 0) ∧ (N <= 384) ∧ (K >= 4096)` predicate does NOT cover N=1136
(N=1136 > 384 — the predicate's N-ceiling cuts off below this rung).
Explicit alias-stack promotion is required until S1 is extended to cover
N>384 in a separate task.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18 (full grid; no upstream-aliased exclusions).
  2. N axis is exactly {1136}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Both dtype rows are complete 9/9.
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N.
  6. Off-by-48 wave-misalignment band membership: 1136 mod 64 == 48
     (same wave class as K-1978 P45 N=816, K-1994 P45 N=880, K-2010 P46
     N=944, K-2031 P47 N=1008 — distinct N-axis rung; mod 128 = 112,
     same mod-128 sub-class as K-1994 N=880 and K-2031 N=1008).
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
    _K2055_P48_SKINNY_N1136_KCOMPL_ALIASSTACK_18,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
)


FZ = _K2055_P48_SKINNY_N1136_KCOMPL_ALIASSTACK_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n1136():
    assert {N for (_, N, _, _) in FZ} == {1136}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_row_balance():
    """Both dtype rows complete 9/9.  Off-by-48 N=1136 has no R-K979 P5
    Clause-3 bf16 alias coverage at this rung; both rows are strictly
    load-bearing.  K-913 §3 LDS-bank-conflict is dtype-invariant on the
    column-narrow N=1136 tile (same fingerprint as K-1978 P45 N=816,
    K-1994 P45 N=880, K-2010 P46 N=944, and K-2031 P47 N=1008)."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    assert bf == fp


def test_envelope_equals_full_n1136_kcompl_grid():
    """The verified-winner envelope is exactly the 18-cell M ∈ {2048,
    4096,8192} × N=1136 × K ∈ {4096,8192,16384} × {bf16,fp16} grid (no
    upstream-aliased exclusions at this off-by-48 rung) — pinned to
    detect any silent contraction (a false-NEGATIVE that would leak
    winner cells back to TB) or expansion (a false-POSITIVE that would
    leak non-winner cells out to HBL) of the K-2055 verification envelope."""
    full = frozenset(
        (M, 1136, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18


def test_off_by_48_wave_misalignment_band_membership():
    """N=1136 sits in the off-by-48 wave-misaligned band (N mod 64 == 48),
    the same wave class as K-1978 P45 N=816, K-1994 P45 N=880, K-2010
    P46 N=944, and K-2031 P47 N=1008.  N=1136 has mod 128 = 112 (same
    mod-128 sub-family as N=880 and N=1008, distinct from N=816 and
    N=944's mod 128 = 48).  Per K-2031, the binding constraint is the
    (mod 64) wave-lane mismatch, not the BLOCK_N=128 packing tail.
    This invariant guards against a silent N-axis typo (e.g. 1132,
    1140, 1152) that would land on a different SIMD wave alignment
    and thus reflect a different mechanism."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 64 == 48
    assert only_n == 1136


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """P30 covers N ∈ {384, 768, 1536}; P48 covers N=1136 — disjoint by
    natural N-axis separation (P48 sits between the P30 N=768 and N=1536
    rungs)."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p31_n256():
    assert FZ & _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28 == set()
    assert FZ & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == set()


def test_sibling_n_firewall_vs_consolidated_p32_p38():
    """K-1881 consolidated P32–P38 covers N ∈ {96, 160, 224, 288, 320,
    352, 384}; K-2055 P48 covers N=1136 — disjoint by natural N-axis
    separation."""
    assert FZ & _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120 == set()


def test_sibling_n_firewall_vs_p40_n544():
    """K-1922 P40 covers N=544; K-2055 P48 covers N=1136 — disjoint by
    natural N-axis separation."""
    assert FZ & _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18 == set()
