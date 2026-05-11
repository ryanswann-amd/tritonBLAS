"""K-1922 P40 (S-002) — 30th-slot N=544 K-COMPLEMENT verified-winner alias-stack tests.

Source measurement: K-1912 paired n=30 HIP-graph hot-cache MI300X / gfx942
(vs the live post-K-1881 oracle) over the 18-cell sub-cohort
M ∈ {2048, 4096, 8192} × N=544 × K ∈ {4096, 8192, 16384} × {bf16, fp16}.

Result: 18/18 cells gate-pass at the strict ≥1.05 ∧ p<0.05 floor; no
upstream alias coverage at the off-by-32 wave-misaligned N=544 rung — all
18 are load-bearing route-OUT entries.  Cohort geomean TB/HBL ≈ 1.51×.

Mechanism (K-913 §3 / R-K1673 dtype-invariance + R-1811 wave-misalignment):
BLOCK_N=128 packs N=544 into 4.25 BLOCK_N tiles per N-row (544 mod 128 = 32
— same off-by-32 band as P32 N=160 = 1.25 tiles, P35 N=288 = 2.25 tiles,
P37 N=352 = 2.75 tiles).  Same SCHEDULER_LDS A4 failure mode as the K-1681 /
K-1710 / K-1781 / K-1812 / K-1824 / K-1832 / K-1843 wave-misaligned
skinny-N class.  persistent_matmul cannot trade tile reshape for
atomic-reduction; hipBLASLt's split-K kernel selection clears the band by
~50% on average.

Per the K-1908 compact-predicate analysis, the existing S1-form
`(N % 64 != 0) ∧ (N <= 384) ∧ (K >= 4096)` predicate does NOT cover N=544
(N=544 > 384 and N % 64 == 32 — the predicate's N-ceiling cuts off below
this rung).  Explicit alias-stack promotion is required until S1 is
extended to cover N=544 in a separate task.  The K-1900 compact-predicate
substitution angle failed at depth-2 closure and is NOT retried here.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18 (full grid; no upstream-aliased exclusions).
  2. N axis is exactly {544}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Both dtype rows are complete 9/9 (off-band N=544 has no R-K979 P5
     Clause-3 bf16 alias — strictly load-bearing on both rows).
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, the consolidated P32–P38 N ∈ {96, 160, 224, 288, 320, 352,
     384}, plus K-1367/K-1397 P13 N ∈ {128, 256}).
  6. Off-by-32 wave-misalignment band membership: 544 mod 128 == 32 (same
     band as P32 N=160, P35 N=288 — distinct N-axis rung).
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
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
)


FZ = _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n544():
    assert {N for (_, N, _, _) in FZ} == {544}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_row_balance():
    """Both dtype rows complete 9/9.  Off-band N=544 has no R-K979 P5
    Clause-3 bf16 alias coverage at this rung; both rows are strictly
    load-bearing.  K-913 §3 LDS-bank-conflict is dtype-invariant on the
    column-narrow N=544 tile (same fingerprint as K-1810 P32 at N=160,
    K-1837 P35 at N=288)."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    assert bf == fp


def test_envelope_equals_full_n544_kcompl_grid():
    """The verified-winner envelope is exactly the 18-cell M ∈ {2048,
    4096,8192} × N=544 × K ∈ {4096,8192,16384} × {bf16,fp16} grid (no
    upstream-aliased exclusions at this off-band rung) — pinned to detect
    any silent contraction (a false-NEGATIVE that would leak winner cells
    back to TB) or expansion (a false-POSITIVE that would leak non-winner
    cells out to HBL) of the K-1912 verification envelope."""
    full = frozenset(
        (M, 544, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18


def test_off_by_32_wave_misalignment_band_membership():
    """N=544 sits in the off-by-32 wave-misaligned band (N mod 128 == 32),
    the same band as P32 N=160, P35 N=288.  This invariant guards against
    a silent N-axis typo (e.g. 540, 560) that would land on a different
    BLOCK_N=128 alignment rung and thus reflect a different mechanism."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 128 == 32
    assert only_n == 544


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """P30 covers N ∈ {384, 768, 1536}; P40 covers N=544 — disjoint by
    natural N-axis separation (P40 sits between the P30 N=384 and N=768
    rungs)."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p31_n256():
    assert FZ & _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28 == set()
    assert FZ & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == set()


def test_sibling_n_firewall_vs_consolidated_p32_p38():
    """K-1881 consolidated P32–P38 covers N ∈ {96, 160, 224, 288, 320,
    352, 384}; K-1922 P40 covers N=544 — disjoint by natural N-axis
    separation."""
    assert FZ & _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120 == set()
