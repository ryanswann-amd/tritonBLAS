"""K-1956 P41 (S-002) — 31st-slot N=608 K-COMPLEMENT verified-winner alias-stack tests.

Source measurement: K-1938 paired n=30 HIP-graph hot-cache MI300X / gfx942
(vs the live post-K-1922 oracle) over the 18-cell sub-cohort
M ∈ {2048, 4096, 8192} × N=608 × K ∈ {4096, 8192, 16384} × {bf16, fp16}.

Result: 18/18 cells gate-pass at the strict observed-ratio < 0.95 ∧
Wilcoxon+Holm q<0.05 floor; no upstream alias coverage at the
wave-misaligned N=608 rung — all 18 are load-bearing route-OUT entries.

Mechanism (K-913 §3 / R-K1673 dtype-invariance + R-1811 wave-misalignment):
BLOCK_N=128 packs N=608 into 4.75 BLOCK_N tiles per N-row (608 mod 64 = 32
— same off-by-32-mod-64 band as K-1922 P40 N=544 = 8.5 mod-64 = 32).
N=608 mod 128 = 96 (one rung above P40 N=544 mod 128 = 32 within the same
mod-64 wave-misalignment band).  Same SCHEDULER_LDS A4 failure mode as
the K-1681 / K-1710 / K-1781 / K-1812 / K-1824 / K-1832 / K-1843 / K-1912
wave-misaligned skinny-N class.  persistent_matmul cannot trade tile
reshape for atomic-reduction; hipBLASLt's split-K kernel selection
clears the band.

Per the K-1908 compact-predicate analysis, the existing S1-form
`(N % 64 != 0) ∧ (N <= 384) ∧ (K >= 4096)` predicate does NOT cover N=608
(N=608 > 384 and N % 64 == 32 — the predicate's N-ceiling cuts off below
this rung).  Explicit alias-stack promotion is required until S1 is
extended to cover N=608 in a separate task.  K-1937 / K-1942 depth-3/
depth-4 closed-form predicate searches both failed at ≥0.98
bit-equivalence; frozenset literal is the proven path.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18 (full grid; no upstream-aliased exclusions).
  2. N axis is exactly {608}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Both dtype rows are complete 9/9 (off-band N=608 has no R-K979 P5
     Clause-3 bf16 alias — strictly load-bearing on both rows).
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, the consolidated P32–P38 N ∈ {96, 160, 224, 288, 320, 352,
     384}, P40 N=544, plus K-1367/K-1397 P13 N ∈ {128, 256}).
  6. Off-by-32-mod-64 wave-misalignment band membership: 608 mod 64 == 32
     (same band as P40 N=544; distinct N-axis rung — N=608 mod 128 == 96
     vs N=544 mod 128 == 32).
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
    _K1956_P41_SKINNY_N608_KCOMPL_ALIASSTACK_18,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
)


FZ = _K1956_P41_SKINNY_N608_KCOMPL_ALIASSTACK_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n608():
    assert {N for (_, N, _, _) in FZ} == {608}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_row_balance():
    """Both dtype rows complete 9/9.  Off-band N=608 has no R-K979 P5
    Clause-3 bf16 alias coverage at this rung; both rows are strictly
    load-bearing.  K-913 §3 LDS-bank-conflict is dtype-invariant on the
    column-narrow N=608 tile (same fingerprint as K-1912 P40 at N=544)."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    assert bf == fp


def test_envelope_equals_full_n608_kcompl_grid():
    """The verified-winner envelope is exactly the 18-cell M ∈ {2048,
    4096,8192} × N=608 × K ∈ {4096,8192,16384} × {bf16,fp16} grid (no
    upstream-aliased exclusions at this off-band rung) — pinned to detect
    any silent contraction (a false-NEGATIVE that would leak winner cells
    back to TB) or expansion (a false-POSITIVE that would leak non-winner
    cells out to HBL) of the K-1938 verification envelope."""
    full = frozenset(
        (M, 608, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18


def test_off_by_32_mod_64_wave_misalignment_band_membership():
    """N=608 sits in the off-by-32-mod-64 wave-misaligned band
    (N mod 64 == 32), the same band as P40 N=544.  This invariant guards
    against a silent N-axis typo (e.g. 600, 616) that would land on a
    different mod-64 alignment rung and thus reflect a different
    mechanism.  Note: N=608 mod 128 == 96, distinct from P40 N=544 mod
    128 == 32 — the two rungs cohabit the mod-64 band but differ on
    BLOCK_N=128 alignment, so each requires its own alias-stack slot."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 64 == 32
    assert only_n == 608
    assert only_n % 128 == 96


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """P30 covers N ∈ {384, 768, 1536}; P41 covers N=608 — disjoint by
    natural N-axis separation (P41 sits between the P30 N=384 and N=768
    rungs, just above the P40 N=544 rung)."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p31_n256():
    assert FZ & _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28 == set()
    assert FZ & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == set()


def test_sibling_n_firewall_vs_consolidated_p32_p38():
    """K-1881 consolidated P32–P38 covers N ∈ {96, 160, 224, 288, 320,
    352, 384}; K-1956 P41 covers N=608 — disjoint by natural N-axis
    separation."""
    assert FZ & _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120 == set()


def test_sibling_n_firewall_vs_p40_n544():
    """P40 covers N=544; P41 covers N=608 — the two adjacent off-by-32-
    mod-64 rungs are disjoint by natural N-axis separation."""
    assert FZ & _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18 == set()
