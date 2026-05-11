"""P34 (S-002) — 25th-slot N=96 K-COMPLEMENT verified-winner subset tests.

Source measurement: K-1818 PMC RCA paired n=30 HIP-graph hot-cache capture/
replay on MI300X / gfx942 across the 18-cell N=96 sub-cohort = M ∈ {2048,
4096, 8192} × N=96 × K ∈ {4096, 8192, 16384} × {bf16, fp16}.  3-engine sweep
(tb_oracle / tb_streamk / hbl) recorded:

  * bf16 N=96 (9/9): cohort UNROUTED in K-1818's measurement (0/9 routed
    via any upstream alias; the wave-misaligned N=96 rung sits BELOW the
    N=128 P28 cliff, where no upstream predicate covers).  All 9 cells
    are load-bearing route-OUT targets (no alias overlap).
  * fp16 N=96 (9/9): cohort UNROUTED in K-1818's measurement (0/9 routed
    via any upstream alias; r_oracle < 1.0 systemically per K-1794 R-
    K1794.FP16-MIRROR-SYSTEMIC-AT-CROSS-BAND-LEVEL extended below N=128).
    All 9 cells are also load-bearing — they close the dtype-mirror gap
    at the wave-misaligned N=96 column-narrow tile.

Mechanism (K-913 §3 / R-K1673 dtype-invariance + R-1811 wave-misalignment):
BLOCK_N=128 packs N=96 into a single wave-misaligned K-block column with
PARTIAL coverage (off-by-32 N rung); SQ_LDS_BANK_CONFLICT/inst stays
elevated AND MFMA-tail inefficiency from wave-misalignment compounds the
band.  persistent_matmul cannot trade tile reshape for atomic-reduction;
hipBLASLt's split-K kernel selection clears the band.  Same fingerprint
productionized at K-1673 P28 (N=128), K-1810 P32 (N=160), K-1775 P31
(N=256), and K-1817 P33 (N=224); now applied to the wave-misaligned N=96
rung BELOW the N=128 cliff.

Note: like K-1817 P33 (N=224), K-1831 P34 (N=96) has NO upstream alias
overlap — both dtype rows are load-bearing route-OUT entries.  This is
the K-1817 alias-overlap-absent discipline (vs K-1810 P32 N=160 alias-
overlap-allowed discipline), reflecting the absence of any upstream alias
coverage at the off-N96 wave-misaligned rung below N=128.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18.
  2. N axis is exactly {96}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Per-shape dtype-mirror (bf16 ↔ fp16 admit sets coincide).
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, P32 N=160, P33 N=224, and the K-1367/K-1397 P13 N ∈ {128,
     256} envelopes).
  6. Envelope is exactly the full 18-cell M ∈ {2048,4096,8192} × N=96
     × K ∈ {4096,8192,16384} × {bf16,fp16} grid.
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
)


FZ = _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n96():
    assert {N for (_, N, _, _) in FZ} == {96}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_mirror_is_complete():
    """Each shape (M, N, K) admitted under bf16 is also admitted under fp16
    (and vice versa).  K-913 §3 LDS-bank-conflict is dtype-invariant on the
    column-narrow N=96 tile (same mechanism as K-1673 P28 at N=128, K-1810
    P32 at N=160, K-1817 P33 at N=224)."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert bf == fp
    assert len(bf) == 9


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p31_n256():
    assert FZ & _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28 == set()
    assert FZ & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == set()


def test_sibling_n_firewall_vs_p32_n160():
    assert FZ & _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18 == set()


def test_sibling_n_firewall_vs_p33_n224():
    """K-1831 P34 (N=96) must be N-axis disjoint from K-1817 P33 (N=224).
    Pairwise-disjointness invariant per K-1175 — the two slots cover
    different N-rungs of the wave-misaligned K-COMPLEMENT N-ladder
    (N=96 BELOW the N=128 cliff vs N=224 ABOVE)."""
    assert FZ & _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18 == set()


def test_envelope_equals_full_n96_kcompl_grid():
    """The verified-winner envelope is exactly the full 18-cell M ∈ {2048,
    4096,8192} × N=96 × K ∈ {4096,8192,16384} × {bf16,fp16} grid — pinned
    to detect any silent contraction or expansion of the K-1818 sub-cohort
    envelope.  Per K-1818, both dtype rows were entirely unrouted (bf16 0/9,
    fp16 0/9 systemic), so all 18 cells are load-bearing route-OUT entries
    (no upstream alias overlap, mirrors K-1817 P33 N=224 discipline)."""
    full = frozenset(
        (M, 96, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18
