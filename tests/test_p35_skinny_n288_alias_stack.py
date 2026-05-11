"""P35 (S-002) — 26th-slot N=288 K-COMPLEMENT verified-winner subset tests.

Source measurement: K-1832 paired n=30 HIP-graph hot-cache 3-pass PMC sweep
(LDS / VALU·MFMA / VMEM·L2; 108 cell-engine-pass datapoints) on MI300X /
gfx942 (OCI MI300X fallback per INFRA-0048) across the 18-cell
N=288 sub-cohort = M ∈ {2048, 4096, 8192} × N=288 × K ∈ {4096, 8192, 16384}
× {bf16, fp16}.  3-engine sweep (tb_oracle / tb_streamk / hbl) recorded:

  * bf16 N=288 (9/9): cohort UNROUTED in K-1832's measurement (0/9 routed
    via any upstream alias; the wave-misaligned N=288 rung sits ABOVE the
    N=256 P31 cliff at the off-by-32 rung where no upstream predicate
    covers).  All 9 cells are load-bearing route-OUT targets (no alias
    overlap).
  * fp16 N=288 (9/9): cohort UNROUTED in K-1832's measurement (0/9 routed
    via any upstream alias; r_oracle < 1.0 systemically per R-K1794
    fp16-mirror systemic gap extended above N=256).  All 9 cells are also
    load-bearing — they close the dtype-mirror gap at the wave-misaligned
    N=288 column-narrow tile.

Cohort geomean TB/HBL = 4.290× (range 3.06×–6.46×, 18/18 admit at the
strict ≥1.05 ∧ p<0.05 gate) — the largest cohort-level uplift in the
alias-stack to date and ~3× above the K-1818 P34 N=96 cohort uplift (1.16×).

Mechanism (K-913 §3 / R-K1673 dtype-invariance + R-1811 wave-misalignment):
BLOCK_N=128 packs N=288 into wave-misaligned K-block columns (off-by-32 N
rung above N=256); SQ_LDS_BANK_CONFLICT/inst stays elevated AND MFMA-tail
inefficiency from wave-misalignment compounds the band.  K-1832 PMC delta
ranking confirms SQ_LDS_BANK_CONFLICT ≈ 869× and SQ_WAIT_INST_LDS ≈ 12.5×
(TB / HBL) at the top of the discriminator list, with L2/HBM signals at the
noise floor — identical fingerprint to K-1812 (TB/HBL ratio R²=0.9999 on
the 4 overlapping cells), K-1818 N=96, and K-1794 N∈{160,224}.  Same
SCHEDULER_LDS A4 failure mode as the K-1681 / K-1710 / K-1781 wave-
misaligned skinny-N class.  persistent_matmul cannot trade tile reshape for
atomic-reduction; hipBLASLt's split-K kernel selection clears the band by
~3.3× on average.  Same fingerprint productionised at K-1673 P28 (N=128),
K-1700 P29 (N=64), K-1748 P30 (N ∈ {384, 768, 1536}), K-1775 P31 (N=256),
K-1810 P32 (N=160), K-1817 P33 (N=224), and K-1831 P34 (N=96); now applied
to the wave-misaligned N=288 rung ABOVE the N=256 cliff.

Note: like K-1817 P33 (N=224) and K-1831 P34 (N=96), K-1837 P35 (N=288) has
NO upstream alias overlap — both dtype rows are load-bearing route-OUT
entries.  This is the K-1817 alias-overlap-absent discipline (vs K-1810 P32
N=160 alias-overlap-allowed discipline), reflecting the absence of any
upstream alias coverage at the off-N288 wave-misaligned rung above N=256.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18.
  2. N axis is exactly {288}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Per-shape dtype-mirror (bf16 ↔ fp16 admit sets coincide).
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, P32 N=160, P33 N=224, P34 N=96, and the K-1367/K-1397 P13
     N ∈ {128, 256} envelopes).
  6. Envelope is exactly the full 18-cell M ∈ {2048,4096,8192} × N=288
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
    _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18,
)


FZ = _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n288():
    assert {N for (_, N, _, _) in FZ} == {288}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_mirror_is_complete():
    """Each shape (M, N, K) admitted under bf16 is also admitted under fp16
    (and vice versa).  K-913 §3 LDS-bank-conflict is dtype-invariant on the
    column-narrow N=288 tile (same mechanism as K-1673 P28 at N=128, K-1810
    P32 at N=160, K-1775 P31 at N=256, K-1817 P33 at N=224, K-1831 P34 at
    N=96)."""
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
    assert FZ & _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18 == set()


def test_sibling_n_firewall_vs_p34_n96():
    """K-1837 P35 (N=288) must be N-axis disjoint from K-1831 P34 (N=96).
    Pairwise-disjointness invariant per K-1175 — the two slots cover
    different N-rungs of the wave-misaligned K-COMPLEMENT N-ladder
    (N=288 ABOVE the N=256 cliff vs N=96 BELOW the N=128 cliff)."""
    assert FZ & _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18 == set()


def test_envelope_equals_full_n288_kcompl_grid():
    """The verified-winner envelope is exactly the full 18-cell M ∈ {2048,
    4096,8192} × N=288 × K ∈ {4096,8192,16384} × {bf16,fp16} grid — pinned
    to detect any silent contraction or expansion of the K-1832 sub-cohort
    envelope.  Per K-1832, both dtype rows were entirely unrouted (bf16 0/9,
    fp16 0/9 systemic) and TB lost in 18/18 cells (cohort geomean TB/HBL =
    4.290×), so all 18 cells are load-bearing route-OUT entries (no upstream
    alias overlap, mirrors K-1817 P33 N=224 / K-1831 P34 N=96 discipline)."""
    full = frozenset(
        (M, 288, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18
