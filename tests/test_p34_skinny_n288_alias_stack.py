"""P34 (S-002) — 25th-slot N=288 K-COMPLEMENT verified-winner subset tests.

Source measurement: paired N ∈ {96, 160, 224, 288} K-COMPLEMENT sweep on
MI300X / gfx942 (paired n=30 HIP-graph hot-cache; bench_paired.csv 4-cell
direct anchor at M=4096 × K ∈ {4096, 16384} × {bf16, fp16} captured under
TRITONBLAS_DISABLE_K971=1 with route-OUT ablated):

  * (4096, 288, 4096, bf16): tb=333.18 µs / hbl=62.07 µs  → tb/hbl 5.37×
  * (4096, 288, 16384, bf16): tb=480.54 µs / hbl=154.32 µs → tb/hbl 3.11×
  * (4096, 288, 4096, fp16): tb=334.28 µs / hbl=64.18 µs  → tb/hbl 5.21×
  * (4096, 288, 16384, fp16): tb=494.03 µs / hbl=159.57 µs → tb/hbl 3.10×

All 4 directly-measured cells exceed the strict tb/hbl ≥ 1.05 admission
gate by 3.0×-5.4×; cohort gmean speedup once routed-OUT is ≥ 4.0×.  The
envelope expands to the standard M ∈ {2048, 4096, 8192} × N=288 × K ∈
{4096, 8192, 16384} × {bf16, fp16} 18-cell grid per the K-1810 / K-1817
wave-misaligned skinny-N alias-stack convention (same M/K/dtype shape as
P32 N=160 and P33 N=224); the remaining 14 cells inherit via dtype-mirror
+ M-axis closure (R-K1673 dtype-invariance + K-913 §3 ±19% cross-M
neighborhood; same envelope shape as P32 N=160 and P33 N=224).

Mechanism (K-913 §3 / R-K1673 dtype-invariance): BLOCK_N=128 packs N=288
into TWO K-block columns (one full BLOCK_N=128 + one wave-misaligned
160-wide remainder); SQ_LDS_BANK_CONFLICT/inst stays elevated and
persistent_matmul cannot trade tile reshape for atomic-reduction.
hipBLASLt's split-K kernel selection clears the band.  Same dtype-
invariant LDS-BC fingerprint as K-1673 P28 (N=128), K-1810 P32 (N=160),
K-1817 P33 (N=224), K-1775 P31 (N=256); now applied to the wave-
misaligned N=288 rung.

Both dtype rows load-bearing — no upstream alias overlap (N=288 falls
above the R-K979 P5 minMN ≤ 192 clause and outside every K-1367/K-1397
P13 N ∈ {128, 256} envelope and the K-1700/K-1748 P29/P30 N ∈ {64, 384,
768, 1536} alias-stacks).  Same load-bearing discipline as K-1817 P33
(N=224) which also has zero upstream alias overlap.

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
     N=256, P32 N=160, P33 N=224, and the K-1367/K-1397 P13 N ∈ {128,
     256} envelopes).
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
    _P34_SKINNY_N288_KCOMPL_VERIFIED_WIN_18,
)


FZ = _P34_SKINNY_N288_KCOMPL_VERIFIED_WIN_18


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
    P32 at N=160, and K-1817 P33 at N=224)."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert bf == fp
    assert len(bf) == 9


def test_paired_anchor_cells_are_admitted():
    """The 4 paired-n30 directly-measured cells (M=4096 × K ∈ {4096, 16384}
    × {bf16, fp16}) from the bench_paired.csv N ∈ {96,160,224,288} sub-cohort
    must be present.  These are the load-bearing measurement anchors that
    establish the >=3x tb/hbl gap; the remaining 14 cells inherit via
    dtype-mirror + M-axis closure per R-K1673 / K-913 §3."""
    anchors = {
        (4096, 288, 4096,  "torch.bfloat16"),
        (4096, 288, 16384, "torch.bfloat16"),
        (4096, 288, 4096,  "torch.float16"),
        (4096, 288, 16384, "torch.float16"),
    }
    assert anchors <= FZ


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
    """P34 (N=288) must be N-axis disjoint from K-1810 P32 (N=160).
    Pairwise-disjointness invariant per K-1175."""
    assert FZ & _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18 == set()


def test_sibling_n_firewall_vs_p33_n224():
    """P34 (N=288) must be N-axis disjoint from K-1817 P33 (N=224).
    Pairwise-disjointness invariant per K-1175 — the two slots cover
    different N-rungs of the contiguous K-COMPLEMENT N-ladder."""
    assert FZ & _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18 == set()


def test_envelope_equals_full_n288_kcompl_grid():
    """The verified-winner envelope is exactly the full 18-cell M ∈ {2048,
    4096,8192} × N=288 × K ∈ {4096,8192,16384} × {bf16,fp16} grid — pinned
    to detect any silent contraction or expansion of the N=288 sub-cohort
    envelope.  Both dtype rows are load-bearing (no upstream alias
    overlap), same discipline as K-1817 P33 (N=224)."""
    full = frozenset(
        (M, 288, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18
