"""P35 (S-002) — 26th-slot N=320 K-COMPLEMENT verified-winner subset tests.

N=320 is the next rung up the wave-misaligned skinny-N K-COMPLEMENT ladder
above P34 (N=288) and below P30 (N=384).  320 mod 64 = 0 is wave-aligned at
the 64-thread level, but 320 / 128 = 2.5 BLOCK_N tiles wastes one full
half-tile of CU mapping (one full BLOCK_N=128 K-block column + one wave-
misaligned 192-wide remainder packed into a second BLOCK_N=128 column with
64 columns idle).

Source measurement: paired n=30 HIP-graph hot-cache anchors at M=4096 ×
N=320 × K ∈ {4096, 16384} × {bf16, fp16} on MI300X / gfx942 captured under
TRITONBLAS_DISABLE_K971=1 with route-OUT ablated.  All anchors clear the
strict ratio_tb_over_hbl ≥ 1.05 admission gate with paired n=30 95% CI
excluding 1.0×.  Envelope expands to the standard 18-cell M ∈ {2048, 4096,
8192} × N=320 × K ∈ {4096, 8192, 16384} × {bf16, fp16} grid per the
K-1810 / K-1817 / K-1835 wave-misaligned skinny-N alias-stack convention
(same M/K/dtype shape as P32 N=160, P33 N=224, and P34 N=288); the
remaining 14 cells inherit via dtype-mirror + M-axis closure (R-K1673
dtype-invariance + K-913 §3 ±19% cross-M neighborhood).

Mechanism (K-913 §3 / R-K1673 dtype-invariance): BLOCK_N=128 packs N=320
into a wave-misaligned K-block column layout where SQ_LDS_BANK_CONFLICT/
inst stays elevated and persistent_matmul cannot trade tile reshape for
atomic-reduction.  hipBLASLt's split-K kernel selection clears the band.
Same dtype-invariant LDS-BC fingerprint as K-1673 P28 (N=128), K-1810 P32
(N=160), K-1817 P33 (N=224), K-1775 P31 (N=256), K-1835 P34 (N=288); now
applied to the wave-misaligned N=320 rung.

Both dtype rows load-bearing — no upstream alias overlap (N=320 falls
above the R-K979 P5 minMN ≤ 192 clause and outside every K-1367/K-1397
P13 N ∈ {128, 256} envelope and the K-1700/K-1748 P29/P30 N ∈ {64, 384,
768, 1536} alias-stacks).  Same load-bearing discipline as K-1817 P33
(N=224) and K-1835 P34 (N=288) which also have zero upstream alias
overlap.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18.
  2. N axis is exactly {320}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Per-shape dtype-mirror (bf16 ↔ fp16 admit sets coincide).
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, P32 N=160, P33 N=224, P34 N=288, and the K-1367/K-1397 P13
     N ∈ {128, 256} envelopes).
  6. Envelope is exactly the full 18-cell M ∈ {2048,4096,8192} × N=320
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
    _P35_SKINNY_N320_KCOMPL_VERIFIED_WIN_18,
)


FZ = _P35_SKINNY_N320_KCOMPL_VERIFIED_WIN_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n320():
    assert {N for (_, N, _, _) in FZ} == {320}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_mirror_is_complete():
    """Each shape (M, N, K) admitted under bf16 is also admitted under fp16
    (and vice versa).  K-913 §3 LDS-bank-conflict is dtype-invariant on the
    wave-misaligned N=320 tile (same mechanism as K-1673 P28 at N=128,
    K-1810 P32 at N=160, K-1817 P33 at N=224, and K-1835 P34 at N=288)."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert bf == fp
    assert len(bf) == 9


def test_paired_anchor_cells_are_admitted():
    """The 4 paired-n30 directly-measured anchor cells (M=4096 × K ∈
    {4096, 16384} × {bf16, fp16}) must be present.  These are the load-
    bearing measurement anchors that establish the strict ratio_tb_over_hbl
    ≥ 1.05 gate with paired n=30 95% CI excluding 1.0×; the remaining 14
    cells inherit via dtype-mirror + M-axis closure per R-K1673 / K-913 §3
    (same envelope convention as P32 N=160, P33 N=224, P34 N=288)."""
    anchors = {
        (4096, 320, 4096,  "torch.bfloat16"),
        (4096, 320, 16384, "torch.bfloat16"),
        (4096, 320, 4096,  "torch.float16"),
        (4096, 320, 16384, "torch.float16"),
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
    assert FZ & _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18 == set()


def test_sibling_n_firewall_vs_p33_n224():
    assert FZ & _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18 == set()


def test_sibling_n_firewall_vs_p34_n288():
    """P35 (N=320) must be N-axis disjoint from K-1835 P34 (N=288).
    Pairwise-disjointness invariant per K-1175 — the two slots cover
    adjacent N-rungs of the contiguous K-COMPLEMENT N-ladder."""
    assert FZ & _P34_SKINNY_N288_KCOMPL_VERIFIED_WIN_18 == set()


def test_envelope_equals_full_n320_kcompl_grid():
    """The verified-winner envelope is exactly the full 18-cell M ∈ {2048,
    4096,8192} × N=320 × K ∈ {4096,8192,16384} × {bf16,fp16} grid — pinned
    to detect any silent contraction or expansion of the N=320 sub-cohort
    envelope.  Both dtype rows are load-bearing (no upstream alias
    overlap), same discipline as K-1817 P33 (N=224) and K-1835 P34
    (N=288)."""
    full = frozenset(
        (M, 320, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18


def test_n_axis_disjoint_from_k1502_corpus():
    """K-1545 GATE-B (corpus drift theorem precondition): every cell in the
    P35 frozenset must have an N value disjoint from the K-1502 / K-1295 /
    K-1316 / longK-family 132-cell reference corpus's N-axis ({1024, 2048,
    4096, 8192, 16384}).  When this holds, the K-1545 append-only theorem
    guarantees zero corpus drift from the P35 promotion (no GPU re-run
    needed on K-1502 reference cells).  Folded in here per the Minimalist
    review — N=320 is structurally outside the corpus N-axis and the pin
    fails loudly the moment that ever changes."""
    K1502_CORPUS_N = frozenset({1024, 2048, 4096, 8192, 16384})
    p35_n_axis = {N for (_, N, _, _) in FZ}
    assert p35_n_axis.isdisjoint(K1502_CORPUS_N), (
        f"K-1545 GATE-B FAIL: P35 N-axis {p35_n_axis} overlaps K-1502 "
        f"corpus N-axis {K1502_CORPUS_N}")
