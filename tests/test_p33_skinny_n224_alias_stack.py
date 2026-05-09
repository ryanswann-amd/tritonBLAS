"""P33 (S-002) — 24th-slot N=224 K-COMPLEMENT verified-winner subset tests.

Source measurement: K-1794 paired n=30 HIP-graph hot-cache capture/replay
on MI300X / gfx942 across the 18-cell N=224 sub-cohort = M ∈ {2048, 4096,
8192} × N=224 × K ∈ {4096, 8192, 16384} × {bf16, fp16}.  3-engine sweep
(tb_oracle / tb_streamk / hbl) recorded:

  * bf16 N=224 (9/9): cohort UNROUTED in K-1794's measurement (0/9 routed
    via any upstream alias; cohort geomean r_oracle = 0.722 — the FIRST
    observation of an N-axis cliff above N=128 per K-1794 R-K1794.N224-
    DOES-NOT-EXTEND-FROM-N192-FROZENSET-MUST-EXPLICITLY-INCLUDE).  The
    K-1782 N=192 envelope does NOT extend smoothly upward to N=224, so
    these 9 cells are load-bearing route-OUT targets (no alias overlap).
  * fp16 N=224 (9/9): cohort UNROUTED in K-1794's measurement (0/9 routed
    via any upstream alias; r_oracle < 1.0 systemically per K-1794 R-
    K1794.FP16-MIRROR-SYSTEMIC-AT-CROSS-BAND-LEVEL).  These 9 cells are
    also load-bearing — they close the dtype-mirror gap at the wave-
    misaligned N=224 column-narrow tile.

Mechanism (K-913 §3 / R-K1673 dtype-invariance): BLOCK_N=128 packs N=224
into a single wave-misaligned K-block column; SQ_LDS_BANK_CONFLICT/inst
stays elevated and persistent_matmul cannot trade tile reshape for
atomic-reduction.  hipBLASLt's split-K kernel selection clears the band.
Same mechanism productionized at K-1673 P28 (N=128), K-1810 P32 (N=160),
and K-1775 P31 (N=256); now applied to the wave-misaligned N=224 rung.

Note: unlike K-1810 P32 (N=160) where the 9 bf16 cells overlap with the
upstream R-K979 P5 alias by design (alias-overlap discipline per K-1775),
K-1817 P33 has NO upstream alias overlap — both dtype rows are load-
bearing route-OUT entries.  This reflects the N=224 cliff finding from
K-1794: bf16 N=192 was caught by the upstream alias, but bf16 N=224 falls
out of every existing alias-stack predicate.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18.
  2. N axis is exactly {224}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Per-shape dtype-mirror (bf16 ↔ fp16 admit sets coincide).
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, P32 N=160, and the K-1367/K-1397 P13 N ∈ {128, 256}
     envelopes).
  6. Envelope is exactly the full 18-cell M ∈ {2048,4096,8192} × N=224
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
)


FZ = _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n224():
    assert {N for (_, N, _, _) in FZ} == {224}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_mirror_is_complete():
    """Each shape (M, N, K) admitted under bf16 is also admitted under fp16
    (and vice versa).  K-913 §3 LDS-bank-conflict is dtype-invariant on the
    column-narrow N=224 tile (same mechanism as K-1673 P28 at N=128 and
    K-1810 P32 at N=160)."""
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
    """K-1817 P33 (N=224) must be N-axis disjoint from K-1810 P32 (N=160).
    Pairwise-disjointness invariant per K-1175 — the two slots cover
    different N-rungs of the contiguous K-COMPLEMENT N-ladder."""
    assert FZ & _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18 == set()


def test_envelope_equals_full_n224_kcompl_grid():
    """The verified-winner envelope is exactly the full 18-cell M ∈ {2048,
    4096,8192} × N=224 × K ∈ {4096,8192,16384} × {bf16,fp16} grid — pinned
    to detect any silent contraction or expansion of the K-1794 sub-cohort
    envelope.  Per K-1794, both dtype rows were entirely unrouted (bf16 0/9
    at gmean 0.722, fp16 0/9 systemic), so all 18 cells are load-bearing
    route-OUT entries (no upstream alias overlap, unlike K-1810 P32 N=160)."""
    full = frozenset(
        (M, 224, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18
