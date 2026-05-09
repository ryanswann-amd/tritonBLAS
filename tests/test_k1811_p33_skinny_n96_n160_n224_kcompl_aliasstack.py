"""K-1811 (S-002) — 24th-slot N∈{96,160,224} K-COMPLEMENT alias-stack pinning tests.

Audit (K-1811, paired n=30 HIP-graph hot-cache on MI300X gfx942 OCI fallback
per INFRA-0048):

  * Cohort: N ∈ {96, 160, 224} × M ∈ {2048, 4096, 8192} × K ∈ {4096, 8192, 16384}
            × dtype ∈ {fp16, bf16}  (54 K-1794 cells).
  * 34 cells satisfy the strict K-1800 ship gate (HBL median ≥ 1.05× TB AND
    paired-diff t-test p < 0.05); cohort prepatch ratio_oracle geomean ≈ 0.696×
    (HBL ≈ 1.44× faster than tritonblas in-Triton persistent_matmul).
  * Sub-cohort decomposition:
      - N=224 fp16+bf16 (18 cells): full uncovered cohort, no upstream slot.
      - N=160 fp16      ( 9 cells): bf16 sibling already routed by R-K979 P5
                                    Clause-3 (minMN ≤ 192 ∧ K ≥ 2048, bf16-only).
      - N=96  fp16      ( 7 cells): bf16 sibling already routed; K=8192 fp16
                                    EXCLUDED because TB wins decisively there
                                    (K-1794 r=1.715/2.683 at M ∈ {2048, 8192};
                                    K-1811 prepatch confirmed r=1.758/2.773).
  * 6-cell adjacency guard band (N ∈ {128, 192} × M ∈ {2048, 4096, 8192}
    × K = 8192 × fp16): 0/6 regress > 2% post-route (K-1811 §4).

Mechanism (K-1795 wave-alignment finding): N ∈ {96, 160, 224} are wave-misaligned
against the MI300X 64-lane wave-front and accumulate LDS-bank-conflict + MFMA tail
inefficiency vs the wave-aligned siblings N ∈ {64, 128, 192} (already routed
by P29/P28/P32 respectively).  Sibling-N firewall vs P28 (N=128), P29 (N=64),
P30 (N ∈ {384,768,1536}), P31 (N ∈ {32,48,80}), P32 (N=192) by N-axis projection.

Invariants (all derived inline from the canonical frozenset, R-1532/R-1720
minimalist-admit-set rule):

  1. Cardinality is exactly 34.
  2. N is exactly {96, 160, 224}.
  3. dtype set is {torch.float16, torch.bfloat16} (bf16 only at N=224).
  4. M ∈ {2048, 4096, 8192}, K ∈ {4096, 8192, 16384} (full K-axis).
  5. Sub-cohort axis-projections are exact.
  6. Sibling-N firewall vs P28 (N=128), P29 (N=64), P30 (N ∈ {384,768,1536}),
     P31 (N ∈ {32,48,80}), P32 (N=192): zero overlap.
  7. K=8192 N=96 fp16 cells (M ∈ {2048, 4096, 8192}) are NOT in the slot
     (the M ∈ {2048, 8192} pair are confirmed TB-wins per K-1794/K-1811).

Cardinality lives in this file (not as an import-time `assert` in
`_route_predicate.py`) per the K-1748 minimalist split.
"""
from __future__ import annotations

from tritonblas._route_predicate import (
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29,
    _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34,
    _K1753_P31_SKINNY_N32_N48_N80_FP16_KCOMPL_ROUTEOUT_36,
    _K1800_P32_SKINNY_N192_FP16_KCOMPL_ALIASSTACK_9,
    _K1811_P33_SKINNY_N96_N160_N224_KCOMPL_ALIASSTACK_34,
)


FZ = _K1811_P33_SKINNY_N96_N160_N224_KCOMPL_ALIASSTACK_34


def test_cardinality_is_34():
    assert len(FZ) == 34


def test_n_axis_is_exactly_96_160_224():
    assert {N for (_, N, _, _) in FZ} == {96, 160, 224}


def test_dtype_set_is_fp16_and_bf16():
    """bf16 only appears at N=224; N ∈ {96, 160} bf16 already routed
    upstream by R-K979 P5 Clause-3 (minMN ≤ 192 ∧ K ≥ 2048, bf16-only)."""
    assert {dt for (_, _, _, dt) in FZ} == {"torch.float16", "torch.bfloat16"}
    bf16_n = {N for (_, N, _, dt) in FZ if dt == "torch.bfloat16"}
    assert bf16_n == {224}


def test_m_axis_is_dense():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}


def test_k_axis_is_kcomplement():
    """K ∈ {4096, 8192, 16384} — the K-COMPLEMENT band per K-1794 cohort."""
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}


def test_n224_subcohort_is_dense_18_cells():
    n224 = {(M, N, K, dt) for (M, N, K, dt) in FZ if N == 224}
    expected = {
        (M, 224, K, dt)
        for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.float16", "torch.bfloat16")
    }
    assert n224 == expected
    assert len(n224) == 18


def test_n160_subcohort_is_fp16_only_9_cells():
    n160 = {(M, N, K, dt) for (M, N, K, dt) in FZ if N == 160}
    expected = {
        (M, 160, K, "torch.float16")
        for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
    }
    assert n160 == expected
    assert len(n160) == 9


def test_n96_subcohort_excludes_2_tb_win_cells_7_cells():
    """K-1794 found (M, 96, 8192, fp16) where TB wins at M ∈ {2048, 8192}
    (r=1.715 at M=2048, r=2.683 at M=8192). K-1811 prepatch confirmed
    r=1.758/2.773. M=4096 is borderline (K-1794 r=0.743) but K-1811 paired
    n=30 confirms HBL at r=0.734, p=0 — kept. Exclude only the 2 TB-win
    cells to stay minimal-admit per R-1532/R-1720."""
    n96 = {(M, N, K, dt) for (M, N, K, dt) in FZ if N == 96}
    expected = {
        (M, 96, K, "torch.float16")
        for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
    } - {
        (2048, 96, 8192, "torch.float16"),
        (8192, 96, 8192, "torch.float16"),
    }
    assert n96 == expected
    assert len(n96) == 7
    # explicit: 2 TB-win cells excluded
    assert (2048, 96, 8192, "torch.float16") not in FZ
    assert (8192, 96, 8192, "torch.float16") not in FZ
    # explicit: borderline M=4096 cell IS included (HBL wins per K-1811)
    assert (4096, 96, 8192, "torch.float16") in FZ


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """K-1748 P30 admits N ∈ {384,768,1536}.  Disjoint by N projection."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p31_ultraskinny():
    """K-1753 P31 admits N ∈ {32,48,80}.  Disjoint by N projection."""
    assert FZ & _K1753_P31_SKINNY_N32_N48_N80_FP16_KCOMPL_ROUTEOUT_36 == set()


def test_sibling_n_firewall_vs_p32_n192():
    """K-1800 P32 admits N=192.  Disjoint by N projection."""
    assert FZ & _K1800_P32_SKINNY_N192_FP16_KCOMPL_ALIASSTACK_9 == set()


def test_full_construction_matches_canonical():
    expected = (
        {
            (M, 224, K, dt)
            for M in (2048, 4096, 8192)
            for K in (4096, 8192, 16384)
            for dt in ("torch.float16", "torch.bfloat16")
        }
        | {
            (M, 160, K, "torch.float16")
            for M in (2048, 4096, 8192)
            for K in (4096, 8192, 16384)
        }
        | (
            {
                (M, 96, K, "torch.float16")
                for M in (2048, 4096, 8192)
                for K in (4096, 8192, 16384)
            }
            - {
                (2048, 96, 8192, "torch.float16"),
                (8192, 96, 8192, "torch.float16"),
            }
        )
    )
    assert FZ == expected
