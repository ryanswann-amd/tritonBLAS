"""K-1825 (S-002) — 25th-slot N=288 K-COMPLEMENT alias-stack pinning tests.

Audit (K-1825, derived from K-1812 paired n=20 HIP-graph hot-cache PMC sweep
on MI300X gfx942, fallback host per INFRA-0048):

  * K-1812 cohort: 24 cells (M=4096 × N ∈ {96,128,160,224,256,288}
                            × K ∈ {4096, 16384} × dtype ∈ {fp16, bf16}).
  * 24/24 satisfy the strict admit gate (HBL median ≥ 1.05× TB AND
    paired-diff t-test p < 0.05); cohort TB/HBL ratios 3.10×–6.69×.
  * 20/24 are already covered by upstream slots:
      - N=128 (4): P13 N=128 K-COMPLEMENT (K-1367).
      - N=96 bf16, N=160 bf16 (4): R-K979 P5 Clause-3 (minMN ≤ 192, K ≥ 2048).
      - N=96 fp16, N=160 fp16 (4): P33 N ∈ {96,160,224} K-COMPLEMENT (K-1811).
      - N=224 (4): P33 N ∈ {96,160,224} K-COMPLEMENT (K-1811).
      - N=256 (4): P21 N=256 K-mid K-COMPLEMENT (K-1503).
  * 4/24 are net-new — N=288 at M=4096 × K ∈ {4096, 16384} × {bf16, fp16}.
  * Per-cell paired-t p-values: 4.83e-205, <1e-300, 1.97e-197, <1e-300
    (computed from K-1812 raw `tb_us` / `hbl_us` arrays, n=20 paired).

Conservative admit set: only the 4 cells K-1812 directly measured.
Extrapolation to M ∈ {2048, 8192} is intentionally deferred per K-1812 §7
("continue cell-by-cell verification per K-1681 / K-1710 / K-1781 precedent")
— a follow-up sweep at M ∈ {2048, 8192} would naturally extend P34 to
12 cells, but absent direct measurement those rows do NOT belong here under
the R-1532 / R-1720 minimalist-admit-set rule.

Sibling-N firewall: N=288 is disjoint from every prior slot's N-axis
projection (P28 N=128, P29 N=64, P30 N ∈ {384,768,1536}, P31 N ∈ {32,48,80},
P32 N=192, P33 N ∈ {96,160,224}, P21 N=256, P13 N ∈ {128,256}).

Cardinality lives in this file (not as an import-time `assert` in
`_route_predicate.py`) per the K-1748 minimalist split.
"""
from __future__ import annotations

from tritonblas._route_predicate import (
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29,
    _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34,
    _K1753_P31_SKINNY_N32_N48_N80_FP16_KCOMPL_ROUTEOUT_36,
    _K1800_P32_SKINNY_N192_FP16_KCOMPL_ALIASSTACK_9,
    _K1811_P33_SKINNY_N96_N160_N224_KCOMPL_ALIASSTACK_34,
    _K1825_P34_SKINNY_N288_KCOMPL_ALIASSTACK_4,
)


FZ = _K1825_P34_SKINNY_N288_KCOMPL_ALIASSTACK_4


def test_cardinality_is_4():
    assert len(FZ) == 4


def test_n_axis_is_exactly_288():
    assert {N for (_, N, _, _) in FZ} == {288}


def test_dtype_set_is_fp16_and_bf16():
    """Both dtypes admitted — K-1812 measured both at every cell."""
    assert {dt for (_, _, _, dt) in FZ} == {"torch.float16", "torch.bfloat16"}


def test_m_axis_is_4096_only():
    """K-1812 measured M=4096 only; extrapolation to M ∈ {2048, 8192}
    deferred per minimal-admit rule (R-1532 / R-1720)."""
    assert {M for (M, _, _, _) in FZ} == {4096}


def test_k_axis_is_kcomplement_4096_16384():
    """K ∈ {4096, 16384} — the K-1812 measured pair."""
    assert {K for (_, _, K, _) in FZ} == {4096, 16384}


def test_full_construction_matches_canonical():
    expected = {
        (4096, 288,  4096, "torch.bfloat16"),
        (4096, 288, 16384, "torch.bfloat16"),
        (4096, 288,  4096, "torch.float16"),
        (4096, 288, 16384, "torch.float16"),
    }
    assert FZ == expected


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """K-1748 P30 admits N ∈ {384, 768, 1536}.  Disjoint by N projection."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p31_ultraskinny():
    """K-1753 P31 admits N ∈ {32, 48, 80}.  Disjoint by N projection."""
    assert FZ & _K1753_P31_SKINNY_N32_N48_N80_FP16_KCOMPL_ROUTEOUT_36 == set()


def test_sibling_n_firewall_vs_p32_n192():
    """K-1800 P32 admits N=192.  Disjoint by N projection."""
    assert FZ & _K1800_P32_SKINNY_N192_FP16_KCOMPL_ALIASSTACK_9 == set()


def test_sibling_n_firewall_vs_p33_n96_n160_n224():
    """K-1811 P33 admits N ∈ {96, 160, 224}.  Disjoint by N projection."""
    assert FZ & _K1811_P33_SKINNY_N96_N160_N224_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p13_n128():
    """K-1367 P13 admits N=128.  Disjoint by N projection."""
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p13_n256():
    """K-1397 P13 admits N=256.  Disjoint by N projection."""
    assert FZ & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == set()


def test_sibling_n_firewall_vs_p21_n256_kmid():
    """K-1503 P21 admits N=256.  Disjoint by N projection."""
    assert FZ & _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT == set()
