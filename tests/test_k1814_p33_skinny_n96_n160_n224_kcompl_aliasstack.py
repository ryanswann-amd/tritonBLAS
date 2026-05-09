"""K-1814 (S-002) — 24th-slot N ∈ {96, 160, 224} K-COMPLEMENT alias-stack pinning tests.

Audit (K-1814, paired n=30 hot-cache HIP-graph replay on MI300X gfx942;
c42 down per INFRA-0048 → OCI fallback):

  * Cohort: N ∈ {96, 160, 224} × M ∈ {2048, 4096, 8192} × K ∈ {4096, 8192, 16384}
            × dtype ∈ {fp16, bf16}  (54 cells = K-1794 cross-band envelope).
  * Admit set = 34 cells:
      - 9 bf16 N=224  (R-K979 P5 dtype-mirror gap; bf16 N=224 0/9 routed
        pre-patch, geomean 0.722× — first observed N-axis cliff above N=128
        in P28's K-COMPLEMENT envelope coverage per K-1794
        R-K1794.N224-DOES-NOT-EXTEND-FROM-N192-FROZENSET).
      - 9 fp16 N=224  (dtype sibling of bf16 N=224 above).
      - 9 fp16 N=160  (R-K979 P5 dtype-mirror gap; bf16 N=160 already
        routed-OUT by K-1673 P28 upstream).
      - 7 fp16 N=96   (M={2048,4096,8192} × K={4096,8192,16384} minus the
        2 cells (M=2048,K=8192) and (M=8192,K=8192) where TB-oracle is
        decisively faster than HBL pre-patch (ratio_med 1.643× and 2.733×
        respectively, paired-t p<0.05 in TB's favor)).
  * bf16 N ∈ {96, 160} EXCLUDED — already routed by K-1673 P28's broader
    K-COMPLEMENT envelope upstream at chain-position 19.

Mechanism (K-913 sec-3, re-confirmed by K-1781/K-1795 PMC RCA):
persistent_matmul on the wave-misaligned column-narrow LDS layout at
N ∈ {96, 160, 224} (none divisible by 64 = wavefront width) incurs
LDS-bank-conflict / MFMA tile mismatch saturation that hipBLASLt avoids
via K-adaptive MT-selection.  The bf16 N=224 cliff confirms the issue
is dtype-invariant and N-axis discontinuous (NOT smoothly decaying).
P33 sidesteps the bottleneck by delegating the 34 admit cells to
hipBLASLt — exactly as P32 does at N=192 and P5 Clause-3 does for
N ≤ 192 bf16.  Sibling-N firewall vs P28 (N=128), P29 (N=64),
P30 (N ∈ {384, 768, 1536}), P31 (N ∈ {32, 48, 80}), P32 (N=192),
and P21 (N=256) by N-axis projection.

Invariants (all derived inline from the canonical frozenset, R-1532/R-1720
minimalist-admit-set rule):

  1. Cardinality is exactly 34.
  2. N is exactly {96, 160, 224}.
  3. dtype is {torch.float16, torch.bfloat16}.
  4. bf16 admits exactly the dense N=224 product (no bf16 N ∈ {96,160}).
  5. fp16 N=160 / N=224 admit the dense product; fp16 N=96 admits exactly
     7 cells, with (M=2048,K=8192) and (M=8192,K=8192) excluded (TB wins).
  6. Sibling-N firewall vs P28 (N=128), P29 (N=64), P30 (N ∈ {384,768,1536}),
     P31 (N ∈ {32,48,80}), P32 (N=192): zero overlap.

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
    _K1814_P33_SKINNY_N96_N160_N224_KCOMPL_ALIASSTACK_34,
)


FZ = _K1814_P33_SKINNY_N96_N160_N224_KCOMPL_ALIASSTACK_34


def test_cardinality_is_34():
    assert len(FZ) == 34


def test_n_axis_is_96_160_224():
    assert {N for (_, N, _, _) in FZ} == {96, 160, 224}


def test_dtype_axis_is_fp16_and_bf16():
    assert {dt for (_, _, _, dt) in FZ} == {"torch.float16", "torch.bfloat16"}


def test_bf16_admits_exactly_n224_dense():
    """bf16 admit set = dense N=224 product; bf16 N ∈ {96,160} already
    routed by K-1673 P28 upstream so excluded from P33 by construction."""
    bf16 = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    expected = {(M, 224, K)
                for M in (2048, 4096, 8192)
                for K in (4096, 8192, 16384)}
    assert bf16 == expected


def test_fp16_n160_and_n224_are_dense():
    fp16_n160 = {(M, K) for (M, N, K, dt) in FZ
                 if dt == "torch.float16" and N == 160}
    fp16_n224 = {(M, K) for (M, N, K, dt) in FZ
                 if dt == "torch.float16" and N == 224}
    dense = {(M, K) for M in (2048, 4096, 8192) for K in (4096, 8192, 16384)}
    assert fp16_n160 == dense
    assert fp16_n224 == dense


def test_fp16_n96_excludes_tb_win_cells():
    """fp16 N=96 (M=2048,K=8192) and (M=8192,K=8192) are TB win zones
    (paired n=30: ratio_med 1.643× and 2.733× in TB's favor with
    p<<0.05) — DELIBERATELY excluded from the admit set."""
    fp16_n96 = {(M, K) for (M, N, K, dt) in FZ
                if dt == "torch.float16" and N == 96}
    expected = {(M, K) for M in (2048, 4096, 8192) for K in (4096, 8192, 16384)}
    expected -= {(2048, 8192), (8192, 8192)}
    assert fp16_n96 == expected
    assert len(fp16_n96) == 7


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
    """K-1800 P32 admits N=192.  Disjoint by N projection
    (P33 admits {96, 160, 224} only)."""
    assert FZ & _K1800_P32_SKINNY_N192_FP16_KCOMPL_ALIASSTACK_9 == set()


def test_full_admit_set_matches_canonical():
    bf16 = {(M, 224, K, "torch.bfloat16")
            for M in (2048, 4096, 8192) for K in (4096, 8192, 16384)}
    fp16_n224 = {(M, 224, K, "torch.float16")
                 for M in (2048, 4096, 8192) for K in (4096, 8192, 16384)}
    fp16_n160 = {(M, 160, K, "torch.float16")
                 for M in (2048, 4096, 8192) for K in (4096, 8192, 16384)}
    fp16_n96 = {(M, 96, K, "torch.float16") for (M, K) in (
        (2048, 4096), (2048, 16384),
        (4096, 4096), (4096, 8192), (4096, 16384),
        (8192, 4096), (8192, 16384),
    )}
    expected = bf16 | fp16_n224 | fp16_n160 | fp16_n96
    assert len(expected) == 34
    assert FZ == expected
