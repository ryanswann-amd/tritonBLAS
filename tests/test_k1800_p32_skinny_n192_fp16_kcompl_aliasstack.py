"""K-1800 (S-002) — 23rd-slot N=192 fp16 K-COMPLEMENT alias-stack pinning tests.

Audit (K-1800, building on K-1782 paired n=30 hot-cache HIP-graph replay
on MI300X gfx942; c42 down per INFRA-0048):

  * Cohort: N=192 × M ∈ {2048, 4096, 8192} × K ∈ {4096, 8192, 16384}
            × dtype ∈ {fp16, bf16}  (18 cells).
  * fp16 sub-cohort (9 cells): 9/9 cells satisfy the strict K-1800 ship
    gate (HBL median ≥ 1.05× TB-direct AND paired-diff t-test p < 0.05);
    cohort fp16 ratio_oracle geomean = 0.667× pre-route (HBL ~1.50×
    faster than tritonblas in-Triton persistent_matmul).
  * bf16 sub-cohort (9 cells): 9/9 already routed-OUT by R-K979 P5
    Clause-3 (minMN ≤ 192 ∧ K ≥ 2048, gated `_dtype_is_bf16`) at chain
    position 4 — never reaches this 23rd-slot, so excluded from P32 by
    construction (would be structural alias-overlap).
  * 6-cell adjacency guard band (N ∈ {128, 256} × M ∈ {2048, 4096, 8192}
    × K = 8192 × fp16): 0/6 regress > 2% post-route (K-1800 §4).

Mechanism (K-913 sec-3, re-confirmed by K-1781 PMC RCA): persistent_matmul
on the N=192 column-narrow LDS layout incurs an LDS-bank-conflict
saturation that is dtype-invariant.  The bf16 row's HBL win therefore
extends cleanly to fp16.  Sibling-N firewall vs P28 (N=128) and P21
(N=256) by N-axis projection.

Invariants (all derived inline from the canonical frozenset, R-1532/R-1720
minimalist-admit-set rule):

  1. Cardinality is exactly 9.
  2. N is exactly {192}.
  3. dtype is {torch.float16} only.
  4. M ∈ {2048, 4096, 8192}, K ∈ {4096, 8192, 16384} (full dense product).
  5. Sibling-N firewall vs P28 (N=128), P29 (N=64), P30 (N ∈ {384,768,1536}),
     P31 (N ∈ {32,48,80}): zero overlap.

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
)


FZ = _K1800_P32_SKINNY_N192_FP16_KCOMPL_ALIASSTACK_9


def test_cardinality_is_9():
    assert len(FZ) == 9


def test_n_axis_is_exactly_192():
    """K-1800 P32 is the N=192 dtype-mirror slot — single N rung."""
    assert {N for (_, N, _, _) in FZ} == {192}


def test_dtype_is_fp16_only():
    """bf16 cells in the same (M, 192, K) cohort already route-OUT
    upstream via R-K979 P5 Clause-3 (minMN ≤ 192 ∧ K ≥ 2048, bf16-only).
    Including bf16 here would be structural alias-overlap, not new work."""
    assert {dt for (_, _, _, dt) in FZ} == {"torch.float16"}


def test_m_axis_is_dense():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}


def test_k_axis_is_kcomplement():
    """K ∈ {4096, 8192, 16384} — the K-COMPLEMENT band per the K-1782
    cohort definition (matches K-1673 P28 N=128 K-axis)."""
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}


def test_sibling_n_firewall_vs_p28_n128():
    """K-1673 P28 admits N=128; K-1800 P32 admits N=192.  N-axis disjoint."""
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """K-1748 P30 admits N ∈ {384,768,1536}.  Disjoint by N projection."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p31_ultraskinny():
    """K-1753 P31 admits N ∈ {32,48,80}.  Disjoint by N projection."""
    assert FZ & _K1753_P31_SKINNY_N32_N48_N80_FP16_KCOMPL_ROUTEOUT_36 == set()


def test_full_dense_construction_matches_canonical():
    """The frozenset is exactly the dense product
    {(M, 192, K, fp16) : M ∈ M-axis, K ∈ K-axis} — alias-row of the bf16
    cohort already caught by R-K979 P5 Clause-3 upstream."""
    expected = {
        (M, 192, K, "torch.float16")
        for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
    }
    assert FZ == expected
