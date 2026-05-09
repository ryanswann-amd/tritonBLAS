"""K-1753 (S-002) — 22nd-slot ultra-skinny N∈{32,48,80} fp16 K-COMPLEMENT
route-OUT pinning tests.

Audit (K-1753): paired n=30 hot-cache HIP-graph replay on MI300X
(gfx942) over the 96-cell grid
N ∈ {32, 48, 80, 96} × M ∈ {1024, 2048, 4096, 8192}
× K ∈ {2048, 8192, 32768} × dtype ∈ {fp16, bf16}.

Findings:
  * 36/36 fp16 cells in N ∈ {32, 48, 80} flagged at the strict 1.05× admit
    gate (cohort fp16 geomean 2.965×, range 1.074×-26.040×).
  * All 48 cohort bf16 cells already routed-OUT by R-K979 P5 Clause-3
    upstream in the dispatch chain (verified out-of-band by
    `scripts/k1753_route_check.py`).  Excluded from this slot — would be
    structural alias-overlap, not new productionization.
  * N=96 swept as the adjacency boundary: 10/12 fp16 flagged = 83.3%,
    below the 95% verdict threshold → ENVELOPE_COVERS, deliberately
    excluded so the slot is contiguous N ∈ {32, 48, 80} and the boundary
    stays uncoupled.

Invariants (all derived inline from the canonical frozenset, R-1532/R-1720
minimalist-admit-set rule):

  1. Cardinality is exactly 36.
  2. N ∈ {32, 48, 80} (boundary N=96 excluded; N=64 belongs to P29).
  3. Sibling-N firewall vs P29 (N=64) and P30 (N ∈ {384, 768, 1536}):
     zero overlap.
  4. dtype is {torch.float16} only — first fp16-only alias-stack slot
     (bf16 path covered by R-K979 P5 Clause-3 upstream).
  5. Full dense M × K product per N: M ∈ {1024, 2048, 4096, 8192},
     K ∈ {2048, 8192, 32768}.

Cardinality lives in this file (not as an import-time `assert` in
`_route_predicate.py`) per the K-1748 minimalist split.
"""
from __future__ import annotations

from tritonblas._route_predicate import (
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29,
    _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34,
    _K1753_P31_SKINNY_N32_N48_N80_FP16_KCOMPL_ROUTEOUT_36,
)


FZ = _K1753_P31_SKINNY_N32_N48_N80_FP16_KCOMPL_ROUTEOUT_36


def test_cardinality_is_36():
    assert len(FZ) == 36


def test_n_axis_is_ultra_skinny():
    """N ∈ {32, 48, 80}: contiguous ultra-skinny band excluding the
    productionized N=64 (P29) and the adjacency-boundary N=96 (83.3%
    flag rate, below 95% gate)."""
    assert {N for (_, N, _, _) in FZ} == {32, 48, 80}


def test_per_n_admit_shape_is_dense_mk():
    """Each N admits the FULL M × K product (no exclusions) — the
    flag rate at the 1.05× gate was 12/12 = 100% per N."""
    by_n = {n: {(M, K, dt) for (M, N, K, dt) in FZ if N == n}
            for n in (32, 48, 80)}
    assert len(by_n[32]) == 12  # 4 M × 3 K × 1 dtype
    assert len(by_n[48]) == 12
    assert len(by_n[80]) == 12


def test_sibling_n_firewall_vs_p29_n64():
    """K-1700 P29 admits N=64 only; K-1753 P31 admits N ∈ {32,48,80}.
    N-axis projection is disjoint by construction."""
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """K-1748 P30 admits N ∈ {384,768,1536}; K-1753 P31 admits N ≤ 80.
    Disjoint by N-axis projection."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_dtype_is_fp16_only():
    """K-1753 P31 is the FIRST fp16-only alias-stack slot.  bf16 cells in
    the same (N, M, K) cohort are already routed by R-K979 P5 Clause-3
    upstream in the dispatch chain — including them here would be a
    structural alias-overlap (silent re-routing) rather than new
    productionization."""
    assert {dt for (_, _, _, dt) in FZ} == {"torch.float16"}


def test_m_axis_is_dense_1024_to_8192():
    """M ∈ {1024, 2048, 4096, 8192} — K-1753 extends K-1709/K-1748's
    M ∈ {2048, 4096, 8192} grid down to M=1024 because at ultra-skinny
    N ≤ 80 the M-block-column count is small enough that even M=1024
    cells lose decisively (e.g., (1024, 32, 32768, fp16) ratio = 26.040×)."""
    assert {M for (M, _, _, _) in FZ} == {1024, 2048, 4096, 8192}


def test_k_axis_is_kcomplement_subset():
    """K ∈ {2048, 8192, 32768} — the canonical K-COMPLEMENT triplet used
    in the K-1700/K-1709 envelope sweep.  K=4096, K=16384 not measured
    in K-1753 (per task brief; K-1732 measured them at N ∈ {48, 80} and
    found dtype-symmetric flags, so the missing-K cells are a
    follow-up audit candidate but not required for this slot)."""
    assert {K for (_, _, K, _) in FZ} == {2048, 8192, 32768}


def test_n96_adjacency_boundary_excluded():
    """N=96 was swept as the adjacency boundary and flagged 10/12 fp16
    cells (83.3%, below the 95% verdict gate).  Ensure it stays out."""
    assert {(M, N, K, dt) for (M, N, K, dt) in FZ if N == 96} == set()


def test_full_dense_construction_matches_canonical():
    """The frozenset is exactly the dense product
    {(M, N, K, fp16) : M ∈ M-axis, N ∈ {32,48,80}, K ∈ K-axis} — no
    bubble-cell exclusions (unlike P29 which dropped (2048,64,4096,fp16)
    at CI95-lo=1.045)."""
    expected = {
        (M, N, K, "torch.float16")
        for N in (32, 48, 80)
        for M in (1024, 2048, 4096, 8192)
        for K in (2048, 8192, 32768)
    }
    assert FZ == expected
