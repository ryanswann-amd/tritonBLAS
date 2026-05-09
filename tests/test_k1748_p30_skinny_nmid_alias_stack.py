"""K-1748 (S-002) — 21st-slot alias-stack pinning tests.

Audit closed by K-1748: the live post-K-1709 oracle had 0/34 cells of the
K-1711 N-mid envelope active (K-1720's 20th-slot proposal was a parallel
branch off K-1685 that never merged into the K-1709 lineage).  Ship the
K-1711 envelope at the 21st alias-stack slot.

Invariants pinned (all derived inline from the canonical frozenset to honor
the R-1532 / R-1720 minimalist-admit-set rule):

  1. Cardinality is exactly 34.
  2. N ∈ {384, 768, 1536}; per-N admit shape matches K-1711's measured
     CI95-gated 10 / 14 / 10 split.
  3. Sibling-N firewall vs P29 (N=64): zero overlap.
  4. K-only members ∈ {2048, 8192, 32768}; M-only members ∈ {2048, 4096, 8192};
     dtype set is exactly {torch.bfloat16, torch.float16}.
  5. Module-load assertion in `_route_predicate` is intact.
"""
from __future__ import annotations

from tritonblas._route_predicate import (
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29,
    _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34,
)


FZ = _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34


def test_cardinality_is_34():
    assert len(FZ) == 34


def test_n_axis_is_skinny_nmid():
    assert {N for (_, N, _, _) in FZ} == {384, 768, 1536}


def test_per_n_admit_shape_matches_k1711_csv():
    by_n = {n: {(M, K, dt) for (M, N, K, dt) in FZ if N == n}
            for n in (384, 768, 1536)}
    assert len(by_n[384]) == 10
    assert len(by_n[768]) == 14   # worst seam
    assert len(by_n[1536]) == 10


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} <= {2048, 8192, 32768}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_mirror_is_complete():
    """Each shape (M, N, K) admitted under bf16 is also admitted under fp16
    (and vice versa) — mirrors the K-1711 CSV where fp16 and bf16 share
    the K-913 §3 LDS-bank-conflict mechanism."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert bf == fp
