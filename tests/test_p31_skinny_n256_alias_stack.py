"""P31 (S-002) — 22nd-slot N=256 K-COMPLEMENT verified-winner subset tests.

Source measurement: TB-native vs HBL-native paired n=30 hot-cache HIP-graph
capture/replay on MI300X / gfx942 with the alias-stack route-OUT logic
ABLATED so the engine path is forced.  Cohort geomean HBL/TB = 1.359×
across the full 30-cell M ∈ {2048,4096,8192} × N=256 × K ∈ {2048,4096,8192,
16384,32768} × {bf16,fp16} grid; 28/30 cells pass the strict ≥1.05 ∧
p<0.05 winner gate.  The 2 LOSER cells at (M=2048, K=2048) — TB-native
beats hipBLASLt by 2% in bf16 (HBL/TB=0.98) and ties in fp16 (HBL/TB=1.02,
p=0.79) — are excluded from the productionized envelope per the
verified-winner constraint.

Winner-subset geomean HBL/TB = 1.392× (n=28), range 1.07× – 2.12×.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants):

  1. Cardinality is exactly 28.
  2. N axis is exactly {256}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {2048, 4096, 8192, 16384, 32768};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. The 2 LOSER cells at (2048, 256, 2048, *) are explicitly excluded.
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, and
     the K-1367 P13 N=128 envelope).
  6. Alias-overlap with the upstream P21 K-mid envelope (17 cells) is
     intentional and complete; alias-overlap with the upstream P13
     K-extremes envelope (11 cells) intersects the winner subset
     completely.
  7. Per-shape dtype-mirror (bf16 ↔ fp16 admit sets coincide) on the
     verified-winner subset.
"""
from __future__ import annotations

from tritonblas._route_predicate import (
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT,
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29,
    _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
)


FZ = _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28
LOSERS = frozenset({
    (2048, 256, 2048, "torch.bfloat16"),
    (2048, 256, 2048, "torch.float16"),
})


def test_cardinality_is_28():
    assert len(FZ) == 28


def test_n_axis_is_skinny_n256():
    assert {N for (_, N, _, _) in FZ} == {256}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {2048, 4096, 8192, 16384, 32768}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_loser_cells_are_excluded():
    """The 2 paired-n30 LOSER cells at (M=2048, K=2048) MUST be excluded
    from the productionized envelope: they violate the verified-winner
    gate (HBL/TB < 1.05 OR p ≥ 0.05) so routing them OUT would regress
    real hot-cache performance."""
    for cell in LOSERS:
        assert cell not in FZ, f"loser cell {cell} must not be in envelope"


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_p21_kmid_alias_overlap_is_complete():
    """All 17 cells in the P21 K-mid envelope (M ∈ {2048,4096,8192} ×
    N=256 × K ∈ {4096,8192,16384}) appear in the verified-winner subset
    (none of them is a (M=2048, K=2048) loser)."""
    assert _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT.issubset(FZ)
    assert len(FZ & _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT) == 17


def test_p13_kextremes_subset_intersection():
    """The 12-cell P13 K-extremes envelope (K ∈ {2048, 32768}) intersects
    the verified-winner subset in exactly 10 cells — the 2 cells at
    (2048, 256, 2048, *) are LOSERS and excluded."""
    overlap = FZ & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12
    assert len(overlap) == 10
    excluded = _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 - FZ
    assert excluded == LOSERS


def test_dtype_mirror_is_complete():
    """Each shape (M, N, K) admitted under bf16 is also admitted under
    fp16 (and vice versa).  The K-913 §3 LDS-bank-conflict mechanism is
    dtype-invariant on the column-narrow N=256 tile, and the 2 excluded
    losers form a (M, K) = (2048, 2048) dtype pair."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert bf == fp


def test_envelope_equals_full_grid_minus_losers():
    """The verified-winner envelope is exactly the full 30-cell N=256
    K-COMPLEMENT grid minus the 2 (M=2048, K=2048) loser cells — pinned
    to detect any silent contraction or expansion of the envelope."""
    full = frozenset(
        (M, 256, K, dt) for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full - LOSERS
    assert len(full) == 30
    assert len(FZ) == 28
