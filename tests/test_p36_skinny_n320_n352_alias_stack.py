"""P36 (S-002) — 27th-slot N ∈ {320, 352} K-COMPLEMENT verified-winner subset tests.

Source measurement: paired n=30 HIP-graph hot-cache + 3-pass rocprofv2 PMC
sweep on MI300X / gfx942 (OCI MI300X fallback per INFRA-0048) across the
36-cell band M ∈ {2048,4096,8192} × N ∈ {320,352} × K ∈ {4096,8192,16384}
× {bf16,fp16}.  Cohort outcome:

  * Verified-winner gate (ratio ≥ 1.05  ∧  paired Student-t p < 0.05  ∧
    CI95-lo > 1.000): 34 / 36 cells PASS.
  * Per-N geomean (TB / HBL): N=320 → 1.539× (17/18), N=352 → 1.473× (17/18).
  * Cohort geomean: 1.506× (range 1.220×–1.902×).
  * 2 cells excluded — both at (M=2048, K=4096, bf16): (2048, 320, 4096,
    bf16) ratio 1.001, p=0.20, CI95=[0.989, 1.014]; (2048, 352, 4096, bf16)
    ratio 0.999, p=0.94, CI95=[0.986, 1.012].  These collapse to parity by
    construction because R_K979 P5 closed-form Clause-3 ALREADY routes them
    in the LIVE oracle (verified — see scripts/trace_routing.py).  Per
    R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST, alias-overlapping cells
    are excluded from new strict-equality slots to keep the slot
    load-bearing and free of redundant double-admit entries (mirrors K-1700
    P29's exclusion of (2048, 64, 4096, fp16)).

Mechanism (3-pass PMC delta, K-913 §3 / R-K1673 / R-1811): 36 / 36 cells
classify LDS_DOMINANT.  Per-cell PMC discriminator ranking (TB / HBL ratio):

  * SQ_LDS_BANK_CONFLICT/inst — 53.7× – 1.78e9× (median 393.9×; HBL ≈ 0)
  * SQ_WAIT_INST_LDS         — 1.97× – 25.26× (median 9.28×)
  * SQ_INSTS_MFMA / SQ_WAVES — 0.67× – 0.95× (TB does LESS MFMA per wave;
                                rules out compute-bound)
  * SQ_INSTS_VMEM / SQ_WAVES — 0.21× – 1.05× (rules out memory-BW)

BLOCK_N=128 packs N=320 (=256+64) and N=352 (=256+96) into wave-misaligned
K-block columns (off-by-64 / -96 N rungs above the N=256 cliff and below
the N=384 P30 productionised slot).  persistent_matmul cannot trade tile
reshape for atomic-reduction; only HBL's split-K kernel selection clears
the band.  Same SCHEDULER_LDS A4 failure mode as K-1681 / K-1710 / K-1781
/ K-1832 wave-misaligned skinny-N class.  hipBLASLt's split-K selection
clears the band by ~1.5× geomean.

Same fingerprint productionised at K-1673 P28 (N=128), K-1700 P29 (N=64),
K-1748 P30 (N ∈ {384, 768, 1536}), K-1775 P31 (N=256), K-1810 P32 (N=160),
K-1817 P33 (N=224), K-1831 P34 (N=96), K-1837 P35 (N=288) — now applied to
the wave-misaligned N ∈ {320, 352} rungs in the gap between P35 and P30.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 34.
  2. N axis is exactly {320, 352}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Per-shape dtype-mirror is symmetric on the M ∈ {4096, 8192} rows
     (the K=4096 bf16 parity exclusion at M=2048 breaks the strict
     full-cohort mirror — invariant 4 is conditioned on M, not global).
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, P32 N=160, P33 N=224, P34 N=96, P35 N=288, plus the
     K-1367/K-1397 P13 N ∈ {128, 256} envelopes).
  6. The exclusion set is exactly {(2048, 320, 4096, bf16),
     (2048, 352, 4096, bf16)} — pinned to detect silent expansion/
     contraction of the parity-band carve-out.
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
    _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18,
    _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18,
    _P36_SKINNY_N320_N352_KCOMPL_VERIFIED_WIN_34,
)


FZ = _P36_SKINNY_N320_N352_KCOMPL_VERIFIED_WIN_34


def test_cardinality_is_34():
    assert len(FZ) == 34


def test_n_axis_is_skinny_n320_n352():
    assert {N for (_, N, _, _) in FZ} == {320, 352}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_mirror_is_complete_for_m_4096_and_8192():
    """The K=4096 bf16 parity exclusion lives exclusively at M=2048; the
    M ∈ {4096, 8192} rows preserve the dtype-mirror invariance (K-913 §3
    LDS-bank-conflict is dtype-invariant on the column-narrow tile)."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16" and M != 2048}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16" and M != 2048}
    assert bf == fp
    # M ∈ {4096, 8192} × N ∈ {320, 352} × K ∈ {4096, 8192, 16384} = 12 shapes per dtype
    assert len(bf) == 12


def test_exclusion_set_is_exactly_the_two_parity_cells():
    """The 2 excluded cells are exactly the parity cells at small-M short-K bf16.
    Pinned to catch silent expansion (admitting a parity cell) or contraction
    (excluding a verified winner)."""
    full = frozenset(
        (M, N, K, dt) for M in (2048, 4096, 8192)
        for N in (320, 352)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    exclusion = full - FZ
    assert exclusion == frozenset({
        (2048, 320, 4096, "torch.bfloat16"),
        (2048, 352, 4096, "torch.bfloat16"),
    })
    assert len(exclusion) == 2


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


def test_sibling_n_firewall_vs_p34_n96():
    assert FZ & _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18 == set()


def test_sibling_n_firewall_vs_p35_n288():
    """P36 (N ∈ {320, 352}) must be N-axis disjoint from P35 (N=288).
    The two slots cover adjacent off-by-32/-64 wave-misaligned rungs above
    the N=256 productionised cliff."""
    assert FZ & _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18 == set()


def test_excluded_parity_cells_are_caught_by_upstream_p5():
    """The 2 parity-excluded cells must already be routed by R_K979 P5
    (closed-form Clause-3); otherwise the LIVE-oracle parity ratio would
    not be ~1.000 and the gate-failure rationale collapses.  Pinned to
    detect any future contraction of the P5 envelope that would silently
    leak these cells out of HBL routing."""
    import torch
    from tritonblas._route_predicate import R_K979_P5_route_to_hbl
    for cell in [(2048, 320, 4096, "torch.bfloat16"),
                 (2048, 352, 4096, "torch.bfloat16")]:
        M, N, K, dt_str = cell
        dt = torch.bfloat16 if "bfloat16" in dt_str else torch.float16
        assert R_K979_P5_route_to_hbl(M, N, K, dt), (
            f"P5 closed-form must already route {cell} for the parity exclusion "
            f"rationale to hold"
        )


def test_envelope_equals_full_n320_n352_kcompl_grid_minus_two():
    """The verified-winner envelope is exactly the full 36-cell grid minus
    the 2 parity exclusions — pinned to detect silent contraction or
    expansion of the cohort.  All 34 admitted cells are NEW route-OUT (no
    upstream alias overlap — N=320/352 are the off-by-64/-96 rungs in the
    gap between N=288 P35 and N=384 P30)."""
    full = frozenset(
        (M, N, K, dt) for M in (2048, 4096, 8192)
        for N in (320, 352)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert len(full) == 36
    assert FZ <= full
    assert len(full - FZ) == 2
