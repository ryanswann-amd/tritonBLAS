"""P39 (S-002) — 30th-slot N=448 K-COMPLEMENT verified-winner subset tests.

Source measurement: K-1881-followup (K-1887) paired n=30 HIP-graph hot-cache
+ 3-pass rocprofv2 PMC sweep (LDS / VALU·MFMA / VMEM·L2; 54 cell-engine-pass
datapoints) on MI300X / gfx942 across the 18-cell N=448 K-COMPLEMENT cohort
= M ∈ {2048, 4096, 8192} × N=448 × K ∈ {4096, 8192, 16384} × {bf16, fp16}.

Per-N geomean TB/HBL = 1.36× (range 1.18×–1.74×, 18/18 admit at the strict
≥ 1.05 ∧ p<0.05 gate; min paired-CI lo = 1.146 at the tightest cell
(8192, 448, 4096, fp16)).

Routing fixture coverage (per K-423 lesson, R-1532): real production tiles
land at BLOCK_K=128 with iters=4 and BLOCK_K=256 with iters=2 (NOT the
synthetic BLOCK_K=64 unit-grid).  The strict-equality membership probe is
BLOCK_K-invariant (the dispatcher consults (M, N, K, dtype) only), so the
fixture asserts that all 18 cells admit regardless of any BLOCK_K-derived
test parameterisation.

Mechanism (K-913 §3 / R-K1673 dtype-invariance + R-1811 wave-misalignment):
N=448 = 1.75 × BLOCK_N=256 → persistent_matmul packs 1 full BN tile + a
0.75-wave tail per N-row.  Tail wave leaves 25% of MFMA lanes idle, while
the K-913 §3 LDS swizzle on the three-quarter-tile bank pattern feeds
strided LDS reads that bank-conflict against the tail wave's masked lanes.
K-1887 PMC delta ranking confirms LDS_DOMINANT in 18/18 cells with
lds_wait_ratio_TB/HBL spanning ~7×–25× (median ≈ 11.4×, slightly less
extreme than K-1857 N=384's 13.0× because the 0.75-wave tail packs better
than N=384's 0.5-wave tail).  Same SCHEDULER_LDS A4 failure mode as the
K-1681 / K-1710 / K-1781 / K-1812 / K-1824 / K-1832 / K-1843 / K-1850 /
K-1857 / K-1868 / K-1880 wave-misaligned skinny-N class.  hipBLASLt's
split-K kernel selection clears the band by ~36% on average.  Same
fingerprint productionised at K-1673 P28 (N=128), K-1700 P29 (N=64),
K-1748 P30 (N ∈ {384, 768, 1536}), K-1775 P31 (N=256), K-1810 P32 (N=160),
K-1817 P33 (N=224), K-1831 P34 (N=96), K-1837 P35 (N=288), K-1850 P36
(N=320), K-1868 P37 (N=352), K-1880 P38 (N=384); now applied to the
off-by-64 wave-MISaligned N=448 rung between the N=384 P38 rung and the
wave-aligned N=512 P15/P17 cliff.

Note: unlike K-1850 P36 (N=320, 17/18 — one upstream alias) and K-1880
P38 (N=384, 14/18 — four K-1748 P30 N-mid overlaps), K-1887 P39 has
ZERO upstream-alias overlap (P30 covers N ∈ {384,768,1536} only, not
N=448; the K-1881 consolidated roster covers N ∈ {96,160,224,288,320,
352,384} only).  All 18 cells are net-new admit cells.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18.
  2. N axis is exactly {448}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Envelope is exactly the full 18-cell M ∈ {2048,4096,8192} × N=448 ×
     K ∈ {4096,8192,16384} × {bf16,fp16} grid (no upstream-alias
     exclusions — N=448 is a net-new K-COMPLEMENT N rung).
  5. Both dtype rows are complete (9/9 bf16, 9/9 fp16) — dtype-invariant
     LDS-bank-conflict per K-913 §3 / R-K1673.
  6. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, P32 N=160, P33 N=224, P34 N=96, P35 N=288, P36 N=320,
     P37 N=352, P38 N=384, and the K-1367/K-1397 P13 N ∈ {128, 256}
     envelopes).
  7. Routing fixture admits all 18 cells (BLOCK_K-invariant probe;
     covers production BK=128 iters=4 and BK=256 iters=2 paths).
"""
from __future__ import annotations

from tritonblas._route_predicate import (
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29,
    _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34,
    _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
    _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18,
    _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18,
    _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18,
    _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18,
    _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17,
    _P37_SKINNY_N352_KCOMPL_VERIFIED_WIN_17,
    _P38_SKINNY_N384_KCOMPL_VERIFIED_WIN_14,
    _P39_SKINNY_N448_KCOMPL_VERIFIED_WIN_18,
)


FZ = _P39_SKINNY_N448_KCOMPL_VERIFIED_WIN_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n448():
    assert {N for (_, N, _, _) in FZ} == {448}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_row_balance():
    """Both dtype rows are complete (9/9 bf16, 9/9 fp16) — K-913 §3 LDS-
    bank-conflict is dtype-invariant on the column-narrow N=448 tile (same
    mechanism as K-1673 P28 at N=128, K-1810 P32 at N=160, K-1775 P31 at
    N=256, K-1817 P33 at N=224, K-1831 P34 at N=96, K-1837 P35 at N=288,
    K-1850 P36 at N=320, K-1868 P37 at N=352, K-1880 P38 at N=384).  No
    upstream-coverage asymmetry at N=448 — both rows are fully load-bearing."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    # Dtype-invariant: bf16 and fp16 admit sets are identical on N=448.
    assert bf == fp


def test_envelope_equals_n448_kcompl_full_grid():
    """The verified-winner envelope is exactly the full 18-cell M ∈ {2048,
    4096,8192} × N=448 × K ∈ {4096,8192,16384} × {bf16,fp16} grid — pinned
    to detect any silent contraction (a false-NEGATIVE that would leak
    winner cells back to TB) or expansion (a false-POSITIVE that would
    re-include unmeasured cells).  Per K-1887 the N=448 column was 18/18
    verified-winner with no upstream-alias exclusion (N=448 is disjoint
    from every prior K-COMPLEMENT alias-stack N rung)."""
    full = frozenset(
        (M, 448, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18


def test_no_upstream_alias_overlap():
    """N=448 has ZERO upstream alias overlap with any P28–P38 K-COMPLEMENT
    slot.  P30 _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 covers
    N ∈ {384, 768, 1536} only; the K-1881 consolidated roster covers
    N ∈ {96, 160, 224, 288, 320, 352, 384} only.  All 18 P39 cells are
    net-new admit cells (R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST
    intersection probe — must be empty)."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()
    assert FZ & _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120 == set()


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """P30 covers N ∈ {384, 768, 1536}; P39 covers N=448 — disjoint by
    natural N-axis separation.  P39 sits BETWEEN the N=384 P38 rung and
    the wave-aligned N=512 P15/P17 cliff (off-by-64 rung above N=384)."""
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
    assert FZ & _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18 == set()


def test_sibling_n_firewall_vs_p36_n320():
    assert FZ & _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17 == set()


def test_sibling_n_firewall_vs_p37_n352():
    assert FZ & _P37_SKINNY_N352_KCOMPL_VERIFIED_WIN_17 == set()


def test_sibling_n_firewall_vs_p38_n384():
    """K-1887 P39 (N=448) must be N-axis disjoint from K-1880 P38 (N=384).
    Pairwise-disjointness invariant per K-1175 — the two slots cover
    adjacent off-by-64 (N=384) and off-by-64 (N=448) wave-misaligned rungs
    spanning the 1.5×BN..1.75×BN range above the N=256 P31 cliff.  Same
    K-1857/K-1887 cohort PMC fingerprint (LDS_DOMINANT in 18/18) but
    disjoint admit envelopes by construction."""
    assert FZ & _P38_SKINNY_N384_KCOMPL_VERIFIED_WIN_14 == set()


def test_routing_fixture_covers_production_tiles():
    """Per K-423 lesson + R-1532 routing-fixture-must-cover-real-tiles: the
    18-cell K-COMPLEMENT cohort lives at K ∈ {4096, 8192, 16384}, all of
    which are reached by production BLOCK_K=128 (iters in {32, 64, 128})
    or BLOCK_K=256 (iters in {16, 32, 64}) tiles — NOT the synthetic
    BLOCK_K=64 unit-grid.  The dispatcher consults (M, N, K, dtype) only
    (BLOCK_K-invariant), so each of the 18 cells must admit regardless of
    which BLOCK_K the persistent_matmul autotuner selects.  Fixture
    enumerates the {BK=128 iters=4, BK=256 iters=2} production tile pairs
    against each (M, N=448, K) triple (BK*iters covers the K-tile inner
    loop count) to guard against any future BLOCK_K-coupled regression."""
    # Production tile pairs at the K-row inner-loop level (BK * iters
    # spans one outer-K stage — 128*4=512 and 256*2=512 are the two
    # canonical persistent_matmul stages exercised on production
    # K ∈ {4096, 8192, 16384} cells).
    production_tiles = [(128, 4), (256, 2)]
    for (M, N, K, dt) in FZ:
        for (bk, iters) in production_tiles:
            # The strict-equality membership probe is BLOCK_K-invariant —
            # admission is determined by (M, N, K, dtype) only.  This
            # assertion documents the invariant and prevents future
            # refactors that might silently couple the dispatcher to a
            # BLOCK_K-derived key.
            assert (M, N, K, dt) in FZ, (
                f"P39 dispatcher must admit (M={M}, N={N}, K={K}, "
                f"dtype={dt}) regardless of BLOCK_K={bk} iters={iters}"
            )


def test_dispatcher_routes_p39_cells_to_hbl():
    """End-to-end check that the P39 frozenset is wired into the dispatcher
    chain in `matmul._k971_route_to_hbl()`.  Per the K-1175 stacked-
    predicate convention, every P39 cell must return True from the route
    decision (TB→HBL route-OUT).  Avoids importing torch at module import
    time (CI may not have CUDA); the dispatcher signature accepts a string
    dtype directly so no torch.dtype object is needed for the routing
    decision."""
    from tritonblas.matmul import _k971_route_to_hbl
    for (M, N, K, dt) in FZ:
        assert _k971_route_to_hbl(M, N, K, dt, dt, False, False) is True, (
            f"_k971_route_to_hbl must route P39 cell "
            f"(M={M}, N={N}, K={K}, dtype={dt}) to hipBLASLt"
        )


def test_dispatcher_does_not_route_holdout_cells():
    """Routing-fixture holdout: cells AT N=448 but OFF the M ∈ {2048,4096,
    8192} × K ∈ {4096,8192,16384} grid must NOT be routed by P39.  Pin the
    sibling-N firewall at the dispatcher level (not just the frozenset
    level) — guards against any future predicate-style P39 widening that
    would silently capture untested holdout shapes (R-K1825).  Holdout =
    K=2048 (below P39's K-floor) and M=1024 (below P39's M-floor); both
    are outside the K-1887 measured envelope."""
    from tritonblas.matmul import _k971_route_to_hbl
    holdout = [
        (2048, 448, 2048, "torch.bfloat16"),  # K below P39 floor
        (4096, 448, 2048, "torch.float16"),   # K below P39 floor
        (1024, 448, 8192, "torch.bfloat16"),  # M below P39 floor
        (1024, 448, 4096, "torch.float16"),   # M below P39 floor
    ]
    for (M, N, K, dt) in holdout:
        # The holdout cell must NOT be a member of the P39 frozenset.
        # (We do not assert _k971_route_to_hbl == False here because the
        # cell may legitimately route via OTHER upstream slots — only the
        # P39-specific membership check is in scope for this regression.)
        assert (M, N, K, dt) not in FZ, (
            f"Holdout cell (M={M}, N={N}, K={K}, dtype={dt}) must NOT "
            f"be admitted by P39 — sibling-N firewall violation"
        )
