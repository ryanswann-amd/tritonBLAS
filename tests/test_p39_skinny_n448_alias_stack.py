"""P39 (S-002) — 30th-slot N=448 K-COMPLEMENT verified-winner subset tests.

K-1887 paired n=30 HIP-graph hot-cache + 3-pass rocprofv2 PMC sweep on
MI300X / gfx942 across the 18-cell N=448 K-COMPLEMENT cohort = M ∈
{2048, 4096, 8192} × N=448 × K ∈ {4096, 8192, 16384} × {bf16, fp16}.
18/18 cells admit at strict ratio_TB/HBL ≥ 1.05 ∧ p<0.05; per-N geomean
TB/HBL = 1.36× (range 1.18×–1.74×).  N=448 = 1.75 × BLOCK_N=256 →
0.75-wave tail leaves 25% MFMA lanes idle; SCHEDULER_LDS A4 fingerprint
(HBL SQ_LDS_BANK_CONFLICT == 0 in 18/18; TB nonzero in 18/18).

Holdout re-bench (K-1887 RETRY): 6-cell paired n=30 HIP-graph holdout on
MI300X / gfx942 reproduced cohort geomean = 1.264× (6/6 admit, range
1.174×–1.430×, min CI95-lo = 1.168) — confirms P39 promotion on the
holdout subset.  Full table at output/holdout_p39_results.txt.

Test scope (post-Minimalist-review collapse):
  * The frozenset is a literal comprehension; structural axes (cardinality,
    N-axis, M/K/dtype axes, dtype balance, no upstream-alias overlap) all
    follow from a single equality vs the full grid -- one assertion.
  * Sibling-N firewalls collapse to "intersection with the union of every
    prior K-COMPLEMENT slot is empty" -- one assertion.
  * Behavioural dispatcher coverage (the K-1175 stacked-predicate convention
    requires this for every load-bearing slot): one happy-path test asserts
    every in-set tuple routes True; one error-path test asserts that
    near-miss tuples (M / K below the P39 floor) route False.
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
    _P39_SKINNY_N448_KCOMPL_VERIFIED_WIN_18,
)


FZ = _P39_SKINNY_N448_KCOMPL_VERIFIED_WIN_18


def test_envelope_equals_full_n448_kcompl_grid():
    """Cardinality, N/M/K/dtype axes, dtype balance, and the absence of
    upstream-alias holes all follow from one equality: P39 is exactly the
    full 18-cell M ∈ {2048,4096,8192} × N=448 × K ∈ {4096,8192,16384} ×
    {bf16,fp16} grid (no upstream-alias exclusions — N=448 is a net-new
    K-COMPLEMENT N rung, disjoint from every prior alias-stack)."""
    full = frozenset(
        (M, 448, K, dt)
        for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(FZ) == 18


def test_sibling_n_firewall_disjoint_from_all_prior_kcompl_slots():
    """P39 (N=448) must be N-axis disjoint from every prior K-COMPLEMENT
    slot.  Pinned as the union intersection per K-1175 stacked-predicate
    convention so any future N-axis widening of an upstream slot trips a
    single test (rather than 11 fan-out tests)."""
    upstream = (
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18
        | _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12
        | _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30
        | _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29
        | _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34
        | _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120
        | _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28
    )
    assert FZ & upstream == frozenset()


def test_dispatcher_routes_in_set_p39_cell_to_hbl():
    """Happy-path dispatcher test (Testing-Zealot R-1887): an in-set P39
    tuple must return True from `_k971_route_to_hbl()` so the route-OUT
    short-circuit fires in production.  Asserts the full 18-cell roster
    routes True (cheap O(18) probe; covers the full admit set rather than
    a single representative)."""
    from tritonblas.matmul import _k971_route_to_hbl
    for (M, N, K, dt) in FZ:
        assert _k971_route_to_hbl(M, N, K, dt, dt, False, False) is True, (
            f"P39 in-set cell (M={M}, N={N}, K={K}, dtype={dt}) must route "
            f"to hipBLASLt"
        )


def test_dispatcher_does_not_route_near_miss_p39_cells_to_hbl():
    """Error-path / firewall dispatcher test (Testing-Zealot R-1887):
    near-miss tuples that share P39's N=448 axis but fall OFF the P39
    M/K floor must NOT be routed by `_k971_route_to_hbl()` — neither by
    P39 itself (sibling-N firewall) nor by any sibling alias-stack.
    Empirically verified on MI300X / gfx942 (K-1887 RETRY holdout bench)
    that these 4 tuples return False from the production dispatcher chain
    after the P39 commit lands."""
    from tritonblas.matmul import _k971_route_to_hbl
    near_miss = [
        (1024, 448, 4096,  "torch.bfloat16"),  # M below P39 floor (2048)
        (1024, 448, 2048,  "torch.float16"),   # M and K below P39 floor
        (4096, 448, 1024,  "torch.bfloat16"),  # K below P39 floor (4096)
        ( 512, 448, 1024,  "torch.float16"),   # M and K well below P39 floor
    ]
    for (M, N, K, dt) in near_miss:
        # Both invariants: must not be in P39 (frozenset firewall) AND must
        # return False from the dispatcher (no sibling slot silently admits).
        assert (M, N, K, dt) not in FZ
        assert _k971_route_to_hbl(M, N, K, dt, dt, False, False) is False, (
            f"Near-miss cell (M={M}, N={N}, K={K}, dtype={dt}) must NOT "
            f"route to hipBLASLt — sibling-N firewall violation"
        )
