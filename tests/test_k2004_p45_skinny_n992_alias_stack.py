"""K-2004 P45 (S-002) — 35th-slot N=992 K-COMPLEMENT verified-winner alias-stack tests.

Source measurement: K-2004 paired n=30 HIP-graph hot-cache MI300X / gfx942
3-engine bench (TB / hipBLASLt / TB-with-alias) vs the live post-K-1963
oracle (`fix/K-1963 84eff0a`, P32-P44 stacked) over the 18-cell sub-cohort
M ∈ {2048, 4096, 8192} × N=992 × K ∈ {4096, 8192, 16384} × {bf16, fp16}.

Wave-/tile-alignment fingerprint:
  992 mod 64  = 32   → off-by-32 on the 64-lane SIMD (wave-misaligned —
                       same residue family as N=864 P42 K-1967, N=608 P41
                       K-1963, N=672 P41 K-1963, N=736 P44 K-1963).
  992 mod 128 = 96   → off-by-96 on BLOCK_N=128 — packs into 7 full BN=128
                       tiles + 1 BN=96 tail (3/4-tile tail; same fingerprint
                       as P41 N=608 and P44 N=736).

Mechanism (R-1811 wave-misalignment + K-913 §3 LDS-bank-conflict): the
persistent_matmul kernel cannot trade tile reshape for atomic-reduction
across the BN=96 tail; hipBLASLt's split-K kernel selection clears the
band.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18 (full grid; no upstream-aliased exclusions).
  2. N axis is exactly {992}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Both dtype rows are complete 9/9.
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, the consolidated P32–P38 N ∈ {96, 160, 224, 288, 320, 352,
     384}, P40 N=544, P41–P44 N ∈ {608, 672, 704, 736}, plus
     K-1367/K-1397 P13 N ∈ {128, 256}).
  6. Off-by-32 / off-by-96-on-128 wave-misalignment band membership:
     992 mod 64 == 32 ∧ 992 mod 128 == 96.
  7. K-1968 disjointness shortcut: N-bucket {992} is disjoint from every
     prior P-rung N-bucket (cardinality of intersection == 0); this
     admits the R-K1968 causal-drift proof for the K-1968 drift gate
     without re-running the 231-cell sweep.
"""
from __future__ import annotations

from tritonblas._route_predicate import (
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29,
    _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34,
    _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120,
    _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
    _K1963_P41_P44_SKINNY_N_QUAD_KCOMPL_ALIASSTACK_72,
    _K2004_P45_SKINNY_N992_KCOMPL_ALIASSTACK_18,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
)


FZ = _K2004_P45_SKINNY_N992_KCOMPL_ALIASSTACK_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n992():
    assert {N for (_, N, _, _) in FZ} == {992}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_row_balance():
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    assert bf == fp


def test_envelope_equals_full_n992_kcompl_grid():
    full = frozenset(
        (M, 992, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18


def test_off_by_32_wave_misalignment_band_membership():
    """N=992 sits in the off-by-32 / off-by-96-on-128 modular family
    (992 mod 64 == 32, 992 mod 128 == 96 — same fingerprint as P41 N=608
    and P44 N=736).  This invariant guards against a silent N-axis typo
    (e.g. 960 = 15×64 wave-aligned, 1024 = 16×64 wave-aligned) that would
    land on a different BLOCK_N=128 alignment rung and thus reflect a
    different mechanism."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 64 == 32
    assert only_n % 128 == 96
    assert only_n == 992


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


def test_sibling_n_firewall_vs_consolidated_p32_p38():
    assert FZ & _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120 == set()


def test_sibling_n_firewall_vs_p40_n544():
    assert FZ & _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18 == set()


def test_sibling_n_firewall_vs_p41_p44_quad():
    """K-1963 P41-P44 covers N ∈ {608, 672, 704, 736}; K-2004 P45 covers
    N=992 — disjoint by natural N-axis separation."""
    assert FZ & _K1963_P41_P44_SKINNY_N_QUAD_KCOMPL_ALIASSTACK_72 == set()


def test_k1968_disjointness_shortcut_n_bucket_intersection_empty():
    """K-1968 R-CAUSAL-DRIFT-PROOF admission: the K-2004 P45 N-bucket
    {992} must be disjoint from every prior K-COMPLEMENT alias-stack
    N-bucket so that the additive Cartesian-product diff cannot causally
    affect any prior P-rung routing path.  This is the structural
    precondition for skipping the 231-cell K-1968 drift sweep on this
    promotion (per F-K1975.K1968-DRIFT-SHORTCUT-CANONICAL)."""
    prior_n_buckets = (
        {N for (_, N, _, _) in _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18}
        | {N for (_, N, _, _) in _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12}
        | {N for (_, N, _, _) in _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30}
        | {N for (_, N, _, _) in _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29}
        | {N for (_, N, _, _) in _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34}
        | {N for (_, N, _, _) in _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28}
        | {N for (_, N, _, _) in _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120}
        | {N for (_, N, _, _) in _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18}
        | {N for (_, N, _, _) in _K1963_P41_P44_SKINNY_N_QUAD_KCOMPL_ALIASSTACK_72}
    )
    assert {992} & prior_n_buckets == set()


def test_n960_wave_aligned_negative_verified_not_in_frozenset():
    """N=960 sits between N=864 (admitted P42 K-1967) and N=992 (this
    K-2004 P45); because 960 mod 64 == 0 (wave-ALIGNED — 960 = 15×64
    exactly) the SCHEDULER_LDS A4 misalignment that drives the
    LDS-DOMINANT advantage simply does not occur, and the K-COMPLEMENT
    alias should not fire.  Per F-K1978.WAVE-ALIGNMENT-CHECK-MUST-PRECEDE-
    LADDER-EXTENSION, this regression-guard test catches any future drift
    that accidentally adds N=960 to the P45 frozenset.

    Likewise N=1024 (1024 mod 64 == 0, 16×64 exactly) is wave-aligned and
    is covered separately by K-1429 P16 not by the K-COMPLEMENT route-OUT."""
    for M in (2048, 4096, 8192):
        for K in (4096, 8192, 16384):
            for dt in ("torch.bfloat16", "torch.float16"):
                assert (M, 960, K, dt) not in FZ
                assert (M, 1024, K, dt) not in FZ
