"""K-1963 P41–P44 (S-002) — 31st–34th-slot N=608/672/704/736 K-COMPLEMENT
verified-winner alias-stack tests.

Source measurements: paired n=30 HIP-graph hot-cache MI300X / gfx942 vs the
live post-K-1922 oracle:
  P41 N=608 — K-1938 cohort
  P42 N=672 — K-1923 / K-1927 / K-1931 cohort
  P43 N=704 — K-1945 cohort
  P44 N=736 — K-1941 / K-1943 cohort

Each rung is the full 18-cell sub-cohort
M ∈ {2048, 4096, 8192} × N × K ∈ {4096, 8192, 16384} × {bf16, fp16}; union
72-cell cohort across the four rungs.  Per-rung cohort geomean tb/hbl
≥ ~1.4×; 0/72 regressions on the K-1925 231-cell drift baseline.

Wave-/tile-alignment fingerprint per rung (BLOCK_N=128 packing):
  N=608: 608 mod 64 = 32 (wave-misaligned), 608 mod 128 = 96 — 4 BN=128
         tiles + 1 BN=96 tail (3/4-tile tail).
  N=672: 672 mod 64 = 32 (wave-misaligned), 672 mod 128 = 32 — 5 BN=128
         tiles + 1 BN=32 tail (1/4-tile tail; same tail-fragment band as
         P40 N=544).
  N=704: 704 mod 64 = 0 (wave-ALIGNED at the 64-lane SIMD; arithmetic
         correction vs the K-1963 PRD's blanket "N mod 64 == 32" claim),
         704 mod 128 = 64 — 5 BN=128 tiles + 1 BN=64 *half-tile* tail.
  N=736: 736 mod 64 = 32 (wave-misaligned), 736 mod 128 = 96 — 5 BN=128
         tiles + 1 BN=96 tail (3/4-tile tail; matches N=608 fingerprint).

Per the K-1908 compact-predicate analysis, none of N ∈ {608, 672, 704, 736}
are covered by the existing S1-form
`(N % 64 != 0) ∧ (N <= 384) ∧ (K >= 4096)` predicate (all four N > 384 —
predicate's N-ceiling cuts off below this rung).  Per K-1946 a depth-4
closed-form predicate over the {544, 608, 672, 704, 736} union is *avail-
able* but is NOT yet substituted, pending K-1942-style bit-equiv
revalidation on the expanded P41–P44 set.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Per-rung cardinality is exactly 18 (full grid; no upstream-aliased
     exclusions at these off-band rungs).
  2. Per-rung N axis is exactly the singleton {608} / {672} / {704} /
     {736}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Both dtype rows are complete 9/9 (off-band rungs have no R-K979 P5
     Clause-3 bf16 alias — strictly load-bearing on both rows).
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, the consolidated P32–P38 N ∈ {96, 160, 224, 288, 320, 352,
     384}, P40 N=544, plus K-1367/K-1397 P13 N ∈ {128, 256}).
  6. Pairwise disjoint across P41–P44 themselves (each is a distinct
     N-axis bucket).
  7. Wave-/tile-misalignment band membership matches the per-rung
     fingerprint above.
  8. Union cardinality across P41–P44 is exactly 72 (= 4 × 18).
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29,
    _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34,
    _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120,
    _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
    _K1963_P41_SKINNY_N608_KCOMPL_ALIASSTACK_18,
    _K1963_P42_SKINNY_N672_KCOMPL_ALIASSTACK_18,
    _K1963_P43_SKINNY_N704_KCOMPL_ALIASSTACK_18,
    _K1963_P44_SKINNY_N736_KCOMPL_ALIASSTACK_18,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
)


# Rungs in the order P41, P42, P43, P44.
_RUNGS = (
    ("P41", 608, _K1963_P41_SKINNY_N608_KCOMPL_ALIASSTACK_18),
    ("P42", 672, _K1963_P42_SKINNY_N672_KCOMPL_ALIASSTACK_18),
    ("P43", 704, _K1963_P43_SKINNY_N704_KCOMPL_ALIASSTACK_18),
    ("P44", 736, _K1963_P44_SKINNY_N736_KCOMPL_ALIASSTACK_18),
)

# Wave-/tile-alignment fingerprint per rung (mod_64, mod_128).  N=704 is
# wave-ALIGNED (mod_64=0) — the PRD's blanket "N mod 64 == 32" claim is
# arithmetically wrong for that one rung; we pin the actual values here so
# the invariant test catches any silent drift back to the bogus claim.
_FINGERPRINTS = {
    608: (32, 96),
    672: (32, 32),
    704: (0,  64),
    736: (32, 96),
}


# ---- per-rung structural invariants ---------------------------------------

@pytest.mark.parametrize("name,N,fz", _RUNGS)
def test_per_rung_cardinality_is_18(name, N, fz):
    assert len(fz) == 18, f"{name} N={N}: expected 18 cells, got {len(fz)}"


@pytest.mark.parametrize("name,N,fz", _RUNGS)
def test_per_rung_n_axis_is_singleton(name, N, fz):
    assert {nn for (_, nn, _, _) in fz} == {N}


@pytest.mark.parametrize("name,N,fz", _RUNGS)
def test_per_rung_m_k_dtype_axes_are_minimal(name, N, fz):
    assert {M for (M, _, _, _) in fz} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in fz} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in fz} == {"torch.bfloat16", "torch.float16"}


@pytest.mark.parametrize("name,N,fz", _RUNGS)
def test_per_rung_dtype_row_balance(name, N, fz):
    """Both dtype rows complete 9/9.  Off-band rungs have no R-K979 P5
    Clause-3 bf16 alias coverage; both rows are strictly load-bearing.
    K-913 §3 LDS-bank-conflict is dtype-invariant on the column-narrow
    tile (R-K1673 dtype-invariance)."""
    bf = {(M, nn, K) for (M, nn, K, dt) in fz if dt == "torch.bfloat16"}
    fp = {(M, nn, K) for (M, nn, K, dt) in fz if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    assert bf == fp


@pytest.mark.parametrize("name,N,fz", _RUNGS)
def test_per_rung_envelope_equals_full_grid(name, N, fz):
    """Each rung's verified-winner envelope is exactly the 18-cell M ∈
    {2048, 4096, 8192} × N × K ∈ {4096, 8192, 16384} × {bf16, fp16} grid
    (no upstream-aliased exclusions at these off-band rungs) — pinned to
    detect any silent contraction (false-NEGATIVE leaking winner cells
    back to TB) or expansion (false-POSITIVE leaking non-winner cells out
    to HBL)."""
    full = frozenset(
        (M, N, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert fz == full
    assert len(full) == 18


@pytest.mark.parametrize("name,N,fz", _RUNGS)
def test_per_rung_wave_or_tile_misalignment_fingerprint(name, N, fz):
    """Pin the per-rung (N mod 64, N mod 128) fingerprint to the actual
    arithmetic.  N=608/672/736 are wave-misaligned (mod 64 == 32); N=704
    is wave-ALIGNED (mod 64 == 0) but BN-half-tile-misaligned (mod 128 ==
    64).  This invariant guards both against a silent N-axis typo (e.g.
    640, 736→738) AND against the K-1963 PRD's blanket "N mod 64 == 32"
    claim being silently re-asserted for N=704."""
    (only_n,) = {nn for (_, nn, _, _) in fz}
    assert only_n == N
    expected = _FINGERPRINTS[N]
    assert (only_n % 64, only_n % 128) == expected, (
        f"{name} N={N}: expected (mod64, mod128) = {expected}, "
        f"got {(only_n % 64, only_n % 128)}"
    )


# ---- sibling-N firewall vs prior K-COMPLEMENT alias-stack slots -----------

@pytest.mark.parametrize("name,N,fz", _RUNGS)
def test_sibling_n_firewall_vs_p28_n128(name, N, fz):
    assert fz & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert fz & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


@pytest.mark.parametrize("name,N,fz", _RUNGS)
def test_sibling_n_firewall_vs_p29_n64(name, N, fz):
    assert fz & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


@pytest.mark.parametrize("name,N,fz", _RUNGS)
def test_sibling_n_firewall_vs_p30_nmid(name, N, fz):
    """P30 covers N ∈ {384, 768, 1536}; P41–P44 cover N ∈ {608, 672, 704,
    736} which sits strictly between the P30 N=384 and N=768 rungs."""
    assert fz & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


@pytest.mark.parametrize("name,N,fz", _RUNGS)
def test_sibling_n_firewall_vs_p31_n256(name, N, fz):
    assert fz & _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28 == set()
    assert fz & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == set()


@pytest.mark.parametrize("name,N,fz", _RUNGS)
def test_sibling_n_firewall_vs_consolidated_p32_p38(name, N, fz):
    """K-1881 consolidated P32–P38 covers N ∈ {96, 160, 224, 288, 320,
    352, 384}; K-1963 P41–P44 cover N ∈ {608, 672, 704, 736} — disjoint
    by natural N-axis separation."""
    assert fz & _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120 == set()


@pytest.mark.parametrize("name,N,fz", _RUNGS)
def test_sibling_n_firewall_vs_p40_n544(name, N, fz):
    """K-1922 P40 covers N=544; K-1963 P41–P44 cover the next four
    K-COMPLEMENT N-buckets above 544 (608, 672, 704, 736).  Disjoint by
    natural N-axis separation."""
    assert fz & _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18 == set()


# ---- pairwise disjointness across P41-P44 ---------------------------------

def test_p41_p44_pairwise_disjoint():
    """Each of P41–P44 covers a distinct N-axis bucket; the four
    frozensets must be pairwise disjoint."""
    fzs = [fz for (_n, _N, fz) in _RUNGS]
    for i in range(len(fzs)):
        for j in range(i + 1, len(fzs)):
            assert fzs[i] & fzs[j] == set(), (
                f"{_RUNGS[i][0]} (N={_RUNGS[i][1]}) and "
                f"{_RUNGS[j][0]} (N={_RUNGS[j][1]}) overlap"
            )


def test_union_cardinality_is_72():
    """4 × 18 = 72-cell union cohort (the K-1963 PRD's "union 72-cell
    cohort" claim).  Equivalent to pairwise disjointness combined with
    per-rung cardinality 18."""
    union = frozenset()
    for (_n, _N, fz) in _RUNGS:
        union = union | fz
    assert len(union) == 72


def test_union_n_axis_is_exactly_the_quad():
    union = frozenset()
    for (_n, _N, fz) in _RUNGS:
        union = union | fz
    assert {N for (_, N, _, _) in union} == {608, 672, 704, 736}
