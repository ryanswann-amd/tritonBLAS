"""K-2031 P47 (S-002) — 37th-slot N=1008 K-COMPLEMENT verified-winner alias-stack tests.

Source measurement: K-2031 paired n=30 HIP-graph hot-cache MI300X / gfx942
(vs the live post-K-1922 oracle, HEAD = 0024a71) over the 18-cell sub-cohort
M ∈ {2048, 4096, 8192} × N=1008 × K ∈ {4096, 8192, 16384} × {bf16, fp16}.

Result: 16/18 cells admit at the ≥1.10× route-out gate, 16/18 strict flips
(BEFORE ≥1.10× → AFTER ≤1.05×); BEFORE/AFTER cohort geomean TB/HBL flips
1.326× → 0.995× (1.334× cohort uplift; AFTER 0.995× sits on the
dispatcher-overhead floor and is below HBL).  18/18 cells satisfy the
≥0.95× HBL parity gate AFTER the patch.  Promoted as the 4th confirmed
wave-misaligned rung in the off-by-48 (mod 64) residue family above
K-1978 P45 N=816, K-1990/K-1994 P45 N=880, and K-2010 P46 N=944
(1008 = 944 + 64 = next 64-stride rung in the off-by-48 class).

Mechanism (K-913 §3 / R-K1673 dtype-invariance + R-1811 wave-misalignment):
BLOCK_N=128 packs N=1008 into 7.875 BLOCK_N tiles per N-row (1008 mod 128
= 112 = 7/8-tile tail).  1008 mod 64 = 48 places it in the off-by-48
wave-misaligned class — the same SIMD-lane mismatch as N=816, N=880, N=944.
Distinct from the mod-128 tail (P45 N=816 has 48-tile tail, P46 N=944 has
48-tile tail, P47 N=1008 has 112-tile tail) — the (mod 64 = 48) wave
misalignment is the binding constraint, not the BLOCK_N=128 packing
remainder.  persistent_matmul cannot trade tile reshape for atomic-reduction;
hipBLASLt's split-K kernel selection clears the band by ~33% on cohort
geomean (range 0.99×–1.63×).

Per the K-1908 compact-predicate analysis, the existing S1-form
`(N % 64 != 0) ∧ (N <= 384) ∧ (K >= 4096)` predicate does NOT cover N=1008
(N=1008 > 384 — the predicate's N-ceiling cuts off below this rung).
Explicit alias-stack promotion is required until S1 is extended to cover
N>384 in a separate task.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18 (full grid; no upstream-aliased exclusions).
  2. N axis is exactly {1008}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Both dtype rows are complete 9/9 (off-by-48 N=1008 has no R-K979 P5
     Clause-3 bf16 alias — strictly load-bearing on both rows).
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, the consolidated P32–P38 N ∈ {96, 160, 224, 288, 320, 352,
     384}, P40 N=544).
  6. Off-by-48 wave-misalignment band membership: 1008 mod 64 == 48
     (same wave class as K-1978 P45 N=816, K-1994 P45 N=880, K-2010 P46
     N=944 — distinct N-axis rung; mod 128 differs from those rungs).
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
    _K2031_P47_SKINNY_N1008_KCOMPL_ALIASSTACK_18,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
)


FZ = _K2031_P47_SKINNY_N1008_KCOMPL_ALIASSTACK_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n1008():
    assert {N for (_, N, _, _) in FZ} == {1008}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_row_balance():
    """Both dtype rows complete 9/9.  Off-by-48 N=1008 has no R-K979 P5
    Clause-3 bf16 alias coverage at this rung; both rows are strictly
    load-bearing.  K-913 §3 LDS-bank-conflict is dtype-invariant on the
    column-narrow N=1008 tile (same fingerprint as K-1978 P45 N=816,
    K-1994 P45 N=880, and K-2010 P46 N=944)."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    assert bf == fp


def test_envelope_equals_full_n1008_kcompl_grid():
    """The verified-winner envelope is exactly the 18-cell M ∈ {2048,
    4096,8192} × N=1008 × K ∈ {4096,8192,16384} × {bf16,fp16} grid (no
    upstream-aliased exclusions at this off-by-48 rung) — pinned to
    detect any silent contraction (a false-NEGATIVE that would leak
    winner cells back to TB) or expansion (a false-POSITIVE that would
    leak non-winner cells out to HBL) of the K-2031 verification envelope."""
    full = frozenset(
        (M, 1008, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18


def test_off_by_48_wave_misalignment_band_membership():
    """N=1008 sits in the off-by-48 wave-misaligned band (N mod 64 == 48),
    the same wave class as K-1978 P45 N=816, K-1994 P45 N=880, and K-2010
    P46 N=944.  Note: 1008 mod 128 == 112, distinct from the prior rungs'
    mod-128 tails (816,880,944 all have mod 128 == 48); the binding
    constraint is the (mod 64) wave-lane mismatch, not the BLOCK_N=128
    packing tail.  This invariant guards against a silent N-axis typo
    (e.g. 1004, 1012, 1024) that would land on a different SIMD wave
    alignment and thus reflect a different mechanism."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 64 == 48
    assert only_n == 1008


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """P30 covers N ∈ {384, 768, 1536}; P47 covers N=1008 — disjoint by
    natural N-axis separation (P47 sits between the P30 N=768 and N=1536
    rungs)."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p31_n256():
    assert FZ & _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28 == set()
    assert FZ & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == set()


def test_sibling_n_firewall_vs_consolidated_p32_p38():
    """K-1881 consolidated P32–P38 covers N ∈ {96, 160, 224, 288, 320,
    352, 384}; K-2031 P47 covers N=1008 — disjoint by natural N-axis
    separation."""
    assert FZ & _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120 == set()


def test_sibling_n_firewall_vs_p40_n544():
    """K-1922 P40 covers N=544; K-2031 P47 covers N=1008 — disjoint by
    natural N-axis separation."""
    assert FZ & _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18 == set()
