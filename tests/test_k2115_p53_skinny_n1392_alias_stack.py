"""K-2115 P53 (S-002) — 43rd-slot N=1392 K-COMPLEMENT verified-winner alias-stack tests.

Source measurement: K-2115 paired n=30 HIP-graph hot-cache MI300X / gfx942
(vs the live fix/K-1922 0024a71 oracle) over the 18-cell sub-cohort
M ∈ {2048, 4096, 8192} × N=1392 × K ∈ {4096, 8192, 16384} × {bf16, fp16}.

Off-by-48 wave-misaligned 9th-rung of the (N % 64 == 48) ladder
(K-2055 P49 N=1136 → K-2075 P50 N=1200 → K-2091 P51 N=1264 →
K-2104 P52 N=1328 → K-2115 P53 N=1392).  1392 mod 64 = 48,
1392 mod 128 = 112 — REJOIN to the N=1264 mod-128 topology, continuing the
48/112/.../48/112 alternation across the residue-48 family.

Mechanism (K-2056 PMC delta-attribution + R-1811 wave-misalignment):
residue-48 grids divide gfx942's CU array more evenly than residue-32
grids → GRBM_GUI_ACTIVE per-wg drops ~24% (Spearman ρ = -0.930, p~1e-5)
despite a +5.5% TCC_READ_REQ_sum L2-traffic premium.  Per the K-2056
retry, the surviving within-tile discriminator is TCC_ACCESS_sum_per_wg
(partial ρ = +0.71, p = 0.014) — i.e. L2 write/atomic traffic, with
VALU_BUSY_per_wg bit-identical at 98280 across all 12 cells.  This is
grid-routing-bound (wave-lane misalignment) not compute-bound;
hipBLASLt's split-K kernel selection clears the band by selecting tile
shapes whose grids land cleanly on the gfx942 CU partition.

Per the K-1908 compact-predicate analysis, the existing S1-form
`(N % 64 != 0) ∧ (N <= 384) ∧ (K >= 4096)` predicate does NOT cover N=1392
(N=1392 > 384 — the predicate's N-ceiling cuts off below this rung).
Explicit alias-stack promotion is required until S1 is extended to absorb
the closed-form `(N % 64 == 48) ∧ (816 ≤ N ≤ 1392)` envelope in a separate
task.  The K-1900 compact-predicate substitution angle failed at depth-2
closure and is NOT retried here.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18 (full grid; no upstream-aliased exclusions).
  2. N axis is exactly {1392}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Both dtype rows are complete 9/9.
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N.
  6. Off-by-48 wave-misalignment band membership: 1392 mod 64 == 48.
  7. mod-128 topology REJOIN: 1392 mod 128 == 112 (same mod-128 residue
     as N=1264 P51, continuing the 48/112/.../48/112 alternation).
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
    _K2115_P53_SKINNY_N1392_KCOMPL_ALIASSTACK_18,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
)


FZ = _K2115_P53_SKINNY_N1392_KCOMPL_ALIASSTACK_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n1392():
    assert {N for (_, N, _, _) in FZ} == {1392}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_row_balance():
    """Both dtype rows complete 9/9.  Off-by-48 N=1392 has no R-K979 P5
    Clause-3 dtype-asymmetric alias coverage at this rung; both rows are
    strictly load-bearing — same fingerprint as the prior P49–P52 ladder
    rungs (K-2055 / K-2075 / K-2091 / K-2104)."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    assert bf == fp


def test_envelope_equals_full_n1392_kcompl_grid():
    """The verified-winner envelope is exactly the 18-cell M ∈ {2048,
    4096,8192} × N=1392 × K ∈ {4096,8192,16384} × {bf16,fp16} grid (no
    upstream-aliased exclusions at this off-by-48 rung) — pinned to detect
    any silent contraction (a false-NEGATIVE that would leak winner cells
    back to TB) or expansion (a false-POSITIVE that would leak non-winner
    cells out to HBL) of the K-2115 verification envelope."""
    full = frozenset(
        (M, 1392, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18


def test_off_by_48_wave_misalignment_band_membership():
    """N=1392 sits in the off-by-48 wave-misaligned ladder (N mod 64 == 48),
    same residue family as P49 N=1136, P50 N=1200, P51 N=1264, P52 N=1328.
    This invariant guards against a silent N-axis typo (e.g. 1390, 1394)
    that would land on a different 64-lane wave alignment and thus reflect
    a different mechanism."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 64 == 48
    assert only_n == 1392


def test_mod_128_topology_rejoin_to_n1264():
    """N=1392 mod 128 == 112, identical mod-128 residue to N=1264 (P51 /
    K-2091).  Continues the 48/112/48/112/.../48/112 alternation across the
    residue-48 family: P49 N=1136 mod 128 = 112, P50 N=1200 mod 128 = 48,
    P51 N=1264 mod 128 = 112, P52 N=1328 mod 128 = 48, P53 N=1392 mod 128
    = 112.  This invariant pins the conjectured BLOCK_N=128 packing-tail
    topology that the K-2056 PMC delta-attribution attributes the band to."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 128 == 112


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    """P30 covers N ∈ {384, 768, 1536}; P53 covers N=1392 — disjoint by
    natural N-axis separation (P53 sits between the P30 N=768 and N=1536
    rungs)."""
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p31_n256():
    assert FZ & _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28 == set()
    assert FZ & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == set()


def test_sibling_n_firewall_vs_consolidated_p32_p38():
    """K-1881 consolidated P32–P38 covers N ∈ {96, 160, 224, 288, 320,
    352, 384}; K-2115 P53 covers N=1392 — disjoint by natural N-axis
    separation."""
    assert FZ & _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120 == set()


def test_sibling_n_firewall_vs_p40_n544():
    """K-1922 P40 covers N=544; K-2115 P53 covers N=1392 — disjoint by
    natural N-axis separation (P53 sits 848 N-lanes above P40 in the same
    explicit-rung promotion lineage)."""
    assert FZ & _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18 == set()
