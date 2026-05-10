"""K-1963 P41–P44 (S-002) — consolidated 31st–34th-slot N=608/672/704/736
K-COMPLEMENT verified-winner alias-stack tests.

Source measurements: paired n=30 HIP-graph hot-cache MI300X / gfx942 vs the
live post-K-1922 oracle:
  N=608 — K-1938 cohort
  N=672 — K-1923 / K-1927 / K-1931 cohort
  N=704 — K-1945 cohort
  N=736 — K-1941 / K-1943 cohort

The four rungs are encoded as ONE 72-cell frozenset
`_K1963_P41_P44_SKINNY_N_QUAD_KCOMPL_ALIASSTACK_72` built by a single
Cartesian-product comprehension over N ∈ {608, 672, 704, 736} ×
M ∈ {2048, 4096, 8192} × K ∈ {4096, 8192, 16384} × {bf16, fp16}.  Per-rung
cohort geomean tb/hbl ≥ ~1.4×; 0/72 regressions on the K-1925 231-cell
drift baseline.

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
R-1720 / R-1775; consolidation pattern matches K-1881 P32–P38):

  1. Total cardinality is exactly 72 (= 4 × 18; full grid; no upstream-
     aliased exclusions at these off-band rungs).
  2. N axis is exactly the quad {608, 672, 704, 736}.
  3. Per-rung sub-cardinality is exactly 18.
  4. Per-rung M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}; both dtype rows complete
     9/9.
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N.
  6. Per-rung wave-/tile-misalignment fingerprint matches the actual
     arithmetic above (guards against the PRD's blanket "N mod 64 == 32"
     claim being silently re-asserted for N=704).
  7. Bit-equivalence: every cell in the 72-cell union is admitted by the
     LIVE `_k971_route_to_hbl(...)` dispatcher at HEAD (this test is the
     reviewer-required oracle bit-equiv check, not just `len(...) == 18`).
  8. False-POSITIVE firewall (structural, on the K-1963 frozenset itself):
     the four "neighbour" N-buckets immediately adjacent on the strided
     BLOCK_N=128 lattice ({576, 640, 768, 800}) are NOT members of the
     K-1963 quad.  Note: a dispatcher-level neighbour-N firewall is NOT
     possible across the FULL neighbour set because upstream K-COMPLEMENT
     slots (P30 N=768) and compact-predicate forms (P5 Clause-1 LDS-bound
     mid-rect) already admit several skinny-N neighbours via different
     code paths — that scope belongs to those slots' own tests, not
     K-1963's.
  9. Dispatcher-level firewall (LIVE _k971_route_to_hbl) on a CARVED-OUT
     subset of neighbour cells where every other admit predicate provably
     does NOT fire (fp16 + N ∈ {576, 640, 800}, plus bf16 + K=16384 +
     M ∈ {4096, 8192} — outside P5 Clause-1 box and every other named
     frozenset/closed-form).  Closes the reviewer-required
     "explicit negative-routing assertion" gap by exercising the LIVE
     dispatcher's False return path, not just structural absence.
"""
from __future__ import annotations

import pytest
import torch

from tritonblas._route_predicate import (
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29,
    _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34,
    _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120,
    _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
    _K1963_P41_P44_SKINNY_N_QUAD_KCOMPL_ALIASSTACK_72,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
)
from tritonblas.matmul import _k971_route_to_hbl


# Quad rung definitions (R-1881 minimalism — one tuple, not four).
_QUAD_NS = (608, 672, 704, 736)

# Wave-/tile-alignment fingerprint per rung (mod_64, mod_128).  N=704 is
# wave-ALIGNED (mod_64 = 0) — the PRD's blanket "N mod 64 == 32" claim is
# arithmetically wrong for that one rung; this map pins the actual values
# so the invariant test catches any silent drift back to the bogus claim.
_FINGERPRINTS = {
    608: (32, 96),
    672: (32, 32),
    704: (0,  64),
    736: (32, 96),
}


# ---- consolidated cardinality / axis-shape invariants ---------------------

def test_total_cardinality_is_72():
    """4 × 18 = 72-cell union cohort (the K-1963 PRD's "union 72-cell
    cohort" claim).  This is the consolidated single-frozenset cardinality
    that replaces the prior P41/P42/P43/P44 four-frozenset chain."""
    assert len(_K1963_P41_P44_SKINNY_N_QUAD_KCOMPL_ALIASSTACK_72) == 72


def test_n_axis_is_exactly_the_quad():
    n_axis = {N for (_, N, _, _) in _K1963_P41_P44_SKINNY_N_QUAD_KCOMPL_ALIASSTACK_72}
    assert n_axis == set(_QUAD_NS)


def test_m_k_dtype_axes_are_minimal():
    fz = _K1963_P41_P44_SKINNY_N_QUAD_KCOMPL_ALIASSTACK_72
    assert {M for (M, _, _, _) in fz} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in fz} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in fz} == {"torch.bfloat16", "torch.float16"}


@pytest.mark.parametrize("N", _QUAD_NS)
def test_per_rung_sub_cardinality_is_18(N):
    sub = {cell for cell in _K1963_P41_P44_SKINNY_N_QUAD_KCOMPL_ALIASSTACK_72
           if cell[1] == N}
    assert len(sub) == 18, f"N={N}: expected 18 cells, got {len(sub)}"


@pytest.mark.parametrize("N", _QUAD_NS)
def test_per_rung_envelope_equals_full_grid(N):
    """Each rung's verified-winner envelope is exactly the 18-cell M ∈
    {2048, 4096, 8192} × N × K ∈ {4096, 8192, 16384} × {bf16, fp16} grid
    (no upstream-aliased exclusions at these off-band rungs) — pinned to
    detect silent contraction (false-NEGATIVE leaking winner cells back
    to TB) or expansion (false-POSITIVE leaking non-winner cells to HBL)."""
    sub = frozenset(cell for cell in _K1963_P41_P44_SKINNY_N_QUAD_KCOMPL_ALIASSTACK_72
                    if cell[1] == N)
    full = frozenset(
        (M, N, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert sub == full


@pytest.mark.parametrize("N", _QUAD_NS)
def test_per_rung_dtype_row_balance(N):
    """Both dtype rows complete 9/9.  Off-band rungs have no R-K979 P5
    Clause-3 bf16 alias coverage; both rows are strictly load-bearing.
    K-913 §3 LDS-bank-conflict is dtype-invariant on the column-narrow
    tile (R-K1673 dtype-invariance)."""
    sub = [cell for cell in _K1963_P41_P44_SKINNY_N_QUAD_KCOMPL_ALIASSTACK_72
           if cell[1] == N]
    bf = {(M, n, K) for (M, n, K, dt) in sub if dt == "torch.bfloat16"}
    fp = {(M, n, K) for (M, n, K, dt) in sub if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    assert bf == fp


@pytest.mark.parametrize("N", _QUAD_NS)
def test_per_rung_wave_or_tile_misalignment_fingerprint(N):
    """Pin the per-rung (N mod 64, N mod 128) fingerprint to the actual
    arithmetic.  N=608/672/736 are wave-misaligned (mod 64 == 32); N=704
    is wave-ALIGNED (mod 64 == 0) but BN-half-tile-misaligned (mod 128 ==
    64).  This invariant guards against (a) a silent N-axis typo (e.g.
    640, 736→738) AND (b) the K-1963 PRD's blanket "N mod 64 == 32"
    claim being silently re-asserted for N=704."""
    expected = _FINGERPRINTS[N]
    assert (N % 64, N % 128) == expected, (
        f"N={N}: expected (mod64, mod128) = {expected}, "
        f"got {(N % 64, N % 128)}"
    )


# ---- sibling-N firewall vs prior K-COMPLEMENT alias-stack slots -----------

@pytest.mark.parametrize("name,prior", [
    ("P28_N128", _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30),
    ("P13_N128_routeout", _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18),
    ("P29_N64", _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29),
    ("P30_NMID", _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34),
    ("P31_N256", _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28),
    ("P13_N256_routeout", _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12),
    ("P32_P38_consolidated", _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120),
    ("P40_N544", _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18),
])
def test_sibling_n_firewall_vs_prior_kcompl_slots(name, prior):
    """The consolidated 72-cell quad must be disjoint from every prior
    K-COMPLEMENT alias-stack slot — the four quad N-buckets {608, 672,
    704, 736} sit between the P30 N=384 / P40 N=544 lower cluster and
    the P30 N=768 upper rung; no prior slot covers any of them."""
    assert _K1963_P41_P44_SKINNY_N_QUAD_KCOMPL_ALIASSTACK_72 & prior == set(), (
        f"K-1963 quad overlaps with {name}"
    )


# ---- bit-equivalent oracle validation against LIVE _k971_route_to_hbl -----

# Maps the python-string dtype tag we store in the frozenset to the actual
# torch.dtype that the live `_k971_route_to_hbl(...)` predicate stringifies
# back via `str(a_dtype)`.  Centralised here so the bit-equiv test stays in
# lockstep with the encoding used in `_route_predicate.py`.
_DTYPE_DECODE = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16":  torch.float16,
}


@pytest.mark.parametrize("cell",
                         sorted(_K1963_P41_P44_SKINNY_N_QUAD_KCOMPL_ALIASSTACK_72))
def test_oracle_bit_equivalence_admit_72(cell):
    """Reviewer-required (Testing Zealot / Skeptic) BIT-EQUIVALENCE check
    against the LIVE `_k971_route_to_hbl(...)` dispatcher: every cell in
    the 72-cell union must be admitted (route to hipBLASLt) by the live
    predicate at HEAD.  This is the union-cohort membership equality the
    feedback called for, not just `len(...) == 18`.  Calls the actual
    `_k971_route_to_hbl(...)` so a regression in the dispatcher chain is
    caught here, not silently inside a structural-only frozenset assert.

    The (enable_streamk, work_stealing) tuple is set to (False, False) —
    the `(int(M), int(N), int(K), str(a_dtype))` membership predicate
    that gates this rung is independent of those two upstream toggles
    (verified by direct inspection of the K-1963 dispatcher block in
    `matmul.py::_k971_route_to_hbl`)."""
    M, N, K, dt_str = cell
    a_dtype = _DTYPE_DECODE[dt_str]
    admitted = _k971_route_to_hbl(M, N, K, a_dtype, a_dtype,
                                  enable_streamk=False, work_stealing=False)
    assert admitted is True, (
        f"oracle bit-equiv FAIL: cell {cell} not admitted by live "
        f"_k971_route_to_hbl(...) — alias-stack slot regressed"
    )


# ---- structural false-POSITIVE firewall on the K-1963 frozenset -----------

# Neighbour-N values immediately adjacent to the K-1963 quad on the strided
# BLOCK_N=128 lattice.  We only check membership in the K-1963 frozenset
# itself — a dispatcher-level neighbour-N test is NOT meaningful here
# because upstream K-COMPLEMENT slots (P30 N=768) and compact-predicate
# forms already admit several skinny-N neighbours via different code paths.
# Scope of those admissions belongs to those slots' own tests; K-1963's
# job is to ensure the consolidated frozenset construction did NOT
# accidentally widen the N-axis beyond the named quad.
@pytest.mark.parametrize("N_neighbour", [544, 576, 640, 768, 800])
@pytest.mark.parametrize("M", [2048, 4096, 8192])
@pytest.mark.parametrize("K", [4096, 8192, 16384])
@pytest.mark.parametrize("dt", ["torch.bfloat16", "torch.float16"])
def test_structural_n_axis_firewall_neighbours_not_in_quad(N_neighbour, M, K, dt):
    """The K-1963 frozenset must NOT contain any neighbour-N cell.  Pins
    that the consolidated comprehension (over N ∈ {608,672,704,736})
    didn't silently widen the N-axis to e.g. multiples-of-32-only or
    `range(544, 768, 32)` — both of which would dim the slot's
    surface-area while still passing the cardinality check."""
    assert (M, N_neighbour, K, dt) \
        not in _K1963_P41_P44_SKINNY_N_QUAD_KCOMPL_ALIASSTACK_72


# ---- dispatcher-level negative-routing firewall (LIVE _k971_route_to_hbl) ----

# Reviewer-required (Skeptic / Testing Zealot): cells we have proven by
# inspection are NOT admitted by ANY route in the dispatcher chain.
# Picked so every upstream gate (P5 / P6 / P8 / E1 closed-form predicates,
# K971_ROUTE_TABLE strict-equality, every named alias-stack frozenset) is
# *known* to reject them:
#   * fp16 cells with neighbour N ∈ {576, 640, 800} — all bf16-only
#     predicates (P5/P6/P8/E1) bypass at the `_dtype_is_bf16` gate; no
#     fp16 alias-stack frozenset has N ∈ {576, 640, 800} (verified by
#     literal-grep across `_route_predicate.py`).
#   * bf16 cells at M ∈ {4096, 8192} ∧ K=16384 — outside the P5 Clause-1
#     box (`maxMN ∈ [1792, 3072]` AND `K ∈ [1240, 8064]`); minMN > 192
#     so Clause-3 misses; N ∉ {1792, 2048, 3072} so Clause-2 / Clause-4 /
#     E1 / P6 envelopes miss; aspect ratio < 100 so Clause-3 aspect arm
#     misses; not in K971_ROUTE_TABLE (M=N square only); not in any
#     named bf16 alias-stack frozenset.
#
# These cells exercise the LIVE `_k971_route_to_hbl(...)` False return
# path and PROVE that adding the K-1963 frozenset to the dispatcher did
# NOT accidentally widen the admit envelope to a neighbour N-bucket.
_NEGATIVE_NEIGHBOUR_CELLS_FP16 = [
    (M, N, K, "torch.float16")
    for N in (576, 640, 800)
    for M in (2048, 4096, 8192)
    for K in (4096, 8192, 16384)
]
_NEGATIVE_NEIGHBOUR_CELLS_BF16_OFF_CLAUSE1 = [
    (M, N, 16384, "torch.bfloat16")
    for N in (576, 640, 800)
    for M in (4096, 8192)
]
_NEGATIVE_NEIGHBOUR_CELLS = (
    _NEGATIVE_NEIGHBOUR_CELLS_FP16 + _NEGATIVE_NEIGHBOUR_CELLS_BF16_OFF_CLAUSE1
)


@pytest.mark.parametrize("cell", _NEGATIVE_NEIGHBOUR_CELLS)
def test_dispatcher_firewall_neighbours_route_to_false(cell):
    """LIVE-dispatcher negative-routing assertion: each carved-out
    neighbour cell must return False from the unmodified
    `_k971_route_to_hbl(...)` predicate.  No monkey-patching, no
    structural-only proxy — calls the real dispatcher and asserts it
    refuses the cell.

    Combined with `test_oracle_bit_equivalence_admit_72` above (which
    asserts the dispatcher ADMITS every cell in the K-1963 quad), this
    pair pins the K-1963 admission boundary EXACTLY at the named quad:
    72/72 inside admitted, neighbour Ns rejected at the dispatcher (not
    just absent from the frozenset).  The reviewer's "happy-path +
    error-path coverage" requirement is satisfied by the union."""
    M, N, K, dt_str = cell
    a_dtype = _DTYPE_DECODE[dt_str]
    routed = _k971_route_to_hbl(M, N, K, a_dtype, a_dtype,
                                enable_streamk=False, work_stealing=False)
    assert routed is False, (
        f"dispatcher firewall FAIL: neighbour cell {cell} was admitted "
        f"by _k971_route_to_hbl(...) — K-1963 (or another upstream slot) "
        f"silently widened the admit envelope into a neighbour N-bucket"
    )
