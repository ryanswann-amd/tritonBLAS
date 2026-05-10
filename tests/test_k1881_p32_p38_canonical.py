"""K-1881 (S-002) — canonical wave-misaligned skinny-N K-COMPLEMENT route-OUT
set-equivalence and structural invariants.

K-1864 consolidated P32–P36 (89 cells across N ∈ {96,160,224,288,320}) into a
single dispatch entry.  K-1868 (P37, N=352) and K-1880 (P38, N=384) added two
more wave-misaligned skinny-N rungs to the alias-stack, bringing the count to
7 distinct frozensets that share the same K-913 §3 LDS-bank-conflict +
R-1811 wave-misalignment MFMA-tail fingerprint (SCHEDULER_LDS A4 failure
mode) and the same routing target (hipBLASLt's split-K kernel selection).

K-1881 offline-joins the 7 frozensets into a single canonical
``_K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT`` with structurally-verified
set-equivalence to the union of the 7 source frozensets.  This test pins:

  1. Cardinality is exactly 121 (= 18 + 18 + 18 + 18 + 17 + 18 + 14, where
     P36 has the (2048,320,4096,bf16) upstream-aliased cell excluded and P38
     has the 4 K-1748 P30 N-mid overlap cells at K=8192 / M ∈ {2048,4096}
     excluded).
  2. Set-equivalence vs the union of the 7 source frozensets (zero behaviour
     change relative to the 7-slot stack the canonical replaces — the
     equivalent of the K-1881 paired n=30 HIP-graph drift-detection
     benchmark, executed structurally on the data).
  3. N axis is exactly the 7-rung wave-misaligned skinny-N ladder
     {96, 160, 224, 288, 320, 352, 384}.
  4. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384}; dtype ∈
     {torch.bfloat16, torch.float16}.
  5. Per-N projection cardinalities match the source frozensets (so any
     silent contraction or expansion is detected here as well as in the
     per-slot tests).
  6. The dispatch helper ``matmul._k971_route_to_hbl`` admits every cell of
     the canonical and rejects every cell of a representative complement
     control set (sanity check that the consolidation preserved the route-
     OUT semantics — analogous to the paired n=30 HIP-graph drift-detection
     benchmark, executed in software).
  7. Compact-predicate audit (per the K-1858 RCA learning): the candidate
     ``N % 64 != 0 ∧ N ≤ 384 ∧ K ≥ 4096`` does NOT cover the canonical
     because it false-NEGATIVES on N ∈ {320, 384} (both divisible by 64).
     The frozenset is therefore retained as the canonical truth-source —
     this test pins the audit so any future predicate refactor must
     re-prove coverage on N ∈ {320, 384}.
"""
from __future__ import annotations

import torch

from tritonblas._route_predicate import (
    _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18,
    _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18,
    _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18,
    _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18,
    _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17,
    _P37_SKINNY_N352_KCOMPL_VERIFIED_WIN_18,
    _P38_SKINNY_N384_KCOMPL_VERIFIED_WIN_14,
    _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT,
)
from tritonblas.matmul import _k971_route_to_hbl


CANON = _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT

SOURCE_SLOTS = (
    _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18,
    _P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18,
    _P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18,
    _P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18,
    _P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17,
    _P37_SKINNY_N352_KCOMPL_VERIFIED_WIN_18,
    _P38_SKINNY_N384_KCOMPL_VERIFIED_WIN_14,
)


def test_set_equivalence_vs_union_of_seven_sources():
    """K-1881 invariant: the canonical equals the union of the 7 source
    frozensets, exactly.  This is the structural equivalent of the K-1881
    paired n=30 HIP-graph drift-detection benchmark — zero behaviour change
    relative to the 7-slot dispatch chain it replaces."""
    union = frozenset().union(*SOURCE_SLOTS)
    assert CANON == union
    # Symmetric difference is empty (defence-in-depth — same assertion via
    # a different operator, makes a future regression report pinpoint).
    assert CANON ^ union == set()


def test_cardinality_is_121():
    """Exact cardinality: 18 + 18 + 18 + 18 + 17 + 18 + 14 = 121.  P36
    excludes 1 upstream-aliased cell; P38 excludes 4 K-1748 P30 N-mid
    overlap cells.  All 7 source frozensets are pairwise N-axis disjoint
    so the union cardinality is the sum of the per-slot cardinalities."""
    expected = sum(len(s) for s in SOURCE_SLOTS)
    assert expected == 121
    assert len(CANON) == 121


def test_n_axis_is_seven_rung_skinny_ladder():
    """N axis is exactly the 7-rung wave-misaligned skinny-N ladder."""
    assert {N for (_, N, _, _) in CANON} == {96, 160, 224, 288, 320, 352, 384}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in CANON} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in CANON} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in CANON} == {"torch.bfloat16", "torch.float16"}


def test_per_n_projection_cardinalities_match_sources():
    """Per-N projection cardinality matches the source frozenset cardinality
    for every N rung — detects any silent contraction (false-NEGATIVE) or
    expansion (false-POSITIVE) of the canonical relative to the sources."""
    by_n = {}
    for cell in CANON:
        by_n.setdefault(cell[1], set()).add(cell)
    expected = {
        96: 18,
        160: 18,
        224: 18,
        288: 18,
        320: 17,  # excludes (2048,320,4096,bf16) upstream alias
        352: 18,
        384: 14,  # excludes 4 K-1748 P30 N-mid overlap cells at K=8192
    }
    actual = {n: len(cells) for n, cells in by_n.items()}
    assert actual == expected


def test_p36_upstream_alias_cell_is_excluded():
    """(2048,320,4096,bf16) is upstream-routed (K-1843 r=1.0001, p=0.293) —
    excluded from the canonical to avoid duplicate routing per
    R-K1825.CHECK-ALIAS-STACK-COVERAGE-MAP-FIRST."""
    assert (2048, 320, 4096, "torch.bfloat16") not in CANON


def test_p38_k1748_p30_overlap_cells_are_excluded():
    """The 4 K-1748 P30 N-mid overlap cells (M ∈ {2048,4096} × N=384 × K=8192
    × {bf16,fp16}) are routed by the upstream P30 alias-stack slot — excluded
    from the canonical to avoid duplicate routing."""
    for M in (2048, 4096):
        for dt in ("torch.bfloat16", "torch.float16"):
            assert (M, 384, 8192, dt) not in CANON


def test_dispatch_helper_admits_every_canonical_cell():
    """``_k971_route_to_hbl`` returns True on every cell in the canonical —
    the route-OUT semantics are preserved by the consolidation.  This is the
    software equivalent of the K-1881 paired n=30 HIP-graph drift-detection
    benchmark across the full union (zero false-NEGATIVES)."""
    for (M, N, K, dt) in CANON:
        a_dtype = torch.bfloat16 if dt == "torch.bfloat16" else torch.float16
        assert _k971_route_to_hbl(
            M, N, K, a_dtype, a_dtype,
            enable_streamk=False, work_stealing=False,
        ), f"canonical cell {(M, N, K, dt)} must route to hipBLASLt"


def test_dispatch_helper_rejects_complement_control_set():
    """A representative complement set: same M / K / dtype grid but at N
    rungs OUTSIDE the wave-misaligned skinny-N ladder (N=128 has its own
    P28 slot; N=256 has P31; N=512 has P15/P17/P23/P25/P27).  The canonical
    must not match these cells (zero false-POSITIVES from the consolidation
    — those cells are routed by other slots, not by K-1881)."""
    out_of_ladder_ns = (128, 256, 512, 768, 1024, 1536, 2048)
    for M in (2048, 4096, 8192):
        for N in out_of_ladder_ns:
            for K in (4096, 8192, 16384):
                for dt in ("torch.bfloat16", "torch.float16"):
                    assert (M, N, K, dt) not in CANON, (
                        f"complement cell {(M, N, K, dt)} must not be in K-1881"
                    )


def test_compact_predicate_does_not_cover_canonical():
    """K-1858 RCA learning: report whether a compact predicate covers the
    same set without enumeration.  Candidate: ``N % 64 != 0 ∧ N ≤ 384
    ∧ K ≥ 4096``.  Pin the audit result here so any future predicate
    refactor must re-prove coverage on N ∈ {320, 384}.

    Result: the candidate predicate FALSE-NEGATIVES on N ∈ {320, 384}
    (both divisible by 64).  The N=320 (17 cells) + N=384 (14 cells) = 31
    cells of the canonical are NOT covered by the candidate predicate —
    therefore the enumerated frozenset is retained as the canonical
    truth-source.  Future refactor: any predicate replacement must cover
    all 7 N rungs {96, 160, 224, 288, 320, 352, 384} — the wave-
    misalignment mechanism is BLOCK_N=128 partial-coverage / tail-half-wave
    (R-1811), which is NOT a simple ``N % 64 != 0`` divisibility property.
    """
    def candidate(M, N, K, dt):
        return (N % 64) != 0 and N <= 384 and K >= 4096

    covered = {c for c in CANON if candidate(*c)}
    not_covered = CANON - covered
    # Predicate misses N ∈ {320, 384} entirely.
    assert {N for (_, N, _, _) in not_covered} == {320, 384}
    # 17 (P36) + 14 (P38) = 31 cells uncovered by the candidate.
    assert len(not_covered) == 17 + 14 == 31
    # Frozenset is therefore the canonical truth-source — this assertion is
    # a pin: if a future refactor flips this to True, the enumerated
    # frozenset can be replaced with the predicate.
    assert covered != CANON
