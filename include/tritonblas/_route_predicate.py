"""Strict-equality MFMA-issue-stall route-OUT table.

Pure data module: a single bf16 ``(M, N, K)`` frozenset of GEMM shapes that
benchmarked faster on hipBLASLt than the in-tree Triton kernel under the
MFMA-issue-stall regime, plus a one-line lookup helper. No closed-form
predicates, no envelopes, no parameters --- the file is consulted at
dispatch time and short-circuits to ``torch.matmul`` for any shape whose
key matches.

The table composes four measurement campaigns that were validated end-to-end
on MI300X with paired n=30 HIP-graph hot-cache timing and 10k-resample
bootstrap CI95, and re-confirmed cross-arch on MI325X / MI355X. Each
sub-set is kept as its own named frozenset so reviewers can audit the
provenance lineage without grepping git history; runtime asserts pin both
the cardinality and pairwise disjointness so a stray duplicate or rename
fails import.

Importable on its own (no ``torch`` / ``triton`` chain) so unit tests can
exercise it without booting a GPU stack.
"""
from __future__ import annotations


def _dtype_is_bf16(dtype) -> bool:
    """Match ``torch.bfloat16`` without importing torch.

    Both the ``torch.bfloat16`` singleton and the literal string
    ``"torch.bfloat16"`` (used by tests) pass; anything else fails.
    """
    return str(dtype) == "torch.bfloat16"


# ---------------------------------------------------------------------------
# 13 mid-square / wide-rect anchors. Per-cell mean speedup hipBLASLt/Triton
# annotated from paired n=30 HIP-graph hot-cache on MI300X / ROCm 7.2;
# range 1.158x-1.365x; cohort geomean 1.226x; CI95-lo >= 1.05x on every cell.
# ---------------------------------------------------------------------------
_MFMA_STALL_ANCHORS_13 = frozenset({
    ( 4480, 3072,  768, "torch.bfloat16"),  # ~1.20x
    (14208, 2048, 1024, "torch.bfloat16"),  # ~1.22x
    ( 5972, 1792,  768, "torch.bfloat16"),  # ~1.16x
    ( 6016, 2048, 1024, "torch.bfloat16"),  # ~1.21x
    (16256, 2048, 1024, "torch.bfloat16"),  # ~1.24x
    ( 8064, 2048, 1024, "torch.bfloat16"),  # ~1.22x
    (18304, 2048, 1024, "torch.bfloat16"),  # ~1.27x
    (20352, 2048, 1024, "torch.bfloat16"),  # ~1.25x
    (22400, 2048, 1024, "torch.bfloat16"),  # ~1.21x
    (24448, 2048, 1024, "torch.bfloat16"),  # ~1.19x
    (26496, 2048, 1024, "torch.bfloat16"),  # ~1.24x
    (25600, 2048,  256, "torch.bfloat16"),  # ~1.30x
    (49152, 2048,  256, "torch.bfloat16"),  # ~1.37x
})

# ---------------------------------------------------------------------------
# 12 held-out neighbors perturbed +/-1 power-of-2 on M, N, or K axes from
# four representative anchors. Validated paired n=30 + 10k-resample paired
# bootstrap; cohort geomean 1.367x; envelope-edge cells noted inline.
# ---------------------------------------------------------------------------
_MFMA_STALL_NEIGHBORS_12 = frozenset({
    # Wide-rect family (anchor 20352x2048x1024) -- full 6-axis coverage
    (20352, 2048,  512, "torch.bfloat16"),  # K/2
    (20352, 2048, 2048, "torch.bfloat16"),  # K*2
    (40704, 2048, 1024, "torch.bfloat16"),  # M*2
    (10176, 2048, 1024, "torch.bfloat16"),  # M/2  (envelope edge: 1.604x)
    (20352, 4096, 1024, "torch.bfloat16"),  # N*2
    (20352, 1024, 1024, "torch.bfloat16"),  # N/2  (cohort MAX: 1.657x)
    # Extreme-M family (anchor 49152x2048x256) -- K-axis only
    (49152, 2048,  128, "torch.bfloat16"),  # K/2
    (49152, 2048,  512, "torch.bfloat16"),  # K*2
    # Only-non-2K-N family (anchor 5972x1792x768) -- N-axis only
    ( 5972,  896,  768, "torch.bfloat16"),  # N/2
    ( 5972, 3584,  768, "torch.bfloat16"),  # N*2
    # Only-N=3072 family (anchor 4480x3072x768) -- M/2 + K*2
    ( 2240, 3072,  768, "torch.bfloat16"),  # M/2
    ( 4480, 3072, 1536, "torch.bfloat16"),  # K*2
})

# ---------------------------------------------------------------------------
# 3 K-interior / M-axis extension admits beyond the anchor + neighbor set.
# A separate K-axis sweep at K in {96, 128, 192, 512, 768} returned a
# negative axis pivot (3/8 admit, 0/2 at K<256) -- only the K=736 single
# cell and the two M-axis extensions at the anchor K=1024 cleared the
# admission gate (hbl/tb >= 1.10x mean AND CI95-lo >= 1.05x). Strict-equality
# only --- the K-floor for any future structural extension MUST remain at
# K >= 256 unless paired with an M-floor strictly above the small-M
# Triton-favoured tail.
# ---------------------------------------------------------------------------
_MFMA_STALL_K_INTERIOR_AND_M_EXTENSION_3 = frozenset({
    (  736, 1792,  736, "torch.bfloat16"),  # K=736 upper edge (hbl/tb=1.315x)
    (10112, 2048, 1024, "torch.bfloat16"),  # M-axis ext (hbl/tb=1.759x)
    (12160, 2048, 1024, "torch.bfloat16"),  # M-axis ext (hbl/tb=1.499x)
})

# ---------------------------------------------------------------------------
# 8 N-axis admits at N=128 (one power-of-2 tier below the natural sub-1024 N
# floor of the anchor set). Validated paired n=30 HIP-graph hot-cache: 8/9
# = 88.9% admit, decisively above the 75% landing threshold and divergent
# from the K-axis negative pivot result. Mechanistic reading: at extreme
# aspect ratios with M >> N=128, hipBLASLt's Tensile schedule (wide unroll
# + many waves) hides the same MFMA-issue-stall bottleneck that the source
# anchors expose at N=2048; per-cell speedups widen 1.14x-3.25x at N=128
# vs 1.16x-1.27x at N=2048. The single rejection (M=4480, K=768) is the
# smallest-M cell and is excluded.
# ---------------------------------------------------------------------------
_MFMA_STALL_N128_ADMITS_8 = frozenset({
    # N=128, K=1024 cluster (5 admits; speedups 1.37x-3.25x)
    ( 6016,  128, 1024, "torch.bfloat16"),  # 1.370x
    ( 8064,  128, 1024, "torch.bfloat16"),  # 1.402x
    (14208,  128, 1024, "torch.bfloat16"),  # 3.254x  (cohort MAX)
    (16256,  128, 1024, "torch.bfloat16"),  # 3.024x
    (22400,  128, 1024, "torch.bfloat16"),  # 2.258x
    # N=128, K=256 cluster (2 admits; extreme-M; 1.14x-1.78x)
    (25600,  128,  256, "torch.bfloat16"),  # 1.144x  (cohort MIN admit)
    (49152,  128,  256, "torch.bfloat16"),  # 1.781x
    # N=128, K=768 cluster (1 admit; M=4480 rejected)
    ( 5972,  128,  768, "torch.bfloat16"),  # 1.227x
})

# Composed 36-cell envelope. Four named provenance frozensets; the dispatch
# path consults the union. Source-campaign lineage is load-bearing for
# future reviewers (precedence-inversion debugging, predicate-classifier
# work, ADR audits) so each measurement campaign keeps its own named set
# with a runtime size + disjointness check.
_MFMA_STALL_ROUTEOUT = (
    _MFMA_STALL_ANCHORS_13
    | _MFMA_STALL_NEIGHBORS_12
    | _MFMA_STALL_K_INTERIOR_AND_M_EXTENSION_3
    | _MFMA_STALL_N128_ADMITS_8
)
assert len(_MFMA_STALL_ROUTEOUT) == 36, (
    "MFMA-stall envelope must be exactly 36 cells (13 anchors + 12 neighbors "
    "+ 3 K-interior/M-axis admits + 8 N=128 admits); a duplicate or stray "
    "entry has crept in.")
# Cross-check: the four sub-sets must be pairwise disjoint by construction.
# Neighbors perturbed AWAY from anchors; K-interior/M-axis admits selected
# from the always-uncovered top catalog minus all anchors and minus all
# neighbors; N=128 admits use N=128 (which is in NO prior cell -- prior
# cells all have N >= 896).
assert _MFMA_STALL_ANCHORS_13.isdisjoint(_MFMA_STALL_NEIGHBORS_12), (
    "Anchors and neighbors overlap; neighbor generation rules require "
    "strict disjointness from the 13 anchors.")
assert _MFMA_STALL_ANCHORS_13.isdisjoint(_MFMA_STALL_K_INTERIOR_AND_M_EXTENSION_3), (
    "K-interior/M-axis admits overlap with anchors; the candidate generator "
    "excluded all anchors by construction.")
assert _MFMA_STALL_NEIGHBORS_12.isdisjoint(_MFMA_STALL_K_INTERIOR_AND_M_EXTENSION_3), (
    "K-interior/M-axis admits overlap with neighbors; the candidate generator "
    "excluded all neighbors by construction.")
assert _MFMA_STALL_N128_ADMITS_8.isdisjoint(_MFMA_STALL_ANCHORS_13), (
    "N=128 admits overlap with anchors; N=128 candidates have N=128 by "
    "construction and no anchor has N=128.")
assert _MFMA_STALL_N128_ADMITS_8.isdisjoint(_MFMA_STALL_NEIGHBORS_12), (
    "N=128 admits overlap with neighbors; N=128 candidates have N=128 by "
    "construction and no neighbor has N=128.")
assert _MFMA_STALL_N128_ADMITS_8.isdisjoint(_MFMA_STALL_K_INTERIOR_AND_M_EXTENSION_3), (
    "N=128 admits overlap with K-interior/M-axis admits; N=128 candidates "
    "have N=128 by construction and no K-interior/M-axis admit has N=128.")


def route_to_hbl(M: int, N: int, K: int, dtype) -> bool:
    """Return True iff ``(M, N, K, dtype)`` is in the 36-cell strict-equality
    MFMA-issue-stall route-OUT envelope.

    bf16-only by design (the entire source measurement scope is bf16; fp16
    parity is tracked separately). Pure data lookup; no closed-form
    predicates, no envelope arithmetic. O(1) frozenset hash.
    """
    if not _dtype_is_bf16(dtype):
        return False
    return (int(M), int(N), int(K), str(dtype)) in _MFMA_STALL_ROUTEOUT
