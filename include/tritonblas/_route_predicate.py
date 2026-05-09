"""bf16 hipBLASLt route-OUT predicate (torch-free, data-only).

Imported by ``matmul.py`` for the runtime dispatch; importable on its own
(no torch / triton chain) so the predicate decision can be exercised
without booting a GPU stack. dtype is matched by ``str(dtype)`` so both
``torch.bfloat16`` and the literal string ``"torch.bfloat16"`` select the
bf16 path.

Layered route-OUT stack consumed by ``matmul._k971_route_to_hbl``:

  1. ``P8_ROUTE_OUT``         strict-equality cohort union (bf16 only).
  2. ``R_E1_route_to_hbl``    closed-form axis-aligned envelope (defense
                              in depth on cells P8 has not enumerated).
  3. ``R_P5_route_to_hbl``    closed-form structural pathology predicate.
  4. ``K971_ROUTE_TABLE``     legacy strict-equality fp16 / bf16 anchors.

Validated on MI300X via paired n=30 HIP-graph hot-cache benchmarks
(B=10000 bootstrap CI95) and cross-arch backtested on MI325X / MI355X for
the N=128 admit cluster — see ``output/stacked_envelope_gist.md`` for the
full per-layer attribution and per-cell speedup numbers.
"""
from __future__ import annotations


def _dtype_is_bf16(dtype) -> bool:
    """Match torch.bfloat16 without importing torch."""
    return str(dtype) == "torch.bfloat16"


# ---------------------------------------------------------------------------
# Legacy strict-equality anchors (fp16 + bf16). Pre-existing K971 cohort;
# kept as a tail of the dispatch chain so legacy mid-square long-K shapes
# continue to route to hipBLASLt regardless of the bf16 stacked layers.
# ---------------------------------------------------------------------------
K971_ROUTE_TABLE = frozenset({
    (1024, 1024, 16384, "torch.bfloat16"),
    (1024, 1024, 16384, "torch.float16"),
    (1024, 1024, 32768, "torch.bfloat16"),
    (1024, 1024, 32768, "torch.float16"),
    (2048, 2048, 16384, "torch.bfloat16"),
    (2048, 2048, 16384, "torch.float16"),
    (2048, 2048, 32768, "torch.bfloat16"),
    (2048, 2048, 32768, "torch.float16"),
})


# ---------------------------------------------------------------------------
# P8 stacked strict-equality route-OUT — 36-cell bf16 union of four
# measurement layers. Composed in the K-1216-validated stacking order:
#
#   P8 anchors            (13 cells)  — MFMA-issue-stall PMC cohort
#                                       (5 PMC matrix + 8 leakage).
#                                       Paired n=30 hbl/tb 1.16x–1.37x;
#                                       cohort geomean 1.226x.
#   P8 neighbors          (12 cells)  — ±1-pow2 perturbations of 4
#                                       representative anchors (S32, S39,
#                                       S18, S24). Paired n=30 hbl/tb
#                                       1.15x–1.66x; geomean 1.367x.
#   E2 axis-extension     ( 3 cells)  — M-axis extension at K=1024 + one
#                                       K=736 pin. K-floor stays at K>=256
#                                       (NEGATIVE_AXIS_PIVOT for K<256;
#                                       Triton-favoured small-M tail).
#   N=128 admit cluster   ( 8 cells)  — extreme-aspect-ratio M>>N=128
#                                       admits; paired n=30 + B=10000
#                                       bootstrap CI95 admit at >=1.10x;
#                                       per-cell speedups 1.14x–3.25x.
#                                       (M=4480, N=128, K=768) deliberately
#                                       OMITTED — measured 0.967x at the
#                                       M-floor of the anchor M-set.
#
# Sub-set provenance is encoded in the trailing-comment "tag" on each
# entry (A=anchor, NB=neighbor, E2=axis-extension, N128=N=128 admit).
# Sub-sets are pairwise disjoint by construction (anchors and neighbors
# share no cells; axis-extension and N=128 admits were drawn from cell
# pools that exclude both).
# ---------------------------------------------------------------------------
P8_ROUTE_OUT = frozenset({
    # ---- P8 anchors (13) ----
    ( 4480, 3072,  768, "torch.bfloat16"),  # A   S24  ~1.20x
    (14208, 2048, 1024, "torch.bfloat16"),  # A   S29  ~1.22x
    ( 5972, 1792,  768, "torch.bfloat16"),  # A   S18  ~1.16x
    ( 6016, 2048, 1024, "torch.bfloat16"),  # A   S25  ~1.21x
    (16256, 2048, 1024, "torch.bfloat16"),  # A   S30  ~1.24x
    ( 8064, 2048, 1024, "torch.bfloat16"),  # A   S26  ~1.22x
    (18304, 2048, 1024, "torch.bfloat16"),  # A   S31  ~1.27x
    (20352, 2048, 1024, "torch.bfloat16"),  # A   S32  ~1.25x
    (22400, 2048, 1024, "torch.bfloat16"),  # A   S33  ~1.21x
    (24448, 2048, 1024, "torch.bfloat16"),  # A   S34  ~1.19x
    (26496, 2048, 1024, "torch.bfloat16"),  # A   S35  ~1.24x
    (25600, 2048,  256, "torch.bfloat16"),  # A   S37  ~1.30x
    (49152, 2048,  256, "torch.bfloat16"),  # A   S39  ~1.37x
    # ---- P8 neighbors (12) ----
    (20352, 2048,  512, "torch.bfloat16"),  # NB  N01  S32 K/2
    (20352, 2048, 2048, "torch.bfloat16"),  # NB  N02  S32 K*2
    (40704, 2048, 1024, "torch.bfloat16"),  # NB  N03  S32 M*2
    (10176, 2048, 1024, "torch.bfloat16"),  # NB  N04  S32 M/2  1.604x
    (20352, 4096, 1024, "torch.bfloat16"),  # NB  N05  S32 N*2
    (20352, 1024, 1024, "torch.bfloat16"),  # NB  N06  S32 N/2  1.657x (max)
    (49152, 2048,  128, "torch.bfloat16"),  # NB  N07  S39 K/2
    (49152, 2048,  512, "torch.bfloat16"),  # NB  N08  S39 K*2
    ( 5972,  896,  768, "torch.bfloat16"),  # NB  N09  S18 N/2
    ( 5972, 3584,  768, "torch.bfloat16"),  # NB  N10  S18 N*2
    ( 2240, 3072,  768, "torch.bfloat16"),  # NB  N11  S24 M/2
    ( 4480, 3072, 1536, "torch.bfloat16"),  # NB  N12  S24 K*2
    # ---- E2 axis-extension admits (3) ----
    (  736, 1792,  736, "torch.bfloat16"),  # E2  K=736 single-cell pin   1.315x
    (10112, 2048, 1024, "torch.bfloat16"),  # E2  M-axis ext at K=1024    1.759x
    (12160, 2048, 1024, "torch.bfloat16"),  # E2  M-axis ext at K=1024    1.499x
    # ---- N=128 admit cluster (8) ----
    ( 6016,  128, 1024, "torch.bfloat16"),  # N128 K=1024  1.370x
    ( 8064,  128, 1024, "torch.bfloat16"),  # N128 K=1024  1.402x
    (14208,  128, 1024, "torch.bfloat16"),  # N128 K=1024  3.254x (max)
    (16256,  128, 1024, "torch.bfloat16"),  # N128 K=1024  3.024x
    (22400,  128, 1024, "torch.bfloat16"),  # N128 K=1024  2.258x
    (25600,  128,  256, "torch.bfloat16"),  # N128 K=256   1.144x
    (49152,  128,  256, "torch.bfloat16"),  # N128 K=256   1.781x
    ( 5972,  128,  768, "torch.bfloat16"),  # N128 K=768   1.227x
    # OMIT (4480, 128, 768, bf16) — 0.967x measured at M-floor; kept out
    # of the union by construction (carve-out by omission).
})


# ---------------------------------------------------------------------------
# E1 axis-aligned envelope. bf16 ∧ N ∈ {1792, 2048, 3072} ∧ K ∈ {256, 768,
# 1024} ∧ M >= 4480 ∧ K >= 256. Stacked AFTER P8 in the dispatch chain so
# P8 wins on overlap (no double-routing); E1 only fires on cells that P8
# does not enumerate, providing defense in depth for shapes outside the
# enumerated cohort. K-floor stays at K>=256 — K=128 is excluded per the
# cross-arch (MI325X / MI355X) result that K<256 is Triton-favoured.
# ---------------------------------------------------------------------------
_E1_NS = frozenset({1792, 2048, 3072})
_E1_KS = frozenset({256, 768, 1024})
_E1_M_FLOOR = 4480
_E1_K_FLOOR = 256


def R_E1_route_to_hbl(M: int, N: int, K: int, dtype) -> bool:
    """E1 envelope: route-OUT for the bf16 axis-aligned cohort."""
    if not _dtype_is_bf16(dtype):
        return False
    if N not in _E1_NS or K not in _E1_KS:
        return False
    if M < _E1_M_FLOOR or K < _E1_K_FLOOR:
        return False
    return True


def R_P5_route_to_hbl(M: int, N: int, K: int, dtype) -> bool:
    """P5 closed-form 4-clause structural pathology predicate (bf16).

    Clause-1: over-tile / LDS-bound mid-rect non-square with K in the
              [1240, 8064] band and one axis pinned to {1792, 2048, 3072}.
    Clause-2: hbl-strong large-M skinny  M >= 5000 ∧ N == 2048 ∧
              K ∈ {256, 1024}.
    Clause-3: tb-weak — extreme aspect (max/min >= 100), or skinny-and-
              long-K (min(M,N) <= 192 ∧ K >= 2048).
    Clause-4: N=1792 shallow-K dual of clause-1; K-floor=512 (8*BK64) to
              silence the K=256 false positive.
    """
    if not _dtype_is_bf16(dtype):
        return False
    minMN, maxMN = min(M, N), max(M, N)
    if (minMN < maxMN
            and 256 <= minMN <= 2304
            and 1792 <= maxMN <= 3072
            and 1240 <= K <= 8064):
        return True
    if M >= 5000 and N == 2048 and K in (256, 1024):
        return True
    if maxMN >= 100 * max(1, minMN):
        return True
    if minMN <= 192 and K >= 2048:
        return True
    if N == 1792 and 512 <= K <= 768 and (M >= 5000 or 256 <= M <= 2048):
        return True
    return False


def _p8_route_out(M: int, N: int, K: int, dtype) -> bool:
    """P8 strict-equality stacked union — bf16 only."""
    if not _dtype_is_bf16(dtype):
        return False
    return (int(M), int(N), int(K), str(dtype)) in P8_ROUTE_OUT
