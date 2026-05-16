"""
Per-shape config dispatch table for tritonBLAS GEMM kernels on MI300X.

This table maps a problem shape ``(M, N, K, dtype)`` to the empirically-best
``(block_m, block_n, block_k, num_stages, num_warps, waves_per_eu, mfma, kpack)``
configuration measured via grid sweeps in the S-002 program.

Background
----------

The Origami heuristic in :mod:`tritonblas.origami` selects tile sizes from a
generic model of the GPU's compute / LDS / register budget. Targeted sweeps
showed it is systematically too small on a number of (M, N, K) shapes; no
single static tile wins across the cohort. Per-shape lookup is the cheapest
way to capture the residual ~5-pp geomean ratio against hipBLASLt without
touching the kernel implementation.

Provenance of each row is annotated in ``_SOURCE``. The dominant sources are:

* **K-3019** — 216-config grid sweep on 5 FP16 TN worst-gap shapes
  (64³, 128³, 256³, 512³, 3072³). Geomean lift over Origami default = 1.111×.
* **K-4437** — Adaptive BMxBNxBK sweep on a 10-shape proxy of K-4394
  worst-gap medium / non-square FP16 TN shapes; 8/10 wins via a 32K-element
  tile family rotated by aspect ratio.
* **K-4484** — Wave-quantization root-cause sweep that recovered
  4096×256×4096 by routing to a smaller BM=BN=64 tile with BK=128.

Cohort selection
----------------

This PR lands the **FP16-only, MI300X-LDS-fitting** measured winners.
After de-duplication and excluding losses / sub-noise deltas / LDS-budget
violations, **7 FP16 entries** ship in this table.

Shapes considered for the table but **deliberately omitted**:

* **K-3019 small-shape winners** (``64³``, ``128³``, ``256³``, ``512³``)
  — The grid-best tiles K-3019 reported all exceed the MI300X 64 KB LDS
  budget at the ``num_stages`` they were recorded for (e.g. ``512³`` best
  ``(256,128,128, ns=3)`` needs ~192 KB).  These rows were present in an
  earlier draft of this PR; they were dead code because
  :func:`tritonblas.origami.OrigamiMatmulSelector._apply_per_shape_override`
  silently falls back to Origami on LDS-budget violations.  Re-landing
  them needs either an LDS-aware tile re-sweep on MI300X (K-3019 was
  swept under a permissive FixedSelector that bypassed the LDS check)
  or a Triton kernel change to admit larger LDS allocations.  Both are
  out of scope for K-4108.
* ``4096×11008×4096`` — K-4437 phase-3 sweep best loses 2–3 % to the
  live Origami / K-4253 routing (this is the "12th" cohort shape the
  reviewer asked about).  Adding it would be a regression with no
  upside, so the table leaves the Origami pick in place.
* The two K-4437 "sub-noise rotation" shapes — same 32 K-element tile
  family as the agreed shapes but the per-shape best is within do_bench
  noise of the cohort default.

BF16 entries are **not** included. K-4539's BF16-vs-FP16 transfer
experiment showed only 7/10 cells agree, with 3 disagreements (2
sub-noise rotations plus a real divergence at 4096×256×4096).
Speculative mirroring would risk silent regressions on the disagreeing
cells without a BF16 benchmark to verify them on this PR. BF16 is
deferred to a follow-up PR backed by its own correctness + perf run.

The dispatch table only fires on exact ``(M, N, K, dtype)`` matches.
Any miss (including every BF16 shape, every FP8 shape, every FP16
shape not listed, and every shape whose entry would overflow LDS)
falls through to the Origami selector. The table is intentionally
narrow so that landing it cannot regress shapes whose best tile is
unknown or whose recorded tile cannot actually run on this hardware.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import torch


class ShapeConfig(NamedTuple):
    """Per-shape kernel configuration override.

    Fields ``waves_per_eu``, ``mfma`` and ``kpack`` are read by the matmul
    dispatcher via ``getattr(selector, "_override_<field>", <default>)`` so
    leaving them at the defaults below is equivalent to "do not override".
    """

    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int = 8
    waves_per_eu: int = 0
    mfma: int = 16
    kpack: int = 1


# Internal lookup key:  (M, N, K, dtype_str)
# dtype_str is one of the strings produced by
# ``OrigamiMatmulSelector.dtype_to_str`` (e.g. "f16", "bf16", "f8").
_DispatchKey = Tuple[int, int, int, str]


# ----------------------------------------------------------------------------
# Per-shape dispatch table — FP16 TN only.
# ----------------------------------------------------------------------------
#
# Format:  (M, N, K) -> ShapeConfig(BM, BN, BK, num_stages, num_warps,
#                                   waves_per_eu, mfma, kpack)

_FP16_TABLE: Dict[Tuple[int, int, int], ShapeConfig] = {
    # ---- K-4437 medium FP16 TN cohort (rotated 32K-element tile) ----
    # All five entries below win 1.5-33% over the Origami default.
    # Tile (256,128,64) at ns=2 uses ~48 KB LDS, fitting MI300X's 64 KB cap.
    (2560, 2560, 2560):  ShapeConfig(256, 128, 64,  2, 8),
    (3072, 3072, 3072):  ShapeConfig(256, 128, 64,  2, 8),
    (8192, 1024, 4096):  ShapeConfig(256, 128, 64,  2, 8),  # skinny-N
    (8192, 1024, 8192):  ShapeConfig(256, 128, 64,  2, 8),  # skinny-N
    (5120, 5120, 13824): ShapeConfig(256, 128, 64,  2, 8),  # large-compute

    # ---- K-4437 long-K shape — already default-optimal but pinned to lock
    #      against future Origami drift. (~64 KB LDS at ns=2, exactly at cap.)
    (2048, 2048, 16384): ShapeConfig(128, 128, 128, 2, 8),

    # ---- K-4484 wave-quantization fix for skinny-N exclusion ----
    # K-4437 picked (256, 128, 64) here and lost 43% to wave quantization
    # (32 work-groups onto 304 CUs). The K-4484 follow-up sweep recovered
    # +38% over the live K-4253 routing with (64, 64, 128, NS=2, NW=4, kpack=2).
    (4096, 256,  4096):  ShapeConfig(64,  64,  128, 2, 4, kpack=2),
}


# Flat lookup table keyed by (M, N, K, dtype_str). FP16-only by design;
# see module docstring "Cohort selection" for the BF16 deferral rationale.
SHAPE_DISPATCH_TABLE: Dict[_DispatchKey, ShapeConfig] = {
    (*_key, "f16"): _cfg for _key, _cfg in _FP16_TABLE.items()
}


# Source / verification metadata for the KB ingest pass. Not consumed at
# runtime; intentionally kept beside the table so the two can drift visibly.
_SOURCE: Dict[Tuple[int, int, int], str] = {
    (3072, 3072, 3072):  "K-4437 (FP16 TN medium cohort, supersedes K-3019)",
    (2560, 2560, 2560):  "K-4437 (FP16 TN medium cohort)",
    (8192, 1024, 4096):  "K-4437 (FP16 TN skinny-N)",
    (8192, 1024, 8192):  "K-4437 (FP16 TN skinny-N)",
    (5120, 5120, 13824): "K-4437 (FP16 TN large-compute)",
    (2048, 2048, 16384): "K-4437 (FP16 TN long-K, locks default)",
    (4096, 256,  4096):  "K-4484 (FP16 TN wave-quantization fix)",
}


def lookup(
    m: int,
    n: int,
    k: int,
    a_dtype_str: str,
) -> Optional[ShapeConfig]:
    """Return the per-shape config override for ``(M, N, K, dtype)`` or ``None``.

    The lookup is intentionally exact-match only: any miss returns ``None`` and
    leaves the Origami heuristic untouched. ``a_dtype_str`` must come from
    ``OrigamiMatmulSelector.dtype_to_str`` to keep the keying consistent with
    the rest of the selector code.
    """
    return SHAPE_DISPATCH_TABLE.get((m, n, k, a_dtype_str))


def lookup_torch(
    m: int,
    n: int,
    k: int,
    a_dtype: torch.dtype,
) -> Optional[ShapeConfig]:
    """Convenience wrapper that accepts a ``torch.dtype`` directly."""
    # Local import to avoid a circular import at module load time.
    from .origami import OrigamiMatmulSelector

    dtype_str = OrigamiMatmulSelector.dtype_to_str.get(a_dtype)
    if dtype_str is None:
        return None
    return lookup(m, n, k, dtype_str)


def covered_shapes() -> Tuple[_DispatchKey, ...]:
    """Return all ``(M, N, K, dtype_str)`` keys currently in the dispatch table."""
    return tuple(SHAPE_DISPATCH_TABLE.keys())
