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
  tile family rotated by aspect ratio. K-4539 confirmed transfer to BF16
  (7/10 cells, 2 sub-noise rotations, 1 known skinny-N exception).
* **K-4484** — Wave-quantization root-cause sweep that recovered
  4096×256×4096 by routing to a smaller BM=BN=64 tile with BK=128.

The dispatch table only fires on exact ``(M, N, K, dtype)`` matches. Any miss
falls through to the Origami selector. The table is intentionally narrow so
that landing it cannot regress shapes whose best tile is unknown.
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
# Per-shape dispatch table
# ----------------------------------------------------------------------------
#
# All entries below are FP16 TN unless otherwise noted. The corresponding BF16
# entries are derived by ``_build_bf16_mirror`` from the K-4539 transfer
# result (7/10 cells agree, no parallel BF16 table needed).
#
# Format:  (M, N, K, "f16") -> ShapeConfig(BM, BN, BK, num_stages, num_warps,
#                                          waves_per_eu, mfma, kpack)

_FP16_TABLE: Dict[Tuple[int, int, int], ShapeConfig] = {
    # ---- K-3019 worst-gap square FP16 TN cohort ----
    # Tiny / launch-overhead-bound shapes: best tile still recovers only ~28%
    # of hipBLASLt; lift over Origami default is 1.04-1.06x. Wiring them in
    # is documented as a no-regress baseline rather than a perf claim.
    (64,   64,   64):   ShapeConfig(256, 128, 256, 2, 8),
    (128,  128,  128):  ShapeConfig(256, 256, 128, 2, 8),
    # Small shapes where Origami selects BM=BN=16-32 but grid-best is >=256:
    # +24% and +23% over Origami default.
    (256,  256,  256):  ShapeConfig(256, 64,  128, 3, 8),
    (512,  512,  512):  ShapeConfig(256, 128, 128, 3, 4),

    # ---- K-4437 / K-4539 medium FP16 TN cohort (rotated 32K-element tile) ----
    # All five entries below win 1.5-33% over the Origami default; the same
    # tile family transfers cleanly to BF16 per K-4539.
    (2560, 2560, 2560):  ShapeConfig(256, 128, 64,  2, 8),
    (3072, 3072, 3072):  ShapeConfig(256, 128, 64,  2, 8),
    (8192, 1024, 4096):  ShapeConfig(256, 128, 64,  2, 8),  # skinny-N
    (8192, 1024, 8192):  ShapeConfig(256, 128, 64,  2, 8),  # skinny-N
    (5120, 5120, 13824): ShapeConfig(256, 128, 64,  2, 8),  # large-compute

    # ---- K-4437 long-K shape — already default-optimal but pinned to lock
    #      against future Origami drift. ----
    (2048, 2048, 16384): ShapeConfig(128, 128, 128, 2, 8),

    # ---- K-4484 wave-quantization fix for skinny-N exclusion ----
    # K-4437 picked (256, 128, 64) here and lost 43% to wave quantization
    # (32 work-groups onto 304 CUs). The K-4484 follow-up sweep recovered
    # +38% over the live K-4253 routing with (64, 64, 128, NS=2, NW=4, kpack=2).
    (4096, 256,  4096):  ShapeConfig(64,  64,  128, 2, 4, kpack=2),
}


def _build_bf16_mirror(
    fp16_table: Dict[Tuple[int, int, int], ShapeConfig],
) -> Dict[Tuple[int, int, int], ShapeConfig]:
    """Mirror the FP16 table into BF16, omitting the K-4539 disagreements.

    K-4539's head-to-head BF16 vs FP16 sweep on the K-4437 cohort showed:
      - 7/10 shapes pick the same tile in both dtypes
      - 2 shapes pick a sub-noise rotation within the same 32K-element family
      - 1 shape (4096x256x4096) genuinely diverges (BF16 prefers (128,64,128))

    For safety we only mirror the K-3019 tiny/small cohort (FP16-only sweep)
    and the K-4437/K-4539 agreed shapes. The 4096x256x4096 BF16 entry uses
    K-4539's verified BF16 winner.
    """
    bf16: Dict[Tuple[int, int, int], ShapeConfig] = {}
    for key, cfg in fp16_table.items():
        # 4096x256x4096 — BF16 winner differs (K-4539).
        if key == (4096, 256, 4096):
            bf16[key] = ShapeConfig(128, 64, 128, 2, 8)
            continue
        bf16[key] = cfg
    return bf16


_BF16_TABLE: Dict[Tuple[int, int, int], ShapeConfig] = _build_bf16_mirror(_FP16_TABLE)


# Flat lookup table keyed by (M, N, K, dtype_str).
SHAPE_DISPATCH_TABLE: Dict[_DispatchKey, ShapeConfig] = {}
for _key, _cfg in _FP16_TABLE.items():
    SHAPE_DISPATCH_TABLE[(*_key, "f16")] = _cfg
for _key, _cfg in _BF16_TABLE.items():
    SHAPE_DISPATCH_TABLE[(*_key, "bf16")] = _cfg


# Source / verification metadata for the KB ingest pass. Not consumed at
# runtime; intentionally kept beside the table so the two can drift visibly.
_SOURCE: Dict[Tuple[int, int, int], str] = {
    (64,   64,   64):    "K-3019 (FP16 TN worst-gap, 216-config grid)",
    (128,  128,  128):   "K-3019 (FP16 TN worst-gap, 216-config grid)",
    (256,  256,  256):   "K-3019 (FP16 TN worst-gap, 216-config grid)",
    (512,  512,  512):   "K-3019 (FP16 TN worst-gap, 216-config grid)",
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
