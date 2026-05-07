# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Per-architecture tile-override registry for the batched matmul (``bmm``) path.

Why this exists
===============
``bmm`` shares the same persistent kernel as the rank-2 ``matmul`` path but
launches it on a 3D grid ``(tiles_per_batch, batch_size)``. The Origami
selector picks tiles assuming a single rank-2 problem, which under-utilizes the
CUs (or uses the wrong number of warps for the chosen tile) once all batches
share one grid. A small, per-arch override on top of Origami closes a large
chunk of the structural gap vs hipBLASLt for batched workloads.

Putting the override behind a registry rather than inside ``matmul.py`` keeps
the rank-2 launcher untouched and makes it explicit what a new arch needs to
do to opt in: drop a new ``Override`` rule list under that arch's key. No
string-matching, no fall-through to a different arch's tunings.

Heuristic constants
===================
The tier thresholds (M·N area, batch size) and the tile / num_warps / kpack /
group_size_m / num_xcds picks were derived from a wide sweep on MI300X
(gfx942, 64 KiB LDS per CU) over the batched residual shapes that
the prior Python-loop emulation under-served. The full sweep methodology
(block sizes, num_warps, num_stages, kpack, mfma_instr_nonkdim,
waves_per_eu, XCDS combinations) and ratio measurements are documented in
the PR description.

Notably ``num_xcds=1`` (no chiplet remap) is the single largest knob for
small-tile batched workloads: with ``B*tiles_per_batch`` tile work units, the
chiplet remap that helps rank-2 large-K problems hurts per-batch L2 locality
(each batch element's operands live in a contiguous L2 region; the remap
scrambles them across XCDs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class TileOverride:
    """One tile-override rule for the batched persistent kernel.

    A rule fires when ``predicate(M, N, K, B, a_dtype, b_dtype)`` returns True
    *and* the resulting tile fits the hardware LDS at the requested
    ``num_stages``. The first matching+fitting rule in the per-arch list wins.

    Fields mirror the constexpr knobs the launcher passes to the kernel:
    ``block_m / block_n / block_k`` are the GEMM tile, ``group_m`` is the
    M-major super-tile width (``0`` means "keep the selector's group_m"),
    ``num_warps`` is the workgroup width, ``kpack`` is the K-axis packing
    factor for MFMA, and ``num_xcds`` is the chiplet-remap divisor (``1`` =
    no remap).

    The optional ``why`` string is for debug logging / docs only; it does not
    affect dispatch.
    """
    predicate: Callable[[int, int, int, int, torch.dtype, torch.dtype], bool]
    block_m: int
    block_n: int
    block_k: int
    group_m: int          # 0 = "keep selector's group_m"
    num_warps: int
    kpack: int
    num_xcds: int         # 1 = no chiplet remap
    why: str = ""


# ───────────────────────── gfx942 (MI300X) tunings ─────────────────────────
# Rules are evaluated in order; first match wins. The shape predicates are
# kept minimal (M*N area + B + dtype) because the sweep showed K-axis was a
# secondary signal at these problem sizes — what matters most is grid
# saturation (B * tiles_per_batch vs CU count) and per-tile work intensity.
#
# Tier 1 — VERY large MN (M*N >= 4M): fp16 B=4 2048^3 lives here.
#   Empirically:
#     Origami default (256x256x64 nw=8 XCDS=8) → ~0.85 boundary
#     This rule (64x128x32 nw=4 XCDS=1)        → ~0.87 (clears 0.85 with margin)
#
# Tier 2 — Medium MN with B>=4: bf16 B=8 1024^3 lives here.
#   Origami picks 64x64x256 (rank-2 optimal) but it under-saturates the CUs.
#   This rule (64x128x32 nw=4 XCDS=1) is the empirical best across the
#   sweep:
#     ratio rises from ~0.07 (Python loop emulation) → ~0.66 (3D-grid w/
#     Origami defaults) → ~0.73 (this rule). Above ~0.73 is structurally
#     bounded by Triton's MFMA scheduling vs hipBLASLt's hand-tuned
#     assembly on small bf16 problems. Closing the residual gap requires
#     a StreamK-on-batched-grid + MFMA-tuning kernel workstream which is
#     out of scope here; the public ``bmm`` docstring documents the cap so
#     callers picking ``bmm`` for these shapes know what to expect.
def _gfx942_tier1(M, N, K, B, ad, bd):
    return B >= 2 and M >= 256 and N >= 256 and K >= 64 and (M * N) >= (4 * 1024 * 1024)


def _gfx942_tier2(M, N, K, B, ad, bd):
    return (
        B >= 4 and M >= 256 and N >= 256 and K >= 64
        and (M * N) >= (1024 * 1024)
    )


_GFX942_RULES: List[TileOverride] = [
    TileOverride(
        predicate=_gfx942_tier1,
        block_m=64, block_n=128, block_k=32,
        group_m=4, num_warps=4, kpack=1, num_xcds=1,
        why="tier1: MN>=4M, no XCD remap, smaller tile saturates CUs better than rank-2 256x256",
    ),
    TileOverride(
        predicate=_gfx942_tier2,
        block_m=64, block_n=128, block_k=32,
        group_m=4, num_warps=4, kpack=1, num_xcds=1,
        why="tier2: MN>=1M & B>=4, no XCD remap, small tile keeps grid saturation high",
    ),
]


# ───────────────── per-arch registry ─────────────────
# Lookup: arch_string -> ordered rule list. Adding a new arch is a single-line
# addition here plus the per-arch sweep that produces its own rule list. The
# launcher (matmul.py) treats an absent arch as "no override" and defers to
# the unmodified Origami pick — safe by default.
_REGISTRY: Dict[str, List[TileOverride]] = {
    "gfx942": _GFX942_RULES,
}


def lookup(
    arch: str,
    M: int, N: int, K: int, B: int,
    a_dtype: torch.dtype, b_dtype: torch.dtype,
    lds_cap: int,
    num_stages: int,
) -> Optional[TileOverride]:
    """Return the first override rule for ``arch`` that matches the shape
    *and* whose tile fits LDS at ``num_stages``. ``None`` = no override.

    Quantized dtypes (int8, fp4, fp8) bypass the override — the rules were
    derived from fp16/bf16 sweeps, and quantized GEMMs have different LDS
    footprint constraints (per-channel scales). Defer those to Origami until
    they get their own sweep.
    """
    rules = _REGISTRY.get(arch)
    if rules is None:
        return None
    if B < 2:
        return None
    try:
        bytes_a = torch.finfo(a_dtype).bits // 8
        bytes_b = torch.finfo(b_dtype).bits // 8
    except TypeError:
        return None  # int8/fp8 quantized — let Origami handle it.

    for rule in rules:
        if not rule.predicate(M, N, K, B, a_dtype, b_dtype):
            continue
        # LDS-fit check at the requested num_stages. Triton's async_copy
        # software pipeline holds (num_stages-1) buffers of A_tile + B_tile;
        # if the tile doesn't fit, the rule is skipped (next rule in the
        # list, or None).
        stages = max(1, num_stages - 1)
        lds = stages * (rule.block_m * rule.block_k * bytes_a +
                        rule.block_k * rule.block_n * bytes_b)
        if lds <= lds_cap:
            return rule
    return None
