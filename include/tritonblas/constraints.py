# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Single-owner module for tritonblas tile + dtype dispatch constraints.

This module is intentionally small and free of GPU / Triton imports so it
can be loaded from anywhere in the package — the dispatch path
(``origami.py``), the public matmul launchers (``matmul.py``), and the
test suite all share a single source of truth for:

* :func:`is_pow2` — pure power-of-two predicate.
* :func:`is_triton_valid_block_tile` — predicate matching Triton's
  ``tl.zeros((BLOCK_M, BLOCK_N))`` accumulator-shape constraint as
  enforced inside ``kernels/stages/gemm_context.py::init_accumulator``.
  A non-pow2 ``BLOCK_M`` / ``BLOCK_N`` / ``BLOCK_K`` raises
  ``ValueError: Shape element N must be a power of 2`` at first compile.
  Mirroring the constraint at the dispatch site lets the dispatcher fail
  closed (return ``None`` / pick a different tile) rather than surfacing
  the compile error deep inside the Triton code path with no pointer
  back to the offending entry.
* :func:`kpack_for_dtype` — tile/dtype-whitelist gate for the
  ``kpack=2`` argument on fp16 / bf16 matmul launches (see
  :mod:`tritonblas.matmul`). The whitelist is empirically derived; see
  :data:`KPACK2_TILE_DTYPE_WHITELIST` for the validated entries.

The codegen site (``init_accumulator``) cannot import this module from
inside its ``@triton.jit`` body, so the docstring there cross-references
this file as the canonical predicate. Any change to the upstream
Triton-AMD accumulator constraint MUST be reflected here as well — a
unit test at ``tests/test_pow2_guard_and_kpack.py`` asserts that every
shipped override entry passes the predicate so the two ends cannot drift
silently.
"""

from __future__ import annotations

import os
from typing import FrozenSet, Tuple

import torch


def is_pow2(n: int) -> bool:
    """Return True iff ``n`` is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def is_triton_valid_block_tile(bm: int, bn: int, bk: int) -> bool:
    """Return True iff ``(BM, BN, BK)`` is a Triton-legal block tile.

    Triton's IR enforces power-of-two shapes on every
    ``tl.zeros((BLOCK_M, BLOCK_N))`` accumulator allocation; the
    constraint lives in
    ``include/tritonblas/kernels/stages/gemm_context.py::init_accumulator``.
    A non-pow2 ``BLOCK_M`` or ``BLOCK_N`` raises ``ValueError`` at first
    compile. ``BLOCK_K`` is included for symmetry — non-pow2 ``BLOCK_K``
    also fails Triton's accumulator-loop unrolling.
    """
    return is_pow2(bm) and is_pow2(bn) and is_pow2(bk)


# ---------------------------------------------------------------------------
# kpack=2 whitelist
# ---------------------------------------------------------------------------
#
# This whitelist is the SOLE production-side allowlist for shipping
# ``kpack=2`` to Triton's matmul launchers. Each entry is a
# ``(BLOCK_M, BLOCK_N, BLOCK_K, a_dtype)`` tuple that crossed the
# ±1.5pp paired-bench noise floor on MI300X / gfx942 with NO row
# regressing below the floor.
#
# ⚠ DO NOT widen this set without first running the paired-bench
# harness on the candidate ``(BM, BN, BK, dtype)`` and confirming:
#
#   (a) median Δ ≥ +1.5pp across at least two independent runs, AND
#   (b) no individual row regresses below -1.5pp.
#
# A previous "asymmetric tile" (block_m != block_n) heuristic was
# rolled back because it falsely included ``128x256x64`` (the long-N
# skinny family), which the paired bench showed regressing on bf16
# with a median Δ inside noise — failing the falsification criterion.
# A fully-unconditional ``kpack=2`` for fp16/bf16 was also tried and
# rolled back: the symmetric ``256x256x64`` tile regressed -8 to -10pp
# on ``8192x8192x8192`` (consistent with the symmetric-tile
# mfma-operand-layout regression documented in prior Triton-AMD work).
#
# CAVEAT: the validated entries below rest on n=2 dtype rows per tile
# (one fp16 + one bf16) at a single shape (8192x1024x8192). This is a
# deliberately narrow allowlist — extend it shape-by-shape with paired
# bench evidence rather than generalizing on tile shape alone.
KPACK2_TILE_DTYPE_WHITELIST: FrozenSet[Tuple[int, int, int, torch.dtype]] = frozenset({
    # Long-M skinny family (8192x1024x8192). Paired bench measured a
    # +2 to +4pp lift at the 256x128x64 tile for both fp16 and bf16
    # across two independent 100-iter cuda.Event runs on MI300X; both
    # dtype rows above the +1.5pp noise floor and zero rows regressing.
    (256, 128, 64, torch.float16),
    (256, 128, 64, torch.bfloat16),
})


def kpack_for_dtype(
    a_dtype: torch.dtype,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
) -> int:
    """Return the ``kpack`` argument the Triton matmul launchers should use.

    Both ``persistent_matmul_lt`` and ``streamk_matmul_lt`` MUST go
    through this helper so the two launch paths cannot drift.

    Policy: whitelist-only. ``kpack=2`` is shipped ONLY for the
    ``(BLOCK_M, BLOCK_N, BLOCK_K, a_dtype)`` tuples in
    :data:`KPACK2_TILE_DTYPE_WHITELIST`; every other dispatch returns
    ``kpack=1``.

    Empirical justification (validated on MI300X / gfx942 with a
    100-iter ``cuda.Event`` paired bench, preallocated outputs, two
    independent runs after clearing ``~/.triton/cache/``):

    * Whitelisted entries (T1 256x128x64 fp16/bf16) measured
      +3.28pp / +3.66pp lift over kpack=1 — the only residual-cohort
      bucket whose median Δ stayed above the ±1.5pp noise floor on both
      runs with ZERO rows regressing.

    * The ``128x256x64`` tile (T2-a, the same tile flipped) FAILED the
      gate: median Δ +0.18pp with bf16 regressing -1.56pp. It is
      therefore EXCLUDED from the whitelist even though it is also
      asymmetric — confirming that the kpack=2 win is not a generic
      "asymmetric tile" property but a specific
      ``(BM, BN, BK, dtype)`` co-occurrence that must be validated case
      by case.

    * The ``256x256x64`` symmetric tile regressed -8.8 to -10.1pp on
      ``8192x8192x8192`` when kpack=2 was shipped unconditionally
      (consistent with the symmetric-tile mfma-operand-layout
      regression documented in prior Triton-AMD work). It is excluded.

    * fp8 / fp4 / fp32 paths always receive ``kpack=1``: the fp8 path
      is gfx950-clamped at codegen time and the fp4 path uses its own
      kernel entry point.

    Set ``TRITONBLAS_FORCE_KPACK1=1`` to clamp the helper to ``kpack=1``
    for every dispatch — used by the paired baseline-vs-fix bench
    harness to isolate the kpack=2 contribution from other selector
    changes without checking out an older revision. Production callers
    leave the env var unset.

    Args:
        a_dtype: A-operand dtype (the fp16 / bf16 selection key).
        block_m, block_n, block_k: Selected tile dimensions. When any
            of them is ``None`` (e.g. an external caller has not yet
            plumbed them through), the helper falls back to the safe
            ``kpack=1`` setting rather than guessing.

    Returns:
        ``2`` iff ``(block_m, block_n, block_k, a_dtype)`` is in the
        validated whitelist; ``1`` otherwise.
    """
    if os.environ.get("TRITONBLAS_FORCE_KPACK1", "0") == "1":
        return 1
    if a_dtype is not torch.float16 and a_dtype is not torch.bfloat16:
        return 1
    if block_m is None or block_n is None or block_k is None:
        return 1
    return 2 if (block_m, block_n, block_k, a_dtype) in KPACK2_TILE_DTYPE_WHITELIST else 1
