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
* :func:`kpack_for_dtype` — centralized ``kpack=2`` gate for fp16 / bf16
  matmul launches (see :mod:`tritonblas.matmul`).

The codegen site (``init_accumulator``) cannot import this module from
inside its ``@triton.jit`` body, so the docstring there cross-references
this file as the canonical predicate. Any change to the upstream
Triton-AMD accumulator constraint MUST be reflected here as well — a
unit test at ``tests/test_pow2_guard.py`` asserts that every shipped
override entry passes the predicate so the two ends cannot drift
silently.
"""

from __future__ import annotations

import os

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


def kpack_for_dtype(
    a_dtype: torch.dtype,
    block_m: int | None = None,
    block_n: int | None = None,
) -> int:
    """Return the ``kpack`` argument the Triton matmul launchers should use.

    Both ``persistent_matmul_lt`` and ``streamk_matmul_lt`` MUST go
    through this helper so the two launch paths cannot drift.

    Policy (validated on MI300X / gfx942 with a 100-iter cuda.Event
    paired bench, preallocated outputs):

    * ``kpack=2`` is shipped ONLY for fp16 / bf16 dispatches that land
      on an **asymmetric** tile (``block_m != block_n``). On the long-M
      skinny family (``8192x1024x8192`` at ``256x128x64``), kpack=2
      produces a measured +3.3 / +3.7pp same-tile lift over kpack=1 —
      the only bucket in the residual cohort that closed cleanly.

    * ``kpack=2`` is NOT shipped for symmetric tiles. A paired bench on
      ``8192x8192x8192`` at ``256x256x64`` measured a -10.14pp /
      -12.31pp regression vs kpack=1 (well outside the ±1.5pp bench
      noise floor and reproduced across two independent runs after
      clearing the Triton kernel cache); the falsification gate from
      the prior taxonomy synthesis demands rollback. Symmetric-tile
      cohort rows therefore retain the existing ``kpack=1`` setting.

    * ``kpack=1`` for fp8 / fp4 / fp32 paths (the fp8 path is
      gfx950-clamped at codegen time and the fp4 path uses its own
      kernel entry point).

    Set ``TRITONBLAS_FORCE_KPACK1=1`` to clamp the helper to ``kpack=1``
    for every dtype — used by the paired baseline-vs-fix bench harness
    to isolate the kpack=2 contribution from other selector changes
    without checking out an older revision. Production callers leave
    the env var unset.

    Args:
        a_dtype: A-operand dtype (the fp16 / bf16 selection key).
        block_m, block_n: Selected tile dimensions. When ``None`` (e.g.
            an external caller has not yet plumbed them through), the
            helper falls back to ``kpack=1`` to stay safely on the
            existing setting.

    Returns:
        ``2`` for the gated fp16 / bf16 + asymmetric-tile case;
        ``1`` otherwise.
    """
    if os.environ.get("TRITONBLAS_FORCE_KPACK1", "0") == "1":
        return 1
    if a_dtype is not torch.float16 and a_dtype is not torch.bfloat16:
        return 1
    # Asymmetric-tile gate: only ship kpack=2 where the paired bench
    # showed a clean win. Symmetric tiles regress on large M=N=K and
    # are kept on the prior kpack=1 setting.
    if block_m is None or block_n is None:
        return 1
    return 2 if block_m != block_n else 1
