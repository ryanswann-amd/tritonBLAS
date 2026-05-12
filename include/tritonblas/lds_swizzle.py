# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
LDS bank-conflict mitigation knobs for tritonblas GEMM kernels.

Background
----------
On gfx94x/gfx95x (MI300/MI350) the LDS (shared memory) is organised in 32 banks
of 4 bytes each.  When 64 lanes of a wave issue a single ``ds_read`` and the
addresses collide on the same bank, the request is serialised — every
collision adds a cycle and inflates ``SQ_LDS_BANK_CONFLICT``.

For long-K, small-square GEMMs (e.g. ``1024 x 1024 x 16384``) the AMD Triton
backend's default shared layout for the A operand has a stride such that
``BLOCK_K * sizeof(elem)`` is a multiple of 128 bytes.  Adjacent rows of
A then alias the same bank set, producing the 1.78 cyc/inst ratio the
performance lessons flag.

Two complementary mitigations are exposed below the user-facing kernel
and can be A/B'd via env vars or kwargs:

1. **Operand swizzle** — bumping ``kpack`` from 1 → 2 packs two contiguous
   K elements into one MFMA load.  This halves the number of distinct
   shared-memory transactions per dot tile and reorders the LDS lane→bank
   mapping so adjacent K slices land on different banks (functionally a
   "1-element padding column" expressed as an XOR swizzle).

2. **MFMA layout swizzle** — switching ``matrix_instr_nonkdim`` from 16
   → 32 changes the MFMA fragment's K stride.  On long-K tiles this
   reorganises the consumer-side lane/bank mapping of the shared loads
   and breaks the conflict pattern produced by the dot's default layout.

3. **K-tile right-sizing** — reducing ``BLOCK_K`` from Origami's
   long-K default (256/512) to 64/32 collapses the LDS row stride to the
   natural cache-line size and the bank-conflict cycle disappears
   entirely.  This is the single biggest knob: it drives the
   SQ_LDS_BANK_CONFLICT/SQ_INSTS_LDS ratio from ~1.78 to 0 on the
   problematic cohort.  Trade-off: 4–8× more K-loop trips, so it is only
   applied for the small-square long-K cohort the rocprof study flagged.

Resolution order (per persistent_matmul_lt call):

* explicit kwargs override everything;
* per-knob env vars (``TRITONBLAS_*``) come next;
* ``TRITONBLAS_LDS_AUTO=1`` (default ON) auto-applies the cohort
  mitigation bundle when ``M,N <= 1024`` and ``K >= 8192``;
* historical defaults otherwise.

Set ``TRITONBLAS_LDS_AUTO=0`` to disable the auto-mitigation entirely
(returns the kernel to its pre-K-3022 behaviour for benchmarks).

The knobs are intentionally orthogonal to the algorithmic kernel code:
they only flow into the Triton compiler's MFMA / async-copy layout
selection, so correctness is unaffected.

Env vars
--------
``TRITONBLAS_KPACK``                — 1 or 2 (default depends on cohort).
``TRITONBLAS_MATRIX_INSTR_NONKDIM`` — 16 or 32 (default depends on cohort).
``TRITONBLAS_WAVES_PER_EU``         — int >= 0 (default 0; 0 = compiler choice).
``TRITONBLAS_NUM_WARPS``            — int (default 8).
``TRITONBLAS_BLOCK_K``              — int, override BLOCK_K.  -1 / unset =
                                      keep selector default unless cohort
                                      override fires.
``TRITONBLAS_LDS_AUTO``             — 1 (default) auto-mitigates the
                                      long-K small-square cohort; 0 disables.
``TRITONBLAS_LDS_SWIZZLE``          — set to ``1`` to apply the recommended
                                      mitigation bundle (kpack=2,
                                      nonkdim=32) regardless of cohort.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


@dataclass(frozen=True)
class LdsSwizzleConfig:
    """Resolved LDS bank-conflict mitigation parameters."""

    kpack: int
    matrix_instr_nonkdim: int
    waves_per_eu: int
    num_warps: int
    # block_k_override is None when the selector's BLOCK_K should be kept,
    # otherwise it is the value to use.
    block_k_override: Optional[int] = None

    def as_kwargs(self) -> dict:
        return {
            "kpack": self.kpack,
            "matrix_instr_nonkdim": self.matrix_instr_nonkdim,
            "waves_per_eu": self.waves_per_eu,
            "num_warps": self.num_warps,
            "block_k_override": self.block_k_override,
        }


# Historical defaults (pre-K-3022).  Kept as a constant so callers can
# detect "user did not override" by passing None.
_BASELINE = LdsSwizzleConfig(
    kpack=1,
    matrix_instr_nonkdim=16,
    waves_per_eu=0,
    num_warps=8,
    block_k_override=None,
)


def _is_long_k_small_square(M: Optional[int], N: Optional[int], K: Optional[int]) -> bool:
    """Cohort the rocprof study flagged as conflict-heavy.

    Square-ish output (M,N <= 1024) with a long K reduction (>= 8192) so the
    Origami selector picks a small BLOCK_M/N with a large BLOCK_K, producing
    an LDS row stride that aliases the 32-bank cycle.
    """
    if M is None or N is None or K is None:
        return False
    return M <= 1024 and N <= 1024 and K >= 8192


def _cohort_mitigation(M: Optional[int], N: Optional[int], K: Optional[int],
                       BLK_M: Optional[int], BLK_N: Optional[int]) -> LdsSwizzleConfig:
    """Mitigation bundle picked from the rocprof sweep on MI300X.

    For the long-K small-square cohort, the BLK_K override + mfma=32 + kpack=2
    drives SQ_LDS_BANK_CONFLICT to 0 in the rocprof measurement.
    """
    # Pick the smallest BLOCK_K that breaks the bank-conflict cycle while
    # preserving alignment.  64 is chosen because it fits the two LDS-bank
    # cycles per row (32 banks × 4 B = 128 B = 64 fp16 elements).
    blk_k = 64
    # mfma_nonkdim must divide BLK_M and BLK_N; fall back to 16 if not.
    mfma = 32
    if BLK_M is not None and BLK_N is not None and (BLK_M % 32 != 0 or BLK_N % 32 != 0):
        mfma = 16
    return LdsSwizzleConfig(
        kpack=2,
        matrix_instr_nonkdim=mfma,
        waves_per_eu=0,
        num_warps=8,
        block_k_override=blk_k,
    )


# Mitigation bundle for TRITONBLAS_LDS_SWIZZLE=1 (no shape needed).
_MITIGATION_GENERIC = LdsSwizzleConfig(
    kpack=2,
    matrix_instr_nonkdim=32,
    waves_per_eu=0,
    num_warps=8,
    block_k_override=64,
)


def resolve_swizzle(
    *,
    kpack: Optional[int] = None,
    matrix_instr_nonkdim: Optional[int] = None,
    waves_per_eu: Optional[int] = None,
    num_warps: Optional[int] = None,
    block_k: Optional[int] = None,
    M: Optional[int] = None,
    N: Optional[int] = None,
    K: Optional[int] = None,
    BLK_M: Optional[int] = None,
    BLK_N: Optional[int] = None,
) -> LdsSwizzleConfig:
    """
    Resolve the LDS swizzle / bank-conflict mitigation knobs.

    Order of precedence:

    1. Explicit keyword arguments (anything not ``None``).
    2. Per-knob env vars (``TRITONBLAS_KPACK`` etc.).
    3. ``TRITONBLAS_LDS_AUTO=1`` (default) → cohort mitigation when shape
       matches the long-K small-square cohort.
    4. ``TRITONBLAS_LDS_SWIZZLE=1`` → generic mitigation bundle.
    5. Historical baseline.
    """

    # Pick the base config: explicit opt-in > cohort detection > baseline.
    auto = _env_bool("TRITONBLAS_LDS_AUTO", default=True)
    swizzle_opt_in = _env_bool("TRITONBLAS_LDS_SWIZZLE")

    if swizzle_opt_in:
        base = _MITIGATION_GENERIC
    elif auto and _is_long_k_small_square(M, N, K):
        base = _cohort_mitigation(M, N, K, BLK_M, BLK_N)
    else:
        base = _BASELINE

    resolved_kpack = (
        kpack
        if kpack is not None
        else _env_int("TRITONBLAS_KPACK", base.kpack)
    )
    resolved_nonkdim = (
        matrix_instr_nonkdim
        if matrix_instr_nonkdim is not None
        else _env_int("TRITONBLAS_MATRIX_INSTR_NONKDIM", base.matrix_instr_nonkdim)
    )
    resolved_wpu = (
        waves_per_eu
        if waves_per_eu is not None
        else _env_int("TRITONBLAS_WAVES_PER_EU", base.waves_per_eu)
    )
    resolved_nwarps = (
        num_warps
        if num_warps is not None
        else _env_int("TRITONBLAS_NUM_WARPS", base.num_warps)
    )

    if block_k is not None:
        resolved_block_k: Optional[int] = block_k
    else:
        env_block_k = _env_int("TRITONBLAS_BLOCK_K", -1)
        if env_block_k > 0:
            resolved_block_k = env_block_k
        else:
            resolved_block_k = base.block_k_override

    if resolved_kpack not in (1, 2):
        raise ValueError(f"kpack must be 1 or 2, got {resolved_kpack}")
    if resolved_nonkdim not in (16, 32):
        raise ValueError(
            f"matrix_instr_nonkdim must be 16 or 32, got {resolved_nonkdim}"
        )
    if resolved_wpu < 0:
        raise ValueError(f"waves_per_eu must be >= 0, got {resolved_wpu}")
    if resolved_nwarps not in (1, 2, 4, 8, 16):
        raise ValueError(
            f"num_warps must be a power of two in [1,16], got {resolved_nwarps}"
        )
    if resolved_block_k is not None and resolved_block_k <= 0:
        raise ValueError(
            f"block_k_override must be a positive int, got {resolved_block_k}"
        )

    return LdsSwizzleConfig(
        kpack=resolved_kpack,
        matrix_instr_nonkdim=resolved_nonkdim,
        waves_per_eu=resolved_wpu,
        num_warps=resolved_nwarps,
        block_k_override=resolved_block_k,
    )


__all__ = [
    "LdsSwizzleConfig",
    "resolve_swizzle",
]
