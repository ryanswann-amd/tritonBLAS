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

Two well-established mitigations live below the user-facing kernel.  Both
are exposed as constexprs / kernel-launch arguments so users (and the
selector) can A/B them:

1. **Operand swizzle** — bumping ``kpack`` from 1 → 2 packs two contiguous
   K elements into one MFMA load.  This halves the number of distinct
   shared-memory transactions per dot tile, which is functionally
   equivalent to a column padding step (the swizzled-shared encoding
   shifts banks every few rows so the doubled vector width stops
   aliasing).

2. **MFMA layout swizzle** — switching ``matrix_instr_nonkdim`` from 16
   → 32 (or vice versa) changes the MFMA fragment's K stride.  On long-K
   tiles this reorganises the consumer-side lane-bank mapping of the
   shared loads, breaking the conflict pattern produced by the dot's
   default layout.

Both knobs are honoured per call:

* explicit kwargs to :func:`tritonblas.matmul.persistent_matmul_lt`
  override everything;
* failing that, the env vars below are read once per process;
* failing that, the historical defaults (``kpack=1``,
  ``matrix_instr_nonkdim=16``, ``waves_per_eu=0``, ``num_warps=8``)
  are used so existing benchmarks are not perturbed.

Env vars
--------
``TRITONBLAS_KPACK``                — 1 or 2 (default 1).
``TRITONBLAS_MATRIX_INSTR_NONKDIM`` — 16 or 32 (default 16).
``TRITONBLAS_WAVES_PER_EU``         — int >= 0 (default 0; 0 = compiler choice).
``TRITONBLAS_NUM_WARPS``            — int (default 8).
``TRITONBLAS_LDS_SWIZZLE``          — set to ``1`` to apply the recommended
                                      mitigation bundle (kpack=2, nonkdim=16)
                                      when no explicit override is provided.

The knobs are intentionally orthogonal to the algorithmic kernel code:
they only flow into the Triton compiler's MFMA / async-copy layout
selection, so correctness is unaffected.
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


def _env_bool(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


@dataclass(frozen=True)
class LdsSwizzleConfig:
    """Resolved LDS bank-conflict mitigation parameters."""

    kpack: int
    matrix_instr_nonkdim: int
    waves_per_eu: int
    num_warps: int

    def as_kwargs(self) -> dict:
        return {
            "kpack": self.kpack,
            "matrix_instr_nonkdim": self.matrix_instr_nonkdim,
            "waves_per_eu": self.waves_per_eu,
            "num_warps": self.num_warps,
        }


# Historical defaults (pre-K-3022).  Kept as a constant so callers can
# detect "user did not override" by passing None.
_BASELINE = LdsSwizzleConfig(
    kpack=1,
    matrix_instr_nonkdim=16,
    waves_per_eu=0,
    num_warps=8,
)

# Mitigation bundle proven on the long-K small-square cohort
# (512x512x16384, 1024x1024x16384) on MI300X. Measured with rocprofv3
# SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS:
#   1024x1024x16384  baseline (kpack=1, nonkdim=16, warps=8 ): 1.7778
#   1024x1024x16384  mitig    (kpack=2, nonkdim=32, warps=16): 0.3810  (-78%, <0.5 target)
#    512x 512x16384  baseline                                : 1.4545
#    512x 512x16384  mitig    (kpack=2, nonkdim=32, warps=16): 0.6486  (-55%)
# kpack=2 packs two K elements per LDS load (operand swizzle that rotates
# bank addressing), nonkdim=32 doubles the MFMA K stride, and num_warps=16
# halves the per-wave LDS row stride so it no longer aliases the 32-bank
# cycle.  Together they reach the conflict-ratio target on the cohort.
_MITIGATION = LdsSwizzleConfig(
    kpack=2,
    matrix_instr_nonkdim=32,
    waves_per_eu=0,
    num_warps=16,
)


def resolve_swizzle(
    *,
    kpack: Optional[int] = None,
    matrix_instr_nonkdim: Optional[int] = None,
    waves_per_eu: Optional[int] = None,
    num_warps: Optional[int] = None,
) -> LdsSwizzleConfig:
    """
    Resolve the LDS swizzle / bank-conflict mitigation knobs.

    Order of precedence:

    1. Explicit keyword arguments (anything not ``None``).
    2. Per-knob env vars (``TRITONBLAS_KPACK`` etc.).
    3. ``TRITONBLAS_LDS_SWIZZLE=1`` opts in to the mitigation bundle.
    4. Historical baseline.
    """

    # Start from baseline or mitigation depending on global opt-in.
    base = _MITIGATION if _env_bool("TRITONBLAS_LDS_SWIZZLE") else _BASELINE

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

    if resolved_kpack not in (1, 2):
        raise ValueError(f"kpack must be 1 or 2, got {resolved_kpack}")
    if resolved_nonkdim not in (16, 32):
        raise ValueError(
            f"matrix_instr_nonkdim must be 16 or 32, got {resolved_nonkdim}"
        )
    if resolved_wpu < 0:
        raise ValueError(f"waves_per_eu must be >= 0, got {resolved_wpu}")
    if resolved_nwarps not in (1, 2, 4, 8, 16):
        raise ValueError(f"num_warps must be a power of two in [1,16], got {resolved_nwarps}")

    return LdsSwizzleConfig(
        kpack=resolved_kpack,
        matrix_instr_nonkdim=resolved_nonkdim,
        waves_per_eu=resolved_wpu,
        num_warps=resolved_nwarps,
    )


__all__ = [
    "LdsSwizzleConfig",
    "resolve_swizzle",
]
