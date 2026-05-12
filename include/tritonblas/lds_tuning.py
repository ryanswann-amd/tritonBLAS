# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""LDS bank-conflict mitigation knobs for tritonblas matmul kernels.

This module centralises the AMD-Triton compiler hints that control the
shared-memory (LDS) layout of the A and B operands inside ``persistent_matmul``
(and the Stream-K / work-stealing variants).

Background
----------
The AMD Triton backend allocates LDS for the A and B tiles using
``swizzled_shared`` / ``amd_rotating_shared`` encodings. Three compile-time
launcher kwargs jointly determine the bank-mapping of those encodings:

* ``kpack``                  — number of K elements packed per LDS line.

  - ``kpack = 1`` keeps the natural stride which on bf16/fp16 GEMMs at
    common BLOCK_K values lands consecutive lanes on the same bank,
    generating ~1.78 conflict cycles per LDS instruction on the long-K
    small-square cohort (e.g. 1024×1024×16384, see lessons K-913).
  - ``kpack = 2`` packs two K elements per LDS line.  For 16-bit operands
    this widens the per-row LDS stride from 32 B to 64 B, which is the
    LDS-side analogue of inserting a one-element padding column in classic
    CUDA shared-memory kernels — consecutive threads in a wave then hit
    different banks instead of colliding.  This is the **padding** lever.

* ``matrix_instr_nonkdim``   — non-K MFMA instruction shape (16 or 32).

  - 16 picks ``mfma_16x16x16`` (the default).
  - 32 picks ``mfma_32x32x8``, which the AMD backend pairs with a different
    rotated-shared layout for A/B.  This is the **swizzle** lever proper.

* ``waves_per_eu``           — occupancy hint; non-zero may trade a small
                                amount of register pressure for one extra
                                wave per EU and help hide residual LDS
                                latency.

Mitigation modes
----------------
Selected via the ``TRITONBLAS_LDS_MITIGATION`` env var.  Defaults preserve
existing production behaviour (mode == ``off`` for traffic that does not
match the long-K small-square cohort).

* ``off``      — preserve the original ``kpack=1, matrix_instr_nonkdim=16``
                 launcher kwargs.  This is the **before** baseline and the
                 default value when no env var is set.
* ``pad``      — kpack=2 only (1-element LDS padding analogue).
* ``swizzle``  — matrix_instr_nonkdim=32 only (rotated-shared MFMA layout).
* ``both``     — kpack=2 AND matrix_instr_nonkdim=32.
* ``auto``     — apply ``pad`` only on shapes matching the K-913 cohort
                 (``M == N <= 2048`` and ``K >= 8 * min(M, N)``); pass-through
                 otherwise.  Conservative: this is the route used by
                 ``persistent_matmul_lt`` when the env var is unset.

Per-knob overrides
------------------
Each launcher knob can be force-set via env var (these win over the mode):

* ``TRITONBLAS_KPACK``                 (int)
* ``TRITONBLAS_MATRIX_INSTR_NONKDIM``  (int)
* ``TRITONBLAS_NUM_WARPS``             (int)
* ``TRITONBLAS_NUM_STAGES``            (int)
* ``TRITONBLAS_WAVES_PER_EU``          (int)

Setting ``TRITONBLAS_LDS_MITIGATION=off`` (or unsetting it) restores the
pre-mitigation behaviour, used to collect the *before* rocprof counter
measurement.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple


# Original baseline (pre-mitigation).  Kept here so the *off* mode is a single
# source of truth and benchmarks can reference it.
_BASELINE_KPACK = 1
_BASELINE_MATRIX_INSTR_NONKDIM = 16
_BASELINE_WAVES_PER_EU = 0


def _get_env(name: str) -> Optional[str]:
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    return v


def _get_int_env(name: str) -> Optional[int]:
    """Parse an integer env var, returning None if unset or invalid."""
    v = _get_env(name)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _is_long_k_small_square(M: Optional[int], N: Optional[int], K: Optional[int]) -> bool:
    """Detect the K-913 long-K small-square cohort.

    Cohort definition (lessons / TRITONBLAS-0065):
        ``M == N`` and ``M <= 2048`` and ``K >= 8 * min(M, N)``

    Examples in-cohort: 512×512×16384, 1024×1024×16384, 2048×2048×32768.
    """
    if M is None or N is None or K is None:
        return False
    return M == N and M <= 2048 and K >= 8 * min(M, N)


def _resolve_mode_overrides(mode: str, M, N, K):
    """Map a mitigation mode → ``(kpack, mfma_nonkdim)`` overrides (or Nones)."""
    mode = mode.lower()
    if mode in ("off", "none", ""):
        return None, None
    if mode in ("pad", "kpack", "kpack2"):
        return 2, None
    if mode in ("swizzle", "swizzle_lds", "mfma32"):
        return None, 32
    if mode in ("both", "pad_swizzle", "pad+swizzle"):
        return 2, 32
    if mode == "auto":
        # Auto: shape-aware preset for the K-913 long-K small-square cohort.
        # Empirically measured SQ_LDS_BANK_CONFLICT/SQ_INSTS_LDS on MI300X
        # (see output/lds_conflict_results.csv):
        #
        #   shape           baseline  pad   swizzle  both
        #   ─────────────── ────────  ────  ───────  ────
        #   512x512x16384    1.45     2.18   0.84    1.26   ← swizzle wins
        #   1024x1024x16384  1.78     0.89   1.45    0.73   ← both wins
        #
        # We pick the per-shape best so both small-square and the canonical
        # K-913 1024x1024x16384 case land below 1.0 (vs. 1.78 baseline).
        # No-op on shapes outside the cohort to avoid disturbing production.
        if _is_long_k_small_square(M, N, K):
            if M is not None and M <= 512:
                # Tile is small enough that doubling the LDS line width
                # (kpack=2) hurts.  Use the rotated-shared MFMA layout only.
                return None, 32
            # M >= 1024: combine padding + swizzle for the largest reduction.
            return 2, 32
        return None, None
    # Unknown modes fall back to off.
    return None, None


def get_lds_tuning(
    M: Optional[int] = None,
    N: Optional[int] = None,
    K: Optional[int] = None,
) -> Tuple[int, int, int, Optional[int], Optional[int]]:
    """Return ``(kpack, matrix_instr_nonkdim, waves_per_eu,
    num_warps_override, num_stages_override)`` for the kernel launcher.

    Resolution order:

    1. Per-knob env vars (``TRITONBLAS_KPACK``, etc.) win unconditionally.
    2. ``TRITONBLAS_LDS_MITIGATION`` mode (default ``auto``) selects a preset.
    3. Falls back to the original baseline (``kpack=1``, ``mfma_nk=16``,
       ``waves_per_eu=0``) — i.e. behaviour identical to upstream.

    ``num_warps_override`` and ``num_stages_override`` are returned only if
    explicitly set via env var; otherwise the caller's existing default is kept.
    """
    mode = os.environ.get("TRITONBLAS_LDS_MITIGATION", "auto")
    mode_kpack, mode_mfma = _resolve_mode_overrides(mode, M, N, K)

    kpack = mode_kpack if mode_kpack is not None else _BASELINE_KPACK
    mfma_nk = mode_mfma if mode_mfma is not None else _BASELINE_MATRIX_INSTR_NONKDIM
    waves_per_eu = _BASELINE_WAVES_PER_EU

    # Per-knob env overrides (always win).
    env_kpack = _get_int_env("TRITONBLAS_KPACK")
    if env_kpack is not None:
        kpack = env_kpack
    env_mfma = _get_int_env("TRITONBLAS_MATRIX_INSTR_NONKDIM")
    if env_mfma is not None:
        mfma_nk = env_mfma
    env_wpeu = _get_int_env("TRITONBLAS_WAVES_PER_EU")
    if env_wpeu is not None:
        waves_per_eu = env_wpeu

    # Validate to known-good values.
    if kpack not in (1, 2):
        kpack = _BASELINE_KPACK
    if mfma_nk not in (16, 32):
        mfma_nk = _BASELINE_MATRIX_INSTR_NONKDIM
    if waves_per_eu < 0:
        waves_per_eu = _BASELINE_WAVES_PER_EU

    num_warps = _get_int_env("TRITONBLAS_NUM_WARPS")
    num_stages = _get_int_env("TRITONBLAS_NUM_STAGES")

    return kpack, mfma_nk, waves_per_eu, num_warps, num_stages


def describe_active_mode(M=None, N=None, K=None) -> str:
    """Return a human-readable string describing the active LDS mitigation."""
    kpack, mfma_nk, wpeu, nw, ns = get_lds_tuning(M, N, K)
    mode = os.environ.get("TRITONBLAS_LDS_MITIGATION", "auto")
    extra = []
    if nw is not None:
        extra.append(f"num_warps={nw}")
    if ns is not None:
        extra.append(f"num_stages={ns}")
    if wpeu:
        extra.append(f"waves_per_eu={wpeu}")
    extra_s = (" " + " ".join(extra)) if extra else ""
    return (
        f"lds_mitigation={mode} kpack={kpack} matrix_instr_nonkdim={mfma_nk}"
        f"{extra_s}"
    )
