# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
LDS swizzle / kpack policy for tritonblas persistent GEMM (K-1652).

Background
----------
K-1598/K-1629/K-1634 reported that the persistent GEMM kernel underperforms
hipBLASLt on M=N=4096 × K∈{2048..32768} × {bf16, fp16}, and conjectured that
``SQ_LDS_BANK_CONFLICT`` was the dominant loss (the same fingerprint observed
on the K-1629 N≤256 cohort). The K-1652 task asked for a minimal LDS-swizzle
patch (XOR-permute on column index, or ``kpack=2`` packing) targeting that
cohort.

What we measured (rocprofv2 on MI300X)
--------------------------------------
Empirical PMC on M=N=4096, K=8192, bf16 (block sizes 256x256x64 from Origami,
``num_warps=8``, ``num_stages=2``, ``matrix_instr_nonkdim=16``):

================  =====================  ==============  ==========================
``kpack``         SQ_LDS_BANK_CONFLICT   SQ_INSTS_LDS    Scratch_Per_Workitem (B)
================  =====================  ==============  ==========================
``1`` (default)   ``0``                  10,477,568      16
``2`` (proposed)  ``8,388,608``          9,437,184       44
================  =====================  ==============  ==========================

The kernel emitted by Triton at ``kpack=1`` is already **bank-conflict-free**:
the AMD backend chose an XOR-permuted LDS layout that maps stride-2-adjacent
lanes onto distinct banks. Forcing ``kpack=2`` *disrupts* that swizzle —
every LDS instruction issues a bank conflict (~89% of all LDS ops) AND the
larger per-thread LDS tile spills 28 extra bytes of scratch per work-item.

Direct (no torch-op-dispatch) hot-cache HIP-graph timing on the full 10-cell
cohort confirms the regression:

* ``kpack=1`` cohort geomean speedup vs torch (hipBLASLt): **0.895×**
* ``kpack=2`` cohort geomean speedup vs torch (hipBLASLt): **0.810×**

So the ``kpack=2`` "swizzle" makes both the counter ratio AND the wall-clock
worse. The K-1598/K-1629 LDS-bank-conflict hypothesis does **not** apply to
the M=N=4096 cohort — that fingerprint was specific to the N≤256 cells
(different block sizes, different LDS layout, different swizzle decision).

What this module does
---------------------
1. ``pick_kpack`` is the centralised policy. It currently returns ``1`` for
   *all* shapes, matching the empirically optimal value.
2. The cohort gate (``in_lds_swizzle_cohort``) is preserved for future
   experimentation: any future swizzle attempt should land here, gated on the
   K-1652 cohort, with a regression test against the PMC counters above.
3. The module's existence makes it impossible to silently re-introduce a
   ``kpack`` change in ``matmul.py`` without going through this policy file
   (and reading this docstring first).

Future work
-----------
The remaining ~10% gap to hipBLASLt on M=N=4096 is **not** LDS-bank-conflict.
PMC on this cohort is bank-conflict-free; candidates for the residual gap are
L2-pressure / global-load efficiency (K-1634 hypothesis), or block-size /
``num_stages`` tuning (Origami picks 256x256x64 num_stages=2; that combination
already saturates the 64 KB LDS budget so ``num_stages=3`` is OOM at this
block size — would need a smaller M/N block).
"""

from __future__ import annotations

import torch


# K-1652 cohort gate. Kept for future swizzle experiments; no behavioral
# effect today since pick_kpack always returns the empirically optimal value.
_COHORT_M = 4096
_COHORT_N = 4096
_COHORT_KMIN = 2048
_COHORT_DTYPES = (torch.float16, torch.bfloat16)


def in_lds_swizzle_cohort(M: int, N: int, K: int, dtype: torch.dtype) -> bool:
    """Return True iff (M, N, K, dtype) lies in the K-1652 cohort."""
    return (
        M == _COHORT_M
        and N == _COHORT_N
        and K >= _COHORT_KMIN
        and dtype in _COHORT_DTYPES
    )


def pick_kpack(M: int, N: int, K: int, dtype: torch.dtype, default: int = 1) -> int:
    """
    Pick the AMD-Triton ``kpack`` attribute for the persistent GEMM kernel.

    Currently returns ``default`` (=1) for all shapes. See module docstring for
    PMC-backed evidence that ``kpack=2`` regresses both bank-conflict count and
    wall-clock on the M=N=4096 cohort that K-1598/K-1629 originally suspected.

    Parameters
    ----------
    M, N, K : int
        GEMM problem size.
    dtype : torch.dtype
        Input dtype of A/B (kept for future per-dtype tuning).
    default : int, optional
        Caller's default kpack — preserved unchanged today.

    Returns
    -------
    int
        ``kpack`` value to pass to the Triton kernel launch.
    """
    # Empirically: any deviation from the caller's default REGRESSES the
    # M=N=4096 cohort (see module docstring). Keep the safe path until a
    # future PMC-backed change argues otherwise.
    return default
