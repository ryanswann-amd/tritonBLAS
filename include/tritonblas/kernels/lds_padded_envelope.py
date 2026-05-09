# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Skinny-N LDS-padded staging envelope predicate.

Background
----------
K-1629's 3-pass rocprofv2 PMC capture on the 6 worst N in {128, 256}
K-COMPLEMENT cells showed the persistent_matmul kernel pays a measurable
``SQ_LDS_BANK_CONFLICT`` / ``SQ_INSTS_LDS`` overhead on these skinny shapes
(consistent with the K-913 long-K small-square finding of 1.78 cyc/inst LDS
and ~41% LDS_WAIT absolute fraction). hipBLASLt does not pay this cost on
the same shapes.

Mechanism
---------
For BN in {128, 256} with the default 16x16x16 MFMA tile and num_warps=8,
each ds_read_b64 issued by the A-tile loader maps consecutive lanes of a
warp to the SAME LDS bank row (stride = BN * sizeof(elem) is a multiple of
the 32-bank stride of an LDS read). The collision pattern is::

    bank(lane) = (lane * stride_elem_bytes / 4) mod 32
              = (lane * BN_bytes / 4) mod 32

For BN=128, fp16: stride = 256 bytes, bank period = (256/4) mod 32 = 0
                  -> all 64 lanes serialise on bank 0
For BN=256, fp16: stride = 512 bytes, bank period = (512/4) mod 32 = 0
                  -> all 64 lanes serialise on bank 0

Mitigation (this envelope's "padded" path):
  * Switch to the 32x32x8 MFMA tile (matrix_instr_nonkdim=32). This
    re-laninates the LDS access so each ds_read_b64 spreads its 64 lanes
    across 4 distinct LDS bank rows, removing the modular collision.
  * Halve num_warps (8 -> 4). Skinny-N has no parallelism to amortise the
    8-warp LDS contention; halving warps directly halves the per-bank
    queue depth. (Also reduces the VGPR pressure that 32x32 MFMA introduces
    so per-CU occupancy is preserved at >= 2 waves.)

These two knobs together produce the same effect as the CUDA "+1 element
stride padding" trick, but in the AMD MFMA codegen vocabulary - they break
the modular bank-collision pattern without changing the K-loop count
(unlike kpack=2, which K-612/K-646/K-683 showed regresses long-K shapes
through latency-hiding collapse - see ``project_context/tritonblas/lessons.md``).

Envelope (load-bearing safety, narrowed by K-1640 paired n=30 evidence)
-----------------------------------------------------------------------
The K-1640 paired n=30 HIP-graph hot-cache sweep on the 6 K-1629 worst
cells found that the bank-conflict mitigation is **only** statistically
beneficial on the square-skinny corner (M == K, N == 128). On the other
five admit candidates the change is statistical noise or a small
regression (paired-t in [-4.13, +1.70]). We therefore narrow the
admit envelope from the original "N in {128,256}, K in [1024,16384]"
shape rectangle to the single proven-winning cell pattern:

  * N == 128                                  # the only width that won
  * M == K                                    # the only square-skinny
                                              # M:K ratio with paired-t > 2
  * K in [2048, 16384]                        # extrapolation cap; we have
                                              # ground-truth only at K=4096,
                                              # so we admit one octave each
                                              # side and cap at the
                                              # K-1633 catalogue ceiling
  * BLOCK_M in {128, 256}, BLOCK_N == 128
  * BLOCK_K in {32, 64}

Anything else falls through to the baseline persistent_matmul launch path
unchanged. In particular:

  * N == 256: refused (K-1640 sweep showed paired-t in [-4.13, -0.50] on
    all three measured cells — not the targeted bank-conflict regime)
  * M != K (skinny-rect rather than square-skinny): refused (K-1640 sweep
    showed (M=4096,K=2048) at -1.93% paired-t=-2.63 and (M=4096,K=8192)
    at noise; only the square-skinny corner won)
  * N >= 512: baseline (the 17 P26 production frozensets live here -
    this gate must not perturb their selection)
  * Quantized GEMM: baseline (32x32 MFMA is not certified for the int8
    epilogue path used by matmul_a8w8_lt; FP4 has its own kernel)
  * Bias path: baseline (the bias epilogue has not been re-verified
    against the 32x32 MFMA layout for skinny-N; conservative)

Rationale references
--------------------
  * K-1629  : PMC capture identifying SQ_LDS_BANK_CONFLICT overhead
  * K-1617  : N-bucket envelope verification
  * K-1623  : P26 routing oracle (productionised frozensets)
  * K-1633  : K-COMPLEMENT cell catalogue
  * K-913   : long-K small-square LDS_WAIT root cause
  * K-612 / K-646 / K-683 : kpack=2 anti-pattern - WHY this gate uses
    matrix_instr_nonkdim=32 + num_warps=4 instead of kpack=2.
"""

from typing import Tuple

# K-COMPLEMENT envelope bounds, NARROWED by K-1640 iter-2 paired n=30
# evidence to the single (M==K, N==128) square-skinny corner that showed a
# statistically-significant win (paired-t = +3.87, +3.04% median speedup
# on cell (4096,128,4096); see workspace gist). The previous, wider
# envelope (N in {128,256}, K in [1024,16384]) admitted 5 cells where the
# change was noise or net-negative — those are now refused.
#
# Do not widen without:
#   (a) re-running the 17-frozenset P26 regression sweep, AND
#   (b) showing paired-t > 2 on the new cells AND no per-cell paired-t < -2.
_LDS_PADDED_N_VALUES = (128,)
_LDS_PADDED_K_LO = 2048
_LDS_PADDED_K_HI = 16384
_LDS_PADDED_BLOCK_M_VALUES = (128, 256)
_LDS_PADDED_BLOCK_N_VALUES = (128,)
_LDS_PADDED_BLOCK_K_VALUES = (32, 64)
# Square-skinny ratio gate. M and K may differ by at most this factor
# (i.e. 0.5 <= M/K <= 2). The (4096,128,4096) winning cell has M/K=1;
# the regressing (4096,128,2048) cell has M/K=2 and the noise
# (4096,128,8192) cell has M/K=0.5 — both at the edge. We therefore
# admit only EXACTLY M==K to stay strictly inside the proven point.
_LDS_PADDED_REQUIRE_M_EQ_K = True

# The codegen knobs the variant path will swap in. These constants are
# intentionally exported so the gist & tests can assert on them and so
# matmul.py only has to know "what the gate decided", not "what knobs to
# flip".
LDS_PADDED_MFMA_NONKDIM = 32   # 32x32x8 MFMA tile (vs default 16x16x16)
LDS_PADDED_NUM_WARPS = 4       # halve warps (vs default 8)
LDS_PADDED_KPACK = 1           # leave kpack alone - K-646 anti-pattern


def should_use_lds_padded_path(
    M: int,
    N: int,
    K: int,
    block_m: int,
    block_n: int,
    block_k: int,
    *,
    quantized: bool = False,
    bias: bool = False,
) -> bool:
    """
    Return True iff the (shape, tile) tuple sits inside the K-1617/K-1633
    skinny-N K-COMPLEMENT envelope and is therefore eligible for the
    LDS-padded staging path.

    This is the single load-bearing gate. The caller MUST route any
    non-eligible shape to the unchanged baseline persistent_matmul launch
    path - see ``persistent_matmul_lt`` in ``matmul.py``.

    Parameters
    ----------
    M, N, K : int
        GEMM problem dimensions.
    block_m, block_n, block_k : int
        Tile dimensions selected by the Origami P26 routing oracle.
    quantized : bool
        True for int8 / fp8 quantized A8W8 GEMMs - those use a different
        epilogue path that has not been re-verified against the 32x32
        MFMA layout, so they are excluded.
    bias : bool
        True if a bias epilogue is requested - excluded for the same
        certification reason.

    Returns
    -------
    bool
        True  -> launch with ``matrix_instr_nonkdim=LDS_PADDED_MFMA_NONKDIM``
                 and ``num_warps=LDS_PADDED_NUM_WARPS``.
        False -> launch with the unchanged baseline parameters.
    """
    # Cheap rejects first - these are the structural exclusions from the
    # P26 production frozensets.
    if quantized:
        return False
    if bias:
        return False

    # Envelope bounds. Order: cheapest predicate first.
    if N not in _LDS_PADDED_N_VALUES:
        return False
    if not (_LDS_PADDED_K_LO <= K <= _LDS_PADDED_K_HI):
        return False
    if block_n not in _LDS_PADDED_BLOCK_N_VALUES:
        return False
    if block_m not in _LDS_PADDED_BLOCK_M_VALUES:
        return False
    if block_k not in _LDS_PADDED_BLOCK_K_VALUES:
        return False

    # Sanity: block_m must not exceed M (otherwise the P26 oracle would
    # not have selected this tile and we'd be in a degenerate corner).
    if block_m > M:
        return False

    # K-1640 iter-2 narrowing: only the M==K square-skinny corner showed
    # a statistically-significant win in the paired n=30 sweep. The
    # off-diagonal M:K ratios (1:2, 2:1) either regressed or showed
    # noise. Refusing those keeps the gate strictly inside the proven
    # point.
    if _LDS_PADDED_REQUIRE_M_EQ_K and M != K:
        return False

    return True


def lds_padded_launch_overrides() -> Tuple[int, int, int]:
    """
    Return ``(matrix_instr_nonkdim, num_warps, kpack)`` to use when
    ``should_use_lds_padded_path`` returned True. Centralised so tests
    and the gist can pin the exact codegen knob set.
    """
    return (LDS_PADDED_MFMA_NONKDIM, LDS_PADDED_NUM_WARPS, LDS_PADDED_KPACK)


__all__ = [
    "should_use_lds_padded_path",
    "lds_padded_launch_overrides",
    "LDS_PADDED_MFMA_NONKDIM",
    "LDS_PADDED_NUM_WARPS",
    "LDS_PADDED_KPACK",
]
