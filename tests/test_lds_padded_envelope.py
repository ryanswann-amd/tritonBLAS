# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Pure-Python unit tests for the K-1640 skinny-N LDS-padded staging envelope.

These tests run on any host (no GPU required) — the envelope predicate is
pure Python over (M, N, K, BM, BN, BK) tuples. The load-bearing safety
contract is:

  1. Every K-1629 worst-cell (the 6 cells the path was designed for) is
     ADMITTED.
  2. Every N >= 512 cell representative of the 17 P26 production
     frozensets is REFUSED — this is what guarantees the path cannot
     regress already-shipped routings.
  3. Quantized + bias paths are REFUSED unconditionally — the envelope
     does not certify those epilogues against the 32x32 MFMA layout.
"""

import pytest

from tritonblas.kernels.lds_padded_envelope import (
    should_use_lds_padded_path,
    lds_padded_launch_overrides,
    LDS_PADDED_MFMA_NONKDIM,
    LDS_PADDED_NUM_WARPS,
    LDS_PADDED_KPACK,
)


# ─────────────────────────────────────────────────────────────────────────────
# K-1640 iter-2 ADMITTED cell — the single (M==K, N==128) square-skinny
# corner that showed a statistically-significant win (paired-t = +3.87,
# +3.04% median speedup) in the K-1640 paired n=30 HIP-graph hot-cache
# sweep. This is the only cell the production gate now admits.
# ─────────────────────────────────────────────────────────────────────────────
K1640_ADMIT_CELLS = [
    # (M,    N,   K,    BM,  BN,  BK)  — the proven winner
    (4096,  128, 4096,  128, 128, 64),
]


# ─────────────────────────────────────────────────────────────────────────────
# K-1640 iter-2 REFUSED cells — the 5 K-1629 cells that the wider iter-1
# envelope admitted but the paired sweep showed were noise or net-negative.
# These MUST now fall through to baseline so we don't ship code that we
# measured as regressing them.
# ─────────────────────────────────────────────────────────────────────────────
K1640_REFUSED_K1629_CELLS = [
    # (M,    N,   K,    BM,  BN,  BK,  reason)
    (4096,  128, 2048,  128, 128, 64, "M!=K (M:K=2:1); paired-t=-2.63 (regress)"),
    (4096,  128, 8192,  128, 128, 64, "M!=K (M:K=1:2); paired-t=+1.70 (noise)"),
    (8192,  256, 2048,  256, 256, 64, "N=256 not admitted; paired-t=-1.54"),
    (8192,  256, 4096,  256, 256, 64, "N=256 not admitted; paired-t=-0.50 (noise)"),
    (8192,  256, 8192,  256, 256, 64, "N=256 not admitted; paired-t=-4.13 (regress)"),
]


# ─────────────────────────────────────────────────────────────────────────────
# K-1633 control cells — N=512 representatives of the 17 P26 frozensets.
# These MUST be refused so the gate cannot perturb production routing.
# ─────────────────────────────────────────────────────────────────────────────
K1633_N512_CONTROL_CELLS = [
    (4096,  512, 2048,  128, 256, 64),
    (4096,  512, 4096,  256, 256, 64),
    (8192,  512, 8192,  256, 256, 64),
]


@pytest.mark.parametrize("M,N,K,BM,BN,BK", K1640_ADMIT_CELLS)
def test_admits_k1640_proven_winner(M, N, K, BM, BN, BK):
    assert should_use_lds_padded_path(M, N, K, BM, BN, BK), (
        f"K-1640 proven-winner cell ({M},{N},{K},{BM},{BN},{BK}) should "
        f"still be admitted (this is the only cell the iter-2 narrowed "
        f"gate admits — losing it loses the entire PR)"
    )


@pytest.mark.parametrize("M,N,K,BM,BN,BK,reason", K1640_REFUSED_K1629_CELLS)
def test_refuses_iter1_cells_that_iter2_sweep_showed_were_not_wins(
    M, N, K, BM, BN, BK, reason,
):
    assert not should_use_lds_padded_path(M, N, K, BM, BN, BK), (
        f"Cell ({M},{N},{K},{BM},{BN},{BK}) was admitted by the iter-1 "
        f"envelope but the K-1640 paired n=30 sweep showed: {reason}. "
        f"The iter-2 narrowed gate must REFUSE it."
    )


@pytest.mark.parametrize("M,N,K,BM,BN,BK", K1633_N512_CONTROL_CELLS)
def test_refuses_n512_control_cells(M, N, K, BM, BN, BK):
    assert not should_use_lds_padded_path(M, N, K, BM, BN, BK), (
        f"N>=512 control cell ({M},{N},{K},{BM},{BN},{BK}) must be refused "
        f"so the 17 P26 frozensets cannot regress"
    )


def test_refuses_n_outside_envelope():
    # N=256, 512, 1024, 2048, 4096, 8192 — all refused by the iter-2 gate
    # (N=256 was refused because the K-1640 paired sweep showed no win)
    for n in (256, 512, 1024, 2048, 4096, 8192):
        assert not should_use_lds_padded_path(4096, n, 4096, 128, min(n, 256), 64), (
            f"N={n} must be refused by the iter-2 gate"
        )


def test_refuses_n64_below_envelope():
    # N<128 lives outside the K-1617 envelope
    assert not should_use_lds_padded_path(4096, 64, 4096, 128, 64, 64)
    assert not should_use_lds_padded_path(4096, 32, 4096, 128, 32, 64)


def test_refuses_short_k():
    # K<2048: K-1640 narrowed lower bound — we have no ground-truth
    # measurement at K<2048, and even at K=2048 the (M=4096,N=128,K=2048)
    # cell regressed -1.93%.
    assert not should_use_lds_padded_path(4096, 128, 1024, 128, 128, 64)
    assert not should_use_lds_padded_path(4096, 128, 512, 128, 128, 64)


def test_refuses_long_k_above_envelope():
    # K>16384: not in the K-1617/K-1633 catalogue — deliberately conservative
    assert not should_use_lds_padded_path(16384, 128, 32768, 128, 128, 64)


def test_refuses_unusual_block_k():
    # The K-COMPLEMENT cells select BK in {32,64}; BK=128 is the long-K
    # corner that the routing oracle assigns to a different path. Use
    # M==K to isolate the BK gate from the M==K gate.
    assert not should_use_lds_padded_path(4096, 128, 4096, 128, 128, 128)
    assert not should_use_lds_padded_path(8192, 128, 8192, 128, 128, 16)


def test_refuses_unequal_M_K():
    # The iter-2 narrowing requires M==K (square-skinny corner). Verify
    # both off-diagonal directions are refused even when N, BM, BN, BK
    # all sit inside the rest of the envelope.
    assert not should_use_lds_padded_path(4096, 128, 8192, 128, 128, 64), \
        "M:K = 1:2 must be refused"
    assert not should_use_lds_padded_path(8192, 128, 4096, 128, 128, 64), \
        "M:K = 2:1 must be refused"


def test_refuses_quantized():
    # int8/fp8 has its own epilogue contract; not certified for 32x32 MFMA
    M, N, K, BM, BN, BK = K1640_ADMIT_CELLS[0]
    assert not should_use_lds_padded_path(M, N, K, BM, BN, BK, quantized=True)


def test_refuses_bias():
    # Bias epilogue not re-verified against the 32x32 MFMA layout for
    # skinny-N — conservative refusal.
    M, N, K, BM, BN, BK = K1640_ADMIT_CELLS[0]
    assert not should_use_lds_padded_path(M, N, K, BM, BN, BK, bias=True)


def test_launch_overrides_pin_codegen_knobs():
    # The PR description and gist both quote these constants — pin them
    # so silent drift is impossible.
    nonkdim, num_warps, kpack = lds_padded_launch_overrides()
    assert nonkdim == 32, "32x32x8 MFMA breaks the bank-conflict pattern"
    assert num_warps == 4, "halve warps to halve LDS queue depth on skinny-N"
    assert kpack == 1, (
        "kpack must remain 1 — K-612/K-646/K-683 anti-pattern: kpack=2 "
        "regresses long-K shapes via latency-hiding collapse"
    )
    # Also assert the module-level constants match the function output —
    # callers that import the constants directly must see the same values.
    assert nonkdim == LDS_PADDED_MFMA_NONKDIM
    assert num_warps == LDS_PADDED_NUM_WARPS
    assert kpack == LDS_PADDED_KPACK
