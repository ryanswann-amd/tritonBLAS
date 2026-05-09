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
# K-1629 worst cells — 6 (M, N, K, BM, BN, BK) tuples that drove this gate.
# These tile assignments mirror the P26 oracle's selection on N∈{128,256}
# K-COMPLEMENT cells (BM=128 or 256, BN matches N, BK=64).
# ─────────────────────────────────────────────────────────────────────────────
K1629_WORST_CELLS = [
    # (M,    N,   K,    BM,  BN,  BK)
    (4096,  128, 2048,  128, 128, 64),
    (4096,  128, 4096,  128, 128, 64),
    (4096,  128, 8192,  128, 128, 64),
    (8192,  256, 2048,  256, 256, 64),
    (8192,  256, 4096,  256, 256, 64),
    (8192,  256, 8192,  256, 256, 64),
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


@pytest.mark.parametrize("M,N,K,BM,BN,BK", K1629_WORST_CELLS)
def test_admits_k1629_worst_cells(M, N, K, BM, BN, BK):
    assert should_use_lds_padded_path(M, N, K, BM, BN, BK), (
        f"K-1629 worst cell ({M},{N},{K},{BM},{BN},{BK}) should be admitted"
    )


@pytest.mark.parametrize("M,N,K,BM,BN,BK", K1633_N512_CONTROL_CELLS)
def test_refuses_n512_control_cells(M, N, K, BM, BN, BK):
    assert not should_use_lds_padded_path(M, N, K, BM, BN, BK), (
        f"N>=512 control cell ({M},{N},{K},{BM},{BN},{BK}) must be refused "
        f"so the 17 P26 frozensets cannot regress"
    )


def test_refuses_n_outside_envelope():
    # N=512, 1024, 2048, 4096, 8192 — all production-routed
    for n in (512, 1024, 2048, 4096, 8192):
        assert not should_use_lds_padded_path(4096, n, 4096, 128, min(n, 256), 64)


def test_refuses_n64_below_envelope():
    # N<128 lives outside the K-1617 envelope
    assert not should_use_lds_padded_path(4096, 64, 4096, 128, 64, 64)
    assert not should_use_lds_padded_path(4096, 32, 4096, 128, 32, 64)


def test_refuses_short_k():
    # K<1024: K-COMPLEMENT lower bound — short-K is amortised by the
    # persistent loop epilogue, this path does not target it
    assert not should_use_lds_padded_path(4096, 128, 512, 128, 128, 64)
    assert not should_use_lds_padded_path(4096, 256, 768, 256, 256, 64)


def test_refuses_long_k_above_envelope():
    # K>16384: not in the K-1617/K-1633 catalogue — deliberately conservative
    assert not should_use_lds_padded_path(4096, 128, 32768, 128, 128, 64)


def test_refuses_unusual_block_k():
    # The K-COMPLEMENT cells select BK in {32,64}; BK=128 is the long-K
    # corner that the routing oracle assigns to a different path
    assert not should_use_lds_padded_path(4096, 128, 4096, 128, 128, 128)
    assert not should_use_lds_padded_path(4096, 256, 8192, 256, 256, 16)


def test_refuses_quantized():
    # int8/fp8 has its own epilogue contract; not certified for 32x32 MFMA
    M, N, K, BM, BN, BK = K1629_WORST_CELLS[0]
    assert not should_use_lds_padded_path(M, N, K, BM, BN, BK, quantized=True)


def test_refuses_bias():
    # Bias epilogue not re-verified against the 32x32 MFMA layout for
    # skinny-N — conservative refusal.
    M, N, K, BM, BN, BK = K1629_WORST_CELLS[0]
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
