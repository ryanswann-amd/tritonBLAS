"""K-857 [S-002] N1 cohort narrow-gate predicate — regression-protected harness.

Background
----------
K-850 PMC + ISA triage of K-837 PMC dataset on the K-790 ranked-gap profile
recommended a (BM,BN)=(256,128) for M>N / (128,256) for M<N asymmetric tile
plus kpack=1->2 for the N1 cohort, mechanistically targeted at LDS bank
collisions on a 128x128-default baseline (composite-14 audit branch).

Live re-verification on `main` (95e2c47, c42/MI300X, n=50 hot HIP-graph,
paired ON/OFF flip via env var) produced **VERDICT: NO-GO** — geomean
ON/OFF = 0.815x across 4 N1-envelope cells (i.e. ON is ~18.5% slower than
OFF), because Origami's default selector on `main` already picks BM=256,
BN=256 (4MB tile) for these cells, which is *larger* than the K-850-
recommended 256x128 (2MB tile). K-850's recommendation was relative to
the composite-14 baseline (default 128x128, 1MB), not current `main`.

See ``output/REPORT.md`` and ``output/k857_paired_n1.csv`` /
``output/k857_paired_guard.csv`` in the K-857 workspace for the full
methodology and per-cell numbers.

Why this file exists despite the NO-GO verdict
----------------------------------------------
The PRD requires an ``is_K857_N1_cohort()`` predicate and disjointness
proofs against neighbouring cohorts (K-756, K-804, K-668, K-746/K-776 FP8)
to prevent the K-654-style bypass regression. The reviewer feedback for
K-857 explicitly asked for this checked in as a real pytest, not as an
ad-hoc shell heredoc.

The override itself was **NOT** landed in production matmul.py — see the
NO-GO verdict above. This file ships **only** the pure-Python predicate
+ tile picker as a documented harness so:

1. The disjointness/boundary logic is regression-protected — a future
   re-test (e.g. when Triton-AMD lifts the pow2(BLOCK)<=256 ceiling, or
   composite-14 lands and shifts Origami's tile picks back toward 128x128)
   can re-import the predicate verbatim and the truth table will already
   be locked in.
2. The K-654-style bypass guard (M==N, max<8192, K out of [2048,4096],
   FP8 dtype, etc.) is enforced by assertion, not by hand-eyeballed code
   review.

Tests are CPU-only (no GPU import / no Triton compile) so they run in
the standard ``python -m pytest tests/`` lane.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch


# ---------------------------------------------------------------------------
# Predicate + tile picker — pure-Python, self-contained, no GPU touch.
# ---------------------------------------------------------------------------

_K857_N1_DTYPES: Tuple[torch.dtype, ...] = (torch.bfloat16, torch.float16)


def is_K857_N1_cohort(M: int, N: int, K: int, dtype: torch.dtype) -> bool:
    """K-850 N1 cohort predicate (M-major OR N-major leg).

    Disjoint by construction from neighbouring cohorts:

    * K-756 (square 2048): excluded by ``M != N``.
    * K-804 (mid-rect, max <= 6144): excluded by ``max(M,N) >= 8192``.
    * K-668 (skinny / micro-N): excluded by ``min(M,N) >= 2048``.
    * K-746 / K-776 (FP8): excluded by ``dtype in {bf16, fp16}``.
    * Out-of-K (e.g. 1024 or 8192): excluded by ``2048 <= K <= 4096``.

    Returns True iff the cell sits inside the K-857 N1 envelope.
    """
    if dtype not in _K857_N1_DTYPES:
        return False
    if M == N:
        return False
    lo, hi = min(M, N), max(M, N)
    if hi < 8192:
        return False
    if not (2048 <= lo <= 4096):
        return False
    if not (2048 <= K <= 4096):
        return False
    return True


def k857_n1_tile(M: int, N: int) -> Tuple[int, int, int, int, int]:
    """Asymmetric tile per K-850 Override #2 (M>N) / #3 (M<N).

    Returns ``(block_m, block_n, block_k, group_m, kpack)``.
    """
    if M > N:
        # M-major leg (e.g. 8192x4096x4096 bf16): BM=256, BN=128
        return (256, 128, 64, 8, 2)
    # N-major leg (e.g. 4096x8192x4096 bf16): BM=128, BN=256
    return (128, 256, 64, 8, 2)


# ---------------------------------------------------------------------------
# Predicate truth table — happy path (N1a / N1b across bf16, fp16).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,M,N,K,dtype",
    [
        ("N1a-bf16", 8192, 4096, 4096, torch.bfloat16),
        ("N1b-bf16", 4096, 8192, 4096, torch.bfloat16),
        ("N1a-fp16", 8192, 4096, 4096, torch.float16),
        ("N1b-fp16", 4096, 8192, 4096, torch.float16),
        # Larger envelope cells (still inside the gate).
        ("N1-edge-Mhi", 16384, 4096, 4096, torch.bfloat16),
        ("N1-edge-K2048", 8192, 4096, 2048, torch.bfloat16),
        ("N1-edge-N2048", 8192, 2048, 4096, torch.bfloat16),
    ],
)
def test_predicate_fires_on_n1_envelope(label, M, N, K, dtype):
    """Every K-850-anchor and envelope-edge N1 cell must satisfy the gate."""
    assert is_K857_N1_cohort(M, N, K, dtype), (
        f"K-857 N1 predicate must fire on {label} ({M}x{N}x{K} {dtype})"
    )


# ---------------------------------------------------------------------------
# Predicate truth table — critical near-miss exclusions. These guard the
# disjointness invariant: if any of these start firing, a future edit has
# silently re-introduced the K-654-style bypass regression.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,M,N,K,dtype,reason",
    [
        # K-756: square 2048 — excluded by M != N.
        ("K-756-square-2048", 2048, 2048, 2048, torch.bfloat16, "M==N"),
        # Square at the K-857 K-bound — still excluded by M==N.
        ("square-4096", 4096, 4096, 4096, torch.bfloat16, "M==N"),
        ("square-8192", 8192, 8192, 4096, torch.bfloat16, "M==N"),
        # K-804 mid-rect (max <= 6144) — excluded by hi < 8192.
        ("K-804-6144x4096", 6144, 4096, 4096, torch.bfloat16, "hi<8192"),
        ("K-804-4096x6144", 4096, 6144, 4096, torch.bfloat16, "hi<8192"),
        # Skinny / K-668 — excluded by lo < 2048.
        ("skinny-N64", 8192, 64, 4096, torch.bfloat16, "lo<2048"),
        ("skinny-N1024", 8192, 1024, 4096, torch.bfloat16, "lo<2048"),
        ("skinny-M1024", 1024, 8192, 4096, torch.bfloat16, "lo<2048"),
        # min upper bound — excluded by lo > 4096.
        ("lo-too-large", 16384, 8192, 4096, torch.bfloat16, "lo>4096"),
        # Out-of-K on either side — excluded by K bounds.
        ("K-1024", 8192, 4096, 1024, torch.bfloat16, "K<2048"),
        ("K-8192", 8192, 4096, 8192, torch.bfloat16, "K>4096"),
        # FP8 / fp32 — excluded by dtype gate (K-746/K-776 disjointness).
        ("fp8-e4m3", 8192, 4096, 4096, torch.float8_e4m3fnuz, "dtype"),
        ("fp32", 8192, 4096, 4096, torch.float32, "dtype"),
    ],
)
def test_predicate_excludes_neighbouring_cohorts(label, M, N, K, dtype, reason):
    """Disjointness guard. A future edit that breaks this re-introduces K-654.

    Each excluded cell documents the *specific* reason it is out of the gate
    (M==N for K-756, hi<8192 for K-804, lo<2048 for K-668, dtype for FP8).
    """
    assert not is_K857_N1_cohort(M, N, K, dtype), (
        f"K-857 N1 predicate must NOT fire on {label} "
        f"({M}x{N}x{K} {dtype}) — reason: {reason}"
    )


# ---------------------------------------------------------------------------
# Tile picker — must produce the K-850 Override #2 / #3 picks exactly.
# ---------------------------------------------------------------------------


def test_tile_pick_m_major_leg():
    """N1a (M>N): K-850 Override #2 -> (BM=256, BN=128, BK=64, GM=8, kpack=2)."""
    bm, bn, bk, gm, kpack = k857_n1_tile(8192, 4096)
    assert (bm, bn, bk, gm, kpack) == (256, 128, 64, 8, 2)


def test_tile_pick_n_major_leg():
    """N1b (M<N): K-850 Override #3 -> (BM=128, BN=256, BK=64, GM=8, kpack=2)."""
    bm, bn, bk, gm, kpack = k857_n1_tile(4096, 8192)
    assert (bm, bn, bk, gm, kpack) == (128, 256, 64, 8, 2)


def test_kpack_is_two_on_both_legs():
    """kpack=2 lever (K-580 / K-834 narrow-gate analog) on both legs."""
    assert k857_n1_tile(8192, 4096)[-1] == 2
    assert k857_n1_tile(4096, 8192)[-1] == 2


def test_tile_areas_are_2x_composite14_baseline():
    """Both K-850 picks are 32768 elements = 2x the 128x128 composite-14 default."""
    for M, N in [(8192, 4096), (4096, 8192)]:
        bm, bn, *_ = k857_n1_tile(M, N)
        assert bm * bn == 32768


# ---------------------------------------------------------------------------
# Override-is-NOT-wired-into-production guard.
#
# The K-857 paired bench produced a NO-GO verdict (geomean 0.815x ON/OFF on
# the 4 N1-envelope cells). The override was deliberately NOT merged into
# matmul.py — this test pins that decision so a future revert/cherry-pick
# does not silently re-enable it.
# ---------------------------------------------------------------------------


def test_override_not_wired_into_production_matmul():
    """K-857 narrow-gate hooks must NOT be present in production matmul.py.

    NO-GO verdict (see output/REPORT.md): the K-850 picks regress on `main`
    because Origami already picks a larger tile. If this test starts failing,
    someone re-landed the override — re-run the K-857 paired bench before
    shipping.
    """
    import inspect

    from tritonblas import matmul as _matmul_mod

    src = inspect.getsource(_matmul_mod)
    forbidden = (
        "is_K857_N1_cohort",
        "_maybe_apply_k857_n1",
        "_K857_N1_OverrideSelector",
        "TRITONBLAS_ENABLE_K857_N1",
        "_k857_kpack",
    )
    leaked = [tok for tok in forbidden if tok in src]
    assert not leaked, (
        "K-857 N1 override hooks leaked into production matmul.py: "
        f"{leaked}. The K-857 experiment was NO-GO; the override must stay "
        "out of the production path. Re-run the paired bench before re-landing."
    )
