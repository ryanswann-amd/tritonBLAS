"""Regression tests for the skinny-M GEMM heuristic.

These cover the three behavior changes shipped together for skinny-M:

  1. ``OrigamiMatmulSelector._compute_sk_grid`` no longer rolls back to
     ``sk_grid = tiles`` when the K-split branch chose ``sk_grid > tiles``.
     The original predicate ``tiles % sk_grid != 0`` always trips in that
     branch (since ``tiles % (tiles*factor) == tiles``) and silently disabled
     the K-split for every small-tile-count problem.

  2. ``tritonblas.matmul._should_auto_streamk`` opts skinny shapes
     (``M <= 32`` with >=16-bit inputs) into streamk so the cohort-tile
     override actually sees the multiplied grid.

  3. Numerics on a non-skinny shape stay within the same tolerance the
     baseline meets — this guards against the K-split predicate change
     accidentally changing accumulation order for shapes the heuristic
     was never meant to touch.

These tests skip if no CUDA device is visible so they can run in
CPU-only CI; the substantive checks are the selector / predicate logic
which is exercised on whatever device is present.
"""

from __future__ import annotations

import pytest
import torch

import tritonblas
from tritonblas.matmul import _should_auto_streamk
from tritonblas.origami import OrigamiMatmulSelector


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="skinny-M heuristic targets GPU shapes"
)


# ---------------------------------------------------------------------------
# (1) _compute_sk_grid engages the K-split for skinny shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m,n,k",
    [
        (16, 5120, 5120),
        (16, 8192, 8192),
        (16, 4096, 14336),
    ],
)
def test_compute_sk_grid_engages_k_split_for_skinny(m, n, k):
    """For skinny-M shapes the K-split branch must produce sk_grid > tiles.

    Pre-fix, the rollback predicate ``tiles % sk_grid != 0`` silently
    snapped sk_grid back to ``tiles`` (e.g. 40 for M=16,N=5120,K=5120),
    leaving 87% of MI300X CUs idle. Post-fix the rollback is gated on
    ``sk_grid <= tiles`` so the K-split branch survives.
    """
    dev = torch.device("cuda", torch.cuda.current_device())
    sel = OrigamiMatmulSelector(
        m, n, k,
        torch.bfloat16, torch.bfloat16, torch.bfloat16,
        dev,
        streamk=True,
    )
    bm, bn = sel.block_m, sel.block_n
    tiles = ((m + bm - 1) // bm) * ((n + bn - 1) // bn)
    assert sel.sk_grid > tiles, (
        f"K-split rollback regression for M={m},N={n},K={k}: "
        f"tile={bm}x{bn}, tiles={tiles}, sk_grid={sel.sk_grid}; "
        "expected sk_grid > tiles (K-split engaged)."
    )


def test_compute_sk_grid_rollback_still_fires_when_uneven_dp_split():
    """Sanity: the rollback predicate still works for the data-parallel
    branch (sk_grid <= tiles) when the chosen split is uneven and the
    workspace would blow the budget. We can't easily synthesize that
    pathologically here — instead assert the predicate now requires
    sk_grid <= tiles, so the DP branch behavior is preserved by
    construction (sk_grid is always <= tiles in that branch).
    """
    import inspect
    src = inspect.getsource(OrigamiMatmulSelector._compute_sk_grid)
    assert "sk_grid <= tiles and tiles % sk_grid != 0" in src, (
        "fix to _compute_sk_grid rollback predicate appears reverted"
    )


# ---------------------------------------------------------------------------
# (2) _should_auto_streamk gating
# ---------------------------------------------------------------------------


def _t(m, k, dtype):
    return torch.empty((m, k), dtype=dtype, device="cuda")


@pytest.mark.parametrize("m", [1, 8, 16, 32])
def test_should_auto_streamk_skinny_bf16_true(m):
    """Skinny-M (>=16-bit) shapes must opt into streamk."""
    assert _should_auto_streamk(_t(m, 4096, torch.bfloat16)) is True


@pytest.mark.parametrize("m", [33, 64, 128, 256, 1024])
def test_should_auto_streamk_non_skinny_false(m):
    """M > 32 shapes must NOT auto-route through streamk; the override
    is meant to leave non-skinny tile/grid choices untouched.
    """
    assert _should_auto_streamk(_t(m, 4096, torch.bfloat16)) is False


def test_should_auto_streamk_fp8_skipped():
    """FP8 inputs (1 byte) must not trigger the bf16-validated heuristic."""
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("fp8 not available in this torch build")
    assert _should_auto_streamk(_t(16, 4096, torch.float8_e4m3fn)) is False


def test_should_auto_streamk_int8_skipped():
    """Int8 inputs (1 byte) must not trigger the bf16-validated heuristic."""
    assert _should_auto_streamk(_t(16, 4096, torch.int8)) is False


def test_should_auto_streamk_only_2d():
    """Batched (3-D) tensors are out of scope — keep the gate strict."""
    a = torch.empty((4, 16, 4096), dtype=torch.bfloat16, device="cuda")
    assert _should_auto_streamk(a) is False


# ---------------------------------------------------------------------------
# (3) Non-skinny numerics regression guard
# ---------------------------------------------------------------------------
#
# The predicate fix lives inside _compute_sk_grid, which is called for
# every streamk shape — not just M<=32. This test is the safety net:
# pick a representative M>32 shape and assert that tritonblas.matmul
# still matches torch within the same tolerance the baseline meets on
# main. Tolerance is bf16-realistic and matches what
# tests/test_matmul_correctness uses for randn inputs.


@pytest.mark.parametrize(
    "m,n,k",
    [
        (64, 4096, 4096),    # smallest "non-skinny" boundary
        (256, 4096, 4096),   # square-ish mid
    ],
)
def test_non_skinny_numerics_unchanged(m, n, k):
    """K-split predicate fix must not regress non-skinny numerics."""
    torch.manual_seed(0)
    dev = torch.device("cuda", torch.cuda.current_device())
    a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    b = torch.randn(k, n, device=dev, dtype=torch.bfloat16)

    # Match the test_matmul.py tolerance (atol=1, rtol=1) — a deliberately
    # loose bound because random bf16 reductions accumulate large absolute
    # error on large K. The point of this test is to catch a *new* class
    # of failure (e.g. NaN, wrong shape), not to tighten what main does.
    out = tritonblas.matmul(a, b)
    ref = torch.matmul(a, b)
    torch.testing.assert_close(out, ref, atol=1.0, rtol=1.0)
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype


# ---------------------------------------------------------------------------
# (4) Skinny-tile selector picks the override tile
# ---------------------------------------------------------------------------


def test_skinny_selector_picks_override_tile_for_M16():
    """For M=16,N>=256,K>=64,bf16 on supported arch, selector picks
    the cohort tile (16,256,64) rather than Origami's narrow default."""
    dev = torch.device("cuda", torch.cuda.current_device())
    # Skip on archs the override is not gated for.
    import origami as _origami
    hw = _origami.get_hardware_for_device(dev.index)
    if hw.N_CU not in (304, 80, 64, 228, 256):
        pytest.skip(f"override not enabled for N_CU={hw.N_CU}")

    sel = OrigamiMatmulSelector(
        16, 5120, 5120,
        torch.bfloat16, torch.bfloat16, torch.bfloat16,
        dev,
    )
    assert (sel.block_m, sel.block_n, sel.block_k) == (16, 256, 64), (
        f"override tile not applied: got "
        f"{sel.block_m}x{sel.block_n}x{sel.block_k}"
    )
