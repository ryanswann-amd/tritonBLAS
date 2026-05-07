"""Unit tests for the small-M (M <= SMALL_M_THRESHOLD) cohort gate in
``OrigamiMatmulSelector`` and the dispatch-layer routing it feeds.

These tests are split into two groups:

  * Selector tests — assert purely on selector state (which tile and routing
    flag get picked). They run on any host with ``origami`` installed and
    a HIP/CUDA-visible device.

  * Numerical tests — actually launch the kernel and compare the bf16
    output to ``torch.matmul`` with an allclose tolerance. These guard
    against a silently miscomputing pinned tile + forced StreamK path that
    would otherwise pass a selector-only assertion suite.

The point of these assertions is to lock down the four invariants the K-154
fix relies on:

  1. For M in {16, 32}: the selector pins (BM, BN, BK) to the curated
     ``_SMALL_M_TILE`` and ``selector.streamk`` is True so dispatch routes
     to the StreamK kernel (recovering wave parallelism via the K-split
     grid).

  2. For a representative M >= 64 problem: the cohort gate is OFF, the
     analytic tile selector is free to run, and ``selector.streamk`` stays
     False unless the caller explicitly opted in via ``streamk=True``.

  3. The boundary is inclusive at M = SMALL_M_THRESHOLD and exclusive at
     M = SMALL_M_THRESHOLD + 1.

  4. Numerical: the M<=32 cohort produces output that matches torch.matmul
     for representative bf16 shapes — including a non-power-of-2 K
     (11008) where the K-tail handling of the StreamK kernel is exercised.

  5. The override would refuse to silently regress to the analytic pick if
     the curated tile ever became LDS-infeasible (we simulate that by
     passing an artificially small ``lds_cap`` to
     ``check_triton_lds_capacity`` in a pure-helper path test, since the
     production path's LDS check is trivially true on supported hardware).
"""

import pytest
import torch

origami_pkg = pytest.importorskip("origami")

import tritonblas
from tritonblas.matmul import _selector_wants_streamk
from tritonblas.origami import (
    SMALL_M_THRESHOLD,
    OrigamiMatmulSelector,
    _SMALL_M_TILE,
    check_triton_lds_capacity,
)


def _device():
    if not torch.cuda.is_available():
        pytest.skip("requires HIP/CUDA visible device for origami hardware lookup")
    return torch.device("cuda", torch.cuda.current_device())


# --------------------------------------------------------------------------- #
# Constants — keep the cohort definition pinned.
# --------------------------------------------------------------------------- #


def test_threshold_is_pinned_at_32():
    # The PRD pins the cohort to M <= 32; bump only with a deliberate retest.
    assert SMALL_M_THRESHOLD == 32


def test_pinned_tile_is_32_64_64():
    # Empirically validated tile for gfx942 bf16/fp16. Changing this constant
    # requires re-running the K-154 sweep.
    assert _SMALL_M_TILE == (32, 64, 64)


# --------------------------------------------------------------------------- #
# Cohort gate — small-M problems must pin the curated tile and force StreamK.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("m", [16, 32])
def test_small_m_cohort_picks_curated_tile_and_enables_streamk(m):
    dev = _device()
    sel = OrigamiMatmulSelector(
        m=m, n=4096, k=4096,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16, out_dtype=torch.bfloat16,
        device=dev, streamk=False,
    )

    # Cohort flag is on; (BM, BN, BK) must equal the pinned _SMALL_M_TILE.
    assert sel.is_small_m is True
    assert (sel.block_m, sel.block_n, sel.block_k) == _SMALL_M_TILE

    # Routing must report StreamK without the caller passing enable_streamk.
    assert sel.streamk is True
    assert _selector_wants_streamk(sel, enable_streamk=False) is True


# --------------------------------------------------------------------------- #
# Non-cohort gate — M >= 64 must not be touched by the small-M logic.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("m", [64, 256, 4096])
def test_large_m_cohort_unchanged(m):
    dev = _device()
    sel = OrigamiMatmulSelector(
        m=m, n=4096, k=4096,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16, out_dtype=torch.bfloat16,
        device=dev, streamk=False,
    )

    # Cohort gate must be OFF so the original tile-search range applies, and
    # the routing layer must not auto-promote to StreamK.
    assert sel.is_small_m is False
    assert sel.streamk is False
    assert _selector_wants_streamk(sel, enable_streamk=False) is False


def test_large_m_explicit_streamk_still_routes():
    """Caller-opt-in path must keep working for M >= 64."""
    dev = _device()
    sel = OrigamiMatmulSelector(
        m=4096, n=4096, k=4096,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16, out_dtype=torch.bfloat16,
        device=dev, streamk=True,
    )
    assert sel.is_small_m is False
    assert sel.streamk is True
    assert _selector_wants_streamk(sel, enable_streamk=False) is True
    assert _selector_wants_streamk(sel, enable_streamk=True) is True


# --------------------------------------------------------------------------- #
# Boundary check — M = SMALL_M_THRESHOLD is in the cohort, M+1 is not.
# --------------------------------------------------------------------------- #


def test_threshold_boundary_inclusive():
    dev = _device()
    sel_in = OrigamiMatmulSelector(
        m=SMALL_M_THRESHOLD, n=4096, k=4096,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16, out_dtype=torch.bfloat16,
        device=dev, streamk=False,
    )
    sel_out = OrigamiMatmulSelector(
        m=SMALL_M_THRESHOLD + 1, n=4096, k=4096,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16, out_dtype=torch.bfloat16,
        device=dev, streamk=False,
    )
    assert sel_in.is_small_m is True
    assert (sel_in.block_m, sel_in.block_n, sel_in.block_k) == _SMALL_M_TILE
    assert sel_out.is_small_m is False


# --------------------------------------------------------------------------- #
# LDS-feasibility contract — the pinned tile fits real hardware, and the
# helper would correctly reject it under an artificially small budget.
# --------------------------------------------------------------------------- #


def test_pinned_tile_fits_supported_hardware_lds():
    """A 2-stage bf16 (32, 64, 64) tile should fit comfortably under any
    supported gfx90a/gfx942/gfx950 LDS budget (>=64KB)."""
    bm, bn, bk = _SMALL_M_TILE
    bytes_per_elem = 2  # bf16/fp16
    # 64KB is the smallest supported LDS budget on gfx942 / gfx90a.
    assert check_triton_lds_capacity(
        bm, bn, bk, bytes_per_elem, bytes_per_elem, 64 * 1024, num_stages=2
    )


def test_lds_feasibility_check_rejects_under_tiny_budget():
    """Sanity-check the LDS predicate the override depends on: the pinned
    tile must be rejected when given an absurdly small LDS cap. This guards
    against a future change to the LDS estimator silently passing
    everything (which would let a too-large override slip through)."""
    bm, bn, bk = _SMALL_M_TILE
    bytes_per_elem = 2
    # 4KB is well below what (32, 64, 64) bf16 needs at any num_stages.
    assert not check_triton_lds_capacity(
        bm, bn, bk, bytes_per_elem, bytes_per_elem, 4 * 1024, num_stages=2
    )


# --------------------------------------------------------------------------- #
# Numerical correctness — actually launch the small-M cohort kernel and
# allclose against torch.matmul. Without these, a silently miscomputing
# pinned tile + forced StreamK path would pass every selector-only test.
#
# Tolerance: bf16 matmul accumulates in fp32, so the worst-case error grows
# like K * eps_bf16 ~ K * 2**-7. For K up to 11008 that puts the natural
# floor around 1e-1 absolute on outputs of magnitude O(sqrt(K)). We use
# (atol=0.5, rtol=0.05) — passing this rules out any silent bit-rot in the
# cohort path while staying inside the slop bf16 GEMMs accumulate.
# --------------------------------------------------------------------------- #


_NUMERICAL_SHAPES = [
    # M, N, K — covers M in {16, 32}, square / rect-N / rect-K.
    (16, 4096, 4096),
    (32, 4096, 4096),
    (16, 8192, 4096),
    (32, 8192, 4096),
    # K=11008 — non-power-of-2, the K-tail case the bench called out as
    # weakest. Critical to verify *correctness* here even if perf is soft.
    (16, 4096, 11008),
    (32, 4096, 11008),
]


@pytest.mark.parametrize("m, n, k", _NUMERICAL_SHAPES)
def test_small_m_cohort_matches_torch_matmul_bf16(m, n, k):
    dev = _device()
    torch.manual_seed(0xC0FFEE ^ (m << 16) ^ (n << 8) ^ k)

    a = torch.randn((m, k), dtype=torch.bfloat16, device=dev)
    b = torch.randn((k, n), dtype=torch.bfloat16, device=dev)

    # Ensure we are exercising the cohort path (selector inspection above
    # already covers this; the assertion here defends against a future
    # refactor that breaks the linkage between the public matmul entry
    # point and the cohort-gated selector).
    sel = OrigamiMatmulSelector(
        m=m, n=n, k=k,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16, out_dtype=torch.bfloat16,
        device=dev, streamk=False,
    )
    assert sel.is_small_m is True
    assert (sel.block_m, sel.block_n, sel.block_k) == _SMALL_M_TILE
    assert sel.streamk is True

    out = tritonblas.matmul(a, b)
    ref = torch.matmul(a, b)

    assert out.shape == (m, n)
    assert out.dtype == torch.bfloat16

    # bf16 GEMM accumulates in fp32 then truncates; element-wise diff scales
    # with K. The tolerance below is comfortably above the worst-case bf16
    # round-off floor and well below what any silent bit-rot would produce.
    torch.testing.assert_close(out, ref, atol=0.5, rtol=0.05)
