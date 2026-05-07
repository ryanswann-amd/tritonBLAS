"""Unit tests for the small-M (M <= SMALL_M_THRESHOLD) cohort gate in
``OrigamiMatmulSelector`` and the dispatch-layer routing it feeds.

These tests do **not** require a GPU kernel launch — they assert purely on
selector state — so they run on any host with ``origami`` installed and a
HIP/CUDA-visible device for ``origami.get_hardware_for_device``.

The point of these assertions is to lock down the four invariants the K-154
fix relies on:

  1. For M in {16, 32}: the selector pins (BM, BN, BK) to the curated
     ``_SMALL_M_TILE`` and ``use_streamk`` is True so dispatch routes to the
     StreamK kernel (recovering wave parallelism via the K-split grid).

  2. For a representative M >= 64 problem: the cohort gate is OFF, the
     analytic tile selector is free to run, and ``use_streamk`` stays False
     unless the caller explicitly opted in via ``streamk=True``.

  3. The boundary is inclusive at M = SMALL_M_THRESHOLD and exclusive at
     M = SMALL_M_THRESHOLD + 1.

  4. The dispatch helper ``_selector_wants_streamk`` reads
     ``selector.use_streamk`` (the read-only property derived from
     construction-time invariants), not the writable ``selector.streamk``
     attribute, so a downstream mutation of ``selector.streamk`` cannot
     silently promote a non-cohort selector onto the StreamK kernel.

  5. The override would refuse to silently regress to the analytic pick if
     the curated tile ever became LDS-infeasible (we simulate that by
     passing an artificially small ``lds_cap`` to
     ``check_triton_lds_capacity`` in a pure-helper path test, since the
     production path's LDS check is trivially true on supported hardware).
"""

import pytest
import torch

origami_pkg = pytest.importorskip("origami")

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
    assert sel.use_streamk is True
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
    assert sel.use_streamk is False
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
    assert sel.use_streamk is True
    assert _selector_wants_streamk(sel, enable_streamk=False) is True
    assert _selector_wants_streamk(sel, enable_streamk=True) is True


def test_dispatch_refuses_silent_promotion_on_non_cohort():
    """If some other code path mutates ``selector.streamk`` on an M >= 64
    selector with ``streamk=False``, the routing layer must still refuse
    to promote to StreamK. ``use_streamk`` is derived from pinned
    construction-time invariants (``_user_streamk``, ``_small_m``) and
    deliberately ignores mutations of the writable ``streamk`` attribute.
    This is the explicit invariant guarding the K-154 change."""
    dev = _device()
    sel = OrigamiMatmulSelector(
        m=4096, n=4096, k=4096,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16, out_dtype=torch.bfloat16,
        device=dev, streamk=False,
    )
    # Simulate a downstream mutation of the writable attribute.
    sel.streamk = True
    assert sel.is_small_m is False
    # use_streamk derives from pinned _user_streamk + _small_m, not from streamk:
    assert sel.use_streamk is False
    # And the dispatch helper does not promote either.
    assert _selector_wants_streamk(sel, enable_streamk=False) is False


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
