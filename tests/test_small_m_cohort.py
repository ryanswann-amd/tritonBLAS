"""Unit tests for the small-M (M <= SMALL_M_THRESHOLD) cohort gate in
``OrigamiMatmulSelector`` and the dispatch-layer routing it feeds.

These tests do **not** require a GPU kernel launch — they assert purely on
selector state — so they run on any host with ``origami`` installed and a
CUDA-visible (or HIP-visible) device for ``origami.get_hardware_for_device``.

The point of these assertions is to lock down the two invariants the K-154
fix relies on:

  1. For M in {16, 32}: the selector picks BM <= 32, biases BN >= 128 when
     N permits, and ``use_streamk`` is True so dispatch routes to the
     StreamK kernel (recovering wave parallelism via the K-split grid).

  2. For a representative M >= 64 problem: the cohort gate is OFF, the
     selector is free to pick BM >= 64, and ``use_streamk`` stays False
     unless the caller explicitly opted in via ``streamk=True``. The
     dispatch helper ``_selector_wants_streamk`` mirrors this and refuses
     to silently promote a non-cohort selector even if its internal
     ``streamk`` attribute is mutated.
"""

import pytest
import torch

origami_pkg = pytest.importorskip("origami")

from tritonblas.matmul import _selector_wants_streamk
from tritonblas.origami import (
    SMALL_M_THRESHOLD,
    OrigamiMatmulSelector,
    _round_up_pow2,
)


def _device():
    if not torch.cuda.is_available():
        pytest.skip("requires HIP/CUDA visible device for origami hardware lookup")
    return torch.device("cuda", torch.cuda.current_device())


# --------------------------------------------------------------------------- #
# Pure helpers — no hardware required.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "x, expected",
    [(0, 1), (1, 1), (2, 2), (3, 4), (8, 8), (9, 16), (16, 16), (17, 32), (32, 32)],
)
def test_round_up_pow2(x, expected):
    assert _round_up_pow2(x) == expected


def test_threshold_is_pinned_at_32():
    # The PRD pins the cohort to M <= 32; bump only with a deliberate retest.
    assert SMALL_M_THRESHOLD == 32


# --------------------------------------------------------------------------- #
# Cohort gate — small-M problems must clamp BM and force StreamK routing.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("m", [16, 32])
def test_small_m_cohort_clamps_bm_and_enables_streamk(m):
    dev = _device()
    sel = OrigamiMatmulSelector(
        m=m, n=4096, k=4096,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16, out_dtype=torch.bfloat16,
        device=dev, streamk=False,
    )

    # Cohort flag is on, BM is clamped to a power of two <= 32, and the
    # tile is wide enough in N to amortize the K-loop on a skinny problem.
    assert sel.is_small_m is True
    assert sel.block_m <= 32, f"BM={sel.block_m} would waste M-dim threads for M={m}"
    assert sel.block_m in (16, 32)
    assert sel.block_n >= 128, f"BN={sel.block_n} too narrow for skinny-M cohort"

    # Routing must report StreamK without the caller passing enable_streamk.
    assert sel.use_streamk is True
    assert _selector_wants_streamk(sel, enable_streamk=False) is True


def test_small_m_falls_back_when_n_is_tiny():
    """When N is itself smaller than the preferred 128 floor, the BN range
    must reopen so we still produce a valid config (regression guard for
    crash-on-empty-candidate-set)."""
    dev = _device()
    sel = OrigamiMatmulSelector(
        m=16, n=64, k=4096,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16, out_dtype=torch.bfloat16,
        device=dev, streamk=False,
    )
    assert sel.is_small_m is True
    assert sel.block_n > 0  # selector did not crash with an empty candidate set


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

    # Cohort gate must be OFF so the original tile-search range applies.
    assert sel.is_small_m is False
    # Selector must be free to pick BM >= 64 (the value the Origami solver
    # would have chosen on main); we just assert the gate did not clamp it.
    assert sel.block_m >= 32  # never clamped down below 32 for M >= 64

    # No selector-level StreamK promotion when the caller did not opt in.
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
    selector, the dispatch helper must still refuse to route to StreamK
    unless either the caller opts in or the cohort gate (``is_small_m``)
    is set. This is the explicit invariant guarding the K-154 change."""
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
    assert sel_out.is_small_m is False
