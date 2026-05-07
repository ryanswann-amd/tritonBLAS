"""Unit tests for the aspect-ratio gate on the symmetric-snap heuristic.

Background
----------
`OrigamiMatmulSelector.__init__` post-processes the tile picked by origami's
`select_config` and snaps an asymmetric (256x?, ?x256) tile to 256x256x64
when the LDS budget allows.  The original heuristic was unconditional and
clobbered origami's asymmetric pick even on highly skewed problems where
the asymmetric tile (typically 256x128x64) outperforms 256x256x64 by 20-30%
TFLOPS on MI300X.

The fix gates the snap by `max(M,N) / min(M,N) <= SYMMETRIC_SNAP_ASPECT_THRESHOLD`
(default 1.5), so square / near-square problems still snap, and large-skinny
problems keep the asymmetric pick.

These tests exercise the *exact* gate the production __init__ uses, by
calling the classmethod `OrigamiMatmulSelector._aspect_allows_symmetric_snap`
that __init__ delegates to -- no replica, no GPU, no origami solver.

The boundary band asp ∈ (1.5, 2.0] is the band whose behavior changed
under this PR: previously every shape in this band snapped to 256x256x64,
now it keeps origami's asymmetric pick.  Both sides of the boundary
(snap-fires and snap-skips) are explicitly asserted.
"""

import math
import pytest

from tritonblas.origami import OrigamiMatmulSelector


# Convenience alias so the test reads like the production check.
_gate = OrigamiMatmulSelector._aspect_allows_symmetric_snap


class TestThresholdConstantSurface:
    """The threshold is the public knob; pin its surface."""

    def test_threshold_is_a_class_constant(self):
        # Lives on the class so downstream callers can override it for hardware
        # variants without monkey-patching __init__.
        assert hasattr(OrigamiMatmulSelector, "SYMMETRIC_SNAP_ASPECT_THRESHOLD")

    def test_threshold_value_default(self):
        # Default 1.5 is conservative: snaps perfectly-square (asp=1.0) and
        # mild skew (asp <= 1.5), skips moderate (asp=2.0) and large (asp>=4)
        # skew where empirical benchmarks show the asymmetric tile wins.
        t = OrigamiMatmulSelector.SYMMETRIC_SNAP_ASPECT_THRESHOLD
        assert isinstance(t, (int, float))
        assert math.isfinite(t)
        assert 1.0 < t < 2.0

    def test_threshold_accessible_from_classmethod(self):
        # Must round-trip through the gate -- not just be a bare attribute.
        assert _gate(1024, 1024) is True            # asp=1.0
        assert _gate(1024, 8192) is False           # asp=8.0


class TestSnapFires:
    """Positive direction -- the snap MUST fire here (asp <= 1.5).

    These prove the gate's both-sides behavior: that the PR did not
    accidentally disable the snap for square / near-square shapes that
    K-712 relies on for its 256x256x64 pick.
    """

    def test_perfect_square_snaps(self):
        # 4096x4096, 8192x8192, 1024x1024 -- canonical K-712 shapes.
        assert _gate(4096, 4096) is True
        assert _gate(8192, 8192) is True
        assert _gate(1024, 1024) is True

    def test_mild_skew_snaps(self):
        # asp=1.25, 1.33, 1.4 -- still considered "square enough".
        assert _gate(1024, 1280) is True            # asp=1.25
        assert _gate(3072, 4096) is True            # asp=1.33
        assert _gate(4096, 3072) is True            # asp=1.33 (transposed)
        assert _gate(1024, 1433) is True            # asp~=1.4

    def test_at_boundary_inclusive(self):
        # asp=1.5 EXACTLY -- inclusive boundary by design (`<=`).
        assert _gate(1024, 1536) is True            # 1536/1024 = 1.5
        assert _gate(2048, 3072) is True            # 3072/2048 = 1.5
        assert _gate(4096, 6144) is True            # 6144/4096 = 1.5

    def test_just_below_boundary(self):
        # asp=1.49 -- comfortably under threshold.
        assert _gate(100, 149) is True              # 1.49


class TestSnapSkips:
    """Negative direction -- the snap MUST NOT fire here (asp > 1.5).

    The band asp ∈ (1.5, 2.0] is the band whose behavior CHANGED in this
    PR -- shapes here used to snap (silently regressing performance) and
    now correctly keep origami's asymmetric tile.
    """

    def test_just_above_boundary(self):
        # asp~=1.501 -- the smallest skew that must skip.  Tightest
        # check that the boundary is truly inclusive on the snap side.
        assert _gate(1000, 1501) is False           # 1.501

    def test_band_just_above_boundary(self):
        # The newly-affected band asp ∈ (1.5, 2.0].  All of these used
        # to snap to 256x256x64 under the legacy heuristic; under the
        # new gate they correctly keep origami's asymmetric pick.
        assert _gate(2048, 3328) is False           # asp~=1.625
        assert _gate(2048, 3584) is False           # asp=1.75
        assert _gate(2048, 3840) is False           # asp~=1.875
        assert _gate(1024, 1800) is False           # asp~=1.758
        # Transposed (N >> M instead of M >> N) -- gate is symmetric.
        assert _gate(3328, 2048) is False
        assert _gate(3584, 2048) is False

    def test_aspect_exactly_2(self):
        # asp=2.0: empirical sweep on 2048x4096 shows 256x128x64 wins
        # over 256x256x64 by ~30% TFLOPS.  Snap MUST skip.
        assert _gate(2048, 4096) is False
        assert _gate(4096, 2048) is False

    def test_large_skinny_skips(self):
        # asp=8.0: K-757 cohort.  These are the headline shapes where
        # the asymmetric pick gives +18-24% TFLOPS uplift.
        assert _gate(8192, 1024) is False
        assert _gate(1024, 8192) is False

    def test_huge_aspect_skips(self):
        # asp=128.0 and beyond: extreme batched-style skew.  Must skip.
        assert _gate(16384, 128) is False
        assert _gate(128, 16384) is False


class TestBoundaryParametrized:
    """Tabular form of the boundary cases for at-a-glance review."""

    @pytest.mark.parametrize("m,n,should_snap,description", [
        (1024, 1024, True,  "asp=1.00 perfect square"),
        (1024, 1280, True,  "asp=1.25 mild skew"),
        (1024, 1433, True,  "asp=1.40 just under boundary"),
        (1024, 1500, True,  "asp~=1.465 just under boundary"),
        (1024, 1536, True,  "asp=1.50 boundary (inclusive on snap side)"),
        (1024, 1537, False, "asp~=1.501 just past threshold"),
        (1000, 1501, False, "asp=1.501 just past threshold (round)"),
        (1024, 1638, False, "asp~=1.60 in newly-affected band"),
        (1024, 1792, False, "asp=1.75 in newly-affected band"),
        (1024, 1843, False, "asp~=1.80 in newly-affected band"),
        (1024, 2048, False, "asp=2.00 (empirical winner: 256x128x64)"),
        (1024, 4096, False, "asp=4.00"),
        (1024, 8192, False, "asp=8.00 K-757 headline shape"),
    ])
    def test_threshold_boundary(self, m, n, should_snap, description):
        assert _gate(m, n) is should_snap, (
            f"asp={max(m,n)/max(min(m,n),1):.4f} ({description}) -- "
            f"expected snap={should_snap} got snap={_gate(m,n)}"
        )


class TestDegenerateInputs:
    """The defensive `max(min(M,N), 1)` divisor must keep degenerate inputs
    from raising; the gate should report 'skip' (large effective aspect)."""

    def test_zero_dim(self):
        # max(min(M,N), 1) keeps the divisor non-zero for M=0 or N=0.
        # Should not raise; effective aspect = max(M,N) -> always > 1.5.
        assert _gate(8192, 0) is False
        assert _gate(0, 8192) is False

    def test_both_zero(self):
        # max(min(0,0), 1) = 1, max(0,0) = 0 -> aspect = 0/1 = 0 <= 1.5.
        # Snap technically fires; this is OK because the rest of __init__
        # catches degenerate problems before reaching the snap.  The test
        # documents the behavior so it can't change silently.
        assert _gate(0, 0) is True

    def test_negative_dims_safe(self):
        # Should not raise (selector itself rejects negative shapes earlier
        # but the gate is defensive too).
        try:
            _gate(-1, 4096)
        except Exception as e:
            pytest.fail(f"gate raised on negative dim: {e!r}")


class TestThresholdOverride:
    """A subclass / runtime monkey-patch must propagate through the gate
    so callers can re-tune for hardware variants without forking the file."""

    def teardown_method(self, method):
        # Restore default in case any test mutated the class attribute.
        OrigamiMatmulSelector.SYMMETRIC_SNAP_ASPECT_THRESHOLD = 1.5

    def test_higher_threshold_extends_snap_zone(self):
        OrigamiMatmulSelector.SYMMETRIC_SNAP_ASPECT_THRESHOLD = 2.5
        # asp=2.0 now falls under the threshold -> snap fires.
        assert _gate(2048, 4096) is True
        # asp=8.0 still skips.
        assert _gate(8192, 1024) is False

    def test_lower_threshold_shrinks_snap_zone(self):
        OrigamiMatmulSelector.SYMMETRIC_SNAP_ASPECT_THRESHOLD = 1.1
        # asp=1.25 now skips.
        assert _gate(1024, 1280) is False
        # asp=1.05 still snaps.
        assert _gate(1000, 1050) is True
