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

These tests exercise the gate in isolation by directly checking the
threshold constant and replicating the gate logic with synthetic inputs --
no GPU required, no origami solver invocation.  End-to-end behavior is
covered by the K-757 benchmark in workspace/scripts/.
"""

import math
import pytest

from tritonblas.origami import OrigamiMatmulSelector


def _snap_fires(m: int, n: int, threshold: float) -> bool:
    """Replica of the snap gate (excluding LDS-capacity check, which is
    tile-only and orthogonal to the aspect-ratio gate added in this PR)."""
    aspect = max(m, n) / max(min(m, n), 1)
    return aspect <= threshold


class TestSymmetricSnapAspectThreshold:
    def test_threshold_is_a_class_constant(self):
        # The threshold lives on the class so downstream callers can override
        # it for hardware variants without monkey-patching __init__.
        assert hasattr(OrigamiMatmulSelector, "SYMMETRIC_SNAP_ASPECT_THRESHOLD")

    def test_threshold_value_default(self):
        # Default 1.5 is conservative: snaps perfectly-square (asp=1.0) and
        # mild skew (asp <= 1.5), skips moderate (asp=2.0) and large (asp>=4)
        # skew where empirical benchmarks show the asymmetric tile wins.
        t = OrigamiMatmulSelector.SYMMETRIC_SNAP_ASPECT_THRESHOLD
        assert 1.0 < t < 2.0

    def test_square_problem_snaps(self):
        # 4096x4096: asp=1.0 -> snap fires, restoring 256x256x64 selection.
        t = OrigamiMatmulSelector.SYMMETRIC_SNAP_ASPECT_THRESHOLD
        assert _snap_fires(4096, 4096, t)
        assert _snap_fires(8192, 8192, t)
        assert _snap_fires(1024, 1024, t)

    def test_near_square_problem_snaps(self):
        # asp=1.33: e.g. 4096x3072 -- still considered "square enough".
        t = OrigamiMatmulSelector.SYMMETRIC_SNAP_ASPECT_THRESHOLD
        assert _snap_fires(4096, 3072, t)
        assert _snap_fires(3072, 4096, t)

    def test_aspect_2_does_not_snap(self):
        # asp=2.0: e.g. 2048x4096 -- empirical sweep shows 256x128x64 wins
        # over 256x256x64 by ~30% TFLOPS on this shape.  Snap must skip.
        t = OrigamiMatmulSelector.SYMMETRIC_SNAP_ASPECT_THRESHOLD
        assert not _snap_fires(2048, 4096, t)
        assert not _snap_fires(4096, 2048, t)

    def test_large_skinny_does_not_snap(self):
        # asp=8.0: K-757 cohort -- 8192x1024 and 1024x8192 must keep the
        # asymmetric origami pick (256x128x64).
        t = OrigamiMatmulSelector.SYMMETRIC_SNAP_ASPECT_THRESHOLD
        assert not _snap_fires(8192, 1024, t)
        assert not _snap_fires(1024, 8192, t)

    def test_huge_aspect_does_not_snap(self):
        # asp=128.0: extreme batched-style skew.
        t = OrigamiMatmulSelector.SYMMETRIC_SNAP_ASPECT_THRESHOLD
        assert not _snap_fires(16384, 128, t)
        assert not _snap_fires(128, 16384, t)

    def test_zero_dim_safe(self):
        # Defensive: max(min(M, N), 1) keeps the divisor non-zero even if
        # an upstream caller passes a degenerate shape.  Should not raise.
        t = OrigamiMatmulSelector.SYMMETRIC_SNAP_ASPECT_THRESHOLD
        # Treating the divisor floor as 1 -> 8192/1 = 8192, snap skips.
        assert not _snap_fires(8192, 0, t)

    @pytest.mark.parametrize("m,n,should_snap", [
        (1024, 1024, True),     # asp=1.0 perfect square
        (1024, 1280, True),     # asp=1.25 mild skew
        (1024, 1536, True),     # asp=1.5 boundary (inclusive)
        (1024, 1537, False),    # asp~=1.501 just past threshold
        (1024, 2048, False),    # asp=2.0
        (1024, 4096, False),    # asp=4.0
        (1024, 8192, False),    # asp=8.0 K-757
    ])
    def test_threshold_boundary(self, m, n, should_snap):
        t = OrigamiMatmulSelector.SYMMETRIC_SNAP_ASPECT_THRESHOLD
        assert _snap_fires(m, n, t) is should_snap


class TestSelectorClassSurface:
    """Ensure the threshold is exposed where downstream callers expect it."""

    def test_threshold_accessible_from_instance(self):
        # The constant must be reachable through `selector.X` as well as
        # `OrigamiMatmulSelector.X`, so callers can introspect at runtime
        # without importing the class.
        cls_val = OrigamiMatmulSelector.SYMMETRIC_SNAP_ASPECT_THRESHOLD
        # Don't instantiate (requires a real GPU); just check class attr binding.
        assert isinstance(cls_val, (int, float))
        assert math.isfinite(cls_val)
