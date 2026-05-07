"""Unit tests for the large-skinny rectangle classifier and the
stream-K recommendation heuristic.

These tests pin the pure-Python decision logic (no GPU required) used by
``tritonblas.matmul`` to auto-enable the persistent stream-K kernel on
shapes with a meaningful last-wave imbalance, plus the informational
``is_large_skinny_shape`` classifier that flags rectangular GEMMs in the
difficult performance regime where last-wave imbalance dominates.

Empirical thresholds (validated on MI300X / gfx942) are documented in
``include/tritonblas/origami.py``.

The tests deliberately also cover:

  * Override paths: explicit ``True`` / ``False`` / ``"auto"`` /
    ``None`` arguments to ``_resolve_enable_streamk``.
  * Negative cases: shapes that should NOT trigger stream-K despite a
    large M*N (small squares, perfectly-aligned grids, very-many-wave
    cases).
  * Error cases: bad argument types raise ``TypeError`` instead of
    silently coercing to a bool.
  * Regression guards: pin that the small-square shapes used in the
    existing regression band do NOT switch kernels under auto mode
    (the dispatch policy must not flip them silently).
"""

import pytest

from tritonblas.origami import (
    is_large_skinny_shape,
    recommend_streamk,
    LARGE_SKINNY_LONG_DIM_MIN,
    LARGE_SKINNY_ASPECT_MIN,
    LARGE_SKINNY_K_MIN,
    STREAMK_MIN_K,
    STREAMK_MAX_WAVES,
    STREAMK_LAST_WAVE_FRAC_MAX,
)


class TestIsLargeSkinny:
    """``is_large_skinny_shape`` is an informational classifier — used to
    record / log when a shape is in the difficult regime where last-wave
    imbalance dominates.  It is a *necessary* (not sufficient) condition
    for the stream-K recommendation to fire on these shapes."""

    def test_target_shapes_qualify(self):
        # Tall-skinny target (1024 x 8192, K=8192) — qualifies
        assert is_large_skinny_shape(1024, 8192, 8192)
        # Wide-skinny target (8192 x 2048, K=4096) — qualifies
        assert is_large_skinny_shape(8192, 2048, 4096)

    def test_near_rectangle_does_not_qualify(self):
        # 6144x4096x4096 is only 1.5:1 aspect, below the 2:1 cutoff.  It
        # still benefits from stream-K (see recommend_streamk) but is
        # *not* "skinny" enough for the informational classifier.
        # Documents the intended split: stream-K recommendation and
        # large-skinny classification are independent.
        assert not is_large_skinny_shape(6144, 4096, 4096)

    def test_square_shapes_do_not_qualify(self):
        assert not is_large_skinny_shape(4096, 4096, 4096)
        assert not is_large_skinny_shape(8192, 8192, 8192)

    def test_short_k_does_not_qualify(self):
        # Even with extreme aspect ratio, a short K means the K-loop is
        # not the bottleneck and tile-shape changes don't pay off.
        assert not is_large_skinny_shape(8192, 1024, 256)

    def test_short_long_dim_does_not_qualify(self):
        # max(M, N) below LARGE_SKINNY_LONG_DIM_MIN: too small to amortise.
        assert not is_large_skinny_shape(2048, 1024, 4096)
        assert not is_large_skinny_shape(1024, 2048, 4096)

    def test_aspect_just_below_cutoff(self):
        # 4096:2400 = 1.71 < 2.0 -> should NOT qualify
        assert not is_large_skinny_shape(4096, 2400, 4096)

    def test_aspect_at_cutoff(self):
        # 4096:2048 = 2.0 -> qualifies (boundary inclusive)
        assert is_large_skinny_shape(4096, 2048, 4096)

    def test_zero_short_dim_does_not_crash(self):
        # Pathological input: short dim 0 must not cause divide-by-zero
        # when computing aspect ratio.  The classifier should reject it
        # cleanly via the K-min / long-dim checks (degenerate shape).
        assert not is_large_skinny_shape(8192, 0, 4096)

    def test_constants_documented(self):
        # Pinning the public threshold names so consumers can see/override
        # them through the module API.
        assert LARGE_SKINNY_LONG_DIM_MIN == 4096
        assert LARGE_SKINNY_ASPECT_MIN == 2.0
        assert LARGE_SKINNY_K_MIN == 2048


class TestRecommendStreamK:
    """``recommend_streamk`` gates auto-enabling the persistent stream-K
    kernel in ``tritonblas.matmul`` / ``tritonblas.addmm``."""

    N_CU = 304  # MI300X

    def test_imbalanced_last_wave_recommends(self):
        """Stream-K wins on shapes with a small partial last wave."""
        # 6144x4096x4096: tiles=24*16=384, waves=1.26, last_wave=80
        # 80 << 0.6 * 304 = 182  ->  True  (validated +9% empirical)
        assert recommend_streamk(6144, 4096, 4096, self.N_CU)

        # 8192x8192x8192: tiles=32*32=1024, waves=3.37, last_wave=112
        # 112 < 182  ->  True  (validated +5% empirical)
        assert recommend_streamk(8192, 8192, 8192, self.N_CU)

    def test_sub_wave_does_not_recommend(self):
        """Shapes with total_tiles < n_cu: K-split overhead beats the gain."""
        # 1024x8192x8192: tiles=4*32=128 < 304  ->  False
        assert not recommend_streamk(1024, 8192, 8192, self.N_CU)
        # 8192x2048x4096: tiles=32*8=256 < 304  ->  False
        assert not recommend_streamk(8192, 2048, 4096, self.N_CU)

    def test_perfect_alignment_does_not_recommend(self):
        # If total_tiles % n_cu == 0, stream-K provides no last-wave fixup.
        # Construct: n_cu * 256 / 256 = n_cu tiles -> 1 full wave, no remainder
        assert not recommend_streamk(304 * 256, 256, 4096, self.N_CU)

    def test_too_many_waves_does_not_recommend(self):
        # waves > STREAMK_MAX_WAVES -> last-wave imbalance is a small
        # fraction of total runtime, stream-K reduction overhead dominates.
        # 16384x16384 with 256 tile: 64*64=4096 tiles, waves=13.5  ->  False
        assert not recommend_streamk(16384, 16384, 4096, self.N_CU)

    def test_short_k_does_not_recommend(self):
        # K < STREAMK_MIN_K: not enough K to absorb stream-K reduction overhead.
        assert not recommend_streamk(6144, 4096, 512, self.N_CU)

    def test_small_shape_does_not_recommend(self):
        # tiles=4, well below n_cu=304  ->  False
        assert not recommend_streamk(1024, 1024, 1024, self.N_CU)

    def test_large_partial_wave_does_not_recommend(self):
        # last_wave / n_cu close to (but above) STREAMK_LAST_WAVE_FRAC_MAX
        # means most CUs are already busy in the last wave; the stream-K
        # reduction cost outweighs the small idle-CU savings.
        # Synthetic: total = 304 + 200 = 504 tiles
        # waves=1.66, last_wave=200, 200 >= int(304*0.6)=182 -> False
        # Achieve 504 tiles via M=504*256, N=256
        assert not recommend_streamk(504 * 256, 256, 4096, self.N_CU)

    def test_streamk_constants_documented(self):
        # Pinning thresholds so a downstream change is intentional, not silent.
        assert STREAMK_MIN_K == 1024
        assert STREAMK_MAX_WAVES == 4.5
        assert STREAMK_LAST_WAVE_FRAC_MAX == 0.6


class TestRegressionGuards:
    """Regression guards: the existing 0.80-0.95 ratio band squares used
    by the regression set must NOT silently switch kernel under
    auto-mode dispatch.  These pin the negative side of the dispatch
    policy: it is the *quietness* of the policy on previously-good
    shapes that prevents downstream surprises."""

    N_CU = 304  # MI300X

    @pytest.mark.parametrize("dim", [1024, 2048, 4096])
    def test_small_square_not_recommended(self, dim):
        # Small squares: total_tiles is well below n_cu, so the
        # ``total_tiles < n_cu`` guard kicks in and the policy returns
        # False.  This is the silent-flip guard the reviewers asked for.
        assert not recommend_streamk(dim, dim, dim, self.N_CU)

    def test_perfect_square_8k_no_silent_flip_for_short_k(self):
        # 8192x8192 with K=512: tiles=1024 (>=304), but K<STREAMK_MIN_K
        # so the policy refuses.  Pins that "lots of tiles" alone is
        # not enough to flip the kernel.
        assert not recommend_streamk(8192, 8192, 512, self.N_CU)

    def test_imbalanced_recommendation_does_not_apply_to_balanced_grid(self):
        # Even when M*N is huge, if the grid divides evenly into the CU
        # count there is no last-wave imbalance to exploit.  Construct a
        # tile grid that is exactly n_cu * 4 = 1216 tiles, no remainder.
        # M=304*4*256=311296, N=256 -> tiles=1216, waves=4, last=0 -> False
        assert not recommend_streamk(304 * 4 * 256, 256, 4096, self.N_CU)


class TestResolveEnableStreamk:
    """``_resolve_enable_streamk`` translates the public
    ``enable_streamk`` argument of ``tritonblas.matmul`` into a concrete
    boolean.  Imported lazily because this module needs MAX_SMS to exist
    on the GPU at import time (skip when no CUDA)."""

    def _get_resolver(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA required to import tritonblas.matmul")
        from tritonblas.matmul import _resolve_enable_streamk
        return _resolve_enable_streamk

    def test_explicit_true_passes_through(self):
        resolve = self._get_resolver()
        # Explicit True is honoured even on a shape the heuristic would reject.
        assert resolve(True, 1024, 1024, 1024) is True

    def test_explicit_false_passes_through(self):
        resolve = self._get_resolver()
        # Explicit False is honoured even on a shape the heuristic would accept.
        assert resolve(False, 8192, 8192, 8192) is False

    def test_none_uses_heuristic(self):
        # None is the back-compat alias for "auto" — both must consult
        # the heuristic and both must agree.
        resolve = self._get_resolver()
        assert isinstance(resolve(None, 6144, 4096, 4096), bool)

    def test_auto_string_uses_heuristic(self):
        # The new explicit "auto" sentinel is the documented default.
        resolve = self._get_resolver()
        assert isinstance(resolve("auto", 6144, 4096, 4096), bool)

    def test_auto_and_none_agree(self):
        # The two ways to opt in to auto mode must produce the same
        # boolean for a given shape, otherwise we have two policies.
        resolve = self._get_resolver()
        for m, n, k in [
            (1024, 1024, 1024),    # small square - False
            (8192, 8192, 8192),    # imbalanced - True
            (1024, 8192, 8192),    # sub-wave - False
            (6144, 4096, 4096),    # imbalanced rectangle - True
        ]:
            assert resolve("auto", m, n, k) == resolve(None, m, n, k)

    def test_auto_does_not_flip_small_squares(self):
        # Concrete regression guard: the small-square shapes that the
        # regression band cares about must come back as False under
        # auto mode, so the auto policy does not silently swap the
        # kernel on them.
        resolve = self._get_resolver()
        for dim in (1024, 2048, 4096):
            assert resolve("auto", dim, dim, dim) is False

    def test_invalid_string_raises(self):
        # A typo like "true"/"on" must NOT silently coerce to True.
        resolve = self._get_resolver()
        with pytest.raises(TypeError):
            resolve("true", 1024, 1024, 1024)
        with pytest.raises(TypeError):
            resolve("on", 1024, 1024, 1024)

    def test_invalid_int_raises(self):
        # Bools-via-int-coercion (0 / 1) must NOT silently coerce.
        # Python's ``bool`` is a subclass of ``int``, so a sloppy check
        # would accept these — the resolver explicitly rejects via
        # ``is True`` / ``is False`` identity checks.
        resolve = self._get_resolver()
        with pytest.raises(TypeError):
            resolve(0, 1024, 1024, 1024)
        with pytest.raises(TypeError):
            resolve(1, 1024, 1024, 1024)


class TestPublicDefaults:
    """The public ``matmul`` / ``addmm`` defaults are part of the API
    contract.  These tests pin them so a downstream change is
    intentional and reviewable, not silent."""

    def test_matmul_and_addmm_default_is_auto(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA required to import tritonblas.matmul")
        import inspect
        from tritonblas.matmul import matmul, addmm

        for fn in (matmul, addmm):
            sig = inspect.signature(fn)
            assert sig.parameters["enable_streamk"].default == "auto", (
                f"{fn.__name__} default must be the explicit 'auto' "
                f"sentinel, not None or False"
            )
