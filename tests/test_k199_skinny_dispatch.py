"""Unit tests for K-199 medium-K skinny dispatch overrides.

Covers (per K-199 review):

* ``_is_in_cohort`` — boundary cases (K=128/512 inclusive, M/N=128 boundary,
  out-of-cohort rejection on each axis).
* ``lookup_k199_override`` — table hit, dtype mismatch miss, M/N/K mismatch
  miss, non-gfx942 arch miss, env-var kill switch, streamk-equivalent
  short-circuit (no streamk argument exists on the lookup itself; the
  selector applies the streamk bypass — see TestSelectorIntegration).
* OrigamiMatmulSelector integration — streamk=True must NOT consume any
  K-199 override even on a known-cohort shape; an LDS-overflowing override
  must silently fall back to the analytical pick rather than raising.

These tests are pure-Python except for the small set of selector-integration
tests, which require an MI300X (gfx942) GPU and origami; they self-skip on
other architectures.
"""

import importlib
import os

import pytest

from tritonblas.k199_skinny_dispatch import (
    _K199_OVERRIDES,
    _is_in_cohort,
    lookup_k199_override,
)


# ---------------------------------------------------------------------------
# Cohort predicate boundary tests (no GPU needed)
# ---------------------------------------------------------------------------


class TestIsInCohort:
    """Boundary tests for ``K in [128, 512] and (M <= 128 or N <= 128)``."""

    def test_k_lower_bound_inclusive(self):
        # K == 128 must be IN.
        assert _is_in_cohort(64, 1024, 128) is True
        assert _is_in_cohort(1024, 64, 128) is True

    def test_k_upper_bound_inclusive(self):
        # K == 512 must be IN.
        assert _is_in_cohort(64, 1024, 512) is True
        assert _is_in_cohort(1024, 64, 512) is True

    def test_k_just_below_lower_bound_excluded(self):
        # K == 127 must be OUT.
        assert _is_in_cohort(64, 1024, 127) is False

    def test_k_just_above_upper_bound_excluded(self):
        # K == 513 must be OUT.
        assert _is_in_cohort(64, 1024, 513) is False

    def test_m_boundary_inclusive(self):
        # M == 128 satisfies the skinny-M side.
        assert _is_in_cohort(128, 4096, 256) is True

    def test_n_boundary_inclusive(self):
        # N == 128 satisfies the skinny-N side.
        assert _is_in_cohort(4096, 128, 256) is True

    def test_just_above_skinny_boundary_excluded(self):
        # M == 129 AND N == 129 (both above 128) is OUT, even with K in range.
        assert _is_in_cohort(129, 129, 256) is False

    def test_skinny_via_either_axis_is_sufficient(self):
        # M small, N large (skinny-M).
        assert _is_in_cohort(16, 8192, 256) is True
        # M large, N small (skinny-N).
        assert _is_in_cohort(8192, 16, 256) is True

    def test_large_mn_in_k_range_excluded(self):
        # Both M and N >> 128 with K in range -> not skinny.
        assert _is_in_cohort(1024, 1024, 256) is False
        assert _is_in_cohort(4096, 2048, 512) is False

    def test_extreme_k_values(self):
        # K == 0 and very large K must be OUT (not [128, 512]).
        assert _is_in_cohort(64, 64, 0) is False
        assert _is_in_cohort(64, 64, 4096) is False


# ---------------------------------------------------------------------------
# Lookup hit / miss / dtype / arch / env-var tests (no GPU needed)
# ---------------------------------------------------------------------------


# Pick one entry that is guaranteed to exist in the shipped table so the
# "hit" assertion does not depend on rebalancing the table later. Validated
# at collection time rather than hard-coded.
_HIT_KEY = next(iter(_K199_OVERRIDES))  # (M, N, K, dtype_str)
_HIT_M, _HIT_N, _HIT_K, _HIT_DTYPE = _HIT_KEY


class TestLookupK199Override:
    """Table lookup hit/miss across dtypes, archs, and the env-var kill switch."""

    def setup_method(self):
        # Every test starts with the kill-switch OFF; teardown restores.
        self._prev_env = os.environ.pop("TRITONBLAS_DISABLE_K199_OVERRIDES", None)

    def teardown_method(self):
        os.environ.pop("TRITONBLAS_DISABLE_K199_OVERRIDES", None)
        if self._prev_env is not None:
            os.environ["TRITONBLAS_DISABLE_K199_OVERRIDES"] = self._prev_env

    def test_hit_returns_full_config(self):
        ovr = lookup_k199_override(_HIT_M, _HIT_N, _HIT_K, _HIT_DTYPE, "gfx942")
        assert ovr is not None, f"expected hit on swept entry {_HIT_KEY}"
        # The override must carry the six dispatch knobs the selector consumes.
        for key in ("BLK_M", "BLK_N", "BLK_K", "num_warps", "num_stages", "split_k"):
            assert key in ovr, f"override missing key {key!r}: {ovr!r}"
        # Per K-199 finding #1: every shipped winner is split_k==1.
        assert ovr["split_k"] == 1

    def test_dtype_mismatch_misses(self):
        # Same (M,N,K) but a dtype that does not appear in the table.
        # Use 'fp32' (the K-199 sweep was bf16 + fp16 only, and fp16
        # produced no kernel-level non-regressing winners).
        assert lookup_k199_override(_HIT_M, _HIT_N, _HIT_K, "fp32", "gfx942") is None

    def test_shape_miss_returns_none(self):
        # In-cohort shape that was not part of the swept set.
        assert lookup_k199_override(48, 1024, 256, "bf16", "gfx942") is None

    def test_out_of_cohort_returns_none(self):
        # Cohort predicate filters out non-skinny / out-of-K-range queries
        # BEFORE the dict lookup, so callers can never accidentally read
        # an out-of-cohort cache line.
        assert lookup_k199_override(1024, 1024, 1024, "bf16", "gfx942") is None
        assert lookup_k199_override(64, 1024, 64, "bf16", "gfx942") is None

    def test_non_gfx942_arch_returns_none(self):
        # Table was tuned on MI300X. Other archs must bypass entirely.
        assert lookup_k199_override(_HIT_M, _HIT_N, _HIT_K, _HIT_DTYPE, "gfx950") is None
        assert lookup_k199_override(_HIT_M, _HIT_N, _HIT_K, _HIT_DTYPE, "gfx90a") is None

    def test_arch_none_does_not_short_circuit(self):
        # arch_name=None means "caller did not check"; we still serve the
        # override (the cohort + (M,N,K,dtype) keying is the real filter).
        ovr = lookup_k199_override(_HIT_M, _HIT_N, _HIT_K, _HIT_DTYPE, None)
        assert ovr is not None

    def test_env_var_kill_switch_disables_lookup(self):
        os.environ["TRITONBLAS_DISABLE_K199_OVERRIDES"] = "1"
        assert lookup_k199_override(_HIT_M, _HIT_N, _HIT_K, _HIT_DTYPE, "gfx942") is None

    def test_env_var_zero_does_not_disable(self):
        # Only "1" disables; any other value (including "0") keeps the table on.
        os.environ["TRITONBLAS_DISABLE_K199_OVERRIDES"] = "0"
        assert lookup_k199_override(_HIT_M, _HIT_N, _HIT_K, _HIT_DTYPE, "gfx942") is not None


# ---------------------------------------------------------------------------
# Table integrity invariants (no GPU needed)
# ---------------------------------------------------------------------------


class TestTableIntegrity:
    """Static invariants the dispatch table must hold to ship safely."""

    def test_every_entry_in_cohort(self):
        # If any entry slipped through that does NOT satisfy the cohort
        # predicate, the lookup would never serve it -- making it dead
        # weight or, worse, a sign the predicate drifted.
        for (M, N, K, _dtype) in _K199_OVERRIDES:
            assert _is_in_cohort(M, N, K), \
                f"table entry ({M},{N},{K}) is OUT of cohort"

    def test_every_entry_is_split_k_one(self):
        # Per K-199 lesson: medium-K skinny is launch/dispatch-bound;
        # split_k > 1 lost on every swept shape. Defend against future
        # additions silently flipping that.
        for key, cfg in _K199_OVERRIDES.items():
            assert cfg.get("split_k") == 1, \
                f"non-split_k=1 entry slipped in: {key} -> {cfg}"

    def test_every_entry_uses_supported_num_stages(self):
        # The sweep covered num_stages in {2, 3}; anything else means a
        # paste error (and num_stages=1 would silently halve LDS budget).
        for key, cfg in _K199_OVERRIDES.items():
            assert cfg["num_stages"] in (2, 3), \
                f"unexpected num_stages in {key} -> {cfg}"

    def test_every_entry_uses_supported_num_warps(self):
        for key, cfg in _K199_OVERRIDES.items():
            assert cfg["num_warps"] in (4, 8), \
                f"unexpected num_warps in {key} -> {cfg}"

    def test_no_fp16_entries_after_filter(self):
        # The fp16 sweep produced no kernel-level non-regressing winners
        # (see verify_kernel_fp16.json); shipping any fp16 entry would
        # contradict the K-199 invariant called out by the Skeptic review.
        fp16_entries = [k for k in _K199_OVERRIDES if k[3] == "fp16"]
        assert fp16_entries == [], \
            f"fp16 entries must be filtered out, got: {fp16_entries}"


# ---------------------------------------------------------------------------
# Selector integration: streamk bypass + LDS fallback
# ---------------------------------------------------------------------------


# These two tests need a real GPU (origami needs hardware info) and the
# table was tuned on gfx942, so we skip on other archs.
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

origami = pytest.importorskip("origami")
_HW = origami.get_hardware_for_device(torch.cuda.current_device())
_ARCH = _HW.arch.name if hasattr(_HW, "arch") and hasattr(_HW.arch, "name") else "unknown"

_requires_gfx942 = pytest.mark.skipif(
    _ARCH != "gfx942",
    reason=f"K-199 dispatch table is gfx942-only (running on {_ARCH})",
)


@_requires_gfx942
class TestSelectorIntegration:
    """OrigamiMatmulSelector must honor the streamk bypass and LDS guard."""

    def _make_selector(self, M, N, K, *, streamk, num_stages=2, dtype=None):
        from tritonblas.origami import OrigamiMatmulSelector

        if dtype is None:
            dtype = torch.bfloat16
        return OrigamiMatmulSelector(
            m=M, n=N, k=K,
            a_dtype=dtype, b_dtype=dtype, out_dtype=dtype,
            device=torch.device("cuda", torch.cuda.current_device()),
            streamk=streamk,
            num_stages=num_stages,
        )

    def test_streamk_true_bypasses_override(self):
        # Pick a known cohort hit; with streamk=True, the override must not
        # be applied at all (the swept winners are split_k=1 / persistent).
        sel = self._make_selector(_HIT_M, _HIT_N, _HIT_K, streamk=True)
        assert sel._k199_override is None
        # And num_warps property must therefore fall through to the
        # caller-default sentinel (None == "use your own default 8").
        assert sel.num_warps is None

    def test_streamk_false_applies_override(self):
        # Same cohort hit with streamk=False must consume the override
        # tile and surface its num_warps via the selector property.
        sel = self._make_selector(_HIT_M, _HIT_N, _HIT_K, streamk=False)
        # Override may still be vetoed by the LDS guard if the entry happens
        # to be too large at num_stages=2; pick an entry whose tile fits.
        if sel._k199_override is None:
            pytest.skip("override LDS-vetoed at num_stages=2 for this entry")
        ovr = sel._k199_override
        assert sel.block_m == ovr["BLK_M"]
        assert sel.block_n == ovr["BLK_N"]
        assert sel.block_k == ovr["BLK_K"]
        assert sel.num_warps == ovr["num_warps"]

    def test_lds_overflow_falls_back_silently(self, monkeypatch):
        # The selector applies an LDS-fit guard to the override tile: if
        # the override would overflow LDS at its requested num_stages, the
        # code path silently falls back to the analytical Origami pick
        # rather than raising.
        #
        # Force this branch by patching ``lookup_k199_override`` (the
        # symbol bound inside ``tritonblas.origami``) to return an
        # absurdly oversized tile that cannot fit any LDS budget. We
        # cannot patch ``check_triton_lds_capacity`` directly because the
        # selector ALSO uses it to pre-filter its analytical config list
        # — patching that would leave no configs at all.
        import tritonblas.origami as _ori

        # 1024x1024x1024 bf16 tile at num_stages=3 is ~12 MiB of LDS,
        # which is orders of magnitude over the gfx942 64 KiB cap.
        oversized = dict(
            BLK_M=1024, BLK_N=1024, BLK_K=1024,
            num_warps=8, num_stages=3, split_k=1,
        )
        monkeypatch.setattr(
            _ori, "lookup_k199_override", lambda *a, **kw: oversized
        )

        # Use any cohort shape with streamk=False so the override branch
        # is entered; the LDS guard must short-circuit it.
        sel = self._make_selector(_HIT_M, _HIT_N, _HIT_K, streamk=False)
        # Override must have been vetoed by the LDS guard.
        assert sel._k199_override is None, (
            "LDS-overflowing override should have been silently rejected"
        )
        # Selector must still have a valid analytical tile (the fallback
        # Origami pick), so no exception, and the num_warps sentinel is
        # the "use your own default" None.
        assert sel.block_m > 0 and sel.block_n > 0 and sel.block_k > 0
        assert sel.num_warps is None
