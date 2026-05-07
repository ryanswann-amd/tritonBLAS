"""
Tests for the SPLIT_K autotune dimension added for small-M / large-K shapes.

Two layers of coverage:

1. Pure-Python unit tests for ``OrigamiMatmulSelector._compute_split_k`` —
   the gating predicate, the {1,2,4,8,16} pick, the ≥4 BLOCK_K-iter guard,
   the Stream-K bypass, and the env-var overrides (both the canonical
   ``TBLAS_*`` names and the long-form ``TRITONBLAS_*`` aliases).  These do
   not require a GPU.

2. Numerical-correctness tests that exercise the *public* API
   (``tritonblas.matmul``) on a small-M shape that should engage SPLIT_K
   and on a saturating shape that should bypass it, verifying agreement
   with ``torch.matmul`` in both cases.

Together these protect the fix against future regressions of either the
selector logic or the kernel/dispatch wiring.
"""

import os

import pytest
import torch

import tritonblas
from tritonblas.origami import OrigamiMatmulSelector


# ──────────────────────────────────────────────────────────────────────────
# 1. Unit tests for the pure decision function (no GPU required).
# ──────────────────────────────────────────────────────────────────────────

# MI300X reference: 304 CUs.  Use this for the gating threshold so the
# numbers in the tests match the production hardware target.
N_CU_MI300X = 304


def _split_k(m, n, k, bm, bn, bk, n_cu=N_CU_MI300X, streamk=False, env=None):
    return OrigamiMatmulSelector._compute_split_k(
        m, n, k, bm, bn, bk, n_cu, streamk, env=env or {}
    )


class TestSplitKGating:
    """Gate fires only when (a) ``total_tiles < N_CU`` (under-saturation)
    AND (b) ``m_tiles == 1`` (the M dimension is collapsed to a single
    row of tiles, so K-splitting is the only remaining parallelism lever).
    This is the causal M-direction under-utilization predicate — no magic
    tile-area constant; the rule keys solely on the chosen tile's
    ``ceil(M/BM)``."""

    def test_under_saturation_engages_split_k(self):
        # M=16, N=4096, BM=16, BN=256 -> m_tiles=1, n_tiles=16, total=16 << 304.
        # K=8192 / BK=64 = 128 K-tiles -> plenty of room (≥4 per shard).
        sk = _split_k(16, 4096, 8192, 16, 256, 64)
        assert sk > 1, "small-M shape (m_tiles==1) should engage SPLIT_K"
        assert sk in (1, 2, 4, 8, 16)

    def test_saturating_shape_bypasses(self):
        # 8192² with 128x128 -> 64*64 = 4096 tiles >> 304: total-tiles gate
        # fires the bypass.
        assert _split_k(8192, 8192, 4096, 128, 128, 64) == 1

    def test_square_shape_bypasses_even_when_under_saturated(self):
        # 4096³ with 256x256 tile -> m_tiles=16, n_tiles=16 (256 total < 304
        # N_CU).  The under-saturation rule alone would fire here, but the
        # M-tile gate (m_tiles=16 != 1) bypasses: the M direction already
        # supplies 16-way row parallelism, so SPLIT_K's atomic-add
        # epilogue is pure overhead.
        assert _split_k(4096, 4096, 4096, 256, 256, 64) == 1

    def test_medium_m_bypasses_even_when_under_saturated(self):
        # M=256, N=8192 with 64x128 tile -> m_tiles=4, n_tiles=64 (256 total
        # < 304 N_CU).  Under-saturated, but M direction has 4 rows of
        # tiles already — bypass to avoid the atomic-add regression
        # observed empirically on this shape.
        assert _split_k(256, 8192, 8192, 64, 128, 128) == 1

    def test_m64_with_small_bm_bypasses(self):
        # The empirical regression case: M=64 with BM=16 (Origami's actual
        # pick for M=64 N=4096..8192) -> m_tiles=4.  Causally distinct from
        # m_tiles=1 because the M direction already supplies 4-way row
        # parallelism, so K-splitting on top burns atomic bandwidth without
        # filling additional CUs.  Locked in as a regression test against
        # the previous tile-area gate that was tuned to fire here.
        assert _split_k(64, 4096, 8192, 16, 64, 256) == 1
        assert _split_k(64, 8192, 16384, 16, 128, 128) == 1

    def test_tiny_problem_bypasses_for_precision(self):
        # Numerical regression guard: M=16, N=16, K=4096 has total_tiles=1.
        # Without the MIN_TILES_FOR_SPLIT_K floor, ideal_split=304 capped
        # at 16 would fire, and 16-way bf16 atomic-add accumulation
        # exceeded the test_matmul_correctness skinny tolerance (~0.2 abs
        # error vs allowed 0.1).  The floor keeps SPLIT_K off for these
        # pathological tiny shapes where the atomic-add precision tax is
        # not amortised by enough per-output-element work.
        assert _split_k(16, 16, 4096, 16, 16, 32) == 1

    def test_at_threshold_bypasses(self):
        # total_tiles == N_CU -> not strictly less than -> SPLIT_K = 1.
        # Use bm=bn=128, m=128, n=304*128 -> 1 * 304 = 304 tiles.
        assert _split_k(128, 128 * 304, 4096, 128, 128, 64) == 1


class TestSplitKValueSelection:
    """Selected SPLIT_K is a power of 2 in {1,2,4,8,16}."""

    @pytest.mark.parametrize("k", [4096, 8192, 16384])
    def test_value_is_power_of_two_in_set(self, k):
        sk = _split_k(16, 4096, k, 16, 256, 64)
        assert sk in (1, 2, 4, 8, 16)

    def test_caps_at_16_even_when_room_for_more(self):
        # 16 output tiles with K=65536 / 64 = 1024 K-tiles -> capacity for
        # 64-way split (and for ideal-from-CU=304/16=19), but the cap
        # clamps to 16.  Use BN=256 so n_tiles=16 (and total_tiles=16,
        # safely above the MIN_TILES_FOR_SPLIT_K floor that bypasses
        # 1-tile pathological shapes).
        sk = _split_k(16, 4096, 65536, 16, 256, 64)
        assert sk == 16

    def test_under_saturation_does_not_overshoot_cu_count(self):
        # 16 tiles, 304 CUs -> ideal split = 304/16 = 19 -> floor pow2 = 16.
        # Floor-pow2 clamp guarantees we never overshoot the CU count by more
        # than 2x.
        sk = _split_k(16, 4096, 16384, 16, 256, 64)
        assert sk == 16
        assert 16 * sk <= 2 * N_CU_MI300X


class TestSplitKMinKTilesPerSplitGuard:
    """SPLIT_K is reduced when the K-loop is too short to amortise atomics."""

    def test_short_k_disables_split(self):
        # K=128 / BK=64 = 2 K-tiles total.  Even SPLIT_K=2 would give 1
        # K-tile per shard, well under the 4-tile floor -> bypass.
        assert _split_k(16, 4096, 128, 16, 256, 64) == 1

    def test_k_just_at_floor_for_split_2(self):
        # 8 K-tiles -> SPLIT_K=2 leaves 4 per shard exactly (>= 4).  Allowed.
        # But ideal_split = 304/16 = 19 -> floor pow2 = 16, then capped by
        # max_split_k_by_k = 8//4 = 2 -> final = 2.
        assert _split_k(16, 4096, 8 * 64, 16, 256, 64) == 2

    def test_medium_k_caps_to_k_budget(self):
        # 16 K-tiles -> max_split_by_k = 16//4 = 4.  Ideal still 16 -> 4.
        assert _split_k(16, 4096, 16 * 64, 16, 256, 64) == 4


class TestSplitKStreamKBypass:
    """Stream-K already partitions K, so layering SPLIT_K on top is wrong."""

    def test_streamk_always_returns_1(self):
        # Same small-M shape that would otherwise engage SPLIT_K.
        assert _split_k(16, 4096, 8192, 16, 256, 64, streamk=True) == 1


class TestSplitKEnvOverrides:
    """Env-var knobs work and are spelled consistently."""

    def test_disable_short_name(self):
        env = {"TBLAS_DISABLE_SPLIT_K": "1"}
        assert _split_k(16, 4096, 8192, 16, 256, 64, env=env) == 1

    def test_disable_long_name_alias(self):
        # Reviewer caught a real bug where benchmark scripts spelled this
        # the long way.  Both names MUST be honored or the disable knob is
        # silently a no-op.
        env = {"TRITONBLAS_DISABLE_SPLIT_K": "1"}
        assert _split_k(16, 4096, 8192, 16, 256, 64, env=env) == 1

    def test_force_short_name(self):
        env = {"TBLAS_SPLIT_K": "8"}
        # Forced even on a saturating shape that would normally bypass.
        assert _split_k(8192, 8192, 4096, 128, 128, 64, env=env) == 8

    def test_force_long_name_alias(self):
        env = {"TRITONBLAS_SPLIT_K": "4"}
        assert _split_k(8192, 8192, 4096, 128, 128, 64, env=env) == 4

    def test_force_clamps_to_legal_set(self):
        env = {"TBLAS_SPLIT_K": "7"}  # not a power of two in {1,2,4,8,16}
        assert _split_k(16, 4096, 8192, 16, 256, 64, env=env) == 1

    def test_disable_takes_precedence_over_force(self):
        env = {"TBLAS_DISABLE_SPLIT_K": "1", "TBLAS_SPLIT_K": "16"}
        assert _split_k(16, 4096, 8192, 16, 256, 64, env=env) == 1


# ──────────────────────────────────────────────────────────────────────────
# 2. End-to-end numerical correctness via the *public* tritonblas.matmul.
#    Requires a GPU.
# ──────────────────────────────────────────────────────────────────────────

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required for numerical tests"
)


@requires_cuda
class TestSplitKNumericalCorrectness:
    """Routes through ``tritonblas.matmul`` (NOT ``matmul_lt``) so the
    user-facing dispatch hook is exercised end-to-end."""

    @staticmethod
    def _run(m, n, k, dtype=torch.bfloat16):
        torch.manual_seed(0)
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype)
        out = torch.empty(m, n, device="cuda", dtype=dtype)
        tritonblas.matmul(a, b, out=out)
        ref = torch.matmul(a.float(), b.float()).to(dtype)
        return out, ref

    def test_small_m_engages_split_k_and_is_correct(self):
        # M=16, N=4096, K=8192: spec's canonical small-M / large-K shape.
        # The selector should pick SPLIT_K > 1 here, so this test guards
        # the kernel + dispatch wiring (zero_init + atomic_add reduction).
        out, ref = self._run(16, 4096, 8192)
        # bf16 GEMM tolerance.
        torch.testing.assert_close(out, ref, atol=8.0, rtol=2e-2)

    def test_large_m_no_regression(self):
        # Saturating shape: SPLIT_K should be 1 (legacy path).  Same kernel
        # tolerance.  Guards against accidentally enabling SPLIT_K for shapes
        # where the legacy path is correct.
        out, ref = self._run(2048, 2048, 4096)
        torch.testing.assert_close(out, ref, atol=8.0, rtol=2e-2)

    def test_disable_env_var_routes_through_legacy_path(self):
        """With ``TBLAS_DISABLE_SPLIT_K=1`` the dispatch must NOT route to
        the split-K kernel; numerical agreement still required."""
        os.environ["TBLAS_DISABLE_SPLIT_K"] = "1"
        try:
            out, ref = self._run(16, 4096, 8192)
            torch.testing.assert_close(out, ref, atol=8.0, rtol=2e-2)
        finally:
            os.environ.pop("TBLAS_DISABLE_SPLIT_K", None)


@requires_cuda
class TestSplitKDispatchViaPublicSelector:
    """Lock the dispatch boundary at ``OrigamiMatmulSelector`` — the same
    object the public ``tritonblas.matmul`` builds for every call.  These
    are the load-bearing dispatch assertions the previous review demanded:
    one positive (small-M engages), one negative (square shape disengages),
    plus an env-var override exercised through the selector itself.

    Without this the ``m_tiles == 1`` gate is only covered by the unit
    tests on ``_compute_split_k`` and not at the public boundary."""

    @staticmethod
    def _selector(m, n, k, dtype=torch.bfloat16):
        return OrigamiMatmulSelector(
            m, n, k, dtype, dtype, dtype, torch.device("cuda:0")
        )

    def test_small_m_selector_engages_split_k(self):
        # Positive dispatch assertion: the spec's canonical M=16 shape
        # MUST come back from the public selector with split_k > 1, or
        # the dispatch hook is not actually wired up for the target case.
        sel = self._selector(16, 4096, 8192)
        assert sel.split_k > 1, (
            f"M=16,N=4096,K=8192 should engage SPLIT_K via the public "
            f"selector dispatch path; got split_k={sel.split_k}"
        )
        assert sel.split_k in (2, 4, 8, 16)

    def test_square_shape_selector_disengages_split_k(self):
        # Negative dispatch assertion: a well-shaped square GEMM where
        # m_tiles > 1 MUST come back with split_k == 1.  Locks in the
        # tile-aware gate (formerly the magic-number area gate the
        # previous reviewer flagged) at the public boundary.
        sel = self._selector(4096, 4096, 4096)
        assert sel.split_k == 1, (
            f"4096^3 square shape must NOT engage SPLIT_K via the public "
            f"selector dispatch path; got split_k={sel.split_k}"
        )

    def test_disable_env_routes_dispatch_through_legacy(self):
        # The env-var disable knob must be honored at the dispatch
        # boundary too — not just inside ``_compute_split_k`` — or the
        # ablation/A-B benchmark workflow regresses to a no-op.
        os.environ["TBLAS_DISABLE_SPLIT_K"] = "1"
        try:
            sel = self._selector(16, 4096, 8192)
            assert sel.split_k == 1, (
                "TBLAS_DISABLE_SPLIT_K=1 must force selector.split_k == 1"
            )
        finally:
            os.environ.pop("TBLAS_DISABLE_SPLIT_K", None)
