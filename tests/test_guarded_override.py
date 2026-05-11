"""K-883/K-901 -- guarded-override harness gating-logic tests (CPU-only).

These tests exercise every layer (L1-L9) of the K-883 9-layer guarded-override
pattern WITHOUT touching CUDA, plus a synthetic LAND-path integration test
that proves the harness can actually approve a valid override (so a bug that
makes the harness reject everything would be caught).

The GPU paired-bench code path is exercised separately by
``scripts/run_falsification_k882.py`` on c42/MI300X.

Test names map 1:1 to the K-883 §5 / §3 layer rules so a CI failure points
at the broken layer.
"""
import math
import os
import threading

import pytest

torch = pytest.importorskip("torch")
from tritonblas.guarded_override import (  # noqa: E402
    CellResult,
    FalsificationVerdict,
    GuardedOverride,
    GuardedOverride as GO,
    OverrideRegistry,
    _classify_delta,
    _compute_verdict,
    _paired_delta_with_ci,
    _t_crit_95,
    apply_override_in_dispatcher,
    lds_budget_for,
    set_arch_lds_budget,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_gate(env="TB_TEST_ENABLE", ticket="K-TEST",
               keys=None, dtype_allowlist=(torch.float16, torch.bfloat16),
               override_fn=None):
    if keys is None:
        keys = [(2048, 2048, 4096, torch.float16),
                (2048, 2048, 8192, torch.bfloat16)]
    return GuardedOverride(
        ticket=ticket, env_var=env, cohort_keys=keys,
        dtype_allowlist=dtype_allowlist,
        override_fn=override_fn or (lambda M, N, K, dt: {"num_stages": 2}),
    )


def _k882_fixture():
    """The K-882 grid-cap prototype, defined here as a TEST FIXTURE rather
    than as a production override.  K-882's verdict is NO-LAND, so it must
    not ship in the package; keeping it here exercises the harness's
    reject-path (regression test that the harness keeps catching this
    structural-no-op pattern)."""
    return GuardedOverride(
        ticket="K-882", env_var="TB_K882_ENABLE",
        cohort_keys=[
            (2048, 2048, 4096, torch.float16),
            (2048, 2048, 4096, torch.bfloat16),
            (2048, 2048, 8192, torch.float16),
            (2048, 2048, 8192, torch.bfloat16),
            (2048, 2048, 16384, torch.float16),
            (2048, 2048, 16384, torch.bfloat16),
        ],
        dtype_allowlist=(torch.float16, torch.bfloat16),
        override_fn=lambda M, N, K, dt: {"grid_cap_to_n_cu": True, "num_stages": 2},
    )


def setup_function(_):
    OverrideRegistry.unregister("K-TEST")
    OverrideRegistry.unregister("K-TEST-2")
    OverrideRegistry.unregister("K-TEST-LAND")
    OverrideRegistry.unregister("K-882")
    for env in ("TB_TEST_ENABLE", "TB_TEST_2_ENABLE", "TB_TEST_LAND_ENABLE",
                "TB_K882_ENABLE"):
        os.environ.pop(env, None)


# ---------------------------------------------------------------------------
# L1 -- strict-equality dispatch table (R4 range-predicate ban)
# ---------------------------------------------------------------------------
def test_l1_strict_equality_no_range_leak():
    g = _make_gate()
    os.environ["TB_TEST_ENABLE"] = "1"
    assert g.lookup(2048, 2048, 4096, torch.float16) is not None
    # Off-by-one on each dimension: must NOT fire (no range bleed).
    assert g.lookup(2048, 2048, 4097, torch.float16) is None
    assert g.lookup(2048, 2049, 4096, torch.float16) is None
    assert g.lookup(2049, 2048, 4096, torch.float16) is None


def test_l1_5tuple_keys_match_with_batch():
    """L1: 5-tuple cohort keys (M,N,K,dt,batch) match only on exact batch."""
    g = _make_gate(keys=[(2048, 2048, 4096, torch.float16, 4)])
    os.environ["TB_TEST_ENABLE"] = "1"
    # Right batch fires.
    assert g.lookup(2048, 2048, 4096, torch.float16, batch=4) == {"num_stages": 2}
    # Wrong batch must not fire.
    assert g.lookup(2048, 2048, 4096, torch.float16, batch=8) is None
    assert g.lookup(2048, 2048, 4096, torch.float16, batch=1) is None


# ---------------------------------------------------------------------------
# L2 -- dtype allowlist guard
# ---------------------------------------------------------------------------
def test_l2_dtype_allowlist_blocks_disallowed_dtype():
    g = _make_gate(dtype_allowlist=(torch.float16,))  # bf16 NOT allowed
    os.environ["TB_TEST_ENABLE"] = "1"
    assert g.lookup(2048, 2048, 4096, torch.float16) is not None
    # bf16 with a key that exists in cohort_keys but NOT in allowlist:
    assert g.lookup(2048, 2048, 8192, torch.bfloat16) is None


def test_l2_dtype_allowlist_blocks_fp32_even_for_matched_shape():
    """L2: fp32 must be blocked even when (M,N,K) is in the cohort table."""
    g = _make_gate()
    os.environ["TB_TEST_ENABLE"] = "1"
    assert g.lookup(2048, 2048, 4096, torch.float32) is None


# ---------------------------------------------------------------------------
# L3 -- env-var killswitch read every call (no lru_cache)
# ---------------------------------------------------------------------------
def test_l3_env_killswitch_off_by_default():
    """L3: every override is OFF unless its env var is explicitly set."""
    g = _make_gate()
    assert g.lookup(2048, 2048, 4096, torch.float16) is None


def test_l3_env_killswitch_on_when_env_set():
    g = _make_gate()
    os.environ["TB_TEST_ENABLE"] = "1"
    assert g.lookup(2048, 2048, 4096, torch.float16) == {"num_stages": 2}


def test_l3_env_killswitch_recognizes_truthy_synonyms():
    g = _make_gate()
    for v in ("1", "true", "TRUE", "yes", "Yes"):
        os.environ["TB_TEST_ENABLE"] = v
        assert g.lookup(2048, 2048, 4096, torch.float16) is not None, v


def test_l3_env_killswitch_rejects_falsey_strings():
    g = _make_gate()
    for v in ("0", "false", "no", "off", ""):
        os.environ["TB_TEST_ENABLE"] = v
        assert g.lookup(2048, 2048, 4096, torch.float16) is None, v


def test_l3_env_killswitch_rechecked_per_call():
    """L3: lru_cache must NOT have memoized the env state; flipping the
    env mid-test changes the next call's result without restart."""
    g = _make_gate()
    os.environ["TB_TEST_ENABLE"] = "1"
    assert g.lookup(2048, 2048, 4096, torch.float16) is not None
    os.environ.pop("TB_TEST_ENABLE", None)
    assert g.lookup(2048, 2048, 4096, torch.float16) is None


# ---------------------------------------------------------------------------
# L4 -- LDS-budget guard at call site
# ---------------------------------------------------------------------------
def test_l4_lds_ok_helper_at_budget_passes():
    # 2 * (128+128) * 64 * 2 = 65536 -> exactly at gfx942 budget
    assert GO.lds_ok(128, 128, 64, num_stages=2, bytes_per_elem=2, arch="gfx942")


def test_l4_lds_ok_helper_over_budget_fails():
    # +1 stage breaks the budget on gfx942
    assert not GO.lds_ok(128, 128, 64, num_stages=3, bytes_per_elem=2, arch="gfx942")


def test_l4_lds_guard_blocks_oversize_tile_via_dispatcher():
    """L4: dispatcher shim refuses to apply an LDS-overstuffing override."""
    g = GuardedOverride(
        ticket="K-TEST", env_var="TB_TEST_ENABLE",
        cohort_keys=[(2048, 2048, 4096, torch.float16)],
        override_fn=lambda M, N, K, dt: {
            "BLOCK_SIZE_M": 1024, "BLOCK_SIZE_N": 1024, "BLOCK_SIZE_K": 1024,
            "num_stages": 4,
        },
    )
    OverrideRegistry.register(g)
    os.environ["TB_TEST_ENABLE"] = "1"
    out = apply_override_in_dispatcher(
        2048, 2048, 4096, torch.float16,
        block_m=128, block_n=128, block_k=64,
        num_stages=2, num_warps=8, waves_per_eu=0,
        kpack=1, mfma_instr_size=16,
        total_tiles=256, n_cu=304,
        bytes_per_elem=2, arch="gfx942",
    )
    assert out.get("lds_blocked") is True


def test_l4_lds_budget_overridable_per_arch():
    """L4: arch-specific budgets can be set/queried."""
    set_arch_lds_budget("gfx-test-arch", 32768)
    assert lds_budget_for("gfx-test-arch") == 32768
    # Default budget kicks in for unknown arch (default is non-zero).
    assert lds_budget_for("gfx-unknown-99") > 0


# ---------------------------------------------------------------------------
# L5 -- composability guard (refuses under work_stealing by default)
# ---------------------------------------------------------------------------
def test_l5_composability_guard_refuses_under_work_stealing():
    g = _make_gate()
    OverrideRegistry.register(g)
    os.environ["TB_TEST_ENABLE"] = "1"
    out = apply_override_in_dispatcher(
        2048, 2048, 4096, torch.float16,
        block_m=128, block_n=128, block_k=64,
        num_stages=4, num_warps=8, waves_per_eu=0,
        kpack=1, mfma_instr_size=16,
        total_tiles=256, n_cu=304,
        work_stealing=True,
    )
    assert out == {}, "L5: work_stealing must suppress the gate"


def test_l5_composability_guard_fires_when_no_work_stealing():
    g = _make_gate()
    OverrideRegistry.register(g)
    os.environ["TB_TEST_ENABLE"] = "1"
    out = apply_override_in_dispatcher(
        2048, 2048, 4096, torch.float16,
        block_m=128, block_n=128, block_k=64,
        num_stages=4, num_warps=8, waves_per_eu=0,
        kpack=1, mfma_instr_size=16,
        total_tiles=256, n_cu=304,
        work_stealing=False,
    )
    assert out.get("fired_ticket") == "K-TEST"


def test_l5_composability_guard_can_be_overridden_explicitly():
    """L5: gate authors can opt in to work-stealing-safe behavior with a
    custom composability_guard (e.g. lambda ws: True)."""
    g = GuardedOverride(
        ticket="K-TEST", env_var="TB_TEST_ENABLE",
        cohort_keys=[(2048, 2048, 4096, torch.float16)],
        override_fn=lambda M, N, K, dt: {"num_stages": 2},
        composability_guard=lambda ws: True,
    )
    OverrideRegistry.register(g)
    os.environ["TB_TEST_ENABLE"] = "1"
    out = apply_override_in_dispatcher(
        2048, 2048, 4096, torch.float16,
        block_m=128, block_n=128, block_k=64,
        num_stages=4, num_warps=8, waves_per_eu=0,
        kpack=1, mfma_instr_size=16,
        total_tiles=256, n_cu=304,
        work_stealing=True,
    )
    assert out.get("fired_ticket") == "K-TEST"


# ---------------------------------------------------------------------------
# L6 -- routing-trace counters (per-key + thread-safe)
# ---------------------------------------------------------------------------
def test_l6_routing_counters_track_fires_and_nonfires():
    g = _make_gate()
    os.environ["TB_TEST_ENABLE"] = "1"
    g.lookup(2048, 2048, 4096, torch.float16)  # fire
    g.lookup(2048, 2048, 4096, torch.float16)  # fire
    g.lookup(2048, 2048, 9999, torch.float16)  # nonfire (OOC)
    g.lookup(2048, 2048, 4096, torch.float64)  # nonfire (dtype out of allowlist)
    snap = g.routing_snapshot()
    assert snap["fire_total"] == 2
    assert snap["nonfire_total"] == 2
    assert snap["per_key_fires"][(2048, 2048, 4096, torch.float16)] == 2


def test_l6_routing_counters_thread_safe():
    g = _make_gate()
    os.environ["TB_TEST_ENABLE"] = "1"
    N = 200

    def worker():
        for _ in range(N):
            g.lookup(2048, 2048, 4096, torch.float16)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    snap = g.routing_snapshot()
    assert snap["fire_total"] == 8 * N


def test_l6_routing_counters_resettable():
    g = _make_gate()
    os.environ["TB_TEST_ENABLE"] = "1"
    g.lookup(2048, 2048, 4096, torch.float16)
    g.reset_counters()
    assert g.routing_snapshot()["fire_total"] == 0


# ---------------------------------------------------------------------------
# L7 -- cohort-size cap (K-883 §5 C3)
# ---------------------------------------------------------------------------
def test_l7_cohort_size_cap_rejects_oversize_table():
    keys = [(i, 2048, 4096, torch.float16) for i in range(40)]
    with pytest.raises(ValueError, match="cohort_size"):
        GuardedOverride(
            ticket="K-TEST", env_var="TB_TEST_ENABLE",
            cohort_keys=keys, override_fn=lambda *a: {},
        )


def test_l7_cohort_size_cap_can_be_raised_with_explicit_justification():
    """L7: authors can raise the cap, but must do so explicitly."""
    keys = [(i, 2048, 4096, torch.float16) for i in range(40)]
    g = GuardedOverride(
        ticket="K-TEST", env_var="TB_TEST_ENABLE",
        cohort_keys=keys, override_fn=lambda *a: {}, cohort_size_cap=50,
    )
    assert len(g.cohort_keys) == 40


# ---------------------------------------------------------------------------
# L8 -- registry idempotence + R1 dispatch-order routing
# ---------------------------------------------------------------------------
def test_l8_registry_refuses_duplicate_ticket():
    g1 = _make_gate(ticket="K-TEST")
    g2 = _make_gate(ticket="K-TEST")
    a = OverrideRegistry.register(g1)
    b = OverrideRegistry.register(g2)
    # idempotent: second register returns the original.
    assert a is g1
    assert b is g1


def test_l8_r1_dispatch_order_first_routing():
    """R1: registry returns the first matching gate in registration order."""
    g1 = _make_gate(ticket="K-TEST", env="TB_TEST_ENABLE",
                    keys=[(2048, 2048, 4096, torch.float16)])
    g1.override_fn = lambda M, N, K, dt: {"num_stages": 2}
    g2 = _make_gate(ticket="K-TEST-2", env="TB_TEST_2_ENABLE",
                    keys=[(2048, 2048, 4096, torch.float16)])
    g2.override_fn = lambda M, N, K, dt: {"num_stages": 4}
    OverrideRegistry.register(g1)
    OverrideRegistry.register(g2)
    os.environ["TB_TEST_ENABLE"] = "1"
    os.environ["TB_TEST_2_ENABLE"] = "1"
    hit = OverrideRegistry.apply(2048, 2048, 4096, torch.float16)
    assert hit is not None
    gate, payload = hit
    assert gate.ticket == "K-TEST"
    assert payload["num_stages"] == 2  # g1 wins, NOT g2


# ---------------------------------------------------------------------------
# L9 -- paired CI (paired_t, n>=30 paired samples, Student-t)
# ---------------------------------------------------------------------------
def test_l9_t_crit_95_returns_at_least_normal_value():
    """L9: 95% CI multiplier is monotonic and converges to 1.96 for large df."""
    assert _t_crit_95(1) > _t_crit_95(30)
    assert _t_crit_95(30) >= 1.96
    assert _t_crit_95(120) >= 1.96
    # df<=0 returns NaN (no CI possible).
    assert math.isnan(_t_crit_95(0))


def test_l9_paired_delta_zero_when_inputs_equal():
    on  = [1.0] * 32
    off = [1.0] * 32
    mean, lo, hi, n = _paired_delta_with_ci(on, off)
    assert n == 32
    assert mean == 0.0
    # CI must straddle 0 for identical inputs (sd=0 -> CI=[0,0]).
    assert lo <= 0.0 <= hi


def test_l9_paired_delta_positive_when_on_faster():
    """On a clean +10% speedup signal the CI must exclude 0 with n=30."""
    on  = [0.90] * 30
    off = [1.00] * 30
    mean, lo, hi, n = _paired_delta_with_ci(on, off)
    assert n == 30
    assert mean == pytest.approx(10.0, abs=1e-6)
    # No variance -> CI collapses to mean.
    assert lo == pytest.approx(10.0, abs=1e-6)
    assert hi == pytest.approx(10.0, abs=1e-6)


def test_l9_paired_delta_ci_excludes_zero_when_signal_clear_of_noise():
    """A real +5% signal under Gaussian noise must exclude 0 from the CI."""
    import random
    random.seed(0)
    off = [1.0 + random.gauss(0, 0.005) for _ in range(50)]
    on  = [v * 0.95 + random.gauss(0, 0.005) for v in off]
    mean, lo, hi, n = _paired_delta_with_ci(on, off)
    assert n == 50
    assert mean > 4.0      # ~5% speedup
    assert lo > 0.0        # CI strictly positive: signal beats noise
    assert hi > mean


# ---------------------------------------------------------------------------
# Synthetic LAND-path integration test (proves the harness can APPROVE)
# ---------------------------------------------------------------------------
def _synthetic_cell(M, K, dtype, in_cohort, off_ms, on_ms, ref_ms,
                    on_vs_off_pct, on_vs_off_ci=(None, None), fired=False,
                    classification=None):
    n = 30
    if classification is None:
        classification = "SPEEDUP" if on_vs_off_pct >= 5.0 else (
            "REGRESS" if on_vs_off_pct <= -5.0 else "IN_BAND")
    lo, hi = on_vs_off_ci if on_vs_off_ci != (None, None) else (
        on_vs_off_pct - 0.5, on_vs_off_pct + 0.5)
    return CellResult(
        M=M, N=2048, K=K, dtype=str(dtype).replace("torch.", ""), batch=1,
        in_cohort=in_cohort,
        ms_off_med=off_ms, ms_on_med=on_ms, ms_ref_med=ref_ms,
        ms_off_raw=[off_ms]*n, ms_on_raw=[on_ms]*n, ms_ref_raw=[ref_ms]*n,
        n_paired=n,
        delta_pct_on_vs_off=on_vs_off_pct,
        delta_pct_on_vs_off_ci_lo=lo,
        delta_pct_on_vs_off_ci_hi=hi,
        delta_pct_on_vs_ref=0.0,
        delta_pct_on_vs_ref_ci_lo=-1.0,
        delta_pct_on_vs_ref_ci_hi=1.0,
        paired_spread_pct=2.0,
        classification=classification,
        fired=fired,
    )


def test_synthetic_land_path_yields_land_verdict():
    """A synthetic override that delivers +10% in-cohort with no OOC bleed
    and a clean routing trace MUST receive a LAND verdict.  This proves the
    harness can actually approve a valid override (not just a reject-only
    machine)."""
    keys = [(2048, 2048, 4096, torch.float16),
            (2048, 2048, 8192, torch.float16)]
    gate = GuardedOverride(
        ticket="K-TEST-LAND", env_var="TB_TEST_LAND_ENABLE",
        cohort_keys=keys,
        override_fn=lambda *a: {"num_stages": 2},
    )
    # In-cohort: clean +10% paired delta with CI excluding 0 -> SPEEDUP
    in_c = [
        _synthetic_cell(2048, 4096, torch.float16, True,
                        off_ms=1.000, on_ms=0.900, ref_ms=1.000,
                        on_vs_off_pct=10.0, on_vs_off_ci=(8.0, 12.0),
                        fired=True, classification="SPEEDUP"),
        _synthetic_cell(2048, 8192, torch.float16, True,
                        off_ms=2.000, on_ms=1.800, ref_ms=2.000,
                        on_vs_off_pct=10.0, on_vs_off_ci=(8.0, 12.0),
                        fired=True, classification="SPEEDUP"),
    ]
    # OOC: in-band only (no regress). Routing didn't fire.
    ooc = [
        _synthetic_cell(512, 512, torch.float16, False,
                        off_ms=1.000, on_ms=0.999, ref_ms=1.000,
                        on_vs_off_pct=0.1, fired=False,
                        classification="IN_BAND"),
        _synthetic_cell(4096, 4096, torch.float16, False,
                        off_ms=1.000, on_ms=1.001, ref_ms=1.000,
                        on_vs_off_pct=-0.1, fired=False,
                        classification="IN_BAND"),
    ]
    snap = {
        "ticket": "K-TEST-LAND",
        "fire_total": 2, "nonfire_total": 2,
        "per_key_fires": {keys[0]: 1, keys[1]: 1},
    }
    verdict = _compute_verdict(gate, in_c + ooc, snap)
    assert verdict.land, verdict.report
    assert verdict.routing_n_intended_fires == 2
    assert verdict.routing_n_ooc_fires == 0
    assert verdict.in_cohort_geomean_speedup >= 1.05


def test_synthetic_no_land_path_yields_no_land_verdict_on_ooc_regress():
    """An override that delivers in-cohort speedup but causes a real OOC
    regression (CI upper bound below 0) MUST be rejected."""
    keys = [(2048, 2048, 4096, torch.float16)]
    gate = GuardedOverride(
        ticket="K-TEST-LAND", env_var="TB_TEST_LAND_ENABLE",
        cohort_keys=keys, override_fn=lambda *a: {"num_stages": 2},
    )
    in_c = [_synthetic_cell(2048, 4096, torch.float16, True,
                            off_ms=1.0, on_ms=0.9, ref_ms=1.0,
                            on_vs_off_pct=10.0, on_vs_off_ci=(8.0, 12.0),
                            fired=True, classification="SPEEDUP")]
    # OOC cell with REAL regression: CI strictly below 0
    ooc = [_synthetic_cell(512, 512, torch.float16, False,
                           off_ms=1.0, on_ms=1.10, ref_ms=1.0,
                           on_vs_off_pct=-10.0, on_vs_off_ci=(-12.0, -8.0),
                           fired=False, classification="REGRESS")]
    snap = {"ticket": "K-TEST-LAND", "fire_total": 1, "nonfire_total": 1,
            "per_key_fires": {keys[0]: 1}}
    verdict = _compute_verdict(gate, in_c + ooc, snap)
    assert not verdict.land
    assert verdict.ooc_n_regress == 1


def test_synthetic_no_land_with_single_trial_noise_is_not_classified_regress():
    """K-901 reviewer requirement: a 'REGRESS' classification whose 95% CI
    upper bound straddles 0 is NOISE, not a true regression -- the verdict
    must NOT count it as a regressor (so cells like K-882's -3.90% point are
    statistically defensible, not single-trial noise)."""
    keys = [(2048, 2048, 4096, torch.float16)]
    gate = GuardedOverride(
        ticket="K-TEST-LAND", env_var="TB_TEST_LAND_ENABLE",
        cohort_keys=keys, override_fn=lambda *a: {"num_stages": 2},
    )
    in_c = [_synthetic_cell(2048, 4096, torch.float16, True,
                            off_ms=1.0, on_ms=0.9, ref_ms=1.0,
                            on_vs_off_pct=10.0, on_vs_off_ci=(8.0, 12.0),
                            fired=True, classification="SPEEDUP")]
    # Classifier flagged REGRESS but CI upper bound is positive -> noise.
    ooc = [_synthetic_cell(512, 512, torch.float16, False,
                           off_ms=1.0, on_ms=1.05, ref_ms=1.0,
                           on_vs_off_pct=-5.0, on_vs_off_ci=(-12.0, +2.0),
                           fired=False, classification="REGRESS")]
    snap = {"ticket": "K-TEST-LAND", "fire_total": 1, "nonfire_total": 1,
            "per_key_fires": {keys[0]: 1}}
    verdict = _compute_verdict(gate, in_c + ooc, snap)
    # Verdict must NOT count this as a regressor.
    assert verdict.ooc_n_regress == 0


# ---------------------------------------------------------------------------
# K-882 fixture (regression test for the harness reject-path)
# ---------------------------------------------------------------------------
def test_k882_fixture_strict_keys_only():
    """K-882 fixture: strict-equality cohort -- adjacent shapes must not fire."""
    g = _k882_fixture()
    OverrideRegistry.register(g)
    os.environ["TB_K882_ENABLE"] = "1"
    assert g.lookup(2048, 2048, 4096, torch.float16) is not None
    assert g.lookup(2048, 2048, 5000, torch.float16) is None
    assert g.lookup(2048, 2048, 4096, torch.float32) is None  # dtype guard


def test_k882_fixture_off_by_default():
    """K-882 fixture is OFF unless TB_K882_ENABLE is set."""
    g = _k882_fixture()
    os.environ.pop("TB_K882_ENABLE", None)
    assert g.lookup(2048, 2048, 4096, torch.float16) is None


def test_k882_not_in_production_registry():
    """K-882 must NOT auto-register on production import (it's a fixture)."""
    # Re-import to be sure overrides/__init__ is not pulled in.
    import importlib
    import tritonblas
    importlib.reload(tritonblas)
    tickets = [g.ticket for g in OverrideRegistry.all()]
    assert "K-882" not in tickets, (
        f"K-882 leaked into production registry: {tickets}.  "
        f"It must live in tests/ only (NO-LAND verdict)."
    )


# ---------------------------------------------------------------------------
# Sanity: classifier helper
# ---------------------------------------------------------------------------
def test_classify_delta_noise_band():
    # spread/2 = 1pp, but noise floor is 2pp -> band = 2pp.
    assert _classify_delta(+1.5, paired_spread_pct=2.0) == "IN_BAND"
    assert _classify_delta(+3.0, paired_spread_pct=2.0) == "SPEEDUP"
    assert _classify_delta(-3.0, paired_spread_pct=2.0) == "REGRESS"
    # Wide spread expands the noise band.
    assert _classify_delta(+3.0, paired_spread_pct=10.0) == "IN_BAND"
