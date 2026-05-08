"""K-883/K-901 — guarded-override harness gating-logic tests (CPU-only).

These tests exercise Layers 1, 2, 3, 5, 6 and the OverrideRegistry routing
order without touching CUDA — the GPU paired-bench code path is exercised
separately in scripts/run_falsification_k882.py on c42/MI300X.

The test names map 1:1 to the K-883 §5 / §3 layer rules so a CI failure
points at the broken layer.
"""
import os
import threading

import pytest

torch = pytest.importorskip("torch")
from tritonblas.guarded_override import (  # noqa: E402
    GuardedOverride,
    OverrideRegistry,
    GuardedOverride as GO,
    apply_override_in_dispatcher,
    lds_budget_for,
)


def _make_gate(env="TB_TEST_ENABLE", ticket="K-TEST",
               keys=None, dtype_allowlist=(torch.float16, torch.bfloat16)):
    if keys is None:
        keys = [(2048, 2048, 4096, torch.float16),
                (2048, 2048, 8192, torch.bfloat16)]
    return GuardedOverride(
        ticket=ticket, env_var=env, cohort_keys=keys,
        dtype_allowlist=dtype_allowlist,
        override_fn=lambda M, N, K, dt: {"num_stages": 2},
    )


def setup_function(_):
    OverrideRegistry.unregister("K-TEST")
    OverrideRegistry.unregister("K-TEST-2")
    os.environ.pop("TB_TEST_ENABLE", None)
    os.environ.pop("TB_TEST_2_ENABLE", None)


def test_l3_env_killswitch_off_by_default():
    """Layer 3: env-var killswitch must be OFF by default (must opt-in)."""
    g = _make_gate()
    assert g.lookup(2048, 2048, 4096, torch.float16) is None


def test_l3_env_killswitch_on_when_env_set():
    g = _make_gate()
    os.environ["TB_TEST_ENABLE"] = "1"
    assert g.lookup(2048, 2048, 4096, torch.float16) == {"num_stages": 2}


def test_l1_strict_equality_no_range_leak():
    """Layer 1: strict-equality only.  Adjacent shapes must not fire."""
    g = _make_gate()
    os.environ["TB_TEST_ENABLE"] = "1"
    assert g.lookup(2048, 2048, 4096, torch.float16) is not None
    # Off-by-one shape, same dtype: must not fire.
    assert g.lookup(2048, 2048, 4097, torch.float16) is None
    assert g.lookup(2048, 2049, 4096, torch.float16) is None
    assert g.lookup(2049, 2048, 4096, torch.float16) is None


def test_l2_dtype_allowlist():
    """Layer 2: dtype must be in allowlist even if (M,N,K) matches."""
    g = _make_gate(dtype_allowlist=(torch.float16,))  # bf16 NOT allowed
    os.environ["TB_TEST_ENABLE"] = "1"
    assert g.lookup(2048, 2048, 4096, torch.float16) is not None
    # bf16 with a key that exists in cohort_keys but is NOT in allowlist:
    assert g.lookup(2048, 2048, 8192, torch.bfloat16) is None


def test_l4_lds_guard_blocks_oversize_tile():
    """Layer 4: LDS over-stuff is detected by the dispatcher shim."""
    # Register a gate that asks for an absurdly large block.
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


def test_l5_composability_guard_refuses_under_work_stealing():
    """Layer 5: default composability guard refuses under work_stealing."""
    g = _make_gate()
    OverrideRegistry.register(g)
    os.environ["TB_TEST_ENABLE"] = "1"
    # work_stealing=True must skip this gate.
    out = apply_override_in_dispatcher(
        2048, 2048, 4096, torch.float16,
        block_m=128, block_n=128, block_k=64,
        num_stages=4, num_warps=8, waves_per_eu=0,
        kpack=1, mfma_instr_size=16,
        total_tiles=256, n_cu=304,
        work_stealing=True,
    )
    assert out == {}, "L5: work_stealing must suppress the gate"
    # And work_stealing=False should fire.
    out2 = apply_override_in_dispatcher(
        2048, 2048, 4096, torch.float16,
        block_m=128, block_n=128, block_k=64,
        num_stages=4, num_warps=8, waves_per_eu=0,
        kpack=1, mfma_instr_size=16,
        total_tiles=256, n_cu=304,
        work_stealing=False,
    )
    assert out2.get("fired_ticket") == "K-TEST"


def test_l6_routing_counters_track_fires_and_nonfires():
    """Layer 6: per-key counters increment correctly on fire/nonfire."""
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
    """Layer 6: counters are thread-safe under parallel callers."""
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


def test_cohort_size_cap_rejects_oversize_table():
    """K-883 §5 C3: cohort size <= 30 by default."""
    keys = [(i, 2048, 4096, torch.float16) for i in range(40)]
    with pytest.raises(ValueError, match="cohort_size"):
        GuardedOverride(
            ticket="K-TEST", env_var="TB_TEST_ENABLE",
            cohort_keys=keys, override_fn=lambda *a: {},
        )


def test_r1_dispatch_order_first_routing():
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


def test_lds_ok_helper():
    """LDS budget helper computes ns*(BM+BN)*BK*bpe correctly."""
    # 2 * (128+128) * 64 * 2 = 65536  -> exactly at budget on gfx942
    assert GO.lds_ok(128, 128, 64, num_stages=2, bytes_per_elem=2, arch="gfx942")
    # ... +1 stage breaks the budget
    assert not GO.lds_ok(128, 128, 64, num_stages=3, bytes_per_elem=2, arch="gfx942")


def test_k882_override_registered_at_import():
    """K-882 must be in the registry as a result of import side effects."""
    import tritonblas  # noqa: F401  (forces overrides/__init__.py import)
    tickets = [g.ticket for g in OverrideRegistry.all()]
    assert "K-882" in tickets, f"K-882 not registered; have {tickets}"


def test_k882_off_by_default():
    """K-882 must be OFF by default (TB_K882_ENABLE not set)."""
    import tritonblas  # noqa: F401
    os.environ.pop("TB_K882_ENABLE", None)
    hit = OverrideRegistry.apply(2048, 2048, 4096, torch.float16)
    assert hit is None, "K-882 must be inert without TB_K882_ENABLE=1"
