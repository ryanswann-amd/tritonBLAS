"""K-935 — guarded-override structural tests (CPU-only).

These tests verify the K-935 cohort-A LDS-pressure mitigation gate:

  - is registered in the production registry on import
  - is fail-closed (env-var OFF -> never fires)
  - fires only on the strict-equality (M,N,K,dtype) cohort keys
  - returns the K-935 payload (kpack=2, num_warps=8, num_stages=2)
  - does not bleed onto neighbour cells in the K-905 cohort-A neighbourhood
  - composes correctly with the work-stealing dispatcher (refuses by default
    per K-883 L5)

GPU paired-CI falsification lives in ``scripts/run_falsification_k935.py``.
"""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

import tritonblas  # noqa: E402  -- triggers override registration
from tritonblas.guarded_override import (  # noqa: E402
    GuardedOverride,
    OverrideRegistry,
    apply_override_in_dispatcher,
)
from tritonblas.overrides.k935_lds_pressure import K935_GATE  # noqa: E402


COHORT = [
    (1024, 1024, 16384, torch.float16),
    (1024, 1024, 16384, torch.bfloat16),
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _disable():
    os.environ.pop("TB_K935_ENABLE", None)


def _enable():
    os.environ["TB_K935_ENABLE"] = "1"


@pytest.fixture(autouse=True)
def _restore_env():
    prev = os.environ.get("TB_K935_ENABLE")
    _disable()
    yield
    if prev is None:
        os.environ.pop("TB_K935_ENABLE", None)
    else:
        os.environ["TB_K935_ENABLE"] = prev


# ----------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------
def test_k935_gate_registered_in_production_singleton():
    tickets = {g.ticket for g in OverrideRegistry.all()}
    assert "K-935" in tickets, (
        "K-935 gate must register on `import tritonblas`; got tickets="
        f"{tickets}"
    )


def test_k935_gate_is_singleton():
    """Registering twice is a no-op (idempotence — K-883 L8)."""
    n_before = len(OverrideRegistry.all())
    OverrideRegistry.register(K935_GATE)
    n_after = len(OverrideRegistry.all())
    assert n_before == n_after


# ----------------------------------------------------------------------------
# Fail-closed semantics (K-883 L3)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", COHORT)
def test_k935_off_by_default(M, N, K, dtype):
    _disable()
    assert K935_GATE.lookup(M, N, K, dtype) is None, (
        "Cohort-A key fired with TB_K935_ENABLE unset; gate is not fail-closed."
    )


# ----------------------------------------------------------------------------
# Cohort enumeration (K-883 L1 strict-equality dispatch)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", COHORT)
def test_k935_fires_on_each_cohort_key(M, N, K, dtype):
    K935_GATE.reset_counters()
    _enable()
    payload = K935_GATE.lookup(M, N, K, dtype)
    assert payload is not None, f"Cohort key ({M},{N},{K},{dtype}) did not fire."
    assert payload["kpack"] == 2
    assert payload["num_warps"] == 8
    assert payload["num_stages"] == 2


@pytest.mark.parametrize(
    "M,N,K,dtype",
    [
        # K-905 cohort-A near-neighbours that MUST NOT fire (R1 strict equality).
        (1024, 1024, 8192, torch.float16),    # smaller K
        (1024, 1024, 16384, torch.float32),   # disallowed dtype
        (2048, 2048, 16384, torch.float16),   # K-882 cohort cell
        (1024, 1024, 32768, torch.bfloat16),  # larger K
        (768, 1024, 16384, torch.float16),    # asymmetric square
        (1024, 768, 16384, torch.bfloat16),
        (512, 512, 16384, torch.float16),
        (1024, 1024, 4096, torch.bfloat16),
    ],
)
def test_k935_does_not_fire_on_neighbours(M, N, K, dtype):
    K935_GATE.reset_counters()
    _enable()
    assert K935_GATE.lookup(M, N, K, dtype) is None, (
        f"Neighbour key ({M},{N},{K},{dtype}) fired -- predicate bleed "
        "(K-654 anti-pattern)."
    )


# ----------------------------------------------------------------------------
# Dtype allowlist (K-883 L2)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int8])
def test_k935_dtype_allowlist_blocks(dtype):
    _enable()
    assert K935_GATE.lookup(1024, 1024, 16384, dtype) is None


# ----------------------------------------------------------------------------
# Routing-trace counters (K-883 L6)
# ----------------------------------------------------------------------------
def test_k935_routing_counters_track_fires_and_nonfires():
    K935_GATE.reset_counters()
    _enable()
    K935_GATE.lookup(1024, 1024, 16384, torch.float16)        # fire
    K935_GATE.lookup(1024, 1024, 16384, torch.bfloat16)       # fire
    K935_GATE.lookup(2048, 2048, 16384, torch.float16)        # nonfire (OOC)
    _disable()
    K935_GATE.lookup(1024, 1024, 16384, torch.float16)        # nonfire (env)
    snap = K935_GATE.routing_snapshot()
    assert snap["fire_total"] == 2
    assert snap["nonfire_total"] == 2
    assert snap["per_key_fires"][(1024, 1024, 16384, torch.float16)] == 1
    assert snap["per_key_fires"][(1024, 1024, 16384, torch.bfloat16)] == 1


# ----------------------------------------------------------------------------
# Composability guard (K-883 L5) -- default refuses under work_stealing.
# ----------------------------------------------------------------------------
def test_k935_refuses_under_work_stealing():
    _enable()
    K935_GATE.reset_counters()
    hit = OverrideRegistry.apply(
        1024, 1024, 16384, torch.float16, work_stealing=True
    )
    assert hit is None, (
        "K-935 fired under work_stealing=True; L5 composability guard broken."
    )


def test_k935_fires_via_registry_apply_without_work_stealing():
    _enable()
    K935_GATE.reset_counters()
    hit = OverrideRegistry.apply(
        1024, 1024, 16384, torch.float16, work_stealing=False
    )
    assert hit is not None
    gate, payload = hit
    assert gate.ticket == "K-935"
    assert payload == {"kpack": 2, "num_warps": 8, "num_stages": 2}


# ----------------------------------------------------------------------------
# Dispatch-site shim integration (K-883 L4 LDS budget enforced)
# ----------------------------------------------------------------------------
def test_k935_dispatch_site_returns_kpack_and_num_warps():
    _enable()
    K935_GATE.reset_counters()
    out = apply_override_in_dispatcher(
        M=1024, N=1024, K=16384, dtype=torch.float16,
        block_m=128, block_n=128, block_k=64,
        num_stages=2, num_warps=8, waves_per_eu=0,
        kpack=1, mfma_instr_size=16,
        total_tiles=64, n_cu=304,
        work_stealing=False, bytes_per_elem=2,
    )
    assert out.get("fired_ticket") == "K-935"
    assert out.get("kpack") == 2
    # num_warps already 8 -- shim only emits diffs, so num_warps should NOT be
    # in the output (no-change diff is omitted).
    assert "num_warps" not in out


def test_k935_dispatch_site_lds_budget_blocks_oversized_tile():
    """Synthetic guard-test: a hypothetical oversized tile triggers L4."""
    _enable()
    K935_GATE.reset_counters()
    # 2 * (256 + 256) * 256 * 2 = 524288 bytes >> 64KiB.  L4 should block.
    out = apply_override_in_dispatcher(
        M=1024, N=1024, K=16384, dtype=torch.float16,
        block_m=256, block_n=256, block_k=256,
        num_stages=2, num_warps=8, waves_per_eu=0,
        kpack=1, mfma_instr_size=16,
        total_tiles=16, n_cu=304,
        work_stealing=False, bytes_per_elem=2,
    )
    assert out.get("lds_blocked") is True
    assert out.get("fired_ticket") == "K-935"
    # When L4 trips, the kpack diff must NOT be applied (caller would skip).
    assert "kpack" not in out


# ----------------------------------------------------------------------------
# Regression — K-905 cohort-A documented bf16/fp16 keys must be present.
# ----------------------------------------------------------------------------
def test_k935_cohort_keys_match_k905_cohortA():
    expected = {
        (1024, 1024, 16384, torch.float16),
        (1024, 1024, 16384, torch.bfloat16),
    }
    assert set(K935_GATE.cohort_keys) == expected


def test_k935_cohort_size_cap_respected():
    """A future PR cannot grow this gate without explicitly raising the cap."""
    assert K935_GATE.cohort_size_cap <= 4
    assert len(K935_GATE.cohort_keys) <= K935_GATE.cohort_size_cap
