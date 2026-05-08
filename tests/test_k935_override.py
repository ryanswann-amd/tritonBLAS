"""K-935 — guarded-override structural tests (CPU-only).

These tests verify the K-935 cohort-A LDS-pressure mitigation gate's
**structural correctness** (predicate, killswitch, dispatch-site shim,
LDS-budget guard).  They do NOT touch CUDA — the GPU paired-CI verdict
lives in ``scripts/run_falsification_k935.py`` (current verdict on
c42/MI300X gfx942: NO-LAND, geomean ON/OFF ~0.99x, V2/V3 PASS).

Per K-901 M2 the gate definition lives only in scripts/ + tests/ until
its harness verdict flips to LAND, so these tests import the canonical
fixture from the script via importlib path manipulation.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

from tritonblas.guarded_override import (  # noqa: E402
    GuardedOverride,
    OverrideRegistry,
    apply_override_in_dispatcher,
)


# ---------------------------------------------------------------------------
# Load the K-935 gate definition from the script (canonical fixture site).
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "run_falsification_k935.py"


def _load_k935_module():
    spec = importlib.util.spec_from_file_location(
        "_k935_falsify", str(_SCRIPT_PATH),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_k935_falsify"] = mod
    spec.loader.exec_module(mod)
    return mod


_k935_mod = _load_k935_module()


@pytest.fixture
def gate():
    """Fresh K-935 gate per test (counters reset, no env leakage)."""
    g = _k935_mod.build_k935_gate()
    OverrideRegistry.register(g)
    yield g
    OverrideRegistry.unregister("K-935")


COHORT = [
    (1024, 1024, 16384, torch.float16),
    (1024, 1024, 16384, torch.bfloat16),
]


# ---------------------------------------------------------------------------
# Env hygiene
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# K-901 M2 — production tree must NOT carry the K-935 gate.
# ---------------------------------------------------------------------------
def test_k935_not_in_production_registry_after_clean_import():
    """Fail-closed: `import tritonblas` registers ZERO concrete overrides."""
    import tritonblas  # noqa: F401
    tickets = {g.ticket for g in OverrideRegistry.all()}
    assert "K-935" not in tickets, (
        "K-935 gate auto-registered in production -- violates K-901 M2 "
        "(fail-closed; only LAND verdicts ship).  Tickets seen: "
        f"{tickets}"
    )


def test_k935_overrides_subpackage_does_not_exist():
    """K-901 M2: NO-LAND prototypes are deleted, not disabled."""
    overrides_dir = _REPO_ROOT / "include" / "tritonblas" / "overrides"
    assert not overrides_dir.exists(), (
        f"include/tritonblas/overrides/ exists -- K-901 M2 says delete "
        f"NO-LAND prototypes from production tree, do not stash them as "
        f"disabled-by-default packages."
    )


# ---------------------------------------------------------------------------
# Idempotence (K-883 L8)
# ---------------------------------------------------------------------------
def test_k935_register_idempotent(gate):
    n_before = len(OverrideRegistry.all())
    OverrideRegistry.register(_k935_mod.build_k935_gate())
    n_after = len(OverrideRegistry.all())
    assert n_before == n_after


# ---------------------------------------------------------------------------
# Fail-closed semantics (K-883 L3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", COHORT)
def test_k935_off_by_default(M, N, K, dtype, gate):
    _disable()
    assert gate.lookup(M, N, K, dtype) is None


# ---------------------------------------------------------------------------
# Killswitch happy/error pair (explicit on each in-cohort key) -- K-883 L7
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", COHORT)
def test_k935_killswitch_pair_lookup(M, N, K, dtype, gate):
    """Env-set fires, env-unset refuses -- on the SAME in-cohort key.

    Guards against silent regressions where the killswitch becomes a
    no-op (e.g. env read at import-time, hardcoded True, predicate
    bypass).  The lookup happy/error pair MUST hold on every in-cohort
    cell, not merely "off by default" before any enable.
    """
    _enable()
    assert gate.lookup(M, N, K, dtype) is not None, (
        f"Killswitch happy path broken: gate refused in-cohort key "
        f"({M},{N},{K},{dtype}) with TB_K935_ENABLE=1."
    )
    _disable()
    assert gate.lookup(M, N, K, dtype) is None, (
        f"Killswitch error path broken: gate fired on in-cohort key "
        f"({M},{N},{K},{dtype}) with TB_K935_ENABLE unset."
    )


@pytest.mark.parametrize("M,N,K,dtype", COHORT)
def test_k935_killswitch_pair_routing_trace(M, N, K, dtype, gate):
    """Same pair, but observed via the routing-trace counters.

    The on->off->on triple verifies fire_total / nonfire_total move
    independently and that disabling the env flips a true would-fire
    into the nonfire bucket (rather than skipping the predicate
    entirely, which would also leave fire_total at 0 and silently mask
    a kill-switch bypass).
    """
    _enable()
    gate.lookup(M, N, K, dtype)
    snap = gate.routing_snapshot()
    assert snap["fire_total"] == 1, snap
    assert snap["nonfire_total"] == 0, snap

    _disable()
    gate.lookup(M, N, K, dtype)
    snap = gate.routing_snapshot()
    assert snap["fire_total"] == 1, snap
    assert snap["nonfire_total"] == 1, snap

    _enable()
    gate.lookup(M, N, K, dtype)
    snap = gate.routing_snapshot()
    assert snap["fire_total"] == 2, snap
    assert snap["nonfire_total"] == 1, snap


@pytest.mark.parametrize("M,N,K,dtype", COHORT)
def test_k935_killswitch_pair_via_registry_apply(M, N, K, dtype, gate):
    """End-to-end: dispatcher entry-point honours the env on EVERY cohort key."""
    _enable()
    hit = OverrideRegistry.apply(M, N, K, dtype, work_stealing=False)
    assert hit is not None and hit[0].ticket == "K-935"

    _disable()
    hit = OverrideRegistry.apply(M, N, K, dtype, work_stealing=False)
    assert hit is None, (
        f"Dispatcher fired K-935 on ({M},{N},{K},{dtype}) with env unset -- "
        f"killswitch bypass."
    )


# ---------------------------------------------------------------------------
# Cohort enumeration (K-883 L1 strict-equality dispatch)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", COHORT)
def test_k935_fires_on_each_cohort_key(M, N, K, dtype, gate):
    _enable()
    payload = gate.lookup(M, N, K, dtype)
    assert payload is not None
    assert payload["kpack"] == 2
    assert payload["num_warps"] == 8
    assert payload["num_stages"] == 2


@pytest.mark.parametrize(
    "M,N,K,dtype",
    [
        (1024, 1024, 8192, torch.float16),
        (1024, 1024, 16384, torch.float32),
        (2048, 2048, 16384, torch.float16),
        (1024, 1024, 32768, torch.bfloat16),
        (768, 1024, 16384, torch.float16),
        (1024, 768, 16384, torch.bfloat16),
        (512, 512, 16384, torch.float16),
        (1024, 1024, 4096, torch.bfloat16),
    ],
)
def test_k935_does_not_fire_on_neighbours(M, N, K, dtype, gate):
    _enable()
    assert gate.lookup(M, N, K, dtype) is None, (
        f"Predicate bleed at ({M},{N},{K},{dtype}) -- K-654 anti-pattern."
    )


# ---------------------------------------------------------------------------
# Dtype allowlist (K-883 L2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int8])
def test_k935_dtype_allowlist_blocks(dtype, gate):
    _enable()
    assert gate.lookup(1024, 1024, 16384, dtype) is None


# ---------------------------------------------------------------------------
# Routing-trace counters (K-883 L6)
# ---------------------------------------------------------------------------
def test_k935_routing_counters(gate):
    _enable()
    gate.lookup(1024, 1024, 16384, torch.float16)
    gate.lookup(1024, 1024, 16384, torch.bfloat16)
    gate.lookup(2048, 2048, 16384, torch.float16)         # OOC nonfire
    _disable()
    gate.lookup(1024, 1024, 16384, torch.float16)         # env nonfire
    snap = gate.routing_snapshot()
    assert snap["fire_total"] == 2
    assert snap["nonfire_total"] == 2
    assert snap["per_key_fires"][(1024, 1024, 16384, torch.float16)] == 1
    assert snap["per_key_fires"][(1024, 1024, 16384, torch.bfloat16)] == 1


# ---------------------------------------------------------------------------
# Composability guard (K-883 L5)
# ---------------------------------------------------------------------------
def test_k935_refuses_under_work_stealing(gate):
    _enable()
    hit = OverrideRegistry.apply(
        1024, 1024, 16384, torch.float16, work_stealing=True
    )
    assert hit is None, "K-935 fired under work_stealing=True; L5 broken."


def test_k935_fires_via_registry_apply(gate):
    _enable()
    hit = OverrideRegistry.apply(
        1024, 1024, 16384, torch.float16, work_stealing=False
    )
    assert hit is not None
    g, payload = hit
    assert g.ticket == "K-935"
    assert payload == {"kpack": 2, "num_warps": 8, "num_stages": 2}


# ---------------------------------------------------------------------------
# Dispatch-site shim integration (K-883 L4 LDS budget enforced)
# ---------------------------------------------------------------------------
def test_k935_dispatch_site_returns_kpack_diff(gate):
    _enable()
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
    # No-change diffs must be omitted from the patch dict.
    assert "num_warps" not in out
    assert "num_stages" not in out


def test_k935_dispatch_site_lds_budget_blocks_oversized_tile(gate):
    """Synthetic guard-test: an oversized tile must trip the L4 LDS guard."""
    _enable()
    # 2 * (256 + 256) * 256 * 2 = 524288 bytes >> 64 KiB -- L4 must block.
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
    assert "kpack" not in out


# ---------------------------------------------------------------------------
# Regression -- cohort spec must match K-905 / K-913 grounding.
# ---------------------------------------------------------------------------
def test_k935_cohort_keys_match_k905_cohortA(gate):
    expected = {
        (1024, 1024, 16384, torch.float16),
        (1024, 1024, 16384, torch.bfloat16),
    }
    assert set(gate.cohort_keys) == expected


def test_k935_cohort_size_cap_respected(gate):
    """A future PR cannot grow this gate without explicitly raising the cap."""
    assert gate.cohort_size_cap <= 4
    assert len(gate.cohort_keys) <= gate.cohort_size_cap
