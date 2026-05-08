"""K-693 N=64 tall-skinny FP16/BF16 tile-override gate tests.

Validates the predicate (`_tskinny_tile_override`) over the full K-668
128-shape cohort:
  M  in {2048, 4096, 8192, 16384}
  N  in {32, 64, 128, 256}
  K  in {1024, 2048, 4096, 8192}
  dt in {float16, bfloat16}

Expected firing classification (per K-697 + K-693 ship gates):
  * K-697 sub-band (N == 32):  M >= 2048 AND K >= 4096   -> 16 shapes
      Stream-K spec: BM=BN=32, BK=256, ns=2, nw=4, kp=1, force_streamk=True
  * K-693 sub-band (N == 64):  M >= 4096 AND K >= 4096   -> 12 shapes
      Persistent spec: BM=32, BN=64, BK=128, ns=2, nw=8, kp=2, force_streamk=False
  * OOC (everything else):                                -> 100 shapes
      Predicate must return None (no leakage).

Also verified:
  * TRITONBLAS_DISABLE_TSKINNY_OVERRIDE=1 disables the gate for all 28
    nominally-firing shapes (predicate returns None).
  * Numerical correctness (rtol/atol) on representative fp16 + bf16 firing
    shapes when CUDA is available (skipped otherwise).
"""

import importlib
import os
import sys
import pytest


def _reload_matmul_module(env_value):
    os.environ["TRITONBLAS_DISABLE_TSKINNY_OVERRIDE"] = env_value
    for m in list(sys.modules):
        if m.startswith("tritonblas"):
            del sys.modules[m]
    return importlib.import_module("tritonblas.matmul")


@pytest.fixture()
def gate_on():
    return _reload_matmul_module("0")


@pytest.fixture()
def gate_off():
    return _reload_matmul_module("1")


import torch


# K-668 128-shape cohort grid.
_M_GRID = (2048, 4096, 8192, 16384)
_N_GRID = (32, 64, 128, 256)
_K_GRID = (1024, 2048, 4096, 8192)
_DT_GRID = (torch.float16, torch.bfloat16)

K668_COHORT = [
    (M, N, K, dt) for dt in _DT_GRID for M in _M_GRID for N in _N_GRID for K in _K_GRID
]
assert len(K668_COHORT) == 128, f"K-668 cohort must be 128 shapes, got {len(K668_COHORT)}"


def _classify(M, N, K, dt):
    """Return ('k693', spec) | ('k697', spec) | ('ooc', None)."""
    if dt not in (torch.float16, torch.bfloat16):
        return ("ooc", None)
    if M < 2048 or K < 4096:
        return ("ooc", None)
    if N == 32:
        return ("k697", {"BM": 32, "BN": 32, "BK": 256, "num_stages": 2,
                         "num_warps": 4, "kpack": 1, "force_streamk": True})
    if N == 64 and M >= 4096:
        return ("k693", {"BM": 32, "BN": 64, "BK": 128, "num_stages": 2,
                         "num_warps": 8, "kpack": 2, "force_streamk": False})
    return ("ooc", None)


# Pre-partition cohort for clarity in test IDs.
K693_FIRE = [s for s in K668_COHORT if _classify(*s)[0] == "k693"]
K697_FIRE = [s for s in K668_COHORT if _classify(*s)[0] == "k697"]
OOC_SHAPES = [s for s in K668_COHORT if _classify(*s)[0] == "ooc"]

# The PRD says: 12 K-693 fires, 16 K-697 fires, 100 OOC.
assert len(K693_FIRE) == 12, f"expected 12 K-693 fire shapes, got {len(K693_FIRE)}"
assert len(K697_FIRE) == 16, f"expected 16 K-697 fire shapes, got {len(K697_FIRE)}"
assert len(OOC_SHAPES) == 100, f"expected 100 OOC shapes, got {len(OOC_SHAPES)}"


@pytest.mark.parametrize("M,N,K,dt", K693_FIRE)
def test_k693_predicate_fires_on_all_12_shapes(gate_on, M, N, K, dt):
    """Predicate must fire on every K-693 N=64 shape with the K-693 ship spec."""
    spec = gate_on._tskinny_tile_override(M, N, K, dt)
    assert spec is not None, f"K-693 expected fire on ({M},{N},{K},{dt})"
    assert spec["BM"] == 32
    assert spec["BN"] == 64
    assert spec["BK"] == 128
    assert spec["num_stages"] == 2
    assert spec["num_warps"] == 8
    assert spec["kpack"] == 2
    assert spec["force_streamk"] is False, "K-693 sub-band uses persistent dispatch"


@pytest.mark.parametrize("M,N,K,dt", K697_FIRE)
def test_k697_predicate_fires_on_all_16_shapes(gate_on, M, N, K, dt):
    """K-697 N=32 sub-band must keep its Stream-K spec — no regression."""
    spec = gate_on._tskinny_tile_override(M, N, K, dt)
    assert spec is not None, f"K-697 expected fire on ({M},{N},{K},{dt})"
    assert spec["BM"] == 32
    assert spec["BN"] == 32
    assert spec["BK"] == 256
    assert spec["num_stages"] == 2
    assert spec["num_warps"] == 4
    assert spec["kpack"] == 1
    assert spec["force_streamk"] is True, "K-697 sub-band requires Stream-K"


@pytest.mark.parametrize("M,N,K,dt", OOC_SHAPES)
def test_predicate_inert_on_all_100_ooc_shapes(gate_on, M, N, K, dt):
    """Zero-leakage audit: predicate must return None on every OOC shape."""
    spec = gate_on._tskinny_tile_override(M, N, K, dt)
    assert spec is None, f"Expected inert OOC on ({M},{N},{K},{dt}); got {spec}"


@pytest.mark.parametrize("M,N,K,dt", K693_FIRE + K697_FIRE)
def test_disable_env_var_inerts_gate(gate_off, M, N, K, dt):
    """TRITONBLAS_DISABLE_TSKINNY_OVERRIDE=1 must turn the gate off entirely."""
    spec = gate_off._tskinny_tile_override(M, N, K, dt)
    assert spec is None, f"Disable env failed on ({M},{N},{K},{dt}); got {spec}"


def test_partition_counts():
    """Sanity: K-693 + K-697 + OOC == full cohort, no overlap."""
    assert len(K693_FIRE) + len(K697_FIRE) + len(OOC_SHAPES) == 128
    s = set(K693_FIRE) | set(K697_FIRE) | set(OOC_SHAPES)
    assert len(s) == 128, "partitions overlap"


# -----------------------------------------------------------------------------
# Numerical correctness — at least one fp16 and one bf16 K-693 firing shape.
# Tolerances follow K-697 ship lessons (K=8192 fp16 accumulation rel_err
# typically <= 5e-3; we use atol/rtol that admit accumulation noise but
# would catch any wrong-output bug).
# -----------------------------------------------------------------------------

NUMERICAL_FIRE_CASES = [
    (4096, 64, 4096, torch.float16),
    (4096, 64, 4096, torch.bfloat16),
    (8192, 64, 8192, torch.float16),
    (8192, 64, 8192, torch.bfloat16),
]


@pytest.mark.parametrize("M,N,K,dt", NUMERICAL_FIRE_CASES)
def test_k693_numerical_correctness(gate_on, M, N, K, dt):
    """Override path must produce numerically correct results within tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import tritonblas
    a = torch.randn(M, K, device="cuda", dtype=dt)
    b = torch.randn(K, N, device="cuda", dtype=dt)
    ref = (a.float() @ b.float()).to(dt)
    got = tritonblas.matmul(a, b)

    # Confirm the override actually fired for this shape.
    spec = gate_on._tskinny_tile_override(M, N, K, dt)
    assert spec is not None and spec["BN"] == 64, "test misclassification"

    # Per-element tolerance (catches wrong-output bugs); permissive enough
    # for K=8192 fp16 accumulation drift. atol scales with magnitude.
    ref_max = ref.abs().max().item()
    diff = (got - ref).abs()
    abs_err = diff.max().item()
    rel_err = abs_err / max(ref_max, 1e-6)
    # Mean error catches systematic bias the max may miss for large tensors.
    mean_rel = diff.mean().item() / max(ref.abs().mean().item(), 1e-6)

    assert rel_err < 1e-2, (
        f"max rel_err {rel_err:.2e} exceeds tolerance for ({M},{N},{K},{dt})"
    )
    assert mean_rel < 5e-3, (
        f"mean rel_err {mean_rel:.2e} exceeds tolerance for ({M},{N},{K},{dt})"
    )
