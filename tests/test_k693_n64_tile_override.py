"""K-693 N=64 tall-skinny FP16/BF16 tile-override gate tests.

Validates the predicate (`_tskinny_tile_override`) for the K-693 sub-band:
* Fires on M >= 4096 ∧ N == 64 ∧ K >= 4096 ∧ dtype in {fp16, bf16}.
* Returns the K-693-verified persistent tile (BM=32, BN=64, BK=128, ns=2,
  nw=8, kp=2, force_streamk=False).
* Does NOT fire on M=2048,N=64 (where Origami's default is competitive).
* Does NOT fire on N != 32 ∧ N != 64 (gate is two disjoint sub-bands).
* Disjoint from K-697 N=32 sub-band (which keeps its Stream-K spec).
* Inert for fp32 / fp8 / square shapes / K<4096.
* TRITONBLAS_DISABLE_TSKINNY_OVERRIDE=1 turns the gate off entirely.

These are pure-predicate tests -- no GPU required.  Numerical / perf
verification is performed by scripts/validate_128_ab.py against the
full K-668 128-shape cohort.
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


# K-693 fire cases (N=64 sub-band)
K693_FIRE = [
    (4096, 64, 4096, torch.float16),
    (4096, 64, 8192, torch.bfloat16),
    (8192, 64, 4096, torch.float16),
    (8192, 64, 8192, torch.bfloat16),
    (16384, 64, 4096, torch.float16),
    (16384, 64, 8192, torch.bfloat16),
]

# K-697 fire cases (N=32 sub-band) -- must keep Stream-K spec
K697_FIRE = [
    (2048, 32, 4096, torch.float16),
    (4096, 32, 8192, torch.bfloat16),
    (16384, 32, 8192, torch.float16),
]

# Out-of-cohort cases: must return None
OOC = [
    # M=2048,N=64 cohort: Origami is competitive there, exclude it
    (2048, 64, 4096, torch.float16),
    (2048, 64, 8192, torch.bfloat16),
    # N=128, N=256: not in either sub-band
    (4096, 128, 8192, torch.float16),
    (4096, 256, 8192, torch.bfloat16),
    (16384, 128, 4096, torch.float16),
    # M < 2048
    (1024, 32, 8192, torch.float16),
    (512, 64, 8192, torch.bfloat16),
    # K < 4096
    (4096, 32, 1024, torch.float16),
    (8192, 64, 2048, torch.bfloat16),
    # Wrong dtype
    (4096, 64, 8192, torch.float32),
    (4096, 32, 8192, torch.float32),
    # Square / non-tall-skinny
    (4096, 4096, 4096, torch.float16),
    (8192, 8192, 8192, torch.bfloat16),
]


@pytest.mark.parametrize("M,N,K,dt", K693_FIRE)
def test_k693_predicate_fires(gate_on, M, N, K, dt):
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
def test_k697_subband_unchanged(gate_on, M, N, K, dt):
    spec = gate_on._tskinny_tile_override(M, N, K, dt)
    assert spec is not None, f"K-697 expected fire on ({M},{N},{K},{dt})"
    assert spec["BM"] == 32
    assert spec["BN"] == 32
    assert spec["BK"] == 256
    assert spec["force_streamk"] is True, "K-697 sub-band requires Stream-K"


@pytest.mark.parametrize("M,N,K,dt", OOC)
def test_predicate_inert_ooc(gate_on, M, N, K, dt):
    spec = gate_on._tskinny_tile_override(M, N, K, dt)
    assert spec is None, f"Expected inert OOC on ({M},{N},{K},{dt}); got {spec}"


@pytest.mark.parametrize("M,N,K,dt", K693_FIRE + K697_FIRE)
def test_disable_env_var(gate_off, M, N, K, dt):
    spec = gate_off._tskinny_tile_override(M, N, K, dt)
    assert spec is None, f"Disable env failed on ({M},{N},{K},{dt}); got {spec}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M,N,K", [(4096, 64, 8192), (8192, 64, 4096), (16384, 64, 8192)])
def test_k693_numerical_correctness(gate_on, M, N, K, dtype):
    """End-to-end: ensure the override path produces numerically correct results."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import tritonblas
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    ref = (a.float() @ b.float()).to(dtype)
    got = tritonblas.matmul(a, b)
    rel_err = (got - ref).abs().max().item() / ref.abs().max().item()
    # K=8192 fp16 accumulation tolerance per K-697 ship: 4.74e-3
    assert rel_err < 1e-2, f"rel_err {rel_err:.2e} too high for ({M},{N},{K},{dtype})"
