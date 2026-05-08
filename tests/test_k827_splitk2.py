"""K-827: routed split-K=2 override for the M=N=2048,
K∈{4096, 8192, 16384}, fp16/bf16 cohort.

The override ships **default-OFF** (opt-in via TRITONBLAS_ENABLE_K827_SPLITK2=1)
because the K-795 prototype currently regresses these 6 cells on the
Triton-AMD compiler backend (per-shard codegen density / atomic-add
overhead — see K-791 PMC and the K-827 PR description).  Landing the
dispatch path makes the override one env-var flip away from being live
once the K-760 backend successor work changes per-shard arithmetic
intensity.

These tests verify three contracts:

    1. Gate isolation, default-OFF: NO cell in the test set fires the
       split-K=2 dispatch.
    2. Gate isolation, env=1: EXACTLY the 6 cohort cells fire, and
       NONE of the 17 non-cohort guard / asymmetric / off-K / off-dtype
       cells fire.
    3. Numerical correctness, env=1: when the override fires, the result
       satisfies torch.testing.assert_close vs an fp32-reference matmul
       cast back to the cohort dtype, with rtol/atol bounds chosen to
       absorb fp16 / bf16 dot-product accumulation noise at K=16384.

The test parameterises over a single subprocess-aware fixture so that
toggling the env var actually re-imports tritonblas with a fresh module
state (the gate is read once at module-import time).
"""
import importlib
import os
import subprocess
import sys
import textwrap

import pytest
import torch


COHORT = [(2048, 2048, K, dt)
          for K in (4096, 8192, 16384)
          for dt in (torch.float16, torch.bfloat16)]

GUARD = [(M, M, K, dt)
         for M in (1024, 4096)
         for K in (4096, 8192, 16384)
         for dt in (torch.float16, torch.bfloat16)]

EXTRAS = [
    (2048, 4096, 16384, torch.float16),  # asymmetric N
    (2048, 1024, 16384, torch.float16),  # asymmetric N
    (2048, 2048,  4080, torch.float16),  # off-K (not in {4096, 8192, 16384})
    (2048, 2048, 12288, torch.float16),  # off-K
    (2048, 2048, 16384, torch.float32),  # off-dtype
]

ALL_CELLS = COHORT + GUARD + EXTRAS

TOL = {
    torch.float16:  dict(rtol=5e-3, atol=5e-1),
    torch.bfloat16: dict(rtol=4e-2, atol=8.0),
}


def _has_cuda():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="requires CUDA / ROCm GPU")


def _run_with_env(env_value):
    """Run a small probe in a subprocess so the env-var read at module
    import time is fresh per parameterisation."""
    code = textwrap.dedent(f"""
        import os, json, sys, torch
        import importlib
        import tritonblas
        tbm = importlib.import_module("tritonblas.matmul")

        fires = []
        orig = tbm.maybe_dispatch_splitk2
        def counting(a, b, out):
            ok = orig(a, b, out)
            if ok:
                fires.append([a.shape[0], b.shape[1], a.shape[1], str(a.dtype)])
            return ok
        tbm.maybe_dispatch_splitk2 = counting

        cells = {[(M, N, K, str(dt)) for (M, N, K, dt) in ALL_CELLS]!r}
        torch.manual_seed(0)
        for (M, N, K, dt_s) in cells:
            dt = getattr(torch, dt_s.split('.')[-1])
            a = torch.randn(M, K, device="cuda", dtype=dt)
            b = torch.randn(K, N, device="cuda", dtype=dt)
            out = torch.empty(M, N, device="cuda", dtype=dt)
            tritonblas.matmul(a, b, out=out)
        json.dump(fires, sys.stdout)
    """)
    env = dict(os.environ)
    if env_value is None:
        env.pop("TRITONBLAS_ENABLE_K827_SPLITK2", None)
    else:
        env["TRITONBLAS_ENABLE_K827_SPLITK2"] = env_value
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, env=env, check=True)
    import json
    return json.loads(proc.stdout)


def test_gate_isolation_default_off():
    """Default OFF: gate fires on ZERO cells across the 23-cell test set."""
    fires = _run_with_env(None)
    assert fires == [], (
        f"Default-OFF gate fired on {len(fires)} cell(s); expected 0. "
        f"Fires: {fires}"
    )


def test_gate_isolation_env_on():
    """Env=1: gate fires on EXACTLY the 6 cohort cells, ZERO guard / extras."""
    fires_raw = _run_with_env("1")
    fired_set = {(M, N, K, dt_s) for (M, N, K, dt_s) in fires_raw}
    expected = {(M, N, K, str(dt)) for (M, N, K, dt) in COHORT}
    miss = expected - fired_set
    extra = fired_set - expected
    assert not miss and not extra, (
        f"Gate-fire set mismatch: missing={sorted(miss)}, unexpected={sorted(extra)}"
    )
    assert len(fires_raw) == len(COHORT), (
        f"Gate fired {len(fires_raw)} times; expected {len(COHORT)}"
    )


@pytest.mark.parametrize("M, N, K, dtype", COHORT,
                         ids=[f"{M}x{N}x{K}-{str(dt).split('.')[-1]}"
                              for (M, N, K, dt) in COHORT])
def test_split_k2_correctness(M, N, K, dtype):
    """env=1: split-K=2 reduction is numerically close to fp32 reference matmul."""
    # Re-run in a subprocess so the env-var-driven gate is hot.
    code = textwrap.dedent(f"""
        import os, sys, torch, importlib
        import tritonblas
        tbm = importlib.import_module("tritonblas.matmul")

        fires = []
        orig = tbm.maybe_dispatch_splitk2
        def counting(a, b, out):
            ok = orig(a, b, out)
            if ok:
                fires.append(1)
            return ok
        tbm.maybe_dispatch_splitk2 = counting

        torch.manual_seed(0)
        dt = getattr(torch, {repr(str(dtype).split('.')[-1])})
        a = torch.randn({M}, {K}, device="cuda", dtype=dt)
        b = torch.randn({K}, {N}, device="cuda", dtype=dt)
        ref = torch.matmul(a.float(), b.float()).to(dt)
        out = torch.empty({M}, {N}, device="cuda", dtype=dt)
        tritonblas.matmul(a, b, out=out)
        torch.cuda.synchronize()

        if not fires:
            print("FAIL gate did not fire", file=sys.stderr); sys.exit(2)

        diff = (out.float() - ref.float()).abs()
        max_abs = float(diff.max().item())
        try:
            torch.testing.assert_close(
                out.float(), ref.float(),
                rtol={TOL[dtype]['rtol']}, atol={TOL[dtype]['atol']},
            )
        except AssertionError as e:
            print(f"FAIL max_abs={{max_abs}}: {{e}}", file=sys.stderr); sys.exit(3)
        print(f"PASS max_abs={{max_abs}}")
    """)
    env = dict(os.environ)
    env["TRITONBLAS_ENABLE_K827_SPLITK2"] = "1"
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        pytest.fail(f"correctness probe failed (rc={proc.returncode}):\n"
                    f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
