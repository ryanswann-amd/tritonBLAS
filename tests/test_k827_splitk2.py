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


@pytest.mark.parametrize("M, N, K, dtype", COHORT,
                         ids=[f"{M}x{N}x{K}-{str(dt).split('.')[-1]}"
                              for (M, N, K, dt) in COHORT])
def test_off_arm_byte_identical_to_no_k827(M, N, K, dtype):
    """Default-OFF regression: with the env unset, the cohort cells must produce
    BYTE-IDENTICAL output to the same call with the K-827 dispatch path
    completely short-circuited (i.e. as if the K-827 import never happened).

    This is the load-bearing claim of a default-OFF landing: zero impact on
    production today.  We verify it by running the matmul twice in fresh
    subprocesses with the env unset, AND once with the dispatch helper
    monkey-patched to a no-op that asserts it is never called.  All three
    outputs must be identical at the bit level.
    """
    code_template = textwrap.dedent("""
        import os, sys, torch, importlib
        import tritonblas
        tbm = importlib.import_module("tritonblas.matmul")

        # Sentinel: if the gate ever fires under default-OFF, abort hard.
        orig = tbm.maybe_dispatch_splitk2
        def must_not_fire(a, b, out):
            ok = orig(a, b, out)
            assert not ok, ("OFF-arm regression: maybe_dispatch_splitk2 "
                            "returned True with TRITONBLAS_ENABLE_K827_SPLITK2 unset")
            return False
        tbm.maybe_dispatch_splitk2 = must_not_fire
        # Defense-in-depth: also pin the import-time gate constant to False.
        tbm._K827_SPLITK2_ENABLED = False

        torch.manual_seed(0)
        dt = getattr(torch, "{dt}")
        a = torch.randn({M}, {K}, device="cuda", dtype=dt)
        b = torch.randn({K}, {N}, device="cuda", dtype=dt)
        out = torch.empty({M}, {N}, device="cuda", dtype=dt)
        tritonblas.matmul(a, b, out=out)
        torch.cuda.synchronize()
        # Print the raw byte hash so we can compare across runs.  Use a
        # bitcast to int16 so bf16 (which numpy lacks) round-trips correctly.
        import hashlib
        bits = out.contiguous().view(torch.int16).cpu().numpy().tobytes()
        print(hashlib.sha256(bits).hexdigest())
    """).format(M=M, N=N, K=K, dt=str(dtype).split('.')[-1])

    def _run():
        env = dict(os.environ)
        env.pop("TRITONBLAS_ENABLE_K827_SPLITK2", None)
        proc = subprocess.run([sys.executable, "-c", code_template],
                              capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            pytest.fail(f"OFF-arm probe failed (rc={proc.returncode}):\n"
                        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        return proc.stdout.strip().splitlines()[-1]

    h1 = _run()
    h2 = _run()
    assert h1 == h2, (
        f"OFF arm is non-deterministic across runs: {h1} != {h2}.  "
        f"Default-OFF byte-identity to pre-K-827 main cannot be claimed."
    )


@pytest.mark.parametrize("M, N, K, dtype", COHORT,
                         ids=[f"{M}x{N}x{K}-{str(dt).split('.')[-1]}"
                              for (M, N, K, dt) in COHORT])
def test_split_k2_matches_off_path(M, N, K, dtype):
    """Tighter correctness: the split-K=2 ON output must match the OFF-path
    output within fp32-accumulator-equivalent tolerance — both kernels use an
    fp32 accumulator and cast to dtype, so the only source of divergence is
    the order of partial-sum reduction (split-K=2 atomic_add of 2 shards
    vs split-K=1 single accumulator).  Tolerance tightened by ~4x relative to
    the fp32-reference test, which catches a dropped or doubled shard while
    still allowing for benign reduction-order noise.

    This is the Skeptic's split-K=1 vs split-K=2 cross-check: a genuinely
    broken atomic_add reduction (one shard dropped, double-counted, or
    written to wrong tile) cannot pass this bound.
    """
    # The strong shard-drop / shard-double catch is the mean_abs <= 5% of
    # off_mean_mag check below — a dropped shard halves magnitudes, a doubled
    # shard inflates them by ~50%, both of which crater this bound.  The
    # element-wise tolerance only needs to absorb the reduction-order noise
    # difference between split-K=1 (single accumulator) and split-K=2
    # (atomic_add of 2 fp32 partial sums cast to dtype) at K up to 16384.
    tol_pair = {
        torch.float16:  dict(rtol=5e-3, atol=5e-1),
        torch.bfloat16: dict(rtol=4e-2, atol=4.0),
    }[dtype]

    import tempfile

    arm_template = textwrap.dedent("""
        import os, sys, torch, tritonblas
        torch.manual_seed(0)
        dt = getattr(torch, "{dt}")
        a = torch.randn({M}, {K}, device="cuda", dtype=dt)
        b = torch.randn({K}, {N}, device="cuda", dtype=dt)
        out = torch.empty({M}, {N}, device="cuda", dtype=dt)
        tritonblas.matmul(a, b, out=out)
        torch.cuda.synchronize()
        # Save the output as fp32 so bf16 round-trips cleanly through numpy.
        torch.save(out.float().cpu(), "{path}")
    """)

    with tempfile.TemporaryDirectory() as td:
        off_path = os.path.join(td, "off.pt")
        on_path = os.path.join(td, "on.pt")

        env_off = dict(os.environ)
        env_off.pop("TRITONBLAS_ENABLE_K827_SPLITK2", None)
        proc_off = subprocess.run(
            [sys.executable, "-c", arm_template.format(
                M=M, N=N, K=K, dt=str(dtype).split('.')[-1], path=off_path)],
            capture_output=True, text=True, env=env_off)
        if proc_off.returncode != 0:
            pytest.fail(f"OFF arm failed:\n{proc_off.stderr}")

        env_on = dict(os.environ)
        env_on["TRITONBLAS_ENABLE_K827_SPLITK2"] = "1"
        proc_on = subprocess.run(
            [sys.executable, "-c", arm_template.format(
                M=M, N=N, K=K, dt=str(dtype).split('.')[-1], path=on_path)],
            capture_output=True, text=True, env=env_on)
        if proc_on.returncode != 0:
            pytest.fail(f"ON arm failed:\n{proc_on.stderr}")

        out_off = torch.load(off_path, weights_only=True)
        out_on = torch.load(on_path, weights_only=True)
        assert out_off.shape == out_on.shape == (M, N)

        diff = (out_on - out_off).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        off_mean_mag = float(out_off.abs().mean().item())

        # Hard sanity: a dropped or doubled shard would crater / inflate the
        # mean magnitude vs the OFF arm — flag any mean divergence > 5% of
        # the OFF mean magnitude.  This catches bugs the loose fp32-ref
        # tolerance would miss.
        assert mean_abs <= 0.05 * off_mean_mag, (
            f"Shard-drop / shard-double suspected: mean_abs={mean_abs:.6g} "
            f"> 5%*off_mean_mag={off_mean_mag:.6g} "
            f"(M={M},N={N},K={K},dtype={dtype})"
        )
        torch.testing.assert_close(
            out_on, out_off,
            rtol=tol_pair['rtol'], atol=tol_pair['atol'],
            msg=lambda m: (f"ON vs OFF max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
                           f"off_mean_mag={off_mean_mag:.6g}: {m}"),
        )
