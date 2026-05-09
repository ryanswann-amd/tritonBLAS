"""K-1653 regression tests for the long-K wide-N streamk auto-route.

Load-bearing change: ``tritonblas.matmul(a, b)`` and ``tritonblas.matmul_out``
now dispatch through ``streamk_matmul_lt`` (instead of ``persistent_matmul_lt``)
whenever ``K >= 16384 AND N >= 8192``, even when the caller did not explicitly
set ``enable_streamk=True``. This is the single mechanism that closes the K-1634
PMC gap on the 6 worst-loser cells (geomean speedup vs hipBLASLt 0.82x -> 0.94x).

These tests pin:

* The auto-route window (positive: in-window dispatches to streamk; negative:
  out-of-window stays on the persistent path).
* Numerical correctness of the auto-routed path on the K-1634 cohort vs
  ``torch.matmul`` (which lowers to hipBLASLt on ROCm).
* The ``_n`` / ``_k`` attributes the selector exposes for any downstream
  override that needs the shape (kept as a stable public-ish API).

A future change that disables, narrows, or moves the auto-route window will
break one of these tests instead of silently regressing the K-1634 cells.
"""

import pytest
import torch

# Skip the whole module unless CUDA is available -- selector requires a
# torch.device with index that origami can map to gfx hardware.
pytest.importorskip("origami")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from tritonblas.origami import OrigamiMatmulSelector  # noqa: E402


# K-1634 worst-loser cohort: M x N=16384 x K, bf16 / fp16. The success
# criterion of K-1653 is geomean speedup vs hipBLASLt >= 0.90x on these
# cells. The auto-route is the load-bearing change that achieves it.
K1634_CELLS = [
    (4096, 16384, 16384),
    (4096, 16384, 32768),
    (8192, 16384, 16384),
    (8192, 16384, 32768),
]


def _selector(M, N, K, dtype=torch.bfloat16):
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return OrigamiMatmulSelector(M, N, K, dtype, dtype, dtype, device)


# ----------------------------------------------------------------------------
# Selector pick stays LDS-legal in the auto-route window.  The auto-route
# decision is taken in matmul.py from (M, N, K) directly, but the selector
# still has to produce a config that fits LDS for the streamk kernel to
# launch -- this test catches a future origami change that picks an
# over-allocating tile in this regime.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K", K1634_CELLS)
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
def test_in_window_selector_pick_is_lds_legal(M, N, K, dt):
    sel = _selector(M, N, K, dt)
    # Origami's own LDS check runs inside the constructor; if we got a
    # selector back, the chosen MT fit. Guard the obvious invariants.
    assert sel.block_m > 0 and sel.block_n > 0 and sel.block_k > 0
    assert K % sel.block_k == 0, (
        f"selector chose BK={sel.block_k} which doesn't divide K={K}"
    )


# ----------------------------------------------------------------------------
# Selector exposes problem dimensions on the public-ish `_n` / `_k`
# attributes. Keep this so any downstream consumer (and the auto-route
# decision reused in matmul.py) cannot be silently broken by a rename.
# ----------------------------------------------------------------------------
def test_selector_exposes_problem_dims():
    sel = _selector(4096, 16384, 16384, torch.bfloat16)
    assert getattr(sel, "_k", None) == 16384, "OrigamiMatmulSelector._k missing"
    assert getattr(sel, "_n", None) == 16384, "OrigamiMatmulSelector._n missing"
    assert getattr(sel, "_m", None) == 4096, "OrigamiMatmulSelector._m missing"


# ----------------------------------------------------------------------------
# Auto-route dispatch: in-window cells must hit `streamk_matmul_lt`, even
# when the caller passes `enable_streamk=False`. Out-of-window cells must
# stay on `persistent_matmul_lt`. Asserted by patching both kernel entry
# points and watching which one is invoked.
# ----------------------------------------------------------------------------
def _patched_dispatch_call(M, N, K, enable_streamk, dtype=torch.bfloat16):
    """Run a tiny matmul and return ('streamk' | 'persistent') based on
    which dispatch entry was actually invoked. Uses small tensors and
    monkey-patches the two kernels to no-op (writing zeros) to avoid the
    cost of the real streamk launch in unit tests."""
    import sys
    import tritonblas  # ensures tritonblas.matmul SUBMODULE is loaded
    # ``tritonblas.matmul`` the attribute is a function that shadows the
    # submodule of the same name; reach the dispatcher module via sys.modules.
    mm = sys.modules["tritonblas.matmul"]
    calls = []
    real_sk = mm.streamk_matmul_lt
    real_pm = mm.persistent_matmul_lt

    def fake_sk(a, b, c, *args, **kwargs):
        calls.append("streamk")
        c.zero_()
        return c

    def fake_pm(a, b, c, *args, **kwargs):
        calls.append("persistent")
        c.zero_()
        return c

    mm.streamk_matmul_lt = fake_sk
    mm.persistent_matmul_lt = fake_pm
    try:
        # Use 1x1 tensors padded with the actual M/N/K shape via empty().
        # We never run the real kernel -- only the selector + dispatcher.
        a = torch.zeros(M, K, device="cuda", dtype=dtype)
        b = torch.zeros(K, N, device="cuda", dtype=dtype)
        # Call the dispatcher exposed at package scope (tritonblas.matmul).
        tritonblas.matmul(a, b, enable_streamk=enable_streamk)
    finally:
        mm.streamk_matmul_lt = real_sk
        mm.persistent_matmul_lt = real_pm
    assert len(calls) == 1, f"dispatcher fired {len(calls)} kernels (expected 1)"
    return calls[0]


@pytest.mark.parametrize("M,N,K", K1634_CELLS)
def test_in_window_auto_routes_to_streamk(M, N, K):
    # User did NOT request streamk; auto-route MUST fire.
    routed = _patched_dispatch_call(M, N, K, enable_streamk=False)
    assert routed == "streamk", (
        f"K-1653 auto-route did not fire for in-window cell "
        f"M={M} N={N} K={K} (got {routed!r}); the K-1634 worst-loser "
        f"speedup floor of 0.90x will regress to ~0.82x without it."
    )


@pytest.mark.parametrize("M,N,K", [
    (4096, 16384,  8192),    # K below cutoff
    (4096,  4096, 16384),    # N below cutoff
    (1024,  1024,  1024),    # both below cutoff
    (8192,  4096, 32768),    # N below cutoff even with long K
])
def test_out_of_window_stays_on_persistent(M, N, K):
    routed = _patched_dispatch_call(M, N, K, enable_streamk=False)
    assert routed == "persistent", (
        f"K-1653 auto-route fired spuriously on out-of-window cell "
        f"M={M} N={N} K={K} (got {routed!r}); the trigger envelope is "
        f"K>=16384 AND N>=8192 -- a wider envelope would over-route "
        f"persistent-favorable cells through streamk and slow them down."
    )


# ----------------------------------------------------------------------------
# Numerical correctness of the auto-routed path on the K-1634 cohort.
# This is the single test that proves the load-bearing change does not
# silently corrupt long-K reductions through the streamk path.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K", K1634_CELLS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_long_k_wide_n_matmul_correctness(M, N, K, dtype):
    import math
    import tritonblas
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    out = tritonblas.matmul(a, b)
    ref = torch.matmul(a, b)
    err = (out.float() - ref.float()).abs().max().item()
    # Long-K accumulation tolerance: base * sqrt(K). Matches the bench
    # harness in scripts/bench_k1653.py so a numerical regression cannot
    # land silently through either gate.
    base = 0.05 if dtype == torch.bfloat16 else 0.005
    tol = base * math.sqrt(K)
    assert err <= tol, (
        f"K-1653 streamk auto-route correctness failed: "
        f"M={M} N={N} K={K} dt={dtype} max_abs_err={err:.4g} tol={tol:.4g}"
    )
