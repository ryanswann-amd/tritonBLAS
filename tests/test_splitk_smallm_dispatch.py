"""K-513: Split-K small-M decode dispatch + correctness tests.

Two pieces under test:
 1. Dispatch gate — only the 32 in-scope shapes route to the new kernel.
 2. Correctness — when dispatched, the result matches torch.matmul to
    within fp16/bf16 mfma tolerance.
"""

import pytest
import torch

import tritonblas
from tritonblas.kernels.splitk_smallm_gemm import (
    should_dispatch_splitk_smallm,
    splitk_smallm_matmul,
    get_splitk_smallm_config,
    set_gate_enabled,
    is_gate_enabled,
)


@pytest.fixture(autouse=True)
def _enable_gate_for_test():
    """Tests must see gate ENABLED (in K-513 the production default is OFF)
    so they can exercise the dispatch logic without worrying about the
    deployment toggle. The previous value is restored on the way out so a
    flipped flag cannot leak from one test to the next."""
    prev = is_gate_enabled()
    set_gate_enabled(True)
    yield
    set_gate_enabled(prev)


# ----- gate ---------------------------------------------------------------

GATED_M = [1, 2, 4, 8]
GATED_K = [4096, 8192]
GATED_N = [1024, 2048, 4096, 8192]
GATED_DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("M", GATED_M)
@pytest.mark.parametrize("K", GATED_K)
@pytest.mark.parametrize("N", GATED_N)
@pytest.mark.parametrize("dtype", GATED_DTYPES)
def test_gate_in_scope(M, N, K, dtype):
    assert should_dispatch_splitk_smallm(M, N, K, dtype, dtype, dtype)
    assert get_splitk_smallm_config(M, N, K, dtype) is not None


@pytest.mark.parametrize(
    "M, N, K, dtype",
    [
        # Out of M range
        (16, 4096, 4096, torch.float16),
        (32, 4096, 8192, torch.bfloat16),
        # Out of K range (the very point of the gate — K-484 lesson)
        (4, 4096, 1024, torch.float16),
        (4, 4096, 2048, torch.bfloat16),
        # Out of N range
        (4, 512, 4096, torch.float16),
        # Wrong dtype
        (4, 4096, 4096, torch.float32),
        # Mixed dtype not allowed
    ],
)
def test_gate_out_of_scope(M, N, K, dtype):
    assert not should_dispatch_splitk_smallm(M, N, K, dtype, dtype, dtype)


def test_gate_mixed_dtype_rejected():
    # Mixed input/output dtype — out of scope.
    assert not should_dispatch_splitk_smallm(
        4, 4096, 4096, torch.float16, torch.bfloat16, torch.float16
    )


# ----- correctness --------------------------------------------------------


def _cpu_fp32_reference(a, b):
    """Reference matmul that does not depend on cuBLAS/hipBLASLt or torch.matmul.

    Some MI300X / hipBLASLt builds have shape-specific launch failures on
    skinny-M GEMMs (notably M=1 with N in {1024, 2048}); this affects the
    reference, not the split-K kernel under test, so we compute the
    reference by moving inputs to CPU fp32 and using torch's CPU mm path.
    """
    a32 = a.detach().to("cpu", dtype=torch.float32)
    b32 = b.detach().to("cpu", dtype=torch.float32)
    return (a32 @ b32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("M", [1, 2, 4, 8])
@pytest.mark.parametrize("N", [1024, 4096])
@pytest.mark.parametrize("K", [4096, 8192])
@pytest.mark.parametrize("dtype", GATED_DTYPES)
def test_splitk_smallm_correctness(M, N, K, dtype):
    """Direct kernel call: split-K small-M result matches an fp32 CPU reference.

    Uses CPU fp32 so the reference never goes through cuBLAS/hipBLASLt and
    cannot be poisoned by environment-specific GEMM launch bugs (those
    would otherwise mask kernel correctness behind reference failures).
    """
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=dtype) * 0.1
    b = torch.randn((K, N), device="cuda", dtype=dtype) * 0.1
    c = torch.empty((M, N), device="cuda", dtype=dtype)

    cfg = get_splitk_smallm_config(M, N, K, dtype)
    assert cfg is not None
    split_k, block_n, block_k = cfg

    splitk_smallm_matmul(a, b, c, split_k=split_k, block_n=block_n, block_k=block_k)
    ref = _cpu_fp32_reference(a, b).to(dtype)

    # bf16/fp16 mfma tolerance for K up to 8192
    atol = 1.0 if dtype == torch.bfloat16 else 0.5
    rtol = 1e-2
    torch.testing.assert_close(c.cpu(), ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("dtype", GATED_DTYPES)
def test_splitk_split_k_choices_agree(dtype):
    """Self-consistency: SPLIT_K in {2,4,8} must produce the same output
    (within mfma tolerance) for the same shape — the split factor is a
    compute-distribution choice and cannot change the math.

    This test never touches torch.matmul / cuBLAS so it cannot regress for
    environment reasons unrelated to the kernel itself."""
    torch.manual_seed(0)
    M, N, K = 4, 4096, 8192
    a = torch.randn((M, K), device="cuda", dtype=dtype) * 0.1
    b = torch.randn((K, N), device="cuda", dtype=dtype) * 0.1
    outs = []
    for sk in (2, 4, 8):
        c = torch.empty((M, N), device="cuda", dtype=dtype)
        splitk_smallm_matmul(a, b, c, split_k=sk, block_n=128, block_k=64)
        outs.append(c)
    # All three must agree to within mfma tolerance for fp16/bf16.
    atol = 0.5 if dtype == torch.float16 else 1.0
    torch.testing.assert_close(outs[0], outs[1], atol=atol, rtol=1e-2)
    torch.testing.assert_close(outs[0], outs[2], atol=atol, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("M", [1, 4])
@pytest.mark.parametrize("N", [2048])
@pytest.mark.parametrize("K", [4096, 8192])
@pytest.mark.parametrize("dtype", GATED_DTYPES)
def test_dispatch_through_matmul(M, N, K, dtype):
    """End-to-end: tritonblas.matmul on a gated shape produces correct result."""
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=dtype) * 0.1
    b = torch.randn((K, N), device="cuda", dtype=dtype) * 0.1
    c = torch.empty((M, N), device="cuda", dtype=dtype)

    tritonblas.matmul(a, b, c)
    ref = _cpu_fp32_reference(a, b).to(dtype)
    atol = 1.0 if dtype == torch.bfloat16 else 0.5
    torch.testing.assert_close(c.cpu(), ref, atol=atol, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_dispatch_out_of_gate_unaffected():
    """A shape outside the gate must not change behaviour — sanity that the
    gate is tight (K-654 lesson)."""
    M, N, K = 64, 4096, 4096
    dtype = torch.float16
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=dtype) * 0.1
    b = torch.randn((K, N), device="cuda", dtype=dtype) * 0.1
    c = torch.empty((M, N), device="cuda", dtype=dtype)

    # Just check it runs without error and gives a correct result via the
    # original persistent path (no kernel dispatch swap).
    tritonblas.matmul(a, b, c)
    ref = _cpu_fp32_reference(a, b).to(dtype)
    torch.testing.assert_close(c.cpu(), ref, atol=1.0, rtol=1e-2)


# ----- gate toggle --------------------------------------------------------


def test_gate_disable_blocks_all_shapes():
    """When the master toggle is off, every shape (including in-scope ones)
    must be rejected by the gate — proves the bench harness's disable path
    actually disables and is not racing with stale state."""
    set_gate_enabled(False)
    try:
        for M in GATED_M:
            for N in GATED_N:
                for K in GATED_K:
                    for dtype in GATED_DTYPES:
                        assert not should_dispatch_splitk_smallm(
                            M, N, K, dtype, dtype, dtype
                        ), f"gate leaked for {M}x{N}x{K} {dtype}"
    finally:
        set_gate_enabled(True)
    # Sanity: re-enabling restores in-scope dispatch.
    assert should_dispatch_splitk_smallm(
        4, 4096, 4096, torch.float16, torch.float16, torch.float16
    )


def test_gate_toggle_returns_previous():
    """set_gate_enabled returns the previous value, regardless of starting
    state. The autouse fixture leaves us with the gate currently enabled."""
    assert is_gate_enabled() is True
    prev = set_gate_enabled(False)
    assert prev is True
    prev = set_gate_enabled(True)
    assert prev is False
    assert is_gate_enabled() is True


def test_gate_default_is_enabled_at_import():
    """Production behaviour: the K-513 amortized cohort comparison on
    MI300X showed split-K beating the K-144 persistent path on every
    measured bucket in the gated cohort (geomean 4.40x; see
    output/cohort_amortized.csv in the K-513 workspace), so the module
    default is ENABLED. This test bypasses the autouse fixture by
    re-importing the module in a fresh namespace."""
    import importlib
    import tritonblas.kernels.splitk_smallm_gemm as fresh
    fresh = importlib.reload(fresh)
    assert fresh._GATE_ENABLED is True, (
        "Gate must default to enabled — measured wins on the gated cohort."
    )
    # Restore for subsequent tests (autouse fixture re-enables anyway, but
    # reload changed the underlying module global, so be explicit).
    fresh.set_gate_enabled(True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_disabled_gate_routes_to_persistent_for_in_scope_shape():
    """End-to-end: an in-scope shape with the gate disabled must still
    produce a correct result (via the K-144 persistent path)."""
    M, N, K = 4, 4096, 4096
    dtype = torch.float16
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=dtype) * 0.1
    b = torch.randn((K, N), device="cuda", dtype=dtype) * 0.1
    c = torch.empty((M, N), device="cuda", dtype=dtype)

    set_gate_enabled(False)
    try:
        tritonblas.matmul(a, b, c)
    finally:
        set_gate_enabled(True)
    ref = _cpu_fp32_reference(a, b).to(dtype)
    torch.testing.assert_close(c.cpu(), ref, atol=1.0, rtol=1e-2)
