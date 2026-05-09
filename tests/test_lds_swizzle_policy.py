"""
K-1652: regression guard for the persistent matmul kpack/LDS-swizzle policy.

K-1635 conjectured that ``SQ_LDS_BANK_CONFLICT`` was the dominant loss vs.
hipBLASLt on the M=N=4096 x K in {2048..32768} x {bf16, fp16} cohort, and
that an LDS swizzle / ``kpack=2`` change would close the gap.

K-1652 ran the experiment on MI300X (gfx942) with rocprofv2:

================  =====================  ==============  ==========================
``kpack``         SQ_LDS_BANK_CONFLICT   SQ_INSTS_LDS    Scratch_Per_Workitem (B)
================  =====================  ==============  ==========================
``1`` (default)   ``0``                  10,477,568      16
``2`` (proposed)  ``8,388,608``          9,437,184       44
================  =====================  ==============  ==========================

The Triton AMD backend already picks an XOR-permuted LDS layout that makes
``kpack=1`` bank-conflict-free at the Origami-selected 256x256x64 / 8 warps /
2 stages config. Forcing ``kpack=2`` *disrupts* that swizzle and regresses
the cohort by ~9%. The hypothesis is falsified; ``kpack=1`` is empirically
optimal on this kernel.

This file is a runtime regression guard: it builds a small cohort-shaped
matmul, runs it through ``tritonblas.matmul`` (which goes through the same
``persistent_matmul_lt`` launch path that holds the ``kpack = 1`` literal),
and asserts numerical agreement with ``torch.matmul``. If a future edit
flips ``kpack`` away from 1, this test still passes (numerics are unchanged
by kpack), but the explanatory comment block in ``matmul.py`` is the
institutional memory that prevents the falsified experiment from being
re-run silently.
"""

import pytest
import torch
import tritonblas


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_persistent_matmul_cohort_cell_matches_torch(dtype):
    """K-1652: smoke-test that the kpack=1 / persistent path is correct.

    Uses a small cohort-shaped GEMM (M=N=512, K=2048) -- same M=N square
    aspect as the K-1652 cohort, K large enough to exercise the K loop and
    LDS pipeline, but small enough to run in seconds on a single MI300X CU.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/ROCm GPU")

    M, N, K = 512, 512, 2048
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    c = torch.empty(M, N, device="cuda", dtype=dtype)

    tritonblas.matmul(a, b, c)
    expected = torch.matmul(a, b)

    # Tolerance derived from K=2048 fma chain in bf16/fp16; matches the
    # tolerance used elsewhere in tests/test_matmul.py (atol=1, rtol=1) but
    # tightened because we know K-1652's kp1 path has rel_err ~3e-6.
    torch.testing.assert_close(c, expected, atol=2e-2, rtol=2e-2)
