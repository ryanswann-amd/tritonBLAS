"""K-1672 (productionizes K-1652): regression guard for the persistent matmul
LDS-swizzle / kpack policy on the M=N=4096 cohort.

Background
----------
K-1635 (and predecessors K-1598/K-1629/K-1634) hypothesised that
``SQ_LDS_BANK_CONFLICT`` was the dominant loss vs. hipBLASLt on the
``M=N=4096 x K in {2048,4096,8192,16384,32768} x {bf16, fp16}`` cohort,
and that an LDS-swizzle / ``kpack=2`` change would close the gap.

K-1652 ran the experiment end-to-end on MI300X (gfx942) with rocprofv2,
and **falsified** the hypothesis:

================  =====================  ==============  ==========================
``kpack``         SQ_LDS_BANK_CONFLICT   SQ_INSTS_LDS    Scratch_Per_Workitem (B)
================  =====================  ==============  ==========================
``1`` (default)   ``0``                  10,477,568      16
``2`` (proposed)  ``8,388,608``          9,437,184       44
================  =====================  ==============  ==========================

The Triton AMD backend already picks an XOR-permuted LDS layout that makes
``kpack=1`` bank-conflict-free at the Origami-selected 256x256x64 / 8 warps /
2 stages config. Forcing ``kpack=2`` *disrupts* that swizzle and regresses
the cohort by ~9% (cohort geomean: 0.895x at kp1 vs 0.810x at kp2). K-1641's
PMC decomposition further showed the residual ~10% gap is dominated by
``SQ_WAIT_INST_LDS/wave`` (LDS pipeline back-pressure, NOT bank conflicts);
the recommended next axis is deeper prefetch staging (BK 64->128 with tile
narrowing), which is out of scope for this PR.

Purpose of these tests
----------------------
Two complementary regression guards:

1. ``test_kpack_pinned_to_one_in_source`` -- a *static* assertion that the
   ``kpack = N`` literal in BOTH ``persistent_matmul_lt`` and
   ``streamk_matmul_lt`` is exactly ``1``. This is the contract that the
   PR is built around: a future contributor flipping the literal to 2
   trips this test even if numerical equivalence is preserved (as it
   would be -- kpack=2 is mathematically identical, just slower).

2. ``test_persistent_matmul_cohort_cell_matches_torch`` -- a *runtime*
   numerical-correctness guard on a small cohort-shaped GEMM, so the
   path that actually consumes the pinned literal is exercised.
"""

import importlib
import inspect
import re

import pytest
import torch
import tritonblas

# Resolve the submodule explicitly: ``tritonblas.matmul`` is shadowed in
# ``tritonblas/__init__.py`` by the re-exported ``matmul`` function, so a
# bare ``from tritonblas import matmul`` would bind the function, not the
# module that defines ``persistent_matmul_lt`` / ``streamk_matmul_lt``.
_tb_matmul_module = importlib.import_module("tritonblas.matmul")


# --- (1) static policy pin --------------------------------------------------
#
# We introspect the source of the two persistent-matmul launch wrappers and
# assert that the ``kpack = <int>`` literal is exactly 1. This is the
# institutional-memory anchor for K-1652's falsified-kpack=2 finding: a future
# contributor flipping the literal cannot silently revert without tripping
# this assertion, even though kpack=2 would still produce numerically
# identical output (so a pure equivalence test would not catch it).

_KPACK_LITERAL_RE = re.compile(r"^\s*kpack\s*=\s*(\d+)\s*(?:#.*)?$", re.MULTILINE)


@pytest.mark.parametrize(
    "fn_name",
    ["persistent_matmul_lt", "streamk_matmul_lt"],
)
def test_kpack_pinned_to_one_in_source(fn_name):
    """K-1652/K-1672: the ``kpack`` literal in the persistent-matmul launch
    wrappers must remain ``1``. PMC evidence: kpack=2 introduces 8.4M LDS
    bank conflicts/kernel and regresses the M=N=4096 cohort ~9%. See the
    comment block above the literal in ``include/tritonblas/matmul.py``.
    """
    fn = getattr(_tb_matmul_module, fn_name)
    src = inspect.getsource(fn)
    matches = _KPACK_LITERAL_RE.findall(src)
    assert matches, (
        f"could not locate a `kpack = <int>` literal in {fn_name}; "
        "if the launch wrapper was refactored, update this test AND the "
        "K-1652/K-1672 comment anchor in include/tritonblas/matmul.py."
    )
    bad = [v for v in matches if int(v) != 1]
    assert not bad, (
        f"{fn_name} has kpack={bad[0]} but K-1652 PMC evidence requires "
        "kpack=1 on the M=N=4096 cohort (kpack=2 -> 8.4M bank conflicts/kernel, "
        "~9% cohort regression). See the comment block above the kpack literal "
        "in include/tritonblas/matmul.py and the gist linked from K-1672 before "
        "changing this value."
    )


# --- (2) runtime numerical correctness --------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_persistent_matmul_cohort_cell_matches_torch(dtype):
    """K-1652/K-1672: smoke-test that the kpack=1 / persistent path is correct.

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
