"""K-677 regression guard: forbid hardcoded num_stages override on the
K-570 large-K square FP16/BF16 cohort.

Background
----------
K-677 (and the K-657 ablation it followed) tested whether bumping
num_stages from 2 to 3 (the "+1 pipeline stage" hypothesis) would close
the hipBLASLt gap on the cohort

    M = N in {1024, 2048, 4096}  x  K in {4096, 8192, 16384}
    dtype in {fp16, bf16}                                     (18 cells)

Result: REFUTED.

  * NS=3 with Origami's selected tile fails to compile on 18/18 cells:
    Triton's gfx942 software pipeliner allocates (NS-1) * per_stage_lds
    bytes of shared memory; per_stage_lds is exactly 65536 bytes for
    every cohort cell (Origami picks {64,64,256}, {128,128,128},
    {256,256,64} for M=N=1024/2048/4096 respectively, giving
    (BLOCK_M+BLOCK_N)*BLOCK_K*sizeof(fp16) = 65536), so NS=3 needs
    2*65536 = 131072 B and Triton raises
    OutOfResources("Required: 131072, Hardware limit: 65536").

  * NS=3 with co-modified (shrunk) tiles compiles but regresses every
    cohort cell (mean +7.28%, M=4096 sub-cohort +14% to +17%) per the
    K-677 wide paired graph-captured sweep (15 candidates x 18 shapes
    x 5 trials).

This test enforces that guard in CI: persistent_matmul_lt's num_stages
must remain whatever the Origami selector returns (default 2). If a
future change tries to silently inject NS=3 (or higher) for this cohort
via a hardcoded gate in matmul.py, this test fails loudly and points
at K-677 so the dead-end is not re-attempted.
"""

import re
from pathlib import Path

import pytest
import torch

import tritonblas  # noqa: F401  (ensure import path is set up)
from tritonblas.matmul import _make_matmul_selector


COHORT_M_N = (1024, 2048, 4096)
COHORT_K = (4096, 8192, 16384)
COHORT_DTYPES = (torch.float16, torch.bfloat16)

EXPECTED_NUM_STAGES = 2  # Origami's default for this cohort


def _cohort_cells():
    for dt in COHORT_DTYPES:
        for m in COHORT_M_N:
            for k in COHORT_K:
                yield (m, m, k, dt)


@pytest.mark.parametrize("m,n,k,dtype", list(_cohort_cells()))
def test_origami_selector_num_stages_is_two_for_K570_cohort(m, n, k, dtype):
    """Selector must return num_stages=2 for every cell of the K-570 cohort.

    See K-677 docstring above. If this fails, someone bumped the default
    or wired in a per-shape NS override; verify against the K-677 sweep
    data before shipping (every NS=3 candidate regressed every cell).
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm device for Origami selector")
    device = torch.device("cuda:0")
    sel = _make_matmul_selector(m, n, k, dtype, dtype, dtype, device)
    assert getattr(sel, "num_stages", None) == EXPECTED_NUM_STAGES, (
        f"K-677 guard: selector.num_stages={sel.num_stages} for "
        f"{m}x{n}x{k} {dtype}; expected {EXPECTED_NUM_STAGES}. "
        f"K-677 sweep showed every NS=3 candidate regresses every cell of "
        f"the M=N in {{1024,2048,4096}} x K in {{4096,8192,16384}} cohort. "
        f"Do not change without re-running the K-677 paired sweep."
    )


def test_no_hardcoded_num_stages_override_in_persistent_matmul_lt():
    """Static guard: persistent_matmul_lt must defer to the selector for
    num_stages, not hardcode an override.

    The accepted shape of the assignment is::

        num_stages = getattr(selector, "num_stages", 2)

    Anything else (e.g. ``num_stages = 3``, or a conditional that bumps
    NS based on M/N/K/dtype) is forbidden by K-677. This is a static
    text check so it runs on CPU-only CI agents too.
    """
    matmul_py = Path(__import__("tritonblas").__file__).parent / "matmul.py"
    src = matmul_py.read_text()

    # Locate the persistent_matmul_lt body and inspect every num_stages
    # assignment inside it.
    func_match = re.search(
        r"^def persistent_matmul_lt\(.*?(?=^def |\Z)",
        src,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert func_match, "could not locate persistent_matmul_lt in matmul.py"
    body = func_match.group(0)

    # Find every top-level assignment of num_stages in the function.
    # Match `num_stages = <rhs>` with whitespace around `=` (i.e. a statement),
    # not the kwarg form `num_stages=num_stages,` used in kernel calls.
    assignments = re.findall(
        r"^\s*num_stages[ \t]+=[ \t]+([^=].*)$",
        body,
        flags=re.MULTILINE,
    )
    assert assignments, "expected at least one num_stages assignment in persistent_matmul_lt"

    allowed_rhs = re.compile(r"""getattr\(\s*selector\s*,\s*["']num_stages["']\s*,\s*2\s*\)""")
    for rhs in assignments:
        rhs_stripped = rhs.strip().rstrip("#").strip()
        # Strip trailing comment
        rhs_no_comment = rhs_stripped.split("#", 1)[0].strip()
        assert allowed_rhs.search(rhs_no_comment), (
            f"K-677 guard: persistent_matmul_lt assigns num_stages = {rhs_no_comment!r}; "
            f"only `getattr(selector, 'num_stages', 2)` is allowed. "
            f"K-677 empirically refuted hardcoded NS overrides on the K-570 cohort; "
            f"see the comment block above this assignment in matmul.py and "
            f"tests/test_num_stages_cohort_guard.py for the rationale."
        )
