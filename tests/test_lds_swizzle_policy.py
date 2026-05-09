"""
Regression guard for K-1652: kpack policy on the persistent GEMM kernel.

Why this test exists
--------------------
K-1598/K-1629/K-1634 conjectured that ``SQ_LDS_BANK_CONFLICT`` was the
dominant loss vs. hipBLASLt on the M=N=4096 x K in {2048..32768} x {bf16, fp16}
cohort, and that an LDS swizzle / ``kpack=2`` change would close the gap.

K-1652 ran the experiment on MI300X (gfx942) with rocprofv2 and direct
hot-cache HIP-graph timing. The hypothesis was *falsified*:

================  =====================  ==============  ==========================
``kpack``         SQ_LDS_BANK_CONFLICT   SQ_INSTS_LDS    Scratch_Per_Workitem (B)
================  =====================  ==============  ==========================
``1`` (default)   ``0``                  10,477,568      16
``2`` (proposed)  ``8,388,608``          9,437,184       44
================  =====================  ==============  ==========================

* ``kpack=1`` cohort geomean speedup vs torch (hipBLASLt): **0.895x**
* ``kpack=2`` cohort geomean speedup vs torch (hipBLASLt): **0.810x**

The Triton AMD backend already picks an XOR-permuted LDS layout that makes
``kpack=1`` bank-conflict-free at the Origami-selected 256x256x64 / 8 warps /
2 stages config. Forcing ``kpack=2`` *disrupts* that swizzle, introducing ~89%
LDS bank-conflict ratio AND 28 extra B/work-item of scratch.

This test is a static regression guard: any future change that flips the
``kpack`` literal in :mod:`tritonblas.matmul` away from ``1`` MUST first
re-run the K-1652 PMC sweep and update the comment block (so the next
engineer cannot silently re-introduce the regression).

The test is GPU-free on purpose: it inspects the source so it runs in CI
without an MI300X allocation.
"""

import re
from pathlib import Path

import pytest


_MATMUL_PY = (
    Path(__file__).resolve().parent.parent
    / "include"
    / "tritonblas"
    / "matmul.py"
)


def _matmul_source() -> str:
    assert _MATMUL_PY.exists(), f"missing {_MATMUL_PY}"
    return _MATMUL_PY.read_text()


def test_persistent_matmul_kpack_is_one():
    """``persistent_matmul_lt`` must launch the kernel with ``kpack=1``.

    K-1652: kpack=2 introduces SQ_LDS_BANK_CONFLICT ~= 0.89 * LDS_INSTS and
    regresses the M=N=4096 cohort by ~9%. Do not change without updating the
    PMC evidence block above the literal in ``include/tritonblas/matmul.py``.
    """
    src = _matmul_source()
    # Must contain at least one ``kpack = 1`` literal assignment in the
    # persistent / streamk launch sites.
    assert (
        len(re.findall(r"^\s*kpack\s*=\s*1\s*$", src, re.MULTILINE)) >= 2
    ), (
        "Expected at least two `kpack = 1` literal assignments in matmul.py "
        "(one in persistent_matmul_lt, one in streamk_matmul_lt). "
        "K-1652: changing this literal away from 1 regresses the M=N=4096 "
        "cohort by ~9% and re-introduces 8.4M LDS bank conflicts per kernel. "
        "If you intentionally changed it, update this test AND re-run the "
        "K-1652 PMC sweep."
    )
    # Must NOT contain any ``kpack = 2`` (the falsified swizzle hypothesis).
    assert (
        re.search(r"^\s*kpack\s*=\s*2\s*$", src, re.MULTILINE) is None
    ), (
        "Found `kpack = 2` in matmul.py. K-1652 PMC evidence: kpack=2 "
        "introduces SQ_LDS_BANK_CONFLICT=8,388,608 (vs 0 at kpack=1) on the "
        "M=N=4096 cohort. Revert or update the PMC comment block."
    )


def test_kpack_pmc_evidence_comment_is_present():
    """The PMC-evidence comment block above ``kpack = 1`` must be intact.

    This guards against a refactor that moves the literal but loses the
    explanation of *why* the value is 1 -- without that, the next engineer
    will re-run the same falsified swizzle experiment.
    """
    src = _matmul_source()
    assert "K-1652" in src, (
        "Expected the K-1652 PMC-evidence comment block in matmul.py near "
        "`kpack = 1`. The comment is the institutional memory that prevents "
        "re-running the falsified LDS-swizzle experiment."
    )
    assert "SQ_LDS_BANK_CONFLICT" in src, (
        "Expected the `SQ_LDS_BANK_CONFLICT` PMC counter name in matmul.py "
        "to anchor the K-1652 evidence block. Restore it or update this test."
    )


def test_no_centralised_kpack_module_resurrected():
    """``include/tritonblas/lds_swizzle.py`` must NOT exist.

    A prior K-1652 attempt landed a 95-LOC ``pick_kpack`` policy module that
    was a no-op wrapper around the literal ``1``. Code review (see ticket
    K-1652 review notes from The Minimalist) flagged it as over-engineering
    and asked for the literal + comment instead. This test prevents the
    no-op module from being resurrected.
    """
    swizzle_path = (
        Path(__file__).resolve().parent.parent
        / "include"
        / "tritonblas"
        / "lds_swizzle.py"
    )
    assert not swizzle_path.exists(), (
        f"{swizzle_path} was resurrected. K-1652 review flagged the wrapper "
        "module as over-engineering for a literal constant. Inline `kpack = 1` "
        "and document the PMC evidence in a comment instead."
    )
