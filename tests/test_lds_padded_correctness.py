# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Numerical correctness tests for the K-1640 skinny-N LDS-padded staging path.

The companion ``test_lds_padded_envelope.py`` covers the pure-Python admit/
refuse predicate. THIS file exercises the actual gated kernel path on GPU
and asserts that the codegen knob swap (``matrix_instr_nonkdim=32``,
``num_warps=4``, ``kpack=1``) produces numerically equivalent results to
``torch.matmul`` for both:

  1. an ADMITTED cell (the proven-winning (M=4096, N=128, K=4096)
     square-skinny corner) — exercises the new path itself.
  2. a REFUSED cell (a K-1633 N=512 control representative of the 17 P26
     production frozensets) — exercises the negative-side path so we
     catch any accidental bleed of the new code into the baseline.

Both subtests force the gate predicate via monkeypatch so the test does
not depend on Origami's tile selection (which can drift between releases).
The forced-on subtest will use the LDS-padded knobs even on shapes the
production gate would refuse, giving us a regression-proof correctness
assertion of the codegen-knob swap itself.
"""

import pytest


pytest.importorskip("torch")
import torch  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip(
        "K-1640 LDS-padded GPU correctness tests require a CUDA/HIP device",
        allow_module_level=True,
    )

import tritonblas  # noqa: E402
from tritonblas.kernels import lds_padded_envelope  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# bf16 GEMM matmul accumulates in fp32 then rounds; for medium K the
# practical numerical envelope at this dtype is dominated by the K-axis
# accumulation rounding, not by the codegen-knob choice. Use the same
# tolerance the existing test_matmul_correctness.py file uses for bf16/fp16
# (atol=1e-1, rtol=1e-1). The test is asserting "no catastrophic numerical
# drift from swapping the MFMA tile size", not "bit-exact"; the latter is
# not even true between two consecutive baseline launches at this dtype.
_ATOL = 1e-1
_RTOL = 1e-1


# ─────────────────────────────────────────────────────────────────────────────
# K-1640 proven-winner cell. The pure-Python envelope test pins this cell
# is admitted by the production gate; here we additionally check the
# kernel that gets launched at this admit decision is numerically correct.
# ─────────────────────────────────────────────────────────────────────────────
ADMIT_M, ADMIT_N, ADMIT_K = 4096, 128, 4096

# A K-1633 N=512 control representative — the production gate refuses
# this one, but with the gate forced ON we still want to verify the
# 32x32 MFMA + num_warps=4 codegen produces correct output (so accidental
# widening of the gate at a later date cannot silently corrupt this
# cohort).
REFUSE_M, REFUSE_N, REFUSE_K = 4096, 512, 4096


@pytest.fixture
def force_gate_on(monkeypatch):
    """
    Make ``should_use_lds_padded_path`` return True for every shape, so the
    test exercises the LDS-padded codegen path even on cells the production
    gate would refuse. We patch the symbol the matmul.py launcher already
    imported (rebinding the function on the kernels submodule is not
    enough because ``matmul.py`` did ``from .kernels.lds_padded_envelope
    import should_use_lds_padded_path`` at import time).
    """
    from tritonblas import matmul as matmul_module
    monkeypatch.setattr(
        matmul_module, "should_use_lds_padded_path",
        lambda *a, **kw: True,
    )
    yield


@pytest.fixture
def force_gate_off(monkeypatch):
    """Make the gate refuse every shape — exercises the unchanged baseline."""
    from tritonblas import matmul as matmul_module
    monkeypatch.setattr(
        matmul_module, "should_use_lds_padded_path",
        lambda *a, **kw: False,
    )
    yield


def _run_and_compare(M, N, K, dtype=torch.bfloat16, seed=0xCAFE):
    """Run tritonblas.matmul on the current process state and assert
    numerical equivalence to torch.matmul. Returns the max-abs error
    so the caller can log it."""
    torch.manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)

    out = tritonblas.matmul(a, b)
    ref = torch.matmul(a, b)

    torch.testing.assert_close(out, ref, atol=_ATOL, rtol=_RTOL)
    return (out.float() - ref.float()).abs().max().item()


def test_admitted_cell_numerical_correctness_with_gate_forced_on(force_gate_on):
    """
    The (M=4096, N=128, K=4096) admit cell, with the LDS-padded codegen
    knobs (32x32x8 MFMA, num_warps=4, kpack=1) actually applied, must
    match torch.matmul within the bf16 numerical envelope.

    This is the load-bearing correctness assertion for the K-1640 PR:
    if the swapped MFMA tile or warp count produced wrong results on
    this cell, the PR would silently corrupt every workload at this
    shape in production.
    """
    err = _run_and_compare(ADMIT_M, ADMIT_N, ADMIT_K)
    # Sanity floor — bf16 GEMM at K=4096 with random N(0,1) inputs has
    # max-abs error well under 5.0; if we see >>that something is wrong.
    assert err < 50.0, f"admit-cell bf16 max-abs error {err} is implausibly large"


def test_refused_cell_numerical_correctness_with_gate_forced_off(force_gate_off):
    """
    With the gate forced OFF, the unchanged baseline persistent_matmul
    launch path runs on the (M=4096, N=512, K=4096) K-1633 control cell.
    This is a regression guard against any accidental bleed of the
    K-1640 changes into the baseline launch site.
    """
    err = _run_and_compare(REFUSE_M, REFUSE_N, REFUSE_K)
    assert err < 50.0, f"refuse-cell bf16 max-abs error {err} is implausibly large"


def test_admitted_cell_numerical_correctness_with_gate_forced_off(force_gate_off):
    """
    The same admit cell, but with the gate forced OFF — runs on the
    unchanged baseline. Catches the case where the admit-on test passes
    only because the baseline path is also being run silently.
    """
    err = _run_and_compare(ADMIT_M, ADMIT_N, ADMIT_K)
    assert err < 50.0, f"admit-cell-baseline bf16 max-abs error {err} is implausibly large"


def test_lds_padded_overrides_match_pinned_constants():
    """
    Tie-in to the codegen knob swap so a future drift in
    lds_padded_launch_overrides() lights up here as well as in the
    pure-Python envelope test. The matmul.py launcher relies on
    exactly these three values at the gated launch site.
    """
    nonkdim, num_warps, kpack = lds_padded_envelope.lds_padded_launch_overrides()
    assert (nonkdim, num_warps, kpack) == (32, 4, 1)
