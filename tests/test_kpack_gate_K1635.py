"""K-1635 regression guards for the LDS-bank-conflict kpack gate.

The patch in `include/tritonblas/matmul.py` sets, in both
``persistent_matmul_lt`` and ``streamk_matmul_lt``::

    _bk_in_kpack_set = BLK_K in (128, 256)
    _tile_kpack2_safe = not (BLK_M == 64 and BLK_N >= 64)
    kpack = 2 if (_bk_in_kpack_set and _tile_kpack2_safe) else 1

i.e. ``kpack`` is bumped 1->2 on the BK in {128, 256} cohort, **excluding**
the (BLK_M==64, BLK_N>=64) tile family that K-1635 PMC measurement showed
regresses (e.g. 256x8192x4096 BK=128 doubled SQ_LDS_BANK_CONFLICT under
unconditional kpack=2; cells lost up to 0.92x in paired timing).

BK=64 keeps kpack=1 unchanged. K-1641's per-cell PMC decomposition on the
M=N=4096 envelope showed TB SQ_LDS_BANK_CONFLICT/inst = 0.0 on every cell
-- the M=N=4096 residual gap is LDS pipeline back-pressure (S3), NOT
bank-conflict, and is not addressable via the kpack knob. Fixing it
requires an Origami tile-table change (K-1641 nominates BK=64->128 OR
BN=512->112) outside the kernel-launch surface this PR touches; left for
a follow-up.

Three flavours of test:

1. **Source-level** (no GPU required): asserts the refined kpack gate is
   present in each callable. A refactor that drops or weakens it fails
   on any host.

2. **Runtime gate capture** (needs GPU + Triton): monkey-patches the
   ``_maybe_wrap``-returned grid callable so we intercept the kwargs
   handed to the kernel. Covers six BLOCK_K / tile combinations spanning
   every branch of the gate (BK in {64,128,256} x tile families).

3. **Parametrised correctness** (needs GPU + Triton): runs a handful of
   cells from the BK in {128, 256} cohort the patch targets and checks
   max-abs-error vs ``torch.matmul`` is within bf16 tolerance.
"""

from __future__ import annotations

import inspect
import re
from typing import Iterable

import pytest


# ---------------------------------------------------------------------------
# 1. Source-level guard - always runs, no GPU required.
# ---------------------------------------------------------------------------

def _kpack_gate_lines(source: str) -> list[str]:
    """Return every assignment line that names ``kpack``."""
    return [
        line.strip()
        for line in source.splitlines()
        if re.match(r"\s*kpack\s*=", line)
    ]


def _has_refined_kpack_gate(source: str) -> bool:
    """Refined gate must mention BLK_K in {128,256} AND exclude (M==64, N>=64)."""
    has_bk_set = "BLK_K in (128, 256)" in source or "_bk_in_kpack_set" in source
    has_tile_exclusion = (
        "BLK_M == 64" in source and "BLK_N >= 64" in source
    ) or "_tile_kpack2_safe" in source
    return has_bk_set and has_tile_exclusion


def test_kpack_gate_present_in_persistent_matmul_lt():
    """Refactors that drop / weaken the K-1635 refined kpack gate must fail."""
    from tritonblas.matmul import persistent_matmul_lt

    src = inspect.getsource(persistent_matmul_lt)
    gate_lines = _kpack_gate_lines(src)

    assert gate_lines, "persistent_matmul_lt no longer assigns kpack at all"
    assert _has_refined_kpack_gate(src), (
        "persistent_matmul_lt K-1635 refined kpack gate is missing or weakened. "
        f"Found kpack lines: {gate_lines!r}. Expected the BLK_K in (128, 256) "
        "set AND the (BLK_M == 64 and BLK_N >= 64) exclusion both present."
    )


def test_kpack_gate_present_in_streamk_matmul_lt():
    """Same refined gate must be mirrored in the streamk path."""
    from tritonblas.matmul import streamk_matmul_lt

    src = inspect.getsource(streamk_matmul_lt)
    gate_lines = _kpack_gate_lines(src)

    assert gate_lines, "streamk_matmul_lt no longer assigns kpack at all"
    assert _has_refined_kpack_gate(src), (
        "streamk_matmul_lt K-1635 refined kpack gate is missing or weakened. "
        f"Found kpack lines: {gate_lines!r}."
    )


# ---------------------------------------------------------------------------
# 2/3. GPU-only tests below. Skip cleanly off-GPU.
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
tritonblas = pytest.importorskip("tritonblas")
gpu_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="K-1635 GPU tests need a GPU"
)


# Cells from the BK in {128, 256} cohort the K-1635 patch actually targets.
# These shapes route to BLOCK_K=128 or BLOCK_K=256 under Origami on gfx942.
_BK_GATE_CELLS = [
    # (M, N, K)
    (128, 4096, 4096),
    (128, 8192, 8192),
    (256, 2048, 4096),
    (512, 1024, 4096),
    (512, 4096, 8192),
]


@gpu_only
@pytest.mark.parametrize("m,n,k", _BK_GATE_CELLS)
def test_K1635_correctness_bk_gated_cohort(m, n, k):
    """bf16 correctness on the BK in {128, 256} cohort the patch targets.

    The patch only changes ``kpack``; arithmetic must be bit-for-bit equal
    in the limit and within bf16 rounding in practice.
    """
    from tritonblas.utils import generate_matmul_inputs

    inputs = generate_matmul_inputs(
        m, n, k, torch.bfloat16, torch.bfloat16, "N", "N", "randn"
    )
    selector = tritonblas.OrigamiMatmulSelector(
        m, n, k,
        inputs.A.dtype, inputs.B.dtype, inputs.C.dtype, inputs.A.device,
        streamk=False,
    )
    config = tritonblas.matmul_preamble(selector)
    tritonblas.matmul_lt(
        inputs.A, inputs.B, inputs.C, selector, config,
        enable_streamk=False, work_stealing=False,
    )

    ref = torch.matmul(inputs.A.float(), inputs.B.float()).to(torch.bfloat16)
    # bf16 has ~3 decimal digits; for K up to 8192 a few-LSB delta is fine.
    torch.testing.assert_close(inputs.C, ref, atol=2.0, rtol=2e-2)


# Each row covers one decision branch of the K-1635 refined kpack gate:
#   (case_id, BLK_M, BLK_N, BLK_K, M, N, expected_kpack)
_GATE_CASES = [
    # BK=64: gate never fires regardless of tile -> kpack stays 1
    ("bk64_smalltile",         64,  64,  64,  256,  256, 1),
    ("bk64_largetile",        128, 128,  64, 4096, 4096, 1),
    # BK=128, BLK_M=128 -> refined gate fires (kpack=2)
    ("bk128_blkm128",         128,  64, 128,  512, 4096, 2),
    # BK=128, BLK_M=64, BLK_N=128 -> refined gate EXCLUDES (kpack=1, no regression)
    ("bk128_blkm64_blkn128",   64, 128, 128,  256, 8192, 1),
    # BK=256, BLK_M=32 -> refined gate fires (kpack=2)
    ("bk256_blkm32",           32,  64, 256,  128, 4096, 2),
    # BK=256, BLK_M=64, BLK_N=32 -> refined gate fires (BLK_N<64, kpack=2)
    ("bk256_blkm64_blkn32",    64,  32, 256,  256, 2048, 2),
    # BK=256, BLK_M=64, BLK_N=64 -> refined gate EXCLUDES (kpack=1)
    ("bk256_blkm64_blkn64",    64,  64, 256,  256, 4096, 1),
]


@gpu_only
@pytest.mark.parametrize(
    "case_id,blk_m,blk_n,block_k,M,N,expected_kpack",
    _GATE_CASES,
    ids=[c[0] for c in _GATE_CASES],
)
def test_K1635_runtime_gate_matches_source(
    monkeypatch, case_id, blk_m, blk_n, block_k, M, N, expected_kpack,
):
    """Capture kpack handed to the kernel for each shape and tile.

    Verifies the K-1635 refined kpack gate fires (kpack=2) only on cells
    where the source comment claims it should, and stays at kpack=1
    everywhere else.
    """
    import importlib
    tb_matmul = importlib.import_module("tritonblas.matmul")

    captured = {}

    class _RecordingGrid:
        def __getitem__(self, _grid):
            def _stub(*args, **kwargs):
                captured["kpack"] = kwargs.get("kpack")
                captured["BLOCK_SIZE_K"] = kwargs.get("BLOCK_SIZE_K")
                return None
            return _stub

    def _fake_maybe_wrap(_kernel, probe_tensor=None):  # noqa: ARG001
        return _RecordingGrid()

    monkeypatch.setattr(tb_matmul, "_maybe_wrap", _fake_maybe_wrap)

    _FakeSelector = type(
        "_FakeSelector",
        (),
        {
            "block_m": blk_m,
            "block_n": blk_n,
            "block_k": block_k,
            "group_m": 8,
            "num_sms": 0,
            "num_stages": 2,
            "COUNTERS_PER_XCD": 1,
            "_hardware": type("HW", (), {"N_CU": 1})(),
        },
    )

    K = block_k * 2  # tiny K dimension; we never launch the kernel.
    a = torch.zeros((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.zeros((K, N), dtype=torch.bfloat16, device="cuda")
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")

    tb_matmul.persistent_matmul_lt(
        a, b, c, _FakeSelector(),
        config=None, bias=None, a_scale=None, b_scale=None,
        quantized=False, work_stealing=False,
    )

    assert captured["BLOCK_SIZE_K"] == block_k
    assert captured["kpack"] == expected_kpack, (
        f"[{case_id}] kpack gate broken: M={M} N={N} "
        f"BLK=({blk_m},{blk_n},{block_k}) should yield kpack={expected_kpack}, "
        f"got kpack={captured['kpack']}"
    )
