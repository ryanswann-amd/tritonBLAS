"""K-1635 regression guards for the LDS-bank-conflict kpack gate.

The patch in `include/tritonblas/matmul.py` sets

    kpack = 2 if BLK_K in (128, 256) else 1

inside both `persistent_matmul_lt` and `streamk_matmul_lt`. Two flavours of
test are provided:

1. **Source-level** (no GPU required): parses the source of the two callables
   with `inspect.getsource` and asserts that the gate literal is present in
   each. This is the regression guard the reviewer asked for - a future
   refactor that silently drops the gate (e.g. by re-introducing
   `kpack = 1`) will fail this test in CI on any host.

2. **Runtime gate capture** (needs GPU + Triton): monkey-patches the
   `_maybe_wrap`-returned grid callable so we intercept the kwargs handed to
   the kernel. We then dispatch `persistent_matmul_lt` with selectors whose
   `block_k` hits 64, 128 and 256 and assert the captured `kpack` matches the
   gate.

3. **Parametrised correctness** (needs GPU + Triton): runs a handful of cells
   from the BK in {128, 256} cohort the patch actually targets and checks
   max-abs-error vs `torch.matmul` is within bf16 tolerance.
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


def test_kpack_gate_present_in_persistent_matmul_lt():
    """Future refactors that drop the BK in {128,256} gate must fail here."""
    from tritonblas.matmul import persistent_matmul_lt

    src = inspect.getsource(persistent_matmul_lt)
    gate_lines = _kpack_gate_lines(src)

    assert gate_lines, "persistent_matmul_lt no longer assigns kpack at all"
    # The gate must be the only kpack assignment and must mention 128 and 256.
    assert any("128" in g and "256" in g and "BLK_K" in g for g in gate_lines), (
        f"persistent_matmul_lt kpack gate missing or weakened. "
        f"Found assignments: {gate_lines!r}. Expected something like "
        f"'kpack = 2 if BLK_K in (128, 256) else 1'."
    )


def test_kpack_gate_present_in_streamk_matmul_lt():
    """Same gate must be mirrored in the streamk path."""
    from tritonblas.matmul import streamk_matmul_lt

    src = inspect.getsource(streamk_matmul_lt)
    gate_lines = _kpack_gate_lines(src)

    assert gate_lines, "streamk_matmul_lt no longer assigns kpack at all"
    assert any("128" in g and "256" in g and "BLK_K" in g for g in gate_lines), (
        f"streamk_matmul_lt kpack gate missing or weakened. "
        f"Found assignments: {gate_lines!r}."
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


@gpu_only
@pytest.mark.parametrize(
    "block_k, expected_kpack",
    [(64, 1), (128, 2), (256, 2)],
)
def test_K1635_runtime_kpack_matches_gate(monkeypatch, block_k, expected_kpack):
    """Capture the kpack actually handed to the kernel for each BLOCK_K.

    Patches the kernel grid object so the call resolves to a recording stub
    instead of launching, then dispatches ``persistent_matmul_lt`` through a
    fake selector whose ``block_k`` we control.
    """
    import importlib
    # `tritonblas.matmul` resolves to the matmul function (re-exported in
    # __init__.py), so we have to import the underlying module by its full
    # dotted path to monkey-patch _maybe_wrap.
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

    # Tile shapes that pair with the requested BLOCK_K. (M,N) chosen so the
    # Origami-style block sizes are valid for tiny inputs.
    tile_shapes = {
        64:  (128, 128, 64,  256, 64),
        128: (128, 128, 128, 32,  128),
        256: (128, 128, 256, 32,  64),
    }
    blk_m, blk_n, blk_k_val, _gsz, _ = tile_shapes[block_k]
    assert blk_k_val == block_k

    # Build the fake selector dynamically so we don't trip Python's class-scope
    # rule that prevents `block_k = block_k` from referring to the outer name.
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

    M, N, K = blk_m, blk_n, block_k * 2  # tiny problem; we never launch.
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
        f"Gate broken: BLOCK_K={block_k} should yield kpack={expected_kpack}, "
        f"got kpack={captured['kpack']}"
    )
