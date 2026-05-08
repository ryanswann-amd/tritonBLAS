"""Negative-result regression guard for the K-451 / K-474 / K-519 / K-609
medium-K square FP16/BF16 cohort kpack investigation.

Background
----------
K-519 hypothesised that the residual ~0.65–0.75x perf gap vs hipBLASLt on the
medium-K (256 <= K <= 1024) M=N square FP16/BF16 cohort selected by the K-451
heuristic was caused by LDS bank conflicts on ``ds_read_b128`` in the
(256, 256, 64) winner-tile kernel (NW=8, NS=2). K-609 was opened to land a
swizzle / padded-stride mitigation. **The mitigation does not exist on this
build of the AMD Triton backend** — every knob reachable from the matmul
call site either regresses the wall-clock or *increases* LDS bank conflicts.

This test pins the negative finding so that future agents do not silently
re-attempt the same dead end. It asserts the default ``kpack = 1`` literal
in :mod:`tritonblas.matmul`, with the rocprofv3 + sweep evidence recorded in
this docstring as an inline citation.

Evidence (current Triton AMD backend on MI300X, captured in the K-609
task workspace `output/`):

* rocprofv3 PMC capture at the K-451 cohort centre point
  (M=N=4096, K=512, fp16, tile=(256,256,64), 40 dispatches per kpack):

  =====================  ===========  ===========
  Counter                kpack=1      kpack=2
  =====================  ===========  ===========
  SQ_LDS_BANK_CONFLICT             0   20,971,520
  SQ_LDS_IDX_ACTIVE      167,772,160  188,743,680
  SQ_INSTS_LDS            25,886,720   23,592,960
  GRBM_GUI_ACTIVE         34,052,320   35,274,511
  =====================  ===========  ===========

  ``kpack=1`` is already conflict-free on this backend; ``kpack=2`` *adds*
  ~21M conflicts (~89% of LDS instructions) by fighting the backend's
  built-in ``swizzled_shared`` / ``amd_rotating_shared`` encoding.

* 5-shape K-451 cohort knob sweep (baseline vs candidate, ratio = baseline/cand,
  >1.0 means candidate is faster; n_warm=20, n_iter=200):

  ================  ========  ========  ========  ========
  shape             kp=2      NS=3      mfma32    WPEU=2
  ================  ========  ========  ========  ========
  2048x2048x256     1.0271x   1.0238x   1.0362x   1.0273x
  2048x2048x512     0.9635x   1.0047x   1.0003x   1.0021x
  2048x2048x1024    0.9920x   FAIL      0.8071x   0.9983x
  3072x3072x512     0.9413x   FAIL      0.7958x   1.0080x
  4096x4096x512     0.9671x   FAIL      0.7257x   1.0056x
  ================  ========  ========  ========  ========

  - ``kpack=2``: ≤+2.7% gain on a single shape, regresses 0.94x–0.96x at
    K=512 / K=1024. Net loss on cohort.
  - ``num_stages=3``: compile/launch failure (LDS overflow) on K>=1024 and
    on the (256,256,64) tile.
  - ``matrix_instr_nonkdim=32``: catastrophic ~0.73x–0.81x on K>=512 cohort.
  - ``waves_per_eu=2``: pure noise (-0.02% to +1.0%).

* 60-shape wall-clock sweep (M=N in [2048, 4096], K in [256, 1024],
  fp16/bf16): per-shape kp2/kp1 in [0.99x, 1.02x], geomean 1.0006x.

The remaining gap vs hipBLASLt on this build is driven by fixed
launch / driver overhead (~260 µs per persistent_matmul call,
near-shape-independent), not LDS stalls. Re-investigate when (a) the
AMD Triton backend exposes ``PaddedSharedEncoding`` directly, or (b)
profiling on a future build shows non-zero LDS bank conflicts on the
(256, 256, 64) winner-tile kernel.

If you arrive here intending to bump ``kpack`` to 2 for this cohort:
*don't*. Re-run ``scripts/k609_kpack_repro.py`` first and confirm the
LDS_BANK_CONFLICT counter is non-zero on the build of record.
"""

from __future__ import annotations

import importlib
import inspect
import re

import pytest

# ``tritonblas.matmul`` is re-exported as a *function* from ``__init__.py``,
# so use the explicit submodule import.
tbm = importlib.import_module("tritonblas.matmul")


_KPACK_LITERAL_RE = re.compile(r"^\s*kpack\s*=\s*(\d+)\s*$", re.MULTILINE)


@pytest.mark.parametrize("fn_name", ["persistent_matmul_lt", "streamk_matmul_lt"])
def test_kpack_default_is_one(fn_name: str) -> None:
    """Default ``kpack`` for both LT entry points must stay at 1.

    K-609 negative result: bumping ``kpack`` to 2 on the K-451 cohort
    (M=N square FP16/BF16, 256 <= K <= 1024, tile (256,256,64))
    *introduces* ~21M LDS bank conflicts on the current Triton AMD backend
    while regressing wall-clock by 4-6% on the (256,256,64) winner
    tile (4096x4096x512) cohort centre. See module docstring for the
    rocprofv3 numbers.
    """
    fn = getattr(tbm, fn_name)
    src = inspect.getsource(fn)
    matches = _KPACK_LITERAL_RE.findall(src)
    assert matches, (
        f"{fn_name} no longer contains a top-level ``kpack = <int>`` "
        "literal. If you've replaced it with a selector, update this "
        "test to call the selector for the K-451 cohort and assert "
        "the returned value is 1 (or rerun scripts/k609_kpack_repro.py "
        "to confirm the LDS bank-conflict counter is still zero with "
        "kpack=2 on the build of record). See K-609 ticket."
    )
    for raw in matches:
        assert int(raw) == 1, (
            f"{fn_name} default kpack changed to {raw}. K-609 evidence "
            "shows kpack=2 introduces ~21M SQ_LDS_BANK_CONFLICT on the "
            "(256,256,64) winner-tile kernel and regresses wall-clock by "
            "up to 0.94x on the K-451 cohort. Re-run "
            "scripts/k609_kpack_repro.py before bumping this default."
        )
