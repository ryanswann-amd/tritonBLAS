"""K-464: medium-K square FP16/BF16 cohort gate predicate tests.

The gate routes M=N in {1024,2048,4096} x K in {256,512} x {fp16,bf16} to
the K-442 winner tile.  These tests assert the predicate fires only on those
shapes and explicitly excludes the neighbor cohorts that share BM/BN
dimensions (K-381/K-388 medium-square K in {1024,2048}; K-428 small-K
K in {128,256,512}) per the K-654 anti-pattern (predicate-narrow gate).

These are pure-Python predicate tests -- no GPU is required -- so they run
in CI alongside the existing test suite.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from tritonblas.matmul import _k464_in_cohort


# -- happy path: every shape in the cohort fires the gate ---------------------

_COHORT_M = (1024, 2048, 4096)
_COHORT_K = (256, 512)
_COHORT_DTYPES = (torch.float16, torch.bfloat16)


@pytest.mark.parametrize("M,K,dtype", list(itertools.product(_COHORT_M, _COHORT_K, _COHORT_DTYPES)))
def test_predicate_fires_on_cohort(M, K, dtype):
    assert _k464_in_cohort(
        M=M, N=M, K=K,
        a_dtype=dtype, b_dtype=dtype, c_dtype=dtype,
        mx_block_size=0, streamk=False,
    )


# -- neighbor cohorts must NOT trigger the gate ------------------------------
# Each entry is (label, kwargs) where label names the neighbor cohort that
# shares BM/BN with the K-464 cohort and would silently regress if the gate
# fired on it.

_NEIGHBOR_CASES = [
    # K-381 / K-388 medium-square cohort: same M=N but K in {1024,2048}
    ("K-381 M=N=1024 K=1024 fp16", dict(M=1024, N=1024, K=1024, a_dtype=torch.float16,
                                        b_dtype=torch.float16, c_dtype=torch.float16,
                                        mx_block_size=0, streamk=False)),
    ("K-388 M=N=2048 K=2048 bf16", dict(M=2048, N=2048, K=2048, a_dtype=torch.bfloat16,
                                        b_dtype=torch.bfloat16, c_dtype=torch.bfloat16,
                                        mx_block_size=0, streamk=False)),
    ("K-388 M=N=4096 K=1024 fp16", dict(M=4096, N=4096, K=1024, a_dtype=torch.float16,
                                        b_dtype=torch.float16, c_dtype=torch.float16,
                                        mx_block_size=0, streamk=False)),
    # K-428 small-K cohort: same M=N but K in {128, 256-but-different-M, 512-but-different-M}
    ("K-428 M=N=512 K=256 fp16", dict(M=512, N=512, K=256, a_dtype=torch.float16,
                                      b_dtype=torch.float16, c_dtype=torch.float16,
                                      mx_block_size=0, streamk=False)),
    ("K-428 M=N=1024 K=128 bf16", dict(M=1024, N=1024, K=128, a_dtype=torch.bfloat16,
                                       b_dtype=torch.bfloat16, c_dtype=torch.bfloat16,
                                       mx_block_size=0, streamk=False)),
    # Boundary: cohort M but K just outside cohort
    ("M=2048 K=384 fp16", dict(M=2048, N=2048, K=384, a_dtype=torch.float16,
                               b_dtype=torch.float16, c_dtype=torch.float16,
                               mx_block_size=0, streamk=False)),
    ("M=4096 K=768 fp16", dict(M=4096, N=4096, K=768, a_dtype=torch.float16,
                               b_dtype=torch.float16, c_dtype=torch.float16,
                               mx_block_size=0, streamk=False)),
    # Non-square: cohort K but M != N
    ("M=1024 N=2048 K=256 fp16", dict(M=1024, N=2048, K=256, a_dtype=torch.float16,
                                      b_dtype=torch.float16, c_dtype=torch.float16,
                                      mx_block_size=0, streamk=False)),
    # Wrong dtype: cohort shape but FP32 / FP8
    ("M=2048 K=512 fp32", dict(M=2048, N=2048, K=512, a_dtype=torch.float32,
                               b_dtype=torch.float32, c_dtype=torch.float32,
                               mx_block_size=0, streamk=False)),
    # Stream-K must always bypass the gate
    ("M=2048 K=512 fp16 streamk", dict(M=2048, N=2048, K=512, a_dtype=torch.float16,
                                       b_dtype=torch.float16, c_dtype=torch.float16,
                                       mx_block_size=0, streamk=True)),
    # MX/FP4 path must bypass the gate
    ("M=2048 K=512 fp16 mx", dict(M=2048, N=2048, K=512, a_dtype=torch.float16,
                                  b_dtype=torch.float16, c_dtype=torch.float16,
                                  mx_block_size=32, streamk=False)),
    # Mixed dtype A vs B
    ("M=2048 K=512 mixed", dict(M=2048, N=2048, K=512, a_dtype=torch.float16,
                                b_dtype=torch.float32, c_dtype=torch.float16,
                                mx_block_size=0, streamk=False)),
]


@pytest.mark.parametrize("label,kwargs", _NEIGHBOR_CASES, ids=[c[0] for c in _NEIGHBOR_CASES])
def test_predicate_excludes_neighbors(label, kwargs):
    assert not _k464_in_cohort(**kwargs), f"K-464 gate must not fire on {label}"


# -- override application sets the K-442 tile and the marker flag ------------

def _import_selector_or_skip():
    try:
        from tritonblas.origami import OrigamiMatmulSelector  # type: ignore
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"OrigamiMatmulSelector unavailable: {e}")
    return OrigamiMatmulSelector


def test_apply_override_sets_k442_winner_tile():
    """If a GPU is available, build a real selector for an in-cohort shape
    and confirm the override mutates it to the K-442 winner tile."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    OrigamiMatmulSelector = _import_selector_or_skip()
    from tritonblas.matmul import _k464_apply_override, _K464_TILE

    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    sel = OrigamiMatmulSelector(
        2048, 2048, 512,
        torch.float16, torch.float16, torch.float16,
        dev,
        mx_block_size=0, streamk=False, num_stages=2,
    )
    _k464_apply_override(sel)
    assert sel.block_m == _K464_TILE["block_m"]
    assert sel.block_n == _K464_TILE["block_n"]
    assert sel.block_k == _K464_TILE["block_k"]
    assert sel.num_stages == _K464_TILE["num_stages"]
    assert sel._k464_num_warps == _K464_TILE["num_warps"]
    assert sel._k464_waves_per_eu == _K464_TILE["waves_per_eu"]
    assert sel._k464_override is True
