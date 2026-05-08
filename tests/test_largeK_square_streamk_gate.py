"""Pin the strict-membership inclusion list for the large-K square fp16/bf16
StreamK gate (cf. ``tritonblas.matmul._largeK_square_streamk_gate``).

These tests import the *real* shipped symbols from ``tritonblas.matmul`` so
that any future refactor of the gate or its call sites cannot silently break
the cohort. They are pure-Python (no GPU kernel launches) but verify both:

  1. The predicate returns the expected value for every documented cell and
     every off-cohort axis (dtype, square-ness, K, M-tier, K-leakage probes).
  2. The two public entry points (``_matmul``, ``_matmul_out``) actually call
     the gate before constructing the Origami selector — verified by source
     inspection so a future refactor that forgets the wiring is caught here.
"""

import importlib
import inspect

import pytest
import torch

from tritonblas.matmul import (
    _LARGEK_SQUARE_STREAMK_CELLS,
    _largeK_square_streamk_gate,
)

# ``tritonblas/__init__.py`` re-exports ``matmul`` (the *function*) into the
# package namespace, which shadows the submodule binding ``tritonblas.matmul``.
# Resolve the submodule explicitly so the wiring tests below can read source
# from the public entry points.
tb_matmul = importlib.import_module("tritonblas.matmul")


# Documented inclusion list — must stay in lock-step with the constant.
_DOCUMENTED_CELLS = frozenset({
    (1024, 4096),
    (2048, 4096),
    (2048, 8192),
    (4096, 4096),
    (4096, 8192),
    (4096, 16384),
})

_GATED_DTYPES = (torch.float16, torch.bfloat16)
_NON_GATED_DTYPES = (torch.float32, torch.float64, torch.int32, torch.int64)


# ---------------------------------------------------------------------------
# Constant invariants
# ---------------------------------------------------------------------------

def test_inclusion_list_is_a_frozenset():
    """Constant must be immutable so callers cannot mutate the cohort at runtime."""
    assert isinstance(_LARGEK_SQUARE_STREAMK_CELLS, frozenset)


def test_cell_set_matches_documented_inclusion_list():
    """The shipped constant must equal the documented 6-cell set exactly."""
    assert _LARGEK_SQUARE_STREAMK_CELLS == _DOCUMENTED_CELLS


# ---------------------------------------------------------------------------
# Positive cases — gate must fire
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,K", sorted(_DOCUMENTED_CELLS))
@pytest.mark.parametrize("dtype", _GATED_DTYPES)
def test_gate_fires_on_every_listed_cell(M, K, dtype):
    """Every (M, K, dtype) in the published inclusion list must trigger the gate."""
    assert _largeK_square_streamk_gate(M, M, K, dtype) is True


# ---------------------------------------------------------------------------
# Negative cases — every axis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", _NON_GATED_DTYPES)
def test_gate_excludes_non_fp16_bf16_dtypes(dtype):
    """fp32/fp64/int dtypes must never enter the cohort."""
    for M, K in _DOCUMENTED_CELLS:
        assert _largeK_square_streamk_gate(M, M, K, dtype) is False


@pytest.mark.parametrize(
    "M,N,K",
    [
        (1024, 2048, 4096),   # M != N
        (2048, 1024, 8192),   # M != N
        (4096, 2048, 16384),  # M != N
        (1024, 4096, 4096),   # M != N (M < N)
    ],
)
@pytest.mark.parametrize("dtype", _GATED_DTYPES)
def test_gate_excludes_rectangular_shapes(M, N, K, dtype):
    """Rectangular (non-square) shapes are out of cohort."""
    assert _largeK_square_streamk_gate(M, N, K, dtype) is False


@pytest.mark.parametrize(
    "M,K",
    [
        # Adjacent regression-guard cohort — small/medium K must never enter.
        (1024, 512), (1024, 1024), (1024, 2048),
        (2048, 512), (2048, 1024), (2048, 2048),
        (4096, 512), (4096, 1024), (4096, 2048),
    ],
)
@pytest.mark.parametrize("dtype", _GATED_DTYPES)
def test_gate_excludes_small_K_guard_cohort(M, K, dtype):
    """The 12-shape adjacent-K guard cohort must remain on the persistent path."""
    assert _largeK_square_streamk_gate(M, M, K, dtype) is False


@pytest.mark.parametrize(
    "M,K",
    [
        # Cells inside the original PRD cohort but explicitly excluded after
        # per-shape A/B testing showed StreamK regresses.
        (1024, 8192), (1024, 16384),
        (2048, 16384),
    ],
)
@pytest.mark.parametrize("dtype", _GATED_DTYPES)
def test_gate_excludes_documented_loser_cells(M, K, dtype):
    """A/B-loser cells that share an M-tier with winning cells must NOT fire."""
    assert (M, K) not in _LARGEK_SQUARE_STREAMK_CELLS  # constant invariant
    assert _largeK_square_streamk_gate(M, M, K, dtype) is False


@pytest.mark.parametrize(
    "M,K",
    [
        # Out-of-cohort M-tiers must never fire even at "looks-like-large-K" K.
        (512, 4096), (512, 8192),
        (8192, 4096), (8192, 8192), (8192, 16384),
    ],
)
@pytest.mark.parametrize("dtype", _GATED_DTYPES)
def test_gate_excludes_out_of_cohort_M_tiers(M, K, dtype):
    assert _largeK_square_streamk_gate(M, M, K, dtype) is False


@pytest.mark.parametrize(
    "K",
    # Skeptic guard: gate must NEVER accept K beyond the enumerated set, even
    # for an in-cohort M tier. K∈{32768, 65536, 131072} are the canonical
    # silent-leakage probes.
    [24576, 32768, 65536, 131072],
)
@pytest.mark.parametrize("M", sorted({m for m, _ in _DOCUMENTED_CELLS}))
@pytest.mark.parametrize("dtype", _GATED_DTYPES)
def test_gate_does_not_leak_to_unbounded_K(M, K, dtype):
    """No range-based branch may accept K outside the enumerated inclusion list."""
    assert (M, K) not in _LARGEK_SQUARE_STREAMK_CELLS
    assert _largeK_square_streamk_gate(M, M, K, dtype) is False


# ---------------------------------------------------------------------------
# Wiring — gate must be called from both public entry points
# ---------------------------------------------------------------------------

def _entry_point_body(entry_name: str) -> str:
    """Return the lexical body of ``def {entry_name}(...)`` from the matmul
    module source. ``_matmul`` / ``_matmul_out`` are wrapped by
    ``torch.library.triton_op`` which captures the original function in a
    closure with no public accessor, so we read the module source instead
    and slice between the ``def`` line and the next top-level ``def`` /
    decorator. This is robust against the triton_op wrapping."""
    full = inspect.getsource(tb_matmul)
    needle = f"\ndef {entry_name}("
    start = full.find(needle)
    assert start != -1, f"could not find def {entry_name}( in module source"
    # Find the next top-level def/decorator line after this one.
    rest = full[start + len(needle):]
    next_top = len(rest)
    for marker in ("\ndef ", "\n@triton_op", "\n@", "\nclass "):
        idx = rest.find(marker)
        if idx != -1 and idx < next_top:
            next_top = idx
    return rest[:next_top]


@pytest.mark.parametrize("entry_name", ["_matmul", "_matmul_out"])
def test_entry_point_calls_gate_before_selector(entry_name):
    """``_matmul`` / ``_matmul_out`` must call ``_largeK_square_streamk_gate``
    and flip ``enable_streamk`` *before* constructing the Origami selector,
    so the cohort routes through the StreamK launcher."""
    body = _entry_point_body(entry_name)
    gate_idx = body.find("_largeK_square_streamk_gate")
    selector_idx = body.find("_make_matmul_selector")
    assert gate_idx != -1, (
        f"{entry_name} no longer calls _largeK_square_streamk_gate — "
        "the cohort fix has been silently removed"
    )
    assert selector_idx != -1, (
        f"{entry_name} unexpectedly does not call _make_matmul_selector"
    )
    assert gate_idx < selector_idx, (
        f"{entry_name} calls the gate AFTER the selector is built — the gate "
        "no longer affects which scheduler the selector chooses"
    )
    # And the body must actually flip the variable, not just call the predicate.
    assert "enable_streamk = True" in body[gate_idx:selector_idx], (
        f"{entry_name} calls the gate but does not assign enable_streamk = True"
    )
