"""Unit tests for the strict-equality MFMA-issue-stall route-OUT envelope.

Pure data tests: no GPU, no torch import paths beyond the public dtype
singleton. Import the data module directly and assert envelope cardinality,
admit / no-leak boundaries, and dispatch behavior.

Run:  python -m pytest tests/test_route_predicate.py -v
"""
from __future__ import annotations

import os

import pytest

from tritonblas._route_predicate import (
    _MFMA_STALL_ANCHORS_13,
    _MFMA_STALL_K_INTERIOR_AND_M_EXTENSION_3,
    _MFMA_STALL_N128_ADMITS_8,
    _MFMA_STALL_NEIGHBORS_12,
    _MFMA_STALL_ROUTEOUT,
    route_to_hbl,
)

BF16 = "torch.bfloat16"


# ---------------------------------------------------------------------------
# Cardinality + disjointness invariants (mirror import-time asserts so any
# regression surfaces in CI even if asserts are stripped under -O).
# ---------------------------------------------------------------------------

def test_envelope_cardinality_is_36():
    assert len(_MFMA_STALL_ROUTEOUT) == 36


def test_anchors_cardinality_is_13():
    assert len(_MFMA_STALL_ANCHORS_13) == 13


def test_neighbors_cardinality_is_12():
    assert len(_MFMA_STALL_NEIGHBORS_12) == 12


def test_k_interior_and_m_extension_cardinality_is_3():
    assert len(_MFMA_STALL_K_INTERIOR_AND_M_EXTENSION_3) == 3


def test_n128_admits_cardinality_is_8():
    assert len(_MFMA_STALL_N128_ADMITS_8) == 8


def test_subsets_are_pairwise_disjoint():
    subsets = [
        _MFMA_STALL_ANCHORS_13,
        _MFMA_STALL_NEIGHBORS_12,
        _MFMA_STALL_K_INTERIOR_AND_M_EXTENSION_3,
        _MFMA_STALL_N128_ADMITS_8,
    ]
    for i, a in enumerate(subsets):
        for b in subsets[i + 1:]:
            assert a.isdisjoint(b), f"{a} overlaps with {b}"


def test_n128_admits_all_have_n_equal_128():
    for (M, N, K, dtype) in _MFMA_STALL_N128_ADMITS_8:
        assert N == 128, f"({M},{N},{K}) is not an N=128 cell"


def test_no_prior_subset_has_n_equal_128():
    for subset in (
        _MFMA_STALL_ANCHORS_13,
        _MFMA_STALL_NEIGHBORS_12,
        _MFMA_STALL_K_INTERIOR_AND_M_EXTENSION_3,
    ):
        for (M, N, K, dtype) in subset:
            assert N != 128, f"prior subset cell ({M},{N},{K}) collides with N=128 admits"


# ---------------------------------------------------------------------------
# Dispatch admit: every cell in the envelope must route OUT.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell", sorted(_MFMA_STALL_ROUTEOUT))
def test_dispatch_admits_every_envelope_cell(cell):
    M, N, K, dtype = cell
    assert route_to_hbl(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# bf16-only: identical (M, N, K) under fp16 / fp32 must NOT route OUT.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell", sorted(_MFMA_STALL_ROUTEOUT))
def test_dispatch_rejects_non_bf16_dtype(cell):
    M, N, K, _dtype = cell
    assert route_to_hbl(M, N, K, "torch.float16") is False
    assert route_to_hbl(M, N, K, "torch.float32") is False


# ---------------------------------------------------------------------------
# No-leak negative controls: shapes adjacent to envelope cells but not in
# the envelope must NOT route OUT. These guard against accidental envelope
# widening from a typo in the data table.
# ---------------------------------------------------------------------------

NO_LEAK_NEGATIVES = [
    # M-perturbed by 1 from anchors -- not in envelope, must not fire
    (4481, 3072, 768, BF16),
    (14209, 2048, 1024, BF16),
    # N-perturbed by 1 from N=128 admits -- not in envelope
    (6016, 129, 1024, BF16),
    (49152, 127, 256, BF16),
    # K-perturbed by 1 -- not in envelope
    (4480, 3072, 769, BF16),
    # The single rejected N=128 candidate (M=4480, K=768) -- documented
    # exclusion; must NOT be admitted by accident.
    (4480, 128, 768, BF16),
    # Tiny shape outside any envelope
    (128, 128, 128, BF16),
    # Large square outside envelope
    (8192, 8192, 8192, BF16),
]


@pytest.mark.parametrize("cell", NO_LEAK_NEGATIVES)
def test_no_leak_negatives_do_not_route_out(cell):
    M, N, K, dtype = cell
    assert (M, N, K, dtype) not in _MFMA_STALL_ROUTEOUT
    assert route_to_hbl(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# String-vs-singleton dtype compatibility: tests pass the literal string
# "torch.bfloat16"; the production code passes the torch.bfloat16 singleton.
# Both must match the same envelope cells.
# ---------------------------------------------------------------------------

def test_string_dtype_matches_envelope():
    for (M, N, K, dtype) in _MFMA_STALL_ROUTEOUT:
        # dtype in the frozenset is already the literal string; round-trip
        # through route_to_hbl with the same literal must admit.
        assert route_to_hbl(M, N, K, dtype) is True


def test_torch_singleton_dtype_matches_envelope():
    torch = pytest.importorskip("torch")
    for (M, N, K, _dtype) in _MFMA_STALL_ROUTEOUT:
        assert route_to_hbl(M, N, K, torch.bfloat16) is True
        assert route_to_hbl(M, N, K, torch.float16) is False
