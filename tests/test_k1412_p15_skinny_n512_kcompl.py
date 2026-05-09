"""K-1412 P15 skinny_N512 K-COMPLEMENT (K-axis extremes) pin tests — torch-free.

Minimal pin coverage for the K-1425 productionisation of the K-1412-validated
12-cell `_K1412_P15_SKINNY_N512_KCOMPL_ROUTEOUT_12` frozenset stacked as the
9th-position envelope on top of the K-1400 P14 91-cell baseline (envelope
grows 91 → 103 cells; +12 admits).

This is the N=512 sibling of the K-1402 / K-1406 K-1397 N=256 K-axis-extremes
productionisation, with N axis bumped 256 → 512 and the same K ∈ {2048, 32768}
extremes scope (R-1397.SKINNY-N-K-COMPLEMENT-IS-EXTREMES-NOT-MID-BAND
generalised to N=512 per R-1412).

Cardinality (=12) and the five cross-frozenset disjointness invariants
(P8 / K971_ROUTE_TABLE / P12 / P13 / P14) are already asserted at module load
in `_route_predicate.py`; we do not duplicate those here per the K-1406
test-trim convention (R-1406.MODULE-LOAD-ASSERTS-MAKE-PYTEST-CARDINALITY-
DISJOINTNESS-DUPS-DEAD-WEIGHT).  The pytest suite focuses on the orthogonal
behavioural pins:

  * exact membership of the K-1412 P15 12 cells
    (M ∈ {2048, 4096, 8192} × N=512 × K ∈ {2048, 32768} × {bf16, fp16})
  * full-chain dispatch routes every admit cell to hipBLASLt (covers the
    9th-position predicate's interaction with the 1st–8th-position chain)
  * carve-out short-circuits (streamk / work-stealing / mismatched-dtype)
    correctly suppress the route-OUT for an admit cell
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K1412_P15_SKINNY_N512_KCOMPL_ROUTEOUT_12,
    k971_route_decision,
)


# Bucket rule per K-1425 success criteria: M ∈ {2048,4096,8192} ×
# N=512 × K ∈ {2048,32768} × {bf16, fp16} = 12 cells.
_EXPECTED_K1412_12_CELLS = frozenset({
    (M, 512, K, dt)
    for M in (2048, 4096, 8192)
    for K in (2048, 32768)
    for dt in ("torch.bfloat16", "torch.float16")
})


def test_k1412_membership_exactly_12_kcomplement_cells():
    """The frozenset must equal the K-COMPLEMENT bucket rule exactly — no
    missing or unknown cells (mirrors K-1397 N=256 with N axis 256 → 512)."""
    diff_missing = sorted(_EXPECTED_K1412_12_CELLS
                          - _K1412_P15_SKINNY_N512_KCOMPL_ROUTEOUT_12)
    diff_unknown = sorted(_K1412_P15_SKINNY_N512_KCOMPL_ROUTEOUT_12
                          - _EXPECTED_K1412_12_CELLS)
    assert _K1412_P15_SKINNY_N512_KCOMPL_ROUTEOUT_12 == _EXPECTED_K1412_12_CELLS, (
        f"K-1412 P15 frozenset diverges from the K-COMPLEMENT bucket rule "
        f"(M ∈ {{2048,4096,8192}} × N=512 × K ∈ {{2048,32768}} × {{bf16,fp16}}).\n"
        f"  missing : {diff_missing}\n"
        f"  unknown : {diff_unknown}")


@pytest.mark.parametrize("M,N,K,dtype", sorted(_EXPECTED_K1412_12_CELLS))
def test_full_dispatch_routes_every_k1412_cell_to_hbl(M, N, K, dtype):
    """Every K-1412 admit cell must route to hipBLASLt under the full
    9-position precedence chain with default carve-out settings."""
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


@pytest.mark.parametrize("enable_streamk,work_stealing,b_dtype", [
    (True,  False, "torch.bfloat16"),  # streamk requested
    (False, True,  "torch.bfloat16"),  # work-stealing requested
    (False, False, "torch.float16"),   # mismatched dtype (a bf16, b fp16)
])
def test_k1412_admit_cell_carveout_short_circuits(
    enable_streamk, work_stealing, b_dtype,
):
    """Admit cells must NOT route to hipBLASLt when streamk / work-stealing
    / mismatched-dtype is requested.  Pick (4096, 512, 32768, bf16) as a
    representative K-1412 admit cell at the K=32768 extreme."""
    a_dtype = "torch.bfloat16"
    M, N, K = 4096, 512, 32768
    assert (M, N, K, a_dtype) in _K1412_P15_SKINNY_N512_KCOMPL_ROUTEOUT_12
    routed = k971_route_decision(
        M, N, K, a_dtype, b_dtype, enable_streamk, work_stealing,
    )
    assert routed is False, (
        f"K-1412 P15 admit cell ({M},{N},{K},{a_dtype}) must NOT route to "
        f"hipBLASLt when carve-out applies (streamk={enable_streamk}, "
        f"work_stealing={work_stealing}, b_dtype={b_dtype}); short-circuit "
        f"in _k971_route_to_hbl is load-bearing.")
