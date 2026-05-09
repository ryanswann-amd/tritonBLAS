"""K-1397 P13 skinny_N256 K-COMPLEMENT pin tests — torch-free.

Minimal pin coverage for the K-1402 productionisation of the K-1397-winning
12-cell `_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12` frozenset stacked as the
8th-position envelope on top of the K-1389 P13 73-cell baseline (envelope
grows 73 → 85 cells; +12 admits).

Cardinality and cross-frozenset disjointness are already asserted at module
load in `_route_predicate.py`; we do not duplicate those here.  The pytest
suite focuses on the orthogonal behavioural pins:

  * exact membership of the K-1397 P13 12 cells
    (M ∈ {2048, 4096, 8192} × N=256 × K ∈ {2048, 32768} × {bf16, fp16})
  * full-chain dispatch routes every admit cell to hipBLASLt (covers the
    8th-position predicate's interaction with the precedence chain)
  * carve-out short-circuits (streamk / work-stealing / mismatched-dtype)
    correctly suppress the route-OUT for an admit cell
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    k971_route_decision,
)


# Bucket rule per K-1402 success criteria: M ∈ {2048,4096,8192} ×
# N=256 × K ∈ {2048,32768} × {bf16, fp16}.
_EXPECTED_K1397_12_CELLS = frozenset({
    (M, 256, K, dt)
    for M in (2048, 4096, 8192)
    for K in (2048, 32768)
    for dt in ("torch.bfloat16", "torch.float16")
})


def test_k1397_membership_exactly_12_kcomplement_cells():
    """The frozenset must equal the bucket rule exactly — no missing or
    unknown cells."""
    diff_missing = sorted(_EXPECTED_K1397_12_CELLS
                          - _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12)
    diff_unknown = sorted(_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12
                          - _EXPECTED_K1397_12_CELLS)
    assert _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == _EXPECTED_K1397_12_CELLS, (
        f"K-1397 P13 frozenset diverges from the K-COMPLEMENT bucket rule "
        f"(M ∈ {{2048,4096,8192}} × N=256 × K ∈ {{2048,32768}} × {{bf16,fp16}}).\n"
        f"  missing : {diff_missing}\n"
        f"  unknown : {diff_unknown}")


@pytest.mark.parametrize("M,N,K,dtype", sorted(_EXPECTED_K1397_12_CELLS))
def test_full_dispatch_routes_every_k1397_cell_to_hbl(M, N, K, dtype):
    """Every K-1397 admit cell must route to hipBLASLt under the full
    8-position precedence chain with default carve-out settings."""
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


@pytest.mark.parametrize("enable_streamk,work_stealing,b_dtype", [
    (True,  False, "torch.bfloat16"),  # streamk requested
    (False, True,  "torch.bfloat16"),  # work-stealing requested
    (False, False, "torch.float16"),   # mismatched dtype (a bf16, b fp16)
])
def test_k1397_admit_cell_carveout_short_circuits(
    enable_streamk, work_stealing, b_dtype,
):
    """Admit cells must NOT route to hipBLASLt when streamk / work-stealing
    / mismatched-dtype is requested.  Pick (4096, 256, 32768, bf16) as a
    representative K-1397 admit cell."""
    a_dtype = "torch.bfloat16"
    M, N, K = 4096, 256, 32768
    assert (M, N, K, a_dtype) in _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12
    routed = k971_route_decision(
        M, N, K, a_dtype, b_dtype, enable_streamk, work_stealing,
    )
    assert routed is False, (
        f"K-1397 P13 admit cell ({M},{N},{K},{a_dtype}) must NOT route to "
        f"hipBLASLt when carve-out applies (streamk={enable_streamk}, "
        f"work_stealing={work_stealing}, b_dtype={b_dtype}); short-circuit "
        f"in _k971_route_to_hbl is load-bearing.")
