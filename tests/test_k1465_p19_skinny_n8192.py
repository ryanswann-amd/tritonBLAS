"""K-1474 P19 skinny_N8192 K=2048-COLUMN pin tests — torch-free.

Minimal pin coverage for the K-1474 productionisation of the
6-cell `_K1474_P19_SKINNY_N8192_KCOMPL_K2048_ROUTEOUT_6` frozenset
stacked as the 12th-position envelope on top of the K-1437 P17 11-predicate
stack (envelope grows 127 -> 133 cells; +6 admits at the K=2048 column of
the N=8192 column-narrow regime).

K-1474 SHRUNK FROM K-1465's predicted 29-cell envelope after re-measurement
on MI300X gfx942 / the current ROCm found that K-1465's predicted speedups did not
reproduce on the current stack.  Productionised set is the 6 K=2048 cells
where re-measurement shows ratio_median >= 1.05 AND CI95-lo >= 1.04.
The other 24 K-1465 sweep cells fall through to native triton dispatch.

The pytest suite focuses on the orthogonal behavioural pins:

  * exact membership of the K-1474 P19 6 cells
    (M in {2048,4096,8192} x N=8192 x K=2048 x {bf16, fp16})
  * full-chain dispatch routes every admit cell to hipBLASLt
  * carve-out short-circuits (streamk / work-stealing / mismatched-dtype)
    correctly suppress the route-OUT for an admit cell
  * P17 (K-1437 EXTENDED, N=1024) cells STILL route to hipBLASLt under the
    new chain (zero regression on the 11th-position envelope -- load-bearing
    stacking invariant)
  * P16 (K-1433 BASE, N=1024) cells STILL route to hipBLASLt
  * the K-1465 sweep cells that were SHRUNK OUT (K >= 4096 at N=8192) are
    NOT admitted by the P19 predicate (firewall against any future
    accidental re-broadening to the K-1465 29-cell set)
  * four nearest-neighbour negative pin tests for shapes one axis off the
    K-1474 P19 admit grid that must NOT be admitted by the P19 predicate
    function (firewall against silent over-routing from a future range/
    pattern predicate refactor)
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K1433_P16_SKINNY_N1024_KCOMPL_ROUTEOUT,
    _K1437_P17_SKINNY_N1024_KCOMPL_EXT_ROUTEOUT_12,
    _K1474_P19_SKINNY_N8192_KCOMPL_K2048_ROUTEOUT_6,
    _K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT,  # backwards-compat alias
    _k1465_p19_skinny_n8192_routeout,
    k971_route_decision,
)


# Bucket rule per K-1474 productionisation: M in {2048,4096,8192} x N=8192
# x K=2048 x dtype in {bf16, fp16}.  The K-1465 sweep cells at K >= 4096
# were SHRUNK OUT by K-1474 re-measurement and fall through to native
# triton dispatch.
_EXPECTED_K1474_6_CELLS = frozenset({
    (M, 8192, 2048, dt)
    for M in (2048, 4096, 8192)
    for dt in ("torch.bfloat16", "torch.float16")
})
assert len(_EXPECTED_K1474_6_CELLS) == 6

# K-1465 sweep cells that were SHRUNK OUT (K >= 4096 at N=8192 on the M
# anchor row) -- these MUST NOT be admitted by the K-1474 P19 predicate.
_K1465_SHRUNK_OUT_CELLS = [
    (M, 8192, K, dt)
    for M in (2048, 4096, 8192)
    for K in (4096, 8192, 16384, 32768)
    for dt in ("torch.bfloat16", "torch.float16")
]
# Drop the lone K-1465 sweep reject which was already excluded
_K1465_SHRUNK_OUT_CELLS = [c for c in _K1465_SHRUNK_OUT_CELLS
                           if c != (2048, 8192, 32768, "torch.float16")]
assert len(_K1465_SHRUNK_OUT_CELLS) == 23


def test_k1474_membership_exactly_6_k2048_column_cells():
    """The frozenset must equal the bucket rule (M anchor row x K=2048
    x both dtypes) exactly -- no missing or unknown cells."""
    diff_missing = sorted(_EXPECTED_K1474_6_CELLS
                          - _K1474_P19_SKINNY_N8192_KCOMPL_K2048_ROUTEOUT_6)
    diff_unknown = sorted(_K1474_P19_SKINNY_N8192_KCOMPL_K2048_ROUTEOUT_6
                          - _EXPECTED_K1474_6_CELLS)
    assert (
        _K1474_P19_SKINNY_N8192_KCOMPL_K2048_ROUTEOUT_6
        == _EXPECTED_K1474_6_CELLS
    ), (
        f"K-1474 P19 frozenset diverges from the K=2048 column bucket rule "
        f"(M in {{2048,4096,8192}} x N=8192 x K=2048 x {{bf16,fp16}}).\n"
        f"  missing : {diff_missing}\n"
        f"  unknown : {diff_unknown}")


def test_k1474_backwards_compat_alias_points_at_shrunken_set():
    """The legacy `_K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT` symbol (kept
    for K-1465 workspace artefact compatibility) must alias the K-1474
    6-cell shrunken set, NOT the K-1465 29-cell predicted set."""
    assert (_K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT
            is _K1474_P19_SKINNY_N8192_KCOMPL_K2048_ROUTEOUT_6), (
        "K-1465 alias must point at K-1474 productionised set.")
    assert len(_K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT) == 6


@pytest.mark.parametrize("M,N,K,dtype", sorted(_EXPECTED_K1474_6_CELLS))
def test_full_dispatch_routes_every_k1474_cell_to_hbl(M, N, K, dtype):
    """Every K-1474 P19 admit cell must route to hipBLASLt under the full
    12-position precedence chain with default carve-out settings."""
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


@pytest.mark.parametrize("M,N,K,dtype", _K1465_SHRUNK_OUT_CELLS)
def test_k1465_shrunk_out_cells_not_admitted_by_p19_predicate(M, N, K, dtype):
    """K-1465's predicted 29-cell envelope included cells at K >= 4096 on
    the N=8192 column.  K-1474 re-measurement showed those cells DO NOT
    win under route-OUT (12 of them are actually faster on tritonblas, would
    REGRESS by 0.4-3.8% if shipped).  The P19 predicate MUST NOT admit
    them.  This is the load-bearing K-1474 SHRUNK-FROM-K1465 firewall."""
    assert (M, N, K, dtype) not in _K1474_P19_SKINNY_N8192_KCOMPL_K2048_ROUTEOUT_6, (
        f"Test scaffolding bug: K-1465 SHRUNK_OUT cell {(M,N,K,dtype)} is in "
        f"the K-1474 productionised 6-cell set.")
    p19_admit = _k1465_p19_skinny_n8192_routeout(M, N, K, dtype)
    assert p19_admit is False, (
        f"K-1474 SHRUNK-FROM-K1465 firewall LEAK: K-1465 sweep cell "
        f"{(M,N,K,dtype)} (NOT in K-1474 6-cell productionised set) was "
        f"admitted by the P19 predicate.  K-1474 re-measurement showed this "
        f"cell either ties or favours tritonblas; routing OUT would regress "
        f"performance.")


@pytest.mark.parametrize("enable_streamk,work_stealing,b_dtype", [
    (True,  False, "torch.bfloat16"),  # streamk requested
    (False, True,  "torch.bfloat16"),  # work-stealing requested
    (False, False, "torch.float16"),   # mismatched dtype (a bf16, b fp16)
])
def test_k1474_admit_cell_carveout_short_circuits(
    enable_streamk, work_stealing, b_dtype,
):
    """Admit cells must NOT route to hipBLASLt when streamk / work-stealing
    / mismatched-dtype is requested.  Pick (2048, 8192, 2048, bf16) -- the
    cell with the highest measured speedup r=1.080x -- as the representative
    carve-out probe."""
    a_dtype = "torch.bfloat16"
    M, N, K = 2048, 8192, 2048
    assert (M, N, K, a_dtype) in _K1474_P19_SKINNY_N8192_KCOMPL_K2048_ROUTEOUT_6
    routed = k971_route_decision(
        M, N, K, a_dtype, b_dtype, enable_streamk, work_stealing,
    )
    assert routed is False, (
        f"K-1474 P19 admit cell ({M},{N},{K},{a_dtype}) must NOT route to "
        f"hipBLASLt when carve-out applies (streamk={enable_streamk}, "
        f"work_stealing={work_stealing}, b_dtype={b_dtype}); short-circuit "
        f"in _k971_route_to_hbl is load-bearing.")


@pytest.mark.parametrize("M,N,K,dtype",
                         sorted(_K1437_P17_SKINNY_N1024_KCOMPL_EXT_ROUTEOUT_12))
def test_k1437_p17_extended_envelope_unchanged_under_p19_stack(M, N, K, dtype):
    """Regression: every K-1437 P17 EXTENDED (11th-position, N=1024) cell
    must STILL route to hipBLASLt under the new 12-position stack."""
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


@pytest.mark.parametrize("M,N,K,dtype",
                         sorted(_K1433_P16_SKINNY_N1024_KCOMPL_ROUTEOUT))
def test_k1433_p16_base_envelope_unchanged_under_p19_stack(M, N, K, dtype):
    """Regression: every K-1433 P16 BASE (10th-position, N=1024) cell must
    STILL route to hipBLASLt under the new 12-position stack."""
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


# ---------------------------------------------------------------------------
# Negative pin tests -- strict-equality firewall against silent over-routing
# from the P19 predicate.  Each negative shape varies exactly ONE axis off
# the K-1474 P19 admit grid.
# ---------------------------------------------------------------------------
_K1474_NEAREST_NEIGHBOUR_NEGATIVES = [
    # (M,    N,    K,     a_dtype,           reason)
    (4096,  4096,  2048, "torch.bfloat16",
     "N=4096 OFF column tier (P19 fixes N=8192)"),
    (4096, 16384,  2048, "torch.bfloat16",
     "N=16384 OFF column tier (one column past N=8192)"),
    (1024,  8192,  2048, "torch.bfloat16",
     "M=1024 OFF anchor row (P19 anchors M in {2048,4096,8192})"),
    (4096,  8192,  2048, "torch.float32",
     "dtype OFF (P19 admits {bf16,fp16}); fp32 not in K-COMPLEMENT cohort"),
]


@pytest.mark.parametrize("M,N,K,dtype,reason",
                         _K1474_NEAREST_NEIGHBOUR_NEGATIVES)
def test_k1474_nearest_neighbour_shapes_not_admitted_by_p19_predicate(
    M, N, K, dtype, reason,
):
    """Negative pin: nearest non-member neighbours one axis off the
    K-1474 P19 admit grid must NOT be admitted by the P19 strict-equality
    predicate function."""
    assert (M, N, K, dtype) not in _K1474_P19_SKINNY_N8192_KCOMPL_K2048_ROUTEOUT_6, (
        f"Test scaffolding bug: neighbour ({M},{N},{K},{dtype}) is actually "
        f"in the P19 admit set -- the negative test would pass trivially.")
    p19_admit = _k1465_p19_skinny_n8192_routeout(M, N, K, dtype)
    assert p19_admit is False, (
        f"P19 strict-equality firewall LEAK: ({M},{N},{K},{dtype}) -- {reason} "
        f"-- P19 predicate returned True, but the K-1474 admit set is exactly "
        f"the 6-cell K=2048 column grid (M in {{2048,4096,8192}} x N=8192 x "
        f"K=2048 x {{bf16,fp16}}); any True outside this set signals an "
        f"over-broad predicate.")
