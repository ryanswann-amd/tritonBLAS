"""K-1412 P15 skinny_N512 K-COMPLEMENT (K-axis extremes) pin tests — torch-free.

Pins for the 12-cell `_K1412_P15_SKINNY_N512_KCOMPL_ROUTEOUT_12` frozenset
stacked at 9th-position of `k971_route_decision`:

  1. Membership: every admit cell is in the frozenset and routes to hipBLASLt.
  2. Non-membership (disjointness contract): adjacent K-axis cells, adjacent
     N-axis cells, and out-of-cohort M cells must NOT match the predicate
     (regression guard for the strict-equality 9th-position carve-out).
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K1412_P15_SKINNY_N512_KCOMPL_ROUTEOUT_12,
    _k1412_p15_skinny_n512_routeout,
    k971_route_decision,
)


_ADMIT_CELLS = sorted(_K1412_P15_SKINNY_N512_KCOMPL_ROUTEOUT_12)


# Adjacent / out-of-cohort cells that MUST NOT match the K-1412 P15 predicate.
# Covers (a) N=512 with K NOT in {2048, 32768} — the K-axis "mid-band" hole
# that K-1409 owns (not in this PR), (b) N != 512 (P13/P14 territory),
# (c) M outside the {2048,4096,8192} anchor row, (d) dtype out-of-cohort.
_NEGATIVE_CELLS = [
    # N=512 K-axis NEIGHBORS (mid-K band — must NOT be admitted by P15)
    (2048, 512,  4096, "torch.bfloat16"),
    (4096, 512,  8192, "torch.float16"),
    (8192, 512, 16384, "torch.bfloat16"),
    # N axis off (sibling N values owned by other predicates / no-op here)
    (4096, 256, 32768, "torch.bfloat16"),  # P14 N=256 territory
    (4096, 128,  2048, "torch.float16"),   # P13 N=128 territory
    (4096, 1024, 2048, "torch.bfloat16"),  # K971_ROUTE_TABLE territory
    # M out-of-cohort (M not in {2048, 4096, 8192} anchor row)
    (1024,  512,  2048, "torch.bfloat16"),
    (16384, 512, 32768, "torch.float16"),
    # dtype out-of-cohort (no fp32 admit)
    (4096, 512, 32768, "torch.float32"),
]


@pytest.mark.parametrize("M,N,K,dtype", _ADMIT_CELLS)
def test_admit_cells_match_predicate_and_route_to_hbl(M, N, K, dtype):
    """Every K-1412 P15 admit cell matches the predicate and routes to
    hipBLASLt under the full 9-position precedence chain."""
    assert _k1412_p15_skinny_n512_routeout(M, N, K, dtype) is True
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


@pytest.mark.parametrize("M,N,K,dtype", _NEGATIVE_CELLS)
def test_adjacent_cells_do_not_match_predicate(M, N, K, dtype):
    """Adjacent K-axis (mid-band), neighbouring N-axis, and out-of-cohort M
    cells MUST NOT be admitted by the K-1412 P15 strict-equality predicate.
    Pins the disjointness contract that lets P15 stack at 9th-position
    without colliding with P13 (N=128), P14 (N=256), or any future P15
    mid-K productionisation."""
    assert _k1412_p15_skinny_n512_routeout(M, N, K, dtype) is False
