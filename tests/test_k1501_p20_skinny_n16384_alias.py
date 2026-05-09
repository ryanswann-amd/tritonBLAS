"""Pin tests for the K-1501 P20 ``skinny_N16384`` K-COMPLEMENT route-OUT
13th-position alias-stack of the K-1478 P19 admit envelope.

The K-1501 P20 frozenset is by-construction the *same object* as the
K-1478 P19 frozenset — `_K1501_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_30 =
_K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30` — so the predicate is
unreachable while P19 is enabled (short-circuit semantics).  This is
deliberate: the slot is load-bearing if K-1478 P19 is ever ablated, and
the alias guarantees the two rules cannot silently diverge.

These tests guard the alias invariant and the predicate behaviour so a
future edit that breaks the coupling (or accidentally re-binds P20 to a
different set) is caught at test time rather than in production.

Coverage:

* **Alias invariant** — `_K1501_P20_...` MUST be the same object as the
  `_K1478_P19_...` frozenset (`is` identity), with identical contents
  and length.
* **Predicate behaviour** — every one of the 30 admit cells must return
  True from `_k1501_p20_skinny_n16384_routeout`, AND must produce the
  same boolean as the P19 predicate on every cell of the underlying
  sweep grid (alias-equivalence).
* **Boundary controls** — cells outside the bucket must not route via P20.
* **Full-stack `k971_route_decision` integration** — every P20 admit cell
  routes True through the now-13-deep dispatch chain (regression guard
  for P5–P19 envelopes that share the route-OUT chain).
* **Prior P5–P19 regression** — at least one representative admit cell
  from every prior K-COMPLEMENT productionized predicate (P12, P13×2,
  P15, P16, P17, P19) still routes to hipBLASLt through the full chain
  after the P20 stack-extension.  This is the post-stack no-regression
  pin requested by the K-1501 PRD.

Cross-arch (MI325X / gfx942) note: P20 is alias-equal to P19 by
construction, so the K-1459-pattern MI325X backtest data for P19
inherits unchanged.  These unit tests exercise only the routing logic
(no GPU required) — the alias invariant is the cross-arch firewall.
"""
from __future__ import annotations

import pytest
import torch

from tritonblas._route_predicate import (
    # P20 (this PR — K-1501)
    _K1501_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_30,
    _k1501_p20_skinny_n16384_routeout,
    # P19 (K-1478) — alias source
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30,
    _k1478_p19_skinny_n16384_routeout,
    # Prior productionized K-COMPLEMENT envelopes — regression representatives
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29,
    _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17,
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# Alias-by-construction invariant — the load-bearing pin.  If this fails
# the whole "P20 is unreachable but ready-to-ablate" design has been
# silently broken.
# ---------------------------------------------------------------------------

def test_p20_frozenset_is_same_object_as_p19():
    """Object-identity invariant (`is`) — P20 MUST be a direct alias of P19,
    not a copy.  A copy would let the two diverge under future edits to
    one of the two definitions; the `is` pin makes any divergence a
    test-time error."""
    assert (
        _K1501_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_30
        is _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30
    )


def test_p20_frozenset_contents_equal_p19():
    """Belt-and-braces: even if `is` were ever weakened to `==`, the two
    sets must contain the same 30 cells (this would catch e.g. a
    well-meaning refactor that re-built P20 from a literal)."""
    assert (
        _K1501_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_30
        == _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30
    )


def test_p20_frozenset_size_is_exactly_30():
    assert len(_K1501_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_30) == 30


# ---------------------------------------------------------------------------
# Predicate alias-equivalence — P20 predicate MUST agree with P19
# predicate on every cell of the underlying sweep grid (3 × 1 × 5 × 2 = 30
# admit cells + a handful of boundary controls).
# ---------------------------------------------------------------------------

P20_ADMIT_GRID = [
    (M, 16384, K, dt)
    for M in (2048, 4096, 8192)
    for K in (2048, 4096, 8192, 16384, 32768)
    for dt in (torch.bfloat16, torch.float16)
]


@pytest.mark.parametrize("M,N,K,dtype", P20_ADMIT_GRID)
def test_p20_predicate_returns_true_for_every_admit_cell(M, N, K, dtype):
    assert _k1501_p20_skinny_n16384_routeout(M, N, K, dtype) is True


@pytest.mark.parametrize("M,N,K,dtype", P20_ADMIT_GRID)
def test_p20_predicate_agrees_with_p19_on_admit_grid(M, N, K, dtype):
    """Alias-equivalence at the predicate level — the two callables must
    return the same boolean for every cell of the K-1478 sweep grid.
    This is the runtime mirror of the `is`-identity invariant on the
    underlying frozenset."""
    assert (
        _k1501_p20_skinny_n16384_routeout(M, N, K, dtype)
        is _k1478_p19_skinny_n16384_routeout(M, N, K, dtype)
    )


# ---------------------------------------------------------------------------
# Boundary controls — cells outside the bucket must NOT route via P20.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype", [
    (1024, 16384,  8192, torch.bfloat16),    # M=1024 — out of bucket
    (16384, 16384, 8192, torch.bfloat16),    # M=16384 — out of bucket
    (4096,  8192,  8192, torch.bfloat16),    # N=8192 — sibling-N firewall (P19@N=8192 territory)
    (4096, 16384,  1024, torch.bfloat16),    # K=1024 — out of bucket
    (4096, 16384, 65536, torch.bfloat16),    # K=65536 — out of bucket
    (4096, 16384,  8192, torch.float32),     # dtype not in {bf16,fp16}
])
def test_p20_predicate_returns_false_outside_bucket(M, N, K, dtype):
    assert _k1501_p20_skinny_n16384_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# Full-stack k971_route_decision integration — every P20 admit cell must
# route True through the now-13-deep dispatch chain.  (In practice these
# cells are caught by P19 at chain pos 12, not P20 at pos 13 — but the
# end-to-end OUT outcome is what matters and is what would regress if
# P19 were ever ablated.)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype", P20_ADMIT_GRID)
def test_full_dispatch_chain_routes_p20_admit_cell_to_hbl(M, N, K, dtype):
    assert k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False,
    ) is True


# ---------------------------------------------------------------------------
# Prior P5-P19 admit-envelope regression — at least one representative
# admit cell from every prior productionized K-COMPLEMENT envelope must
# still route OUT through the full chain after the P20 stack-extension.
# This is the post-stack no-regression pin called for in the K-1501 PRD.
#
# Each tuple: (rule-tag, (M, N, K, dtype)).  Cells are picked to be a
# strict member of the named frozenset so this test fails noisily if a
# future refactor changes the frozenset contents.
# ---------------------------------------------------------------------------

PRIOR_ENVELOPE_REPRESENTATIVES = [
    ("P12-K1295-square_mid",
     next(iter(_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4))),
    ("P13-K1367-skinny_N128",
     next(iter(_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18))),
    ("P13-K1397-skinny_N256",
     next(iter(_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12))),
    ("P15-K1409-skinny_N512_extremes",
     next(iter(_K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT))),
    ("P16-K1429-skinny_N1024",
     next(iter(_K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29))),
    ("P17-K1437-skinny_N512_base",
     next(iter(_K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17))),
    ("P19-K1478-skinny_N16384",
     next(iter(_K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30))),
]


@pytest.mark.parametrize("rule_tag,cell_key", PRIOR_ENVELOPE_REPRESENTATIVES)
def test_prior_envelope_representative_still_routes_to_hbl(rule_tag, cell_key):
    """Post-stack no-regression: a representative cell from every prior
    productionized K-COMPLEMENT envelope must still route True through
    the 13-deep `k971_route_decision` chain after the P20 alias-stack
    extension.  The `rule_tag` is purely diagnostic — failure messages
    will name the offending rule."""
    M, N, K, dtype_str = cell_key
    dtype = (torch.bfloat16 if dtype_str == "torch.bfloat16"
             else torch.float16 if dtype_str == "torch.float16"
             else torch.float32)
    assert k971_route_decision(
        int(M), int(N), int(K), dtype, dtype,
        enable_streamk=False, work_stealing=False,
    ) is True, f"regression: {rule_tag} representative {cell_key} no longer routes OUT"
