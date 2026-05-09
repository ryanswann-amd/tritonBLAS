"""K-1501 P20 ``skinny_N16384`` K-COMPLEMENT route-OUT — minimal pin set.

P20 is by-construction the *same object* as K-1478 P19
(`_K1501_P20_... is _K1478_P19_...`).  Once that identity holds, every
other behavioural property of P20 (set contents, predicate output,
admit-grid coverage, cross-arch validity) follows by definition.

So this file pins exactly four things:

* the alias identity (load-bearing — catches any future copy/divergence);
* one full-stack route-OUT smoke through ``k971_route_decision`` to
  confirm the 13-deep dispatch chain still routes a P20/P19 admit cell
  to hipBLASLt after the stack-extension;
* envelope-boundary negative pins — non-admit cells (out-of-bucket M,
  N, K, or dtype) MUST return False from the P20 predicate so future
  frozenset drift cannot silently expand the admit envelope;
* a post-stack negative-routing smoke through ``k971_route_decision``
  for a non-admit cell, confirming P20 does not over-claim through
  the full dispatch chain.
"""
from __future__ import annotations

import pytest
import torch

from tritonblas._route_predicate import (
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30,
    _K1501_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_30,
    _k1501_p20_skinny_n16384_routeout,
    k971_route_decision,
)


def test_p20_is_alias_of_p19():
    """Object-identity invariant — P20 MUST be the SAME object as P19,
    not a copy.  Any future edit that breaks this (e.g. rebuilding P20
    from a literal) is caught here."""
    assert (
        _K1501_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_30
        is _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30
    )


def test_full_stack_routes_admit_cell_out():
    """Post-stack no-regression smoke: a representative P19/P20 admit
    cell still routes True through the now-13-deep dispatch chain."""
    assert k971_route_decision(
        4096, 16384, 8192, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False,
    ) is True


# ---------------------------------------------------------------------------
# Envelope-boundary negative pins (Testing Zealot review fix).
#
# The P20 frozenset is exactly the K-1478 P19 30-cell admit envelope:
#   M ∈ {2048, 4096, 8192} × N = 16384 ×
#   K ∈ {2048, 4096, 8192, 16384, 32768} × dtype ∈ {bf16, fp16}
#
# Every cell BELOW must lie strictly OUTSIDE that envelope along at
# least one axis (M / N / K / dtype) and therefore MUST return False
# from `_k1501_p20_skinny_n16384_routeout`.  Pins each axis-boundary
# explicitly so any future re-author of the P20 frozenset (e.g.
# accidental literal copy that drifts from P19) is caught.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype_str,axis", [
    # M-axis boundary (M not in {2048, 4096, 8192})
    (1024, 16384, 8192, "torch.bfloat16", "M=1024 below admit row"),
    (16384, 16384, 8192, "torch.bfloat16", "M=16384 above admit row"),
    # N-axis boundary (N != 16384) — sibling-N firewall vs P15-P17/P19 cousins
    (4096,  8192, 8192, "torch.bfloat16", "N=8192 sibling not P20"),
    (4096, 32768, 8192, "torch.bfloat16", "N=32768 above admit column"),
    (4096,  1024, 8192, "torch.bfloat16", "N=1024 P16 sibling not P20"),
    # K-axis boundary (K not in {2048, 4096, 8192, 16384, 32768})
    (4096, 16384, 1024, "torch.bfloat16", "K=1024 below admit grid"),
    (4096, 16384, 65536, "torch.bfloat16", "K=65536 above admit grid"),
    # dtype boundary (not in {bf16, fp16})
    (4096, 16384, 8192, "torch.float32", "fp32 dtype outside admit set"),
])
def test_p20_predicate_returns_false_outside_envelope(M, N, K, dtype_str, axis):
    """Negative pin: any cell outside the 30-cell admit envelope MUST
    return False from `_k1501_p20_skinny_n16384_routeout`.  Catches
    future P20-frozenset drift that would silently widen the envelope.
    """
    dtype = (
        torch.bfloat16 if dtype_str == "torch.bfloat16"
        else torch.float16 if dtype_str == "torch.float16"
        else torch.float32
    )
    assert _k1501_p20_skinny_n16384_routeout(M, N, K, dtype) is False, (
        f"P20 predicate must reject out-of-envelope cell on axis: {axis} "
        f"(M={M}, N={N}, K={K}, dtype={dtype_str})"
    )


def test_full_stack_does_not_route_out_for_non_admit_cell():
    """Post-stack negative-routing smoke: a cell strictly outside the
    P20/P19 envelope (M=1024 — below admit row) MUST NOT be routed to
    hipBLASLt by the now-13-deep dispatch chain via the P20 predicate.

    Together with `test_full_stack_routes_admit_cell_out` this pins both
    sides of the boundary at the dispatch-chain level.
    """
    assert k971_route_decision(
        1024, 16384, 8192, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False,
    ) is False
