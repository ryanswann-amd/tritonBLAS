"""K-1501 P20 ``skinny_N16384`` K-COMPLEMENT route-OUT — minimal pin set.

P20 is by-construction the *same object* as K-1478 P19
(`_K1501_P20_... is _K1478_P19_...`).  Once that identity holds, every
other behavioural property of P20 (set contents, predicate output,
admit-grid coverage, cross-arch validity) follows by definition.

So this file pins exactly two things:

* the alias identity (load-bearing — catches any future copy/divergence);
* one full-stack route-OUT smoke through ``k971_route_decision`` to
  confirm the 13-deep dispatch chain still routes a P20/P19 admit cell
  to hipBLASLt after the stack-extension.
"""
from __future__ import annotations

import torch

from tritonblas._route_predicate import (
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30,
    _K1501_P20_SKINNY_N16384_KCOMPL_ROUTEOUT_30,
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
