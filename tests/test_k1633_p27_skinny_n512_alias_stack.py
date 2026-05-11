"""K-1633 P27 — alias of K-1552 P23.  Routing covered transitively."""
from tritonblas._route_predicate import (
    _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30 as _P23,
    _K1633_P27_SKINNY_N512_KCOMPL_ALIASSTACK_30 as _P27,
)


def test_p27_is_identity_alias_of_p23():
    # Identity catches a future rebind to a value-equal duplicate.
    assert _P27 is _P23


def test_p27_cardinality_30():
    assert len(_P27) == 30
