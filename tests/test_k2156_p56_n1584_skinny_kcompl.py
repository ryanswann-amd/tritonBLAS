"""K-2156 (S-002) — invariants for P56 N=1584 K-COMPLEMENT alias-stack.

Per minimalist src/tests split (R-1532 / R-1720 / R-1775): the
`_route_predicate.py` module holds the *data* (frozenset literal); this
file holds the *invariants* — cardinality, residue class, dtype set,
disjointness with prior off-by-48 rungs, and the 18-cell membership
expansion.

The frozenset itself is a single-N slice (N=1584 only) over the canonical
M ∈ {2048, 4096, 8192} × K ∈ {4096, 8192, 16384} × {bf16, fp16} envelope,
i.e. 3 × 3 × 2 = 18 cells.

12th-rung extension of the off-by-48 contiguous ladder branched off
fix/K-2127@9c4f984 (matches K-2091 / K-2092 / K-2107 / K-2127 lineage).
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18,
    _K2156_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18,
)

EXPECTED_M = (2048, 4096, 8192)
EXPECTED_K = (4096, 8192, 16384)
EXPECTED_DT = ("torch.bfloat16", "torch.float16")
EXPECTED_N = 1584
EXPECTED_CARDINALITY = 18


def test_cardinality_is_18():
    assert len(_K2156_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18) == EXPECTED_CARDINALITY


def test_n_is_residue_48_mod_64():
    # Off-by-48 wave-misaligned membership invariant (mod-128 = 48 — same
    # sub-cycle as N=1456 / N=1328 / N=1200 / N=1072 / N=944 / N=816).
    for (M, N, K, dt) in _K2156_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18:
        assert N == EXPECTED_N
        assert N % 64 == 48
        assert N % 128 == 48


def test_full_grid_membership():
    expected = {
        (M, EXPECTED_N, K, dt)
        for M in EXPECTED_M
        for K in EXPECTED_K
        for dt in EXPECTED_DT
    }
    assert _K2156_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18 == expected


def test_dtype_set_is_bf16_fp16_only():
    dts = {dt for (_, _, _, dt) in _K2156_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18}
    assert dts == set(EXPECTED_DT)


def test_disjoint_with_prior_off_by_48_rungs():
    # K-2156 is a single-N slice at N=1584; every prior off-by-48 rung is at
    # a strictly different N (816..1456) so disjointness must hold trivially.
    from tritonblas._route_predicate import (
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
    )
    assert _K2156_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18.isdisjoint(
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18
    )
    assert _K2156_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18.isdisjoint(
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18
    )


def test_n1584_is_next_plus_64_above_n1520():
    # Sanity: N=1584 is the next contiguous +64 step beyond K-2127 P55 N=1520
    # (and +128 above the K-2127 P54 N=1456 admit).
    assert EXPECTED_N - 64 == 1520
    assert EXPECTED_N - 128 == 1456


# --- Dispatcher integration: verify the predicate is wired into the route ---

def test_dispatcher_routes_n1584_cells_to_hbl():
    """Verify the +1 LOC membership-check in `_k971_route_to_hbl` actually
    fires for every (M, N=1584, K, dt) cell in the K-2156 frozenset.

    This test guards against the dead-code regression flagged in K-2156 v1
    review: the frozenset must be CONSULTED by the live dispatcher, not
    just imported.  Pure-Python — no GPU / triton / origami needed.
    """
    # Import the live dispatcher predicate; this also exercises the +1 LOC
    # import line we added at the top of matmul.py.
    from tritonblas.matmul import _k971_route_to_hbl
    import torch as _torch

    dt_to_torch = {
        "torch.bfloat16": _torch.bfloat16,
        "torch.float16": _torch.float16,
    }
    routed = 0
    for (M, N, K, dt_str) in _K2156_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18:
        a_dtype = dt_to_torch[dt_str]
        b_dtype = a_dtype
        # enable_streamk=False, work_stealing=False — most common dispatch
        # path; route-OUT must fire regardless because it's the cohort
        # membership check, not a streamk gate.
        assert _k971_route_to_hbl(
            M, N, K, a_dtype, b_dtype, False, False
        ), f"dispatcher failed to route ({M},{N},{K},{dt_str}) to HBL"
        routed += 1
    assert routed == 18


@pytest.mark.parametrize("M,N,K", [
    # Immediate N neighbours of N=1584 — must NOT be routed by K-2156.
    (4096, 1583, 8192),
    (4096, 1585, 8192),
    (4096, 1648, 8192),  # next-rung not-yet-admitted (N=1584+64)
    (4096, 1520, 8192),  # K-2127 P55 N=1520 — routed by K-2127, not K-2156
    # N=1584 but at NOT-canonical M / K / dtype envelope cells.
    (1024, 1584, 8192),  # M=1024 not in cohort
    (4096, 1584, 2048),  # K=2048 not in cohort
])
def test_n1584_off_grid_not_in_k2156_frozenset(M, N, K):
    """K-2156 frozenset must NOT contain neighbours / off-grid cells; only
    the canonical 18-cell envelope is admitted by P56."""
    for dt in EXPECTED_DT:
        assert (M, N, K, dt) not in _K2156_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18


def test_frozenset_is_immutable():
    assert isinstance(_K2156_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18, frozenset)


@pytest.mark.parametrize("M,N,K", [
    (4096, 1583, 8192),  # N off-by-one (under)
    (4096, 1585, 8192),  # N off-by-one (over)
    (4096, 1648, 8192),  # next-rung not-yet-admitted (N=1584+64)
    (1024, 1584, 8192),  # M=1024 off-grid
    (4096, 1584, 2048),  # K=2048 off-grid
])
def test_dispatcher_does_not_route_n1584_boundary_cells(M, N, K):
    """Dispatcher must NOT route near-miss boundary cells through K-2156.

    Guards the +1 LOC membership-check boundary: a cell that differs from
    the K-2156 cohort by even ±1 in N or by being at an off-cohort M/K
    must NOT be routed to HBL by `_k971_route_to_hbl` (no other frozenset
    in the live alias-stack contains these cells).
    """
    from tritonblas.matmul import _k971_route_to_hbl
    import torch as _torch
    for dt in (_torch.bfloat16, _torch.float16):
        assert not _k971_route_to_hbl(M, N, K, dt, dt, False, False), (
            f"dispatcher unexpectedly routed boundary cell ({M},{N},{K},{dt}) to HBL"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
