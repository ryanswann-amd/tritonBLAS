"""Invariants for the N=1840 residue-48 K-COMPLEMENT alias-stack route.

Per minimalist src/tests split (R-1532 / R-1720 / R-1775): the
`_route_predicate.py` module holds the *data* (frozenset literal); this
file holds the *invariants* — cardinality, residue class, dtype set,
disjointness with prior off-by-48 rungs, the 18-cell membership
expansion, adjacency firewalls (the cells immediately neighbouring
N=1840 in the prior admit set must remain un-routed), and an explicit
dispatcher-contract check exercising ``_k971_route_to_hbl`` with
positive admit cells, off-residue / off-by-one negatives, wrong-dtype
negatives, and the immediate-neighbour adjacency firewall cells.

The frozenset itself is a single-N slice (N=1840 only) over the full
canonical 18-cell envelope: M in {2048,4096,8192} x K in {4096,8192,
16384} x {bf16, fp16}.

16th-rung extension of the off-by-48 contiguous ladder.
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _SKINNY_N1840_RESIDUE48_HBL_ROUTE,
)

EXPECTED_M = (2048, 4096, 8192)
EXPECTED_K = (4096, 8192, 16384)
EXPECTED_DTYPES = ("torch.bfloat16", "torch.float16")
EXPECTED_N = 1840
EXPECTED_CARDINALITY = 18


def test_cardinality_is_18():
    assert len(_SKINNY_N1840_RESIDUE48_HBL_ROUTE) == EXPECTED_CARDINALITY


def test_n_is_residue_48_mod_64():
    # Off-by-48 wave-misaligned membership invariant
    for (M, N, K, dt) in _SKINNY_N1840_RESIDUE48_HBL_ROUTE:
        assert N == EXPECTED_N
        assert N % 64 == 48
        # mod 128 == 48 returns to the N=1456 / N=1648 sub-class -- admit
        # identically (only mod-64==48 binds).
        assert N % 128 == 48


def test_full_grid_membership():
    expected = {
        (M, EXPECTED_N, K, dt)
        for M in EXPECTED_M
        for K in EXPECTED_K
        for dt in EXPECTED_DTYPES
    }
    assert _SKINNY_N1840_RESIDUE48_HBL_ROUTE == expected


def test_dtype_set_is_bf16_fp16_only():
    dts = {dt for (_, _, _, dt) in _SKINNY_N1840_RESIDUE48_HBL_ROUTE}
    assert dts == {"torch.bfloat16", "torch.float16"}


def test_full_3x3_grid_per_dtype():
    for dt in EXPECTED_DTYPES:
        cells = {
            (M, K) for (M, _, K, d) in _SKINNY_N1840_RESIDUE48_HBL_ROUTE
            if d == dt
        }
        expected = {(M, K) for M in EXPECTED_M for K in EXPECTED_K}
        assert cells == expected, f"Missing 3x3 grid for dtype {dt}"


def test_disjoint_with_prior_off_by_48_rungs():
    # Single-N slice at N=1840; every prior off-by-48 rung is at a
    # strictly different N so disjointness must hold trivially.
    from tritonblas._route_predicate import (
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18,
        _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18,
        _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8,
        _K2183_P57_SKINNY_N1712_KCOMPL_ALIASSTACK_8,
        _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10,
    )
    assert _SKINNY_N1840_RESIDUE48_HBL_ROUTE.isdisjoint(
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18
    )
    assert _SKINNY_N1840_RESIDUE48_HBL_ROUTE.isdisjoint(
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18
    )
    assert _SKINNY_N1840_RESIDUE48_HBL_ROUTE.isdisjoint(
        _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18
    )
    assert _SKINNY_N1840_RESIDUE48_HBL_ROUTE.isdisjoint(
        _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8
    )
    assert _SKINNY_N1840_RESIDUE48_HBL_ROUTE.isdisjoint(
        _K2183_P57_SKINNY_N1712_KCOMPL_ALIASSTACK_8
    )
    assert _SKINNY_N1840_RESIDUE48_HBL_ROUTE.isdisjoint(
        _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10
    )


def test_n1840_extends_off_by_48_ladder_above_n1776():
    # Sanity: N=1840 is the next admitted residue-48 step above N=1776
    # (+64 step on the residue-48 lattice).
    assert (EXPECTED_N - 1776) % 64 == 0
    assert (EXPECTED_N - 1776) == 64
    assert EXPECTED_N > 1776


def test_adjacency_firewalls_n1840_neighbors():
    # The cells immediately neighbouring N=1840 on the residue-48 lattice
    # (N=1776 is the prior admit immediately below; N=1904 is the next
    # un-audited rung above) and off-residue N must remain OUTSIDE the
    # N=1840 frozenset -- preserve the per-N audit handles.
    forbidden = [
        # N=1776 (prior admit, lives in its own frozenset, not this one)
        (4096, 1776, 8192, "torch.bfloat16"),
        # N=1904 (next un-audited residue-48 rung above 1840)
        (4096, 1904, 8192, "torch.bfloat16"),
        # Off-residue N (1856 is wave-aligned mod 64 == 0, not residue-48)
        (4096, 1856, 8192, "torch.bfloat16"),
        # Off-residue N (1824 is residue-32, not residue-48)
        (4096, 1824, 8192, "torch.bfloat16"),
        # Wrong dtype (fp32) at admit cell
        (4096, 1840, 8192, "torch.float32"),
    ]
    for cell in forbidden:
        assert cell not in _SKINNY_N1840_RESIDUE48_HBL_ROUTE, (
            f"N=1840 frozenset must NOT contain firewall cell {cell}"
        )


# ---------------------------------------------------------------------------
# Dispatcher-contract tests: exercise ``_k971_route_to_hbl`` directly with
# positive admits, off-by-one N negatives, off-residue N negatives, wrong-
# dtype negatives, and adjacency firewall cells.  Catches regressions that
# pure frozenset-shape tests would miss (e.g. wiring the membership check
# behind the wrong dtype branch, or failing the int() coercion).
# ---------------------------------------------------------------------------


def _has_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


_TORCH_AVAILABLE = _has_torch()
_REQUIRES_TORCH = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="torch not importable; dispatcher contract test skipped",
)


def _route(M, N, K, dtype):
    """Thin wrapper around the dispatcher predicate.

    The torch dependency is only needed because matmul.py imports torch at
    module load time.
    """
    from tritonblas.matmul import _k971_route_to_hbl
    # The dispatcher signature is (M, N, K, a_dtype, b_dtype, enable_streamk,
    # work_stealing); membership-check uses a_dtype only for these admits.
    return _k971_route_to_hbl(M, N, K, dtype, dtype, False, False)


@_REQUIRES_TORCH
def test_dispatcher_routes_positive_admit_cells():
    # Every cell in the admit envelope must route True for both bf16 and fp16.
    import torch
    dtype_map = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}
    for (M, N, K, dt_str) in _SKINNY_N1840_RESIDUE48_HBL_ROUTE:
        dt = dtype_map[dt_str]
        assert _route(M, N, K, dt) is True, (
            f"Dispatcher must route admit cell ({M},{N},{K},{dt_str}) -> True"
        )


@_REQUIRES_TORCH
def test_dispatcher_rejects_off_by_one_n_neighbors():
    # Off-by-one N values around 1840 (not on residue-48 lattice) must NOT
    # be routed by *this* admit.  We pick N values that no other K-COMPLEMENT
    # alias-stack slot covers (1839, 1841, 1842) and a representative M/K/dt
    # from the canonical envelope.
    import torch
    for N_neg in (1839, 1841, 1842):
        assert _route(4096, N_neg, 8192, torch.bfloat16) is False, (
            f"Dispatcher must NOT route off-by-one N={N_neg} -> True"
        )


@_REQUIRES_TORCH
def test_dispatcher_rejects_adjacent_residue48_n_above():
    # N=1904 is the next un-audited residue-48 rung above 1840.  It must
    # remain OUTSIDE the routed set until a future ticket admits it.
    import torch
    assert _route(4096, 1904, 8192, torch.bfloat16) is False
    assert _route(4096, 1904, 8192, torch.float16) is False


@_REQUIRES_TORCH
def test_dispatcher_rejects_wave_aligned_n_neighbor():
    # N=1856 is wave-aligned (mod 64 == 0) — different mechanism class,
    # must not be admitted.
    import torch
    assert _route(4096, 1856, 8192, torch.bfloat16) is False
    assert _route(4096, 1856, 8192, torch.float16) is False


@_REQUIRES_TORCH
def test_dispatcher_rejects_wrong_dtype_at_admit_cell():
    # fp32 at an otherwise-admitted (M,N,K) tuple must NOT route — the
    # frozenset key includes the dtype string, so fp32 is a structural miss.
    import torch
    assert _route(4096, 1840, 8192, torch.float32) is False
    assert _route(2048, 1840, 4096, torch.float32) is False


@_REQUIRES_TORCH
def test_dispatcher_routes_prior_rung_n1776_independently():
    # Sanity: the prior rung (N=1776) must still route True via *its own*
    # frozenset — proves we did not accidentally break the prior dispatcher
    # entry while wiring this one.
    import torch
    assert _route(4096, 1776, 8192, torch.bfloat16) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
