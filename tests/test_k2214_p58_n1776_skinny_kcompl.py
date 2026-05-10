"""K-2214 (S-002) — invariants for P58 N=1776 K-COMPLEMENT alias-stack.

Per minimalist src/tests split (R-1532 / R-1720 / R-1775): the
`_route_predicate.py` module holds the *data* (frozenset literal); this
file holds the *invariants* — cardinality, residue class, dtype set,
disjointness with prior off-by-48 rungs, the 10-cell membership
expansion, and adjacency firewalls (the cells immediately neighbouring
N=1776 in the K-2183 14-cell admit set must remain un-routed).

The frozenset itself is a single-N slice (N=1776 only) over the
task-specified verification cohort: 9 bf16 cells (M in {2048,4096,8192}
x K in {4096,8192,16384}) + 1 fp16 spot-check at (M=4096, K=8192) =
10 cells (smaller than the canonical 18-cell envelope used by K-2127 /
K-2158 — extension to the full grid is deferred to a follow-up
rung-stitch task).

15th-rung extension of the off-by-48 contiguous ladder branched off
fix/K-2183@9375e7a (which itself branches off fix/K-2169@161e017 ->
fix/K-2158@beab03e -> fix/K-2127@9c4f984 -> fix/K-1922@0024a71 —
matches K-2091 / K-2092 / K-2107 / K-2127 / K-2158 / K-2169 / K-2183
lineage).
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10,
)

EXPECTED_BF16_M = (2048, 4096, 8192)
EXPECTED_BF16_K = (4096, 8192, 16384)
EXPECTED_FP16_SPOT = (4096, 1776, 8192, "torch.float16")
EXPECTED_N = 1776
EXPECTED_CARDINALITY = 10


def test_cardinality_is_10():
    assert len(_K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10) == EXPECTED_CARDINALITY


def test_n_is_residue_48_mod_64():
    # Off-by-48 wave-misaligned membership invariant
    for (M, N, K, dt) in _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10:
        assert N == EXPECTED_N
        assert N % 64 == 48
        # mod 128 == 112 is the K-2055-2066 / K-2071-2091 / K-2169 sub-class
        # — admit identically per K-2150 R-K2150.MOD-64-IS-BINDING.
        assert N % 128 == 112


def test_full_grid_membership():
    expected_bf16 = {
        (M, EXPECTED_N, K, "torch.bfloat16")
        for M in EXPECTED_BF16_M
        for K in EXPECTED_BF16_K
    }
    expected = expected_bf16 | {EXPECTED_FP16_SPOT}
    assert _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10 == expected


def test_dtype_set_is_bf16_fp16_only():
    dts = {dt for (_, _, _, dt) in _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10}
    assert dts == {"torch.bfloat16", "torch.float16"}


def test_bf16_cells_are_full_3x3_grid():
    bf16_cells = {
        (M, K) for (M, _, K, dt) in _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10
        if dt == "torch.bfloat16"
    }
    expected_bf16 = {(M, K) for M in EXPECTED_BF16_M for K in EXPECTED_BF16_K}
    assert bf16_cells == expected_bf16


def test_fp16_spot_is_single_cell():
    fp16_cells = [
        cell for cell in _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10
        if cell[3] == "torch.float16"
    ]
    assert len(fp16_cells) == 1
    assert fp16_cells[0] == EXPECTED_FP16_SPOT


def test_disjoint_with_prior_off_by_48_rungs():
    # K-2214 is a single-N slice at N=1776; every prior off-by-48 rung is
    # at a strictly different N (544..1712) so disjointness must hold
    # trivially.
    from tritonblas._route_predicate import (
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18,
        _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18,
        _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8,
        _K2183_P57_SKINNY_N1712_KCOMPL_ALIASSTACK_8,
    )
    assert _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10.isdisjoint(
        _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18
    )
    assert _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10.isdisjoint(
        _K2127_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18
    )
    assert _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10.isdisjoint(
        _K2158_P55_SKINNY_N1584_KCOMPL_ALIASSTACK_18
    )
    assert _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10.isdisjoint(
        _K2169_P56_SKINNY_N1648_KCOMPL_ALIASSTACK_8
    )
    assert _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10.isdisjoint(
        _K2183_P57_SKINNY_N1712_KCOMPL_ALIASSTACK_8
    )


def test_n1776_extends_off_by_48_ladder_above_n1712():
    # Sanity: N=1776 is the next admitted residue-48 step above K-2183 P57
    # N=1712 in this rung sequence (+64 step on the residue-48 lattice).
    assert (EXPECTED_N - 1712) % 64 == 0
    assert (EXPECTED_N - 1712) == 64
    assert EXPECTED_N > 1712


def test_adjacency_firewalls_n1776_neighbors():
    # The cells immediately neighbouring N=1776 on the residue-48 lattice
    # (N=1712 is the K-2183 admit immediately below; N=1840 is the next
    # un-audited rung above) and the off-cohort fp16 cells at N=1776 must
    # remain OUTSIDE the K-2214 frozenset — preserve the per-N audit
    # handles per the K-1175 stacked-predicate convention.
    forbidden = [
        # fp16 at non-spot M/K cells (only fp16 admit is M=4096, K=8192)
        (2048, 1776, 4096, "torch.float16"),
        (2048, 1776, 8192, "torch.float16"),
        (2048, 1776, 16384, "torch.float16"),
        (4096, 1776, 4096, "torch.float16"),
        (4096, 1776, 16384, "torch.float16"),
        (8192, 1776, 4096, "torch.float16"),
        (8192, 1776, 8192, "torch.float16"),
        (8192, 1776, 16384, "torch.float16"),
        # N=1712 (K-2183 admit, lives in its own frozenset, not this one)
        (4096, 1712, 8192, "torch.bfloat16"),
        # N=1840 (next un-audited residue-48 rung above 1776)
        (4096, 1840, 8192, "torch.bfloat16"),
        # Off-residue N (1792 is wave-aligned, not residue-48)
        (4096, 1792, 8192, "torch.bfloat16"),
    ]
    for cell in forbidden:
        assert cell not in _K2214_P58_SKINNY_N1776_KCOMPL_ALIASSTACK_10, (
            f"K-2214 frozenset must NOT contain firewall cell {cell}"
        )


# ---------------------------------------------------------------------------
# Behavioral routing tests — exercise the dispatcher contract
# ---------------------------------------------------------------------------
#
# Per K-2214 retry feedback (Testing Zealot REVISE): the structural-only
# tests above lock down the *data* (frozenset literal cardinality / residue
# / disjointness) but do not exercise `_k971_route_to_hbl(...)` itself —
# the actual function consumed by `tritonblas.matmul`.  The tests below
# commit the routing contract: N=1776 cohort cells return True, the
# immediately-prior K-2183 N=1712 admit still returns True (no
# regression on prior rungs), and the residue-48 adjacency firewall
# (N=1840 next un-audited rung above; N=1712 ± 64 boundaries; the
# wave-aligned N=1792 alias) returns False.  These convert the previous
# attempt's ad-hoc smoke check into committed regression-protected
# assertions.

# Importing matmul triggers a torch.cuda call at import time; tests that
# only need the data layer skip the import to stay CPU-portable.  The
# routing tests gate on torch+matmul availability so they no-op cleanly
# on a CPU-only invariant runner.
try:
    import torch  # noqa: F401
    from tritonblas.matmul import _k971_route_to_hbl
    _ROUTING_AVAILABLE = True
    _IMPORT_ERR: Exception | None = None
except Exception as _e:  # pragma: no cover — environment-conditional
    _ROUTING_AVAILABLE = False
    _IMPORT_ERR = _e


pytestmark_routing = pytest.mark.skipif(
    not _ROUTING_AVAILABLE,
    reason=f"torch/tritonblas.matmul not importable: {_IMPORT_ERR!r}",
)


@pytestmark_routing
def test_routing_n1776_cohort_cells_route_to_hbl():
    """K-2214 P58 admit: every cell of the 10-cell cohort must route True."""
    import torch
    bf16 = torch.bfloat16
    fp16 = torch.float16
    cohort = [
        # 9 bf16 cells
        (2048, 1776, 4096,  bf16),
        (2048, 1776, 8192,  bf16),
        (2048, 1776, 16384, bf16),
        (4096, 1776, 4096,  bf16),
        (4096, 1776, 8192,  bf16),
        (4096, 1776, 16384, bf16),
        (8192, 1776, 4096,  bf16),
        (8192, 1776, 8192,  bf16),
        (8192, 1776, 16384, bf16),
        # 1 fp16 spot
        (4096, 1776, 8192,  fp16),
    ]
    for (M, N, K, dt) in cohort:
        assert _k971_route_to_hbl(M, N, K, dt, dt, False, False), (
            f"K-2214 cohort cell (M={M}, N={N}, K={K}, dt={dt}) "
            "MUST route to hipBLASLt"
        )


@pytestmark_routing
def test_routing_n1712_prior_rung_still_routed():
    """K-2183 P57 admit (N=1712) must still route True post-K-2214 patch.

    Guards against accidentally clobbering the prior rung when extending
    the alias-stack by one slot.  Anchors at the K-2169 peak-amplification
    cell (M=4096, K=16384, fp16) plus a bf16 sample.
    """
    import torch
    assert _k971_route_to_hbl(
        4096, 1712, 16384, torch.float16, torch.float16, False, False
    ), "K-2183 P57 N=1712 fp16 anchor MUST remain routed (no prior-rung regression)"
    assert _k971_route_to_hbl(
        4096, 1712, 8192, torch.bfloat16, torch.bfloat16, False, False
    ), "K-2183 P57 N=1712 bf16 anchor MUST remain routed (no prior-rung regression)"


@pytestmark_routing
def test_routing_adjacency_firewall_n1840_not_routed():
    """N=1840 = next un-audited residue-48 rung above N=1776 — MUST NOT route.

    The K-2214 PR explicitly defers any rung above N=1776; a False here is
    the firewall protecting future rungs from being silently absorbed.
    """
    import torch
    assert not _k971_route_to_hbl(
        4096, 1840, 8192, torch.bfloat16, torch.bfloat16, False, False
    ), "N=1840 (next residue-48 rung) MUST remain un-routed — out of K-2214 scope"


@pytestmark_routing
def test_routing_off_residue_n1792_not_routed():
    """N=1792 is wave-aligned (mod 64 == 0), NOT residue-48 — MUST NOT route.

    Prevents a regression where a sloppy refactor (e.g. dropping the
    explicit-membership check for a coarse residue gate) would over-admit
    wave-aligned N values that already perform at HBL parity natively.
    """
    import torch
    assert not _k971_route_to_hbl(
        4096, 1792, 8192, torch.bfloat16, torch.bfloat16, False, False
    ), "N=1792 (wave-aligned, off-residue) MUST remain un-routed"


@pytestmark_routing
def test_routing_off_cohort_n1776_fp16_cells_not_routed():
    """Only the (M=4096, K=8192, fp16) fp16 spot is admitted at N=1776.

    Other fp16 (M, K) combinations at N=1776 must remain un-routed —
    enforces the task-specified 10-cell cohort boundary at the dispatcher
    level (not just the data-literal level).
    """
    import torch
    fp16 = torch.float16
    off_cohort_fp16 = [
        (2048, 1776, 4096,  fp16),
        (2048, 1776, 8192,  fp16),
        (2048, 1776, 16384, fp16),
        (4096, 1776, 4096,  fp16),
        (4096, 1776, 16384, fp16),
        (8192, 1776, 4096,  fp16),
        (8192, 1776, 8192,  fp16),
        (8192, 1776, 16384, fp16),
    ]
    for (M, N, K, dt) in off_cohort_fp16:
        assert not _k971_route_to_hbl(M, N, K, dt, dt, False, False), (
            f"Off-cohort fp16 cell (M={M}, N={N}, K={K}) at N=1776 MUST NOT route "
            "— K-2214 fp16 admit is single-spot at (M=4096, K=8192) only"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
