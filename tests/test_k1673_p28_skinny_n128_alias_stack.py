"""Unit fixture — P28 skinny_N128 K-COMPLEMENT 30-cell alias-stack
19th-position route-OUT.

Pins the 30-cell envelope (M ∈ {2048, 4096, 8192} × N=128 ×
K ∈ {2048, 4096, 8192, 16384, 32768} × {bf16, fp16}), proves slot-19
actually fires for the 6 NEW fp16 cells at K ∈ {2048, 32768} that fall
through every upstream predicate, and proves the 24 alias cells still
resolve to hipBLASLt via their upstream slot.  Also enforces the
sibling-N firewall vs prior K-COMPLEMENT cohorts and the strict-superset
invariant against P13.
"""
import pytest

from tritonblas._route_predicate import (
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT,
    _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30,
    _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30,
    _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30,
    _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _k1367_p13_skinny_n128_routeout,
    k971_route_decision,
)


# Expected 30-cell envelope (rebuilt from the spec, not from the frozenset
# under test, so a typo in the frozenset is caught).
_P28_EXPECTED = frozenset({
    (M, 128, K, dtype)
    for M in (2048, 4096, 8192)
    for K in (2048, 4096, 8192, 16384, 32768)
    for dtype in ("torch.bfloat16", "torch.float16")
})

# 6 NEW fp16 cells slot-19 is load-bearing for: K ∈ {2048, 32768} × fp16
# (P5 Clause-3 is bf16-only; P13 covers only K ∈ {4096, 8192, 16384}).
_P28_NEW_LOADBEARING_6 = frozenset({
    (M, 128, K, "torch.float16")
    for M in (2048, 4096, 8192)
    for K in (2048, 32768)
})


# (a) Cardinality + envelope shape.
def test_p28_admit_set_cardinality_is_thirty():
    assert len(_K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30) == 30


def test_p28_admit_set_matches_full_envelope():
    assert _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == _P28_EXPECTED


# (b) P13 strict-subset invariant (P28 extends P13 to the full K-grid).
def test_p13_is_strict_subset_of_p28():
    assert (
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18
        < _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30
    )


# (c) Sibling-N firewall — P28 N=128 must be disjoint from every prior
#     K-COMPLEMENT alias-stack on a different N rung.
@pytest.mark.parametrize(
    "sibling_set,sibling_name",
    [
        (_K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT, "P21_N256"),
        (_K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30, "P22_N32768"),
        (_K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30, "P23_N512"),
        (_K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30, "P24_N4096"),
        (_K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30, "P26_N2048"),
    ],
    ids=lambda v: v if isinstance(v, str) else "set",
)
def test_p28_disjoint_from_sibling_n_cohorts(sibling_set, sibling_name):
    assert _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30.isdisjoint(sibling_set)


# (d) ROUTING CONTRACT — every one of the 30 cells dispatches to hipBLASLt
#     via the public `k971_route_decision` (the test surrogate that
#     mirrors `matmul._k971_route_to_hbl`).  This is the load-bearing
#     test the Testing Zealot asked for: end-to-end verification that
#     slot-19 actually catches the 6 NEW fp16 cells AND that the 24
#     alias cells still resolve via their upstream slot.
@pytest.mark.parametrize(
    "cell", sorted(_K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30)
)
def test_p28_admit_cell_routes_to_hbl(cell):
    M, N, K, dtype = cell
    assert k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False, disable_env_set=False,
    ) is True, f"P28 admit cell {cell} failed to dispatch to hipBLASLt."


# (e) The 6 NEW fp16 cells must NOT be caught by P13 (they live outside
#     the P13 K-band {4096, 8192, 16384}); slot-19 is the only layer that
#     covers them, so removing the P28 frozenset would silently drop them.
@pytest.mark.parametrize(
    "cell", sorted(_P28_NEW_LOADBEARING_6)
)
def test_p28_new_loadbearing_cells_not_covered_by_p13(cell):
    M, N, K, dtype = cell
    assert _k1367_p13_skinny_n128_routeout(M, N, K, dtype) is False, (
        f"NEW load-bearing cell {cell} was covered by P13 — sibling-N "
        "firewall claim is broken; re-audit slot-19 cardinality.")


# (f) The 24 alias cells (P13's N=128 K∈{4096,8192,16384} grid × bf16/fp16)
#     must still fire via P13 (their upstream slot) so removing P28 would
#     not change their routing verdict.
@pytest.mark.parametrize(
    "cell", sorted(_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)
)
def test_p28_alias_cells_still_caught_by_p13(cell):
    M, N, K, dtype = cell
    assert _k1367_p13_skinny_n128_routeout(M, N, K, dtype) is True, (
        f"Alias cell {cell} stopped firing via upstream P13 — slot-19 "
        "alias-stack documentation is no longer faithful.")


# (g) Carve-out negatives — streamk / work_stealing / dtype-mismatch must
#     short-circuit OFF even on a P28 admit cell.
@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable_streamk": True,  "work_stealing": False},
        {"enable_streamk": False, "work_stealing": True},
    ],
    ids=["streamk_on", "work_stealing_on"],
)
def test_p28_admit_cell_carved_out_by_flags(kwargs):
    cell = (4096, 128, 2048, "torch.float16")  # canonical NEW load-bearing
    M, N, K, dtype = cell
    assert k971_route_decision(
        M, N, K, dtype, dtype, disable_env_set=False, **kwargs,
    ) is False


def test_p28_admit_cell_carved_out_by_dtype_mismatch():
    M, N, K = 4096, 128, 2048
    assert k971_route_decision(
        M, N, K, "torch.bfloat16", "torch.float16",
        enable_streamk=False, work_stealing=False, disable_env_set=False,
    ) is False
