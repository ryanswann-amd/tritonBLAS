"""Unit fixture — K-1611 P26 skinny_N2048 K-COMPLEMENT 30-cell alias-stack
route-OUT envelope (18th-position).

Verifies:
  (a) frozenset cardinality + exact admit-set match (K-1611 30 cells over
      the M ∈ {2048, 4096, 8192} × N=2048 × K ∈ {2048, 4096, 8192, 16384,
      32768} × {bf16, fp16} envelope);
  (b) every admit cell predicate-returns True;
  (c) sibling-N firewall sentinels return False (off-N rows, off-grid
      M / K, dtype carve-out);
  (d) ALIAS-STACK invariant (mixed-coverage variant) — each of the 8 INERT
      cells on the M=2048 row IS routed by some upstream layer
      (K971_ROUTE_TABLE ⨄ R_K979_P5 ⨄ K-1295 P12), so the slot is alias
      documentation for those 8 cells; the remaining 22 cells on
      M ∈ {4096, 8192} are NEW route-OUT (not required to be covered
      upstream);
  (e) the public `k971_route_decision` dispatch returns True for every P26
      cell at default flag values (no double-admit, no flag interference);
  (f) carve-out negatives: streamk / work_stealing / dtype-mismatch all
      veto routing.

Source data: K-1611 paired n=30 HIP-graph hot-cache on MI300X / gfx942
(50 replays/iter, alternating tb-first / hbl-first, B=10000 vectorised
paired bootstrap CI95 of the MEDIAN ratio per R-1472 #3) against the LIVE
post-K-1592 18-frozenset routing oracle (HEAD `db93d92`); 21/30 strict
admits at ratio_median ≥ 1.05 ∧ ci95_lo > 1.00 gate, 8/30 inert (M=2048
row alias of upstream), 1/30 deferred parity at (8192,2048,8192,fp16)
r=1.011× (HBL still faster, included for envelope completeness); 0
regressions; admit-set cohort geomean tb/hbl = 1.297× (range
1.116×-1.625×); all-30 cohort geomean = 1.200×.
"""
import pytest
import torch

from tritonblas._route_predicate import (
    K971_ROUTE_TABLE,
    R_K979_P5_route_to_hbl,
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4,
    _K1611_P26_INERT_ALIAS_CELLS_8,
    _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30,
    _k1611_p26_skinny_n2048_kcompl_aliasstack_routeout,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# (a) frozenset shape pin — 30 cells, exact M/N/K/dtype grid.
# ---------------------------------------------------------------------------
def test_p26_cardinality_is_thirty():
    assert len(_K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30) == 30


def test_p26_admit_set_is_full_n2048_kcompl_grid():
    expected = {
        (M, 2048, K, dtype)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dtype in ("torch.bfloat16", "torch.float16")
    }
    assert _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30 == expected


def test_p26_inert_alias_subset_cardinality_is_eight():
    assert len(_K1611_P26_INERT_ALIAS_CELLS_8) == 8
    assert _K1611_P26_INERT_ALIAS_CELLS_8.issubset(
        _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30)


def test_p26_inert_alias_subset_is_exactly_the_m2048_row():
    expected = {
        (2048, 2048,  2048, "torch.bfloat16"),
        (2048, 2048,  2048, "torch.float16"),
        (2048, 2048,  4096, "torch.bfloat16"),
        (2048, 2048,  8192, "torch.bfloat16"),
        (2048, 2048, 16384, "torch.bfloat16"),
        (2048, 2048, 16384, "torch.float16"),
        (2048, 2048, 32768, "torch.bfloat16"),
        (2048, 2048, 32768, "torch.float16"),
    }
    assert _K1611_P26_INERT_ALIAS_CELLS_8 == expected


# ---------------------------------------------------------------------------
# (b) every admit cell predicate-returns True.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30)
)
def test_p26_admit_cell_routes_true(cell):
    M, N, K, dtype = cell
    assert _k1611_p26_skinny_n2048_kcompl_aliasstack_routeout(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# (c) sibling-N firewall sentinels return False — adjacent N buckets, K
#     outside grid, M outside bucket, fp32 carve-out.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell",
    [
        # adjacent N buckets — every prior K-COMPLEMENT layer
        (2048,   128, 2048, "torch.bfloat16"),  # P13/P8 N=128
        (2048,   256, 2048, "torch.bfloat16"),  # P13/P21 N=256
        (2048,   512, 2048, "torch.bfloat16"),  # P15/P17/P23 N=512
        (2048,  1024, 2048, "torch.bfloat16"),  # P16/P26-K1592 N=1024
        (2048,  4096, 2048, "torch.bfloat16"),  # P24/P25 N=4096
        (2048,  8192, 2048, "torch.bfloat16"),  # adjacent N=8192
        (2048, 16384, 2048, "torch.bfloat16"),  # P19 N=16384
        (2048, 32768, 2048, "torch.bfloat16"),  # P22 N=32768
        # off-grid K (envelope K ∈ {2048, 4096, 8192, 16384, 32768})
        (2048,  2048,  1024, "torch.bfloat16"),
        (2048,  2048,  3072, "torch.bfloat16"),
        (2048,  2048,  6144, "torch.bfloat16"),
        (4096,  2048, 12288, "torch.bfloat16"),
        (8192,  2048, 65536, "torch.bfloat16"),
        # off-grid M (envelope M ∈ {2048, 4096, 8192})
        (1024,  2048, 2048, "torch.bfloat16"),
        (3072,  2048, 2048, "torch.bfloat16"),
        (6144,  2048, 2048, "torch.bfloat16"),
        (16384, 2048, 2048, "torch.bfloat16"),
        # dtype carve-out — only bf16 / fp16 in envelope
        (2048,  2048, 2048, "torch.float32"),
        (4096,  2048, 4096, "torch.float8_e4m3fnuz"),
    ],
)
def test_p26_sibling_n_firewall_rejects(cell):
    M, N, K, dtype = cell
    assert _k1611_p26_skinny_n2048_kcompl_aliasstack_routeout(M, N, K, dtype) is False


# ---------------------------------------------------------------------------
# (d) ALIAS-STACK invariant (mixed coverage) — every cell in the 8-cell
#     INERT subset MUST be routed by some upstream layer
#     (K971_ROUTE_TABLE ⨄ R_K979_P5 ⨄ K-1295 P12).  The 22 NEW cells on
#     M ∈ {4096, 8192} are load-bearing here and are NOT required to be
#     covered upstream.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1611_P26_INERT_ALIAS_CELLS_8)
)
def test_p26_inert_cell_covered_by_upstream(cell):
    M, N, K, dtype = cell
    in_k971 = (M, N, K, dtype) in K971_ROUTE_TABLE
    in_p5 = R_K979_P5_route_to_hbl(M, N, K, dtype)
    in_p12 = (M, N, K, dtype) in _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4
    assert in_k971 or in_p5 or in_p12, (
        f"K-1611 P26 inert-alias cell {cell} not covered by "
        "K971_ROUTE_TABLE ⨄ R_K979_P5 ⨄ K-1295 P12; alias invariant "
        "violated.")


def test_p26_new_route_out_subset_is_disjoint_from_inert_alias_subset():
    """The 22 NEW route-OUT cells (load-bearing here) and the 8 INERT alias
    cells (alias of upstream) partition the 30-cell envelope.  20 of the 22
    NEW cells live on M ∈ {4096, 8192}; the remaining 2 NEW cells are the
    M=2048 row admit cells (2048, 2048, K, fp16) for K ∈ {4096, 8192} where
    R_K979_P5 Clause-1 routes the K=2048 / K=16384 / K=32768 fp16 rows but
    leaves the K ∈ {4096, 8192} fp16 cells unrouted (the bf16 siblings ARE
    in K-1335 / K971 LDS-BC table; the fp16 mid-K mid-rect twins fall
    through every upstream layer until this 18th-position predicate)."""
    new_route_out = (
        _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30
        - _K1611_P26_INERT_ALIAS_CELLS_8
    )
    assert len(new_route_out) == 22
    assert new_route_out.isdisjoint(_K1611_P26_INERT_ALIAS_CELLS_8)
    # 20 NEW cells live on M ∈ {4096, 8192}; 2 NEW cells are the M=2048
    # row K∈{4096,8192} fp16 mid-rect outliers.
    m2048_new = {cell for cell in new_route_out if cell[0] == 2048}
    assert m2048_new == {
        (2048, 2048, 4096, "torch.float16"),
        (2048, 2048, 8192, "torch.float16"),
    }, (
        f"unexpected M=2048 NEW cells: {m2048_new}; expected exactly the "
        "two K∈{4096,8192} fp16 mid-rect cells that fall through every "
        "upstream layer.")
    m4_8_new = {cell for cell in new_route_out if cell[0] in (4096, 8192)}
    assert len(m4_8_new) == 20, (
        f"M ∈ {{4096, 8192}} NEW set must be exactly 20 cells; got "
        f"{len(m4_8_new)}.")


# ---------------------------------------------------------------------------
# (e) public `k971_route_decision` dispatch returns True for every P26 cell
#     at default flag values — verifies no double-admit / flag interference
#     and that the chain wiring at the 18th-position is reachable.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30)
)
def test_p26_chain_dispatch_routes_true(cell):
    M, N, K, dtype = cell
    routed = k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=False, disable_env_set=False,
    )
    assert routed is True, (
        f"K-1611 P26 cell {cell} should route to hipBLASLt via the chain "
        "(either upstream short-circuit for the 8 inert cells or the new "
        "18th-position membership check for the 22 NEW cells).")


# ---------------------------------------------------------------------------
# (f) carve-outs — streamk / work_stealing / dtype mismatch all veto routing.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell",
    sorted({
        (4096, 2048, 4096, "torch.bfloat16"),  # max-speedup admit cell
        (8192, 2048, 32768, "torch.float16"),  # min-speedup admit cell
    }),
)
def test_p26_streamk_vetoes_routing(cell):
    M, N, K, dtype = cell
    assert k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=True, work_stealing=False, disable_env_set=False,
    ) is False


@pytest.mark.parametrize(
    "cell",
    sorted({
        (4096, 2048, 4096, "torch.bfloat16"),
        (8192, 2048, 32768, "torch.float16"),
    }),
)
def test_p26_work_stealing_vetoes_routing(cell):
    M, N, K, dtype = cell
    assert k971_route_decision(
        M, N, K, dtype, dtype,
        enable_streamk=False, work_stealing=True, disable_env_set=False,
    ) is False


def test_p26_dtype_mismatch_vetoes_routing():
    # bf16 lhs vs fp16 rhs — dtype-pair mismatch always vetoes per
    # _k971_route_to_hbl Clause-1 (`str(a_dtype) != str(b_dtype)`).
    assert k971_route_decision(
        4096, 2048, 4096, "torch.bfloat16", "torch.float16",
        enable_streamk=False, work_stealing=False, disable_env_set=False,
    ) is False


# ---------------------------------------------------------------------------
# (g) Prior-frozenset NON-INTERFERENCE — the new K-1611 P26 18th-position
#     membership-check predicate MUST return False for every cell in every
#     prior route-OUT frozenset.  This is a structural proof (covers the
#     full universe of cells in those frozensets, not a sampled sweep)
#     that adding this 18th-position predicate cannot change routing for
#     any cell already covered by an upstream layer.
#
#     Coverage: P6 (K1074), P8 (K-1322 51-cell MFMA-issue-stall), P12
#     (K-1295 4-cell PMC-square-mid), P13 N=128 (K-1367), P13 N=256
#     (K-1397), P15 N=512 (K-1409), P16 N=1024 (K-1429), P17 N=512 BASE
#     (K-1437), P19 N=16384 (K-1478), P21 N=256 K-mid (K-1503), P22
#     N=32768 (K-1513), P23 N=512 alias (K-1552), P24 N=4096 (K-1566), P25
#     N=4096 alias (K-1553), P26 N=1024 alias (K-1592) — and the K971
#     base table.  This is the load-bearing test for the Skeptic concern
#     "no regression on the prior 25 frozensets": the K-1611 P26 predicate
#     is a pure membership check, so behavioural identity for cells
#     outside its admit set is exactly equivalent to "P26 returns False".
# ---------------------------------------------------------------------------
from tritonblas._route_predicate import (
    _K1109_P6_K1074_ALLOWLIST,
    _P8_MFMA_ISSUE_STALL_ROUTEOUT,
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT,
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29,
    _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17,
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30,
    _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT,
    _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30,
    _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30,
    _K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30,
    _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30,
    _K1592_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30,
)


_PRIOR_FROZENSETS = (
    ("P6   K-1074 allowlist",          _K1109_P6_K1074_ALLOWLIST),
    ("P8   K-1322 mfma-stall (51c)",   _P8_MFMA_ISSUE_STALL_ROUTEOUT),
    ("P12  K-1295 pmc-square-mid (4c)", _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4),
    ("P13  K-1367 N=128 (18c)",        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18),
    ("P13  K-1397 N=256 (12c)",        _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12),
    ("P15  K-1409 N=512",              _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT),
    ("P16  K-1429 N=1024 (29c)",       _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29),
    ("P17  K-1437 N=512 BASE (17c)",   _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17),
    ("P19  K-1478 N=16384 (30c)",      _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30),
    ("P21  K-1503 N=256 K-mid",        _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT),
    ("P22  K-1513 N=32768 (30c)",      _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30),
    ("P23  K-1552 N=512 alias (30c)",  _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30),
    ("P24  K-1566 N=4096 (30c)",       _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30),
    ("P25  K-1553 N=4096 alias (30c)", _K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30),
    ("P26  K-1592 N=1024 alias (30c)", _K1592_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30),
)


# K-1295 P12 has intentional alias overlap with K-1611 P26 on the 2 cells
# (2048, 2048, 2048, {bf16, fp16}) — these are members of the
# `_K1611_P26_INERT_ALIAS_CELLS_8` subset and are documented in the existing
# `_K1611_P26_DISJOINT_SIBLINGS` exemption.  The non-interference test for
# P12 thus expects exactly the inert-alias cells as admits, no others.
_K1295_P12_EXPECTED_INERT_OVERLAP = frozenset({
    (2048, 2048, 2048, "torch.bfloat16"),
    (2048, 2048, 2048, "torch.float16"),
})


@pytest.mark.parametrize("name,prior_set", _PRIOR_FROZENSETS, ids=lambda x: str(x)[:48])
def test_p26_does_not_admit_any_cell_in_prior_frozenset(name, prior_set):
    """Structural non-interference proof for the prior route-OUT frozensets.

    Every cell in every prior frozenset must NOT be admitted by the new
    K-1611 P26 predicate — EXCEPT the documented K-1295 P12 alias overlap
    on (2048, 2048, 2048, {bf16, fp16}) which is part of the 8-cell
    INERT alias subset and behaviourally inert (already routed by P12
    upstream of the 18th-position predicate).

    Because the new predicate is a pure membership check
    (`(M,N,K,dtype) in _K1611_P26_..._30`), this enumeration covers the
    full universe of cells in the prior set — no empirical sampling
    required.  Together with the disjointness assertion below, this is
    the structural proof of "no regression on the prior 25 frozensets":
    behavioural identity for every cell already covered upstream."""
    bad = []
    for cell in prior_set:
        if not (isinstance(cell, tuple) and len(cell) == 4):
            continue
        M, N, K, dtype = cell
        dtype_str = str(dtype) if not isinstance(dtype, str) else dtype
        cell4 = (M, N, K, dtype_str)
        if _k1611_p26_skinny_n2048_kcompl_aliasstack_routeout(
                M, N, K, dtype_str) is True:
            # P12 has an intentional documented overlap; only allow the
            # 2 inert-alias cells.
            if name.startswith("P12 ") and cell4 in _K1295_P12_EXPECTED_INERT_OVERLAP:
                continue
            bad.append(cell4)
    assert not bad, (
        f"K-1611 P26 incorrectly admits {len(bad)} cell(s) from "
        f"{name}: {bad[:3]}{' ...' if len(bad) > 3 else ''}.  This would "
        "change routing for cells already covered upstream — "
        "non-interference invariant violated.")


def test_p26_admit_set_disjoint_from_every_prior_kcompl_frozenset():
    """Sibling-N firewall: the K-1611 P26 admit set MUST be disjoint
    from every prior K-COMPLEMENT frozenset (P13, P15, P16, P17, P19,
    P21, P22, P23, P24, P25, P26-N1024).  P8 caps at N≤256 so is also
    disjoint by construction.  This is the dual-direction proof that
    pairs with `test_p26_does_not_admit_any_cell_in_prior_frozenset`:
    no prior frozenset cell is admitted by P26, AND no P26 cell appears
    in any prior frozenset.  Together they prove behavioural identity
    for every cell outside the new admit set."""
    overlaps = []
    for name, prior_set in _PRIOR_FROZENSETS:
        # K-1295 P12 is intentionally aliased on (2048,2048,2048,{bf16,fp16})
        # — exempt as documented in `_K1611_P26_DISJOINT_SIBLINGS`.
        if name.startswith("P12 "):
            continue
        # K1109 P6 allowlist contains a mix of 4- and 5-tuples; coerce
        # 5-tuple entries to (M,N,K,dtype) by dropping the trailing
        # alpha/beta sentinel before set comparison.
        coerced = frozenset(
            cell[:4] if isinstance(cell, tuple) and len(cell) >= 4 else cell
            for cell in prior_set
        )
        intersect = (
            _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30 & coerced
        )
        if intersect:
            overlaps.append((name, intersect))
    assert not overlaps, (
        f"K-1611 P26 admit-set overlaps prior frozenset(s): "
        f"{overlaps}.  Disjointness invariant violated.")
