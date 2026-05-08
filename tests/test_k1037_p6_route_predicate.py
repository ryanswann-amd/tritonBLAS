"""K-1037 P6 MFMA-issue-stall route-OUT predicate — adversarial held-out tests.

Mirrors the K-1031 / K-1043 / K-1055 protocol used to harden P5 Clause-4:

  * P6 MUST FIRE on the K-1017 MFMA-issue-stall cohort {S24, S29}
    (closed-form replacement for K-983 R1's lookup-style backstop).
  * P6 MUST NOT FIRE on the remaining 16 K-1017 cells (TN cohort) —
    the per-cell verdict has to match K-1037's ``p6_classifier_pmc.csv``.
  * P6 MUST NOT FIRE on any K-984+K-989 ship anchor (14 unique shapes) —
    structural orthogonality vs the existing P5 route lattice.
  * P6 MUST NOT FIRE on any K-950 9-cell LAND set — zero new LAND
    leakage on top of K-1062's K-floor narrowing.
  * P6 MUST NOT FIRE on the K-1028 fp16 cohort (bf16-only carve-out) —
    structural orthogonality vs the fp16 K-905/K-971 anchor table.
  * The full ``_k971_route_to_hbl`` dispatch helper MUST be no-op when the
    ``TRITONBLAS_ENABLE_K1037_P6`` env-var is unset (default-OFF feature
    flag) and MUST consult P6 only when the env-var is set to ``"1"``.
  * The ``TRITONBLAS_DISABLE_K971`` global kill-switch MUST still veto
    routing even when P6 would otherwise fire.

Importable on its own — no GPU required.
"""
from __future__ import annotations

import pytest
import torch

from tritonblas._route_predicate import (
    R_K1037_P6_mfma_issue_stall_route_to_hbl,
    R_K979_P5_route_to_hbl,
    K971_ROUTE_TABLE,
)
from tritonblas.matmul import _k971_route_to_hbl


# ---------------------------------------------------------------------------
# K-1017 18-cell calibration cohort  (source: K-1017 k1017_route_action_table.csv)
#   (cid, M, N, K, dtype, bottleneck_class, expected_p6_verdict)
# ---------------------------------------------------------------------------
K1017_18CELL_COHORT = [
    ("S03",   384, 409600,  384, torch.bfloat16, "Occupancy-bound",       False),
    ("S07",  2048,   1792,  256, torch.bfloat16, "Occupancy-bound",       False),
    ("S09",   384, 409600,  256, torch.bfloat16, "Occupancy-bound",       False),
    ("S14",   128, 409600,  384, torch.bfloat16, "Occupancy-bound",       False),
    ("S15",   384, 409600,  128, torch.bfloat16, "Occupancy-bound",       False),
    ("S16",   256,    256, 2048, torch.bfloat16, "Occupancy-bound",       False),
    ("S24",  4480,   3072,  768, torch.bfloat16, "MFMA-issue-stall",      True),   # TP
    ("S26",  8064,   2048, 1024, torch.bfloat16, "Occupancy-bound",       False),
    ("S29", 14208,   2048, 1024, torch.bfloat16, "MFMA-issue-stall",      True),   # TP
    ("S30", 16256,   2048, 1024, torch.bfloat16, "Occupancy-bound",       False),
    ("S31", 18304,   2048, 1024, torch.bfloat16, "Occupancy-bound",       False),
    ("S32", 20352,   2048, 1024, torch.bfloat16, "Occupancy-bound",       False),
    ("S33", 22400,   2048, 1024, torch.bfloat16, "Occupancy-bound",       False),
    ("S34", 24448,   2048, 1024, torch.bfloat16, "Occupancy-bound",       False),
    ("S35", 26496,   2048, 1024, torch.bfloat16, "Occupancy-bound",       False),
    ("S37", 25600,   2048,  256, torch.bfloat16, "Occupancy-bound",       False),
    ("S38",   384,    128,  200, torch.bfloat16, "HBM/L2-stall-bound",    False),
    ("S39", 49152,   2048,  256, torch.bfloat16, "Occupancy-bound",       False),
]

# K-984+K-989 union — every shape MUST stay outside P6's firing set so the
# ship lattice is uncrowded (P5 owns these via Clauses 1/2/3/4; P6 must not
# duplicate or shadow).
K984_K989_UNION = [
    ("S04",  2304,   2048, 4800),
    ("S05",   256,   1792, 2048),
    ("S06",   736,   1792,  736),
    ("S10",   512,    192, 2048),
    ("S17",   768,   1792, 5972),
    ("S18",  5972,   1792,  768),
    ("S21",   768,   3072, 4480),  # N=3072 neighbour — distinguished by K
    ("S22",  1024,   2048, 6016),
    ("S23",  1024,   2048, 8064),
    ("S25",  6016,   2048, 1024),  # N=2048 K=1024 neighbour — distinguished by M
    ("S27", 10112,   2048, 1024),  # idem
    ("S28", 12160,   2048, 1024),  # idem (just outside lower M-bound 12288)
    ("S36",    30, 786432,  200),
    ("S40",  1024,   2048, 1240),
]

# K-950 9-cell LAND harness.
K950_LAND_CELLS = [
    ("L01", 1024, 1024,   512, torch.bfloat16),
    ("L02",  512,  512,  1024, torch.bfloat16),
    ("L03", 2048,  768,  1024, torch.bfloat16),
    ("L04",  768, 2048,  1024, torch.bfloat16),  # N=2048 K=1024 neighbour
    ("L05", 4096, 4096,  4096, torch.bfloat16),
    ("L06", 2048, 2048, 16384, torch.float16),
    ("L07", 1024, 1024, 16384, torch.bfloat16),
    ("L08", 4096, 4096,  8192, torch.bfloat16),
    ("L09", 1024, 1024, 16384, torch.float16),
]

# K-1028 fp16 cohort — bf16-only carve-out coverage.
K1028_FP16_NEGATIVES = [
    ("K1028-S04",  2304, 2048, 4800, torch.float16),
    ("K1028-S17",   768, 1792, 5972, torch.float16),
    ("K1028-S27", 10112, 2048, 1024, torch.float16),
]

# K-1050 negative-control sample (Occupancy bucket — same N=2048 neighbours).
K1050_NEGATIVE_CONTROLS = [
    ("S03_K1050",   384, 409600, 384, torch.bfloat16),
    ("S16_K1050",   256,    256,2048, torch.bfloat16),
]

# K-1055 paired-PMC neighbours (held-out sanity).
K1055_NEGATIVE_CONTROLS = [
    ("K984-S04",  2304, 2048, 4800, torch.bfloat16),
    ("K984-S10",   512,  192, 2048, torch.bfloat16),
    ("K984-S27", 10112, 2048, 1024, torch.bfloat16),
    ("K984-S28", 12160, 2048, 1024, torch.bfloat16),
]


# ---------------------------------------------------------------------------
# Predicate-only checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cid,M,N,K,dtype,kclass,expected", K1017_18CELL_COHORT,
                         ids=[c[0] for c in K1017_18CELL_COHORT])
def test_p6_matches_k1017_18cell_truth(cid, M, N, K, dtype, kclass, expected):
    """Per-cell P6 verdict must match K-1037's calibration verdict (TP=2/2,
    TN=16/16, perfect agreement on the K-1017 18-cell cohort)."""
    got = R_K1037_P6_mfma_issue_stall_route_to_hbl(M, N, K, dtype)
    assert got is expected, (
        f"K-1017 {cid} ({M},{N},{K}) {dtype} class={kclass} — "
        f"P6 returned {got}, expected {expected}")


def test_p6_fires_on_exactly_two_cells():
    """The K-1017 MFMA-issue-stall cohort is exactly {S24, S29} — P6 must
    fire on exactly two of the 18 cells, no more, no less."""
    fires = [cid for cid, M, N, K, dt, _, _ in K1017_18CELL_COHORT
             if R_K1037_P6_mfma_issue_stall_route_to_hbl(M, N, K, dt)]
    assert set(fires) == {"S24", "S29"}, (
        f"unexpected P6 firing set: {fires} (expected {{S24, S29}})")


@pytest.mark.parametrize("sid,M,N,K", K984_K989_UNION,
                         ids=[c[0] for c in K984_K989_UNION])
def test_p6_no_leak_on_k984_k989_union(sid, M, N, K):
    """P6 must not shadow K-984+K-989 anchors (P5 owns them via Clauses
    1/2/3/4 — P6 is strictly additive)."""
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(
        M, N, K, torch.bfloat16) is False, (
        f"P6 leaked onto K-984+K-989 anchor {sid} ({M},{N},{K})")


@pytest.mark.parametrize("cid,M,N,K,dtype", K950_LAND_CELLS,
                         ids=[c[0] for c in K950_LAND_CELLS])
def test_p6_no_leak_on_k950_land(cid, M, N, K, dtype):
    """Zero LAND-leakage requirement: P6 must produce False on every cell in
    the K-950 9-cell harness (the K-1031/K-1043 admission gate)."""
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(M, N, K, dtype) is False, (
        f"P6 leaked into K-950 LAND cell {cid} ({M},{N},{K}) {dtype}")


@pytest.mark.parametrize("cid,M,N,K,dtype", K1028_FP16_NEGATIVES,
                         ids=[c[0] for c in K1028_FP16_NEGATIVES])
def test_p6_bf16_only_carveout(cid, M, N, K, dtype):
    """Cohort scope is bf16 only — the fp16 carve-out short-circuits to
    False just like P5's bf16-only gating."""
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(M, N, K, dtype) is False, (
        f"P6 fired on fp16 cell {cid} ({M},{N},{K}) — bf16-only carve-out broken")


@pytest.mark.parametrize("cid,M,N,K,dtype", K1050_NEGATIVE_CONTROLS,
                         ids=[c[0] for c in K1050_NEGATIVE_CONTROLS])
def test_p6_no_leak_on_k1050_negatives(cid, M, N, K, dtype):
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(M, N, K, dtype) is False


@pytest.mark.parametrize("cid,M,N,K,dtype", K1055_NEGATIVE_CONTROLS,
                         ids=[c[0] for c in K1055_NEGATIVE_CONTROLS])
def test_p6_no_leak_on_k1055_negatives(cid, M, N, K, dtype):
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(M, N, K, dtype) is False


def test_p6_fp32_int8_short_circuit():
    """Non-bf16 dtypes short-circuit even on the S24 / S29 anchors."""
    for M, N, K in [(4480, 3072, 768), (14208, 2048, 1024)]:
        assert R_K1037_P6_mfma_issue_stall_route_to_hbl(M, N, K, torch.bfloat16) is True
        assert R_K1037_P6_mfma_issue_stall_route_to_hbl(M, N, K, torch.float16) is False
        assert R_K1037_P6_mfma_issue_stall_route_to_hbl(M, N, K, torch.float32) is False
        assert R_K1037_P6_mfma_issue_stall_route_to_hbl(M, N, K, torch.int8) is False


# ---------------------------------------------------------------------------
# Bound-tightness checks — adjacent shapes must straddle the firing set.
# ---------------------------------------------------------------------------
def test_p6_clause_a_M_lower_bound_excludes_3072_strict():
    """Clause-A lower bound is strict (M > 3072 not >=) — guards against
    accidental K-989 anchor S21 (768, 3072, 4480) wrap-around."""
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(3072, 3072, 768, torch.bfloat16) is False
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(3073, 3072, 768, torch.bfloat16) is True


def test_p6_clause_a_M_upper_bound_excludes_5000_strict():
    """Clause-A upper bound is strict (M < 5000) — anything at or above
    that escapes (no LAND evidence beyond S24)."""
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(4999, 3072, 768, torch.bfloat16) is True
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(5000, 3072, 768, torch.bfloat16) is False


def test_p6_clause_a_K_floor_silences_below_BK64x8():
    """Clause-A K-floor is 512 (8·BK64) per K-1062 convention."""
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(4480, 3072, 511, torch.bfloat16) is False
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(4480, 3072, 512, torch.bfloat16) is True
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(4480, 3072, 1024, torch.bfloat16) is True
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(4480, 3072, 1025, torch.bfloat16) is False


def test_p6_clause_b_M_window_excludes_K984_S28_neighbour():
    """Clause-B lower bound 12288 keeps K-984+K-989 anchor S28 (12160,
    2048, 1024) outside the firing set — checked at the 12160/12288 boundary."""
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(12160, 2048, 1024, torch.bfloat16) is False
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(12288, 2048, 1024, torch.bfloat16) is True


def test_p6_clause_b_M_window_excludes_S30_neighbour():
    """Clause-B upper bound 16128 keeps K-1017 cell S30 (16256, 2048,
    1024) outside (S30 is Occupancy-bound, P5 Clause-2 already covers it)."""
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(16127, 2048, 1024, torch.bfloat16) is True
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(16128, 2048, 1024, torch.bfloat16) is False
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(16256, 2048, 1024, torch.bfloat16) is False


def test_p6_clause_b_K_must_equal_1024():
    """Clause-B K is pinned (== 1024) — the MFMA-stall fingerprint requires
    the K-tied tile-grid waves-per-CU ratio to sit close to HBL."""
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(14208, 2048, 1023, torch.bfloat16) is False
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(14208, 2048, 1024, torch.bfloat16) is True
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(14208, 2048, 1025, torch.bfloat16) is False


def test_p6_clause_orthogonality_to_p5():
    """Audit invariant: every cell P6 fires on must NOT already be caught by
    a P5 clause that owns it for a different reason — except the documented
    S29 overlap (P5 Clause-2 also fires).  S24 must be P6-only."""
    # S24 — P6 owns this cell (P5 misses it).
    assert R_K979_P5_route_to_hbl(4480, 3072, 768, torch.bfloat16) is False
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(4480, 3072, 768, torch.bfloat16) is True
    # S29 — P5 already catches via Clause-2; P6 is defence-in-depth.
    assert R_K979_P5_route_to_hbl(14208, 2048, 1024, torch.bfloat16) is True
    assert R_K1037_P6_mfma_issue_stall_route_to_hbl(14208, 2048, 1024, torch.bfloat16) is True


# ---------------------------------------------------------------------------
# Full-dispatch checks — feature-flag guard, kill-switch interaction.
# ---------------------------------------------------------------------------
S24 = (4480, 3072, 768)
S29 = (14208, 2048, 1024)


def test_full_dispatch_p6_default_off(monkeypatch):
    """With TRITONBLAS_ENABLE_K1037_P6 unset, dispatch ignores P6 — S24
    flows through as `False` (P5 misses it; strict-equality table absent).
    """
    monkeypatch.delenv("TRITONBLAS_ENABLE_K1037_P6", raising=False)
    monkeypatch.delenv("TRITONBLAS_DISABLE_K971",     raising=False)
    M, N, K = S24
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is False


def test_full_dispatch_p6_enabled_routes_S24(monkeypatch):
    """With TRITONBLAS_ENABLE_K1037_P6=1, S24 routes to hipBLASLt."""
    monkeypatch.setenv("TRITONBLAS_ENABLE_K1037_P6", "1")
    monkeypatch.delenv("TRITONBLAS_DISABLE_K971", raising=False)
    M, N, K = S24
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is True


def test_full_dispatch_p6_enabled_routes_S29(monkeypatch):
    """S29 already routes via P5 Clause-2 — confirms P6-enabled flag does
    not break the existing dispatch."""
    monkeypatch.setenv("TRITONBLAS_ENABLE_K1037_P6", "1")
    monkeypatch.delenv("TRITONBLAS_DISABLE_K971", raising=False)
    M, N, K = S29
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is True


def test_full_dispatch_p6_enabled_obeys_kill_env(monkeypatch):
    """Global K-971 kill-switch must veto P6 routing too — TRITONBLAS_DISABLE_K971
    is the single L3 disable lever (per K-883 R1)."""
    monkeypatch.setenv("TRITONBLAS_ENABLE_K1037_P6", "1")
    monkeypatch.setenv("TRITONBLAS_DISABLE_K971",    "1")
    M, N, K = S24
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=False) is False


def test_full_dispatch_p6_enabled_obeys_streamk_carveout(monkeypatch):
    monkeypatch.setenv("TRITONBLAS_ENABLE_K1037_P6", "1")
    monkeypatch.delenv("TRITONBLAS_DISABLE_K971", raising=False)
    M, N, K = S24
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=True, work_stealing=False) is False


def test_full_dispatch_p6_enabled_obeys_workstealing_carveout(monkeypatch):
    monkeypatch.setenv("TRITONBLAS_ENABLE_K1037_P6", "1")
    monkeypatch.delenv("TRITONBLAS_DISABLE_K971", raising=False)
    M, N, K = S24
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.bfloat16,
        enable_streamk=False, work_stealing=True) is False


def test_full_dispatch_p6_enabled_obeys_dtype_mismatch_carveout(monkeypatch):
    monkeypatch.setenv("TRITONBLAS_ENABLE_K1037_P6", "1")
    monkeypatch.delenv("TRITONBLAS_DISABLE_K971", raising=False)
    M, N, K = S24
    assert _k971_route_to_hbl(
        M, N, K, torch.bfloat16, torch.float16,
        enable_streamk=False, work_stealing=False) is False


def test_full_dispatch_p6_disabled_does_not_affect_p5_cells(monkeypatch):
    """K-984+K-989 anchors must keep routing whether or not P6 is enabled."""
    for sid, M, N, K in K984_K989_UNION:
        monkeypatch.delenv("TRITONBLAS_ENABLE_K1037_P6", raising=False)
        monkeypatch.delenv("TRITONBLAS_DISABLE_K971", raising=False)
        off = _k971_route_to_hbl(
            M, N, K, torch.bfloat16, torch.bfloat16,
            enable_streamk=False, work_stealing=False)
        monkeypatch.setenv("TRITONBLAS_ENABLE_K1037_P6", "1")
        on = _k971_route_to_hbl(
            M, N, K, torch.bfloat16, torch.bfloat16,
            enable_streamk=False, work_stealing=False)
        assert off is True and on is True, (
            f"K-984+K-989 anchor {sid} ({M},{N},{K}) routing flipped — "
            f"off={off} on={on}")


def test_p6_zero_K950_leak_full_dispatch(monkeypatch):
    """End-to-end sanity: enabling P6 must NOT route any K-950 LAND cell that
    isn't already in the strict-equality K-905 anchor allowlist."""
    K950_ANCHOR_COLLISIONS = frozenset({
        (1024, 1024, 16384, "torch.bfloat16"),
        (1024, 1024, 16384, "torch.float16"),
        (2048, 2048, 16384, "torch.float16"),
    })
    monkeypatch.setenv("TRITONBLAS_ENABLE_K1037_P6", "1")
    monkeypatch.delenv("TRITONBLAS_DISABLE_K971", raising=False)
    for cid, M, N, K, dtype in K950_LAND_CELLS:
        routed = _k971_route_to_hbl(
            M, N, K, dtype, dtype,
            enable_streamk=False, work_stealing=False)
        expected = (M, N, K, str(dtype)) in K950_ANCHOR_COLLISIONS
        assert routed is expected, (
            f"K-950 LAND cell {cid} routed={routed} expected={expected} "
            f"under TRITONBLAS_ENABLE_K1037_P6=1 — P6 leaked outside its cohort")


# ---------------------------------------------------------------------------
# Coupling check — matmul.py imports the helper module's P6 binding.
# ---------------------------------------------------------------------------
def test_matmul_module_uses_helper_module_p6():
    import sys
    _m = sys.modules["tritonblas.matmul"]
    assert _m._R_K1037_P6_mfma_issue_stall_route_to_hbl is \
        R_K1037_P6_mfma_issue_stall_route_to_hbl
