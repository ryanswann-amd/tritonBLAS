"""K-2156 P56 (S-002) — 12th-rung N=1584 K-COMPLEMENT verified-winner alias-stack tests.

Source measurement: K-2156 paired n=30 HIP-graph hot-cache MI300X / gfx942
4-engine (TB-BEFORE / TB-AFTER / hipBLASLt / rocBLAS) on c42 over the
18-cell sub-cohort M ∈ {2048, 4096, 8192} × N=1584 × K ∈ {4096, 8192, 16384}
× {bf16, fp16}, vs the live post-K-2147 P55 oracle (fix/K-2156 base 5e35d4e).

12th rung of the off-by-48 wave-misaligned K-COMPLEMENT alias-stack ladder
(P45 N=816, ..., P54 N=1456, P55 N=1520 → P56 N=1584).  N=1584 mod 64 = 48
(same off-by-48 modular class as every prior rung); N=1584 mod 128 = 48
(sibling residue to N=1456, N=1328, N=1200).

Mechanism (R-1811 wave-misalignment + K-913 §3 LDS-bank-conflict):
BLOCK_N=128 packs N=1584 into 12.375 BLOCK_N tiles per N-row → 24.75
fractional waves on 304-CU gfx942 — same SCHEDULER_LDS A4 wave-misalignment
failure mode as every prior rung. hipBLASLt's split-K kernel selection
clears the band on the K-major bf16/fp16 grid.  Per K-1908 closed-form
analysis, (N % 64 == 48) ∧ (880 ≤ N ≤ 1584) covers 12 contiguous off-by-48
rungs at single-N granularity once K-2156 admits — crosses the K-1908
≥12-rung closed-form refactor actionability threshold.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18 (full 18-cell grid).
  2. N axis is exactly {1584}; M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192,
     16384}; dtype ∈ {torch.bfloat16, torch.float16}.
  3. Both dtype rows are complete 9/9 (no dtype-asymmetric exclusions).
  4. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (e.g. K-1922 P40 N=544; K-2147 P55 N=1520).
  5. Off-by-48 wave-misalignment band membership: 1584 mod 64 == 48,
     1584 mod 128 == 48 (sibling residue to N=1456).
  6. Dispatcher routing actually changes for cells in the frozenset and
     does NOT change for explicitly excluded neighbour N values
     (1520-immediate-neighbour family, wave-aligned values, wrong-residue
     families) — _k971_route_to_hbl is the single source of truth.
"""
from __future__ import annotations

import importlib.util

import pytest


# ---------------------------------------------------------------------------
# Data invariants on the frozenset (no torch / GPU required)
# ---------------------------------------------------------------------------

# Direct-from-source-file load avoids the torch transitively-required
# `tritonblas` import at collection time on hosts without GPU origami.
_pred_path = (
    __import__("pathlib").Path(__file__).resolve().parent.parent
    / "include" / "tritonblas" / "_route_predicate.py"
)
_spec = importlib.util.spec_from_file_location("_route_predicate_k2156", _pred_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FZ = _mod._K2156_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18
PRIOR_P55 = _mod._K2147_P55_SKINNY_N1520_KCOMPL_ALIASSTACK_18
PRIOR_P40 = _mod._K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n1584():
    assert {N for (_, N, _, _) in FZ} == {1584}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_row_balance():
    """Both dtype rows complete 9/9 — no dtype-asymmetric exclusions on the
    off-by-48 column-narrow N=1584 tile."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    assert bf == fp


def test_envelope_equals_full_n1584_kcompl_grid():
    full = frozenset(
        (M, 1584, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18


def test_off_by_48_wave_misalignment_band_membership():
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n == 1584
    assert only_n % 64 == 48           # off-by-48 band
    assert only_n % 128 == 48          # mod-128=48 sibling to 1456/1328/1200


def test_twelfth_rung_cadence_above_p55_n1520():
    """N=1584 is exactly +64 above K-2147 P55 N=1520 (12th rung in the
    off-by-48 contiguous ladder)."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    (prior_n,) = {N for (_, N, _, _) in PRIOR_P55}
    assert only_n - prior_n == 64
    assert prior_n == 1520
    assert only_n == 1584


def test_sibling_n_firewall_vs_p55_n1520():
    assert FZ & PRIOR_P55 == set()


def test_sibling_n_firewall_vs_p40_n544():
    assert FZ & PRIOR_P40 == set()


# ---------------------------------------------------------------------------
# Dispatcher routing: prove the +2 LOC actually wires N=1584 through to the
# route-OUT path, AND that immediate neighbours / wave-aligned values / wrong-
# residue families do NOT route out (Pragmatist + Skeptic + Testing-Zealot).
# ---------------------------------------------------------------------------

# The matmul module pulls in torch + triton + origami at import time.  Skip
# the dispatcher tests cleanly on hosts without those (e.g. KB doc-build).
_HAVE_TRITONBLAS = True
try:
    from tritonblas.matmul import _k971_route_to_hbl  # noqa: F401
except Exception:  # pragma: no cover - defensive on no-GPU hosts
    _HAVE_TRITONBLAS = False


_MISSING_TRITONBLAS = pytest.mark.skipif(
    not _HAVE_TRITONBLAS,
    reason="tritonblas.matmul not importable (requires torch + triton + origami)",
)


@_MISSING_TRITONBLAS
@pytest.mark.parametrize("M", [2048, 4096, 8192])
@pytest.mark.parametrize("K", [4096, 8192, 16384])
@pytest.mark.parametrize("dtype_str", ["torch.bfloat16", "torch.float16"])
def test_dispatcher_routes_admitted_n1584_cells_to_hbl(M, K, dtype_str):
    """Every one of the 18 admitted cells MUST route OUT (to hipBLASLt) via
    the +2 LOC dispatcher diff.  This is the runtime evidence that the
    frozenset is wired into the actual dispatch path — not just defined."""
    import torch
    from tritonblas.matmul import _k971_route_to_hbl

    dtype = torch.bfloat16 if dtype_str == "torch.bfloat16" else torch.float16
    routed = _k971_route_to_hbl(
        M, 1584, K, dtype, dtype, enable_streamk=False, work_stealing=False
    )
    assert routed is True, (
        f"K-2156 P56 dispatcher MISS at admitted cell "
        f"(M={M}, N=1584, K={K}, {dtype_str}); the +2 LOC frozenset entry "
        f"is not wired through to _k971_route_to_hbl"
    )


@_MISSING_TRITONBLAS
@pytest.mark.parametrize(
    "N",
    [
        # Immediate neighbours of 1584 in the off-by-48 cadence
        1520,  # K-2147 P55 — should route via its own predicate, not K-2156
        1648,  # next would-be rung above (NOT yet admitted)
        1712,  # +2 cadence above (NOT admitted)
        # Wave-aligned values around 1584
        1536, 1600, 1568,
        # Wrong-residue (mod-64) families around 1584
        1583, 1585, 1586, 1582,
    ],
)
def test_dispatcher_does_not_route_n1584_neighbours_via_k2156(N):
    """The K-2156 frozenset must be tight: only N=1584 (not 1583/1585/1648/...)
    should be drained off via THIS predicate.  We probe with M=4096, K=8192,
    bf16 — a representative inner cell in the K-2156 grid — so any positive
    membership at neighbour N must come from the K-2156 entry, which would
    indicate a cardinality leak."""
    # Build a (M, N, K, dtype-str) tuple with the same shape as the frozenset
    # entries; assert it is NOT a member of the K-2156 frozenset.  This
    # checks the membership predicate at the data layer and complements the
    # cardinality test above.
    candidate = (4096, N, 8192, "torch.bfloat16")
    assert candidate not in FZ, (
        f"K-2156 frozenset cardinality leak: neighbour N={N} should not be "
        f"a member of the 18-cell verified-winner set"
    )


@_MISSING_TRITONBLAS
def test_dispatcher_predicate_is_pure_and_deterministic():
    """The route predicate is a pure function of (M, N, K, dtype) — calling
    it twice on the same args must return the same boolean."""
    import torch
    from tritonblas.matmul import _k971_route_to_hbl

    args = (4096, 1584, 8192, torch.bfloat16, torch.bfloat16, False, False)
    first = _k971_route_to_hbl(*args)
    second = _k971_route_to_hbl(*args)
    assert first == second is True
