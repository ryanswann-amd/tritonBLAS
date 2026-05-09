"""K-1480 P19 routing oracle — torch-free pin tests for the
12th-position frozenset `_K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT`
in `tritonblas/_route_predicate.py`.

Walks every cell in the K-1465 / K-1468 N=8192 K-COMPLEMENT sweep
(30 cells, M ∈ {2048, 4096, 8192} × N=8192 ×
K ∈ {2048, 4096, 8192, 16384, 32768} × {bf16, fp16}) and verifies:

  * every ADMIT cell  (11) -> _k1465_p19_skinny_n8192_routeout(...) == True
  * every REJECT cell (19) -> _k1465_p19_skinny_n8192_routeout(...) == False
  * cardinality pinned at 11
  * ADMIT and REJECT partition the full 30-cell sweep
  * dtype keys use the torch-free string convention
  * every ADMIT cell sits strictly on the N=8192 column
  * the test fixture matches the in-source frozenset (drift guard)
  * predicate accepts both torch-dtype singleton repr and string
  * 4 NEW cells (added vs the K-1468 ≥1.10 staging frozenset by
    widening to ≥1.05x strict-CI) are explicitly pinned ADMIT

Each cell is its own parametrised test case so a single-cell drift
shows up as a single-cell failure (per K-1175 disjointness/coverage
convention).

Run:  python3 -m pytest tests/test_k1465_p19_skinny_n8192_kcompl.py -v
(no GPU required — host-key membership test only).
"""
from __future__ import annotations

import sys
import pathlib

# Allow running from a fresh checkout without `pip install -e .`
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "include"))

import pytest  # noqa: E402

from tritonblas._route_predicate import (  # noqa: E402
    _K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT,
    _k1465_p19_skinny_n8192_routeout,
)


# --- ground-truth admit / reject lists from the K-1468 paired-n30 sweep ------
# 11 admit cells (≥1.05x strict-CI gate: ratio_median ≥ 1.05 AND
# ratio_ci99_lo ≥ 1.05) — derived from the K-1468 30-cell N=8192 sweep
# CSV.  Comments give r_med / CI99-lo for traceability against
# `output/p19_n8192_per_cell_table.md`.
ADMIT_CELLS = [
    (2048, 8192,  2048, "torch.float16"),    # r=1.139  CI99-lo=1.137
    (2048, 8192,  4096, "torch.float16"),    # r=1.135  CI99-lo=1.130
    (2048, 8192,  8192, "torch.bfloat16"),   # r=1.102  CI99-lo=1.084 — only bf16 admit
    (2048, 8192,  8192, "torch.float16"),    # r=1.410  CI99-lo=1.396
    (2048, 8192, 16384, "torch.float16"),    # r=1.383  CI99-lo=1.383
    (2048, 8192, 32768, "torch.float16"),    # r=1.406  CI99-lo=1.405
    (4096, 8192,  2048, "torch.float16"),    # r=1.102  CI99-lo=1.100
    (4096, 8192,  4096, "torch.float16"),    # r=1.107  CI99-lo=1.103
    (4096, 8192, 16384, "torch.float16"),    # r=1.056  CI99-lo=1.055
    (4096, 8192, 32768, "torch.float16"),    # r=1.060  CI99-lo=1.059
    (8192, 8192,  2048, "torch.float16"),    # r=1.098  CI99-lo=1.091
]

# 19 reject cells (full negative space — every sweep cell that fails the
# ≥1.05x strict-CI admit gate).  12 are bf16 cells where tritonblas
# either wins outright (r<1.0; bf16 inversion at N=8192) or is within
# the dispatch-overhead crossover band; 7 are fp16 cells where the
# ratio is <1.05 or whose CI99-lo dips below 1.05.
REJECT_CELLS = [
    (2048, 8192,  2048, "torch.bfloat16"),   # r=0.974  TB faster
    (2048, 8192,  4096, "torch.bfloat16"),   # r=0.898  TB faster
    (2048, 8192, 16384, "torch.bfloat16"),   # r=0.994  TB faster
    (2048, 8192, 32768, "torch.bfloat16"),   # r=1.053  CI99-lo=1.046 — fails 1.05 CI gate
    (4096, 8192,  2048, "torch.bfloat16"),   # r=0.916  TB faster
    (4096, 8192,  4096, "torch.bfloat16"),   # r=0.855  TB faster
    (4096, 8192,  8192, "torch.bfloat16"),   # r=0.763  TB much faster
    (4096, 8192,  8192, "torch.float16"),    # r=1.046  fails 1.05 gate
    (4096, 8192, 16384, "torch.bfloat16"),   # r=0.739  TB much faster (max bf16 win)
    (4096, 8192, 32768, "torch.bfloat16"),   # r=0.777  TB much faster
    (8192, 8192,  2048, "torch.bfloat16"),   # r=0.907  TB faster
    (8192, 8192,  4096, "torch.bfloat16"),   # r=0.834  TB faster
    (8192, 8192,  4096, "torch.float16"),    # r=1.012  inside crossover
    (8192, 8192,  8192, "torch.bfloat16"),   # r=1.047  fails 1.05 gate
    (8192, 8192,  8192, "torch.float16"),    # r=1.041  fails 1.05 gate
    (8192, 8192, 16384, "torch.bfloat16"),   # r=0.761  TB much faster
    (8192, 8192, 16384, "torch.float16"),    # r=0.990  TB faster
    (8192, 8192, 32768, "torch.bfloat16"),   # r=0.783  TB much faster
    (8192, 8192, 32768, "torch.float16"),    # r=0.987  TB faster
]

# 4 NEW cells admitted by the K-1480 ≥1.05x widening that the K-1468
# ≥1.10 staging gate had rejected.  These are the load-bearing change
# of K-1480 vs K-1468 — pin each one explicitly so any future tightening
# of the gate (or accidental rollback) trips a named single-cell test.
NEW_VS_K1468_CELLS = [
    (4096, 8192,  2048, "torch.float16"),    # r=1.102  K-1468 admitted at exactly 1.10
    (4096, 8192,  4096, "torch.float16"),    # r=1.107  K-1468 admitted
    (4096, 8192, 16384, "torch.float16"),    # r=1.056  NEW @ ≥1.05 (K-1468 reject @ ≥1.10)
    (4096, 8192, 32768, "torch.float16"),    # r=1.060  NEW @ ≥1.05 (K-1468 reject @ ≥1.10)
    (8192, 8192,  2048, "torch.float16"),    # r=1.098  NEW @ ≥1.05 (K-1468 reject @ ≥1.10)
]

# Sanity — derived constants used by parametrised tests.
assert len(ADMIT_CELLS) == 11
assert len(REJECT_CELLS) == 19
assert len(ADMIT_CELLS) + len(REJECT_CELLS) == 30
assert set(ADMIT_CELLS).isdisjoint(set(REJECT_CELLS))


# --------------------------------------------------------------------------
# Per-cell parametrised pin tests — one test case per sweep cell so the
# admit/reject space is explicitly enumerated (30 cases total).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cell", ADMIT_CELLS, ids=lambda c: f"ADMIT_{c[0]}x{c[1]}x{c[2]}_{c[3].split('.')[-1]}")
def test_admit_cell_routes_to_hbl(cell):
    """Each of the 11 admit cells must fire the 12th-position predicate."""
    assert _k1465_p19_skinny_n8192_routeout(*cell), (
        f"K-1480 P19 12th-position predicate misses admit cell {cell}"
    )


@pytest.mark.parametrize("cell", REJECT_CELLS, ids=lambda c: f"REJECT_{c[0]}x{c[1]}x{c[2]}_{c[3].split('.')[-1]}")
def test_reject_cell_does_not_fire(cell):
    """Each of the 19 reject cells must NOT fire the 12th-position predicate
    — they fall through to the earlier predicate stack (K-971..K-1442)
    or to the native triton kernel."""
    assert not _k1465_p19_skinny_n8192_routeout(*cell), (
        f"K-1480 P19 12th-position predicate spuriously fires on reject cell {cell}"
    )


# --------------------------------------------------------------------------
# Aggregate / structural invariants.
# --------------------------------------------------------------------------
def test_cardinality_pinned_at_11():
    assert len(_K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT) == 11


def test_admit_reject_partition_is_30():
    """ADMIT + REJECT == full 30-cell sweep, and the two lists are disjoint."""
    assert len(ADMIT_CELLS) + len(REJECT_CELLS) == 30
    assert set(ADMIT_CELLS).isdisjoint(set(REJECT_CELLS))


def test_admit_set_matches_in_source_frozenset():
    """The ground-truth ADMIT_CELLS list must equal the productionised
    frozenset.  Guards against drift between this test fixture and the
    constant under test in `_route_predicate.py`."""
    assert set(ADMIT_CELLS) == set(_K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT)


def test_dtype_canonical_string_keys():
    """Keys must use the torch-free string convention 'torch.float16' /
    'torch.bfloat16' (matches the rest of the K-1175 predicate chain)."""
    for c in _K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT:
        assert c[3] in {"torch.float16", "torch.bfloat16"}, c


def test_n_axis_is_8192_for_every_admit_cell():
    """All admit cells must be on the N=8192 column (this is the
    K-COMPLEMENT N=8192 frozenset by construction)."""
    for c in _K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT:
        assert c[1] == 8192, c


def test_predicate_accepts_torch_dtype_or_string():
    """`_k1465_p19_skinny_n8192_routeout` must accept either a torch dtype
    object or its string repr — the K-1175 host-key tuple convention."""
    # Stand-in object whose repr matches the canonical key — keeps the test
    # torch-free.
    class _StubDtype:
        def __init__(self, s): self._s = s
        def __str__(self): return self._s
    assert _k1465_p19_skinny_n8192_routeout(2048, 8192, 2048, _StubDtype("torch.float16"))
    assert not _k1465_p19_skinny_n8192_routeout(2048, 8192, 2048, _StubDtype("torch.bfloat16"))
    # Plain string also works.
    assert _k1465_p19_skinny_n8192_routeout(8192, 8192, 2048, "torch.float16")
    # Cached _dtype_str fast-path bypasses the str(dtype) call.
    assert _k1465_p19_skinny_n8192_routeout(
        2048, 8192, 2048, None, _dtype_str="torch.float16"
    )


@pytest.mark.parametrize("cell", NEW_VS_K1468_CELLS, ids=lambda c: f"NEW_{c[0]}x{c[1]}x{c[2]}_{c[3].split('.')[-1]}")
def test_new_cells_vs_k1468_are_admitted(cell):
    """The K-1480 ≥1.05x widening admits cells the K-1468 ≥1.10 staging
    gate had rejected.  Pin each one explicitly so any tightening of the
    gate (or accidental rollback to the K-1468 7-cell set) trips a named
    single-cell test."""
    assert cell in _K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT, (
        f"K-1480 NEW-vs-K-1468 admit cell {cell} missing from frozenset — "
        "did the gate revert to the K-1468 ≥1.10 7-cell set?"
    )
    assert _k1465_p19_skinny_n8192_routeout(*cell)


def test_routes_through_k971_route_decision():
    """Integration: the 12th-position predicate must be wired into
    `k971_route_decision` so admit cells route OUT to hipBLASLt end-to-end
    (not just match the standalone frozenset)."""
    from tritonblas._route_predicate import k971_route_decision
    for cell in ADMIT_CELLS:
        M, N, K, dtype_str = cell
        decision = k971_route_decision(
            M, N, K, dtype_str, dtype_str,
            enable_streamk=False, work_stealing=False,
        )
        assert decision is True, (
            f"k971_route_decision({cell}) returned {decision}; admit cell "
            "must route OUT to hipBLASLt"
        )


if __name__ == "__main__":
    # Allow direct execution as a smoke test (no pytest required).
    for cell in ADMIT_CELLS:
        assert _k1465_p19_skinny_n8192_routeout(*cell), f"admit miss {cell}"
    for cell in REJECT_CELLS:
        assert not _k1465_p19_skinny_n8192_routeout(*cell), f"reject spurious {cell}"
    test_cardinality_pinned_at_11()
    test_admit_reject_partition_is_30()
    test_admit_set_matches_in_source_frozenset()
    test_dtype_canonical_string_keys()
    test_n_axis_is_8192_for_every_admit_cell()
    test_predicate_accepts_torch_dtype_or_string()
    for cell in NEW_VS_K1468_CELLS:
        assert cell in _K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT, f"NEW miss {cell}"
    test_routes_through_k971_route_decision()
    print(
        f"OK: {len(ADMIT_CELLS)} admit / {len(REJECT_CELLS)} reject (30 total); "
        f"{len(NEW_VS_K1468_CELLS)} NEW-vs-K-1468 cells pinned; "
        "all routing-oracle invariants hold; integration via "
        "k971_route_decision verified."
    )
