"""K-1480 P19 routing oracle — torch-free pin tests for the
`_K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT` frozenset.

Walks every cell in the K-1465 / K-1468 N=8192 K-COMPLEMENT sweep
(30 cells, M ∈ {2048, 4096, 8192} × N=8192 ×
K ∈ {2048, 4096, 8192, 16384, 32768} × {bf16, fp16}) and verifies:

  * 11 ADMIT cells -> _k1465_p19_route_to_hbl(...) == True
  * 19 REJECT cells -> _k1465_p19_route_to_hbl(...) == False
  * cardinality pinned at 11
  * ADMIT and REJECT partition the full 30-cell sweep
  * dtype keys use the torch-free string convention
  * every ADMIT cell sits strictly on the N=8192 column

Run:  python3 -m pytest tests/test_k1465_p19_skinny_n8192_kcompl.py -v
(no GPU required — host-key membership test only).
"""
from __future__ import annotations

import sys, pathlib
# Allow running from a fresh checkout without `pip install -e .`
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "include"))

from tritonblas._k1465_p19_skinny_n8192_kcompl import (  # noqa: E402
    _K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT,
    _k1465_p19_route_to_hbl,
)


# --- ground-truth admit / reject lists from the K-1468 sweep CSV --------------
ADMIT_CELLS = [
    (2048, 8192,  2048, "torch.float16"),
    (2048, 8192,  4096, "torch.float16"),
    (2048, 8192,  8192, "torch.bfloat16"),
    (2048, 8192,  8192, "torch.float16"),
    (2048, 8192, 16384, "torch.float16"),
    (2048, 8192, 32768, "torch.float16"),
    (4096, 8192,  2048, "torch.float16"),
    (4096, 8192,  4096, "torch.float16"),
    (4096, 8192, 16384, "torch.float16"),
    (4096, 8192, 32768, "torch.float16"),
    (8192, 8192,  2048, "torch.float16"),
]

REJECT_CELLS = [
    (2048, 8192,  2048, "torch.bfloat16"),   # r=0.974 — TB faster
    (2048, 8192,  4096, "torch.bfloat16"),   # r=0.898
    (2048, 8192, 16384, "torch.bfloat16"),   # r=0.994
    (2048, 8192, 32768, "torch.bfloat16"),   # r=1.053  CI99-lo=1.046 (just below 1.05)
    (4096, 8192,  2048, "torch.bfloat16"),   # r=0.916
    (4096, 8192,  4096, "torch.bfloat16"),   # r=0.855
    (4096, 8192,  8192, "torch.bfloat16"),   # r=0.763
    (4096, 8192,  8192, "torch.float16"),    # r=1.046 — fails 1.05 gate
    (4096, 8192, 16384, "torch.bfloat16"),   # r=0.739
    (4096, 8192, 32768, "torch.bfloat16"),   # r=0.777
    (8192, 8192,  2048, "torch.bfloat16"),   # r=0.907
    (8192, 8192,  4096, "torch.bfloat16"),   # r=0.834
    (8192, 8192,  4096, "torch.float16"),    # r=1.012
    (8192, 8192,  8192, "torch.bfloat16"),   # r=1.047 — fails 1.05 gate
    (8192, 8192,  8192, "torch.float16"),    # r=1.041
    (8192, 8192, 16384, "torch.bfloat16"),   # r=0.761
    (8192, 8192, 16384, "torch.float16"),    # r=0.990
    (8192, 8192, 32768, "torch.bfloat16"),   # r=0.783
    (8192, 8192, 32768, "torch.float16"),    # r=0.987
]


def test_admit_cells_route_to_hbl():
    """All 11 K-1480 admit cells must fire the 12th-position predicate."""
    misses = [c for c in ADMIT_CELLS if not _k1465_p19_route_to_hbl(*c)]
    assert not misses, f"K-1480 P19 misses admit cells: {misses}"


def test_reject_cells_do_not_fire_p19():
    """Cells outside the K-1480 admit set must NOT fire the 12th
    predicate — they fall through to the earlier predicate stack
    (K-971 .. K-1458) or to the native triton kernel."""
    spurious = [c for c in REJECT_CELLS if _k1465_p19_route_to_hbl(*c)]
    assert not spurious, f"K-1480 P19 fires on reject cells: {spurious}"


def test_cardinality_pinned():
    assert len(_K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT) == 11


def test_admit_reject_partition_is_30():
    """Sweep cardinality invariant — ADMIT + REJECT == full 30-cell sweep,
    and the two lists are disjoint."""
    assert len(ADMIT_CELLS) + len(REJECT_CELLS) == 30
    assert set(ADMIT_CELLS).isdisjoint(set(REJECT_CELLS))


def test_admit_set_matches_frozenset():
    """The ground-truth ADMIT_CELLS list must equal the productionised
    frozenset — guards against drift between the test fixture and the
    constant under test."""
    assert set(ADMIT_CELLS) == set(_K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT)


def test_dtype_canonical():
    """Keys must use the torch-free string convention 'torch.float16' /
    'torch.bfloat16' (matches the rest of the K-1175 predicate chain)."""
    for c in _K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT:
        assert c[3] in {"torch.float16", "torch.bfloat16"}, c


def test_n_axis_is_8192():
    """All admit cells must be on the N=8192 column (this is the
    K-COMPLEMENT N=8192 frozenset by construction)."""
    for c in _K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT:
        assert c[1] == 8192, c


def test_predicate_accepts_torch_dtype_or_string():
    """`_k1465_p19_route_to_hbl` must accept either a torch dtype object
    or its string repr — the K-1175 host-key tuple convention."""
    # Exercise the str(dtype) canonicalisation path via a stand-in object
    # whose repr matches the canonical key — keeps the test torch-free.
    class _StubDtype:
        def __init__(self, s): self._s = s
        def __str__(self): return self._s
    assert _k1465_p19_route_to_hbl(2048, 8192, 2048, _StubDtype("torch.float16"))
    assert not _k1465_p19_route_to_hbl(2048, 8192, 2048, _StubDtype("torch.bfloat16"))
    # Plain string also works.
    assert _k1465_p19_route_to_hbl(8192, 8192, 2048, "torch.float16")


if __name__ == "__main__":
    test_admit_cells_route_to_hbl()
    test_reject_cells_do_not_fire_p19()
    test_cardinality_pinned()
    test_admit_reject_partition_is_30()
    test_admit_set_matches_frozenset()
    test_dtype_canonical()
    test_n_axis_is_8192()
    test_predicate_accepts_torch_dtype_or_string()
    print(f"OK: {len(ADMIT_CELLS)} admit / {len(REJECT_CELLS)} reject; "
          f"all 8 routing-oracle invariants hold")
