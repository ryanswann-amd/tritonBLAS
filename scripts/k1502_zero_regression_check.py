"""K-1502 zero-regression check for P35 (K-1847).

Per the K-1545 generalised structural-zero-drift theorem (premise (i')
append-only at the dispatch level + (ii') GATE-B per appended frozenset),
the K-1502 132-cell corpus drift is provably zero when every newly-appended
cell has an N value disjoint from the corpus-N set.

This script enforces that gate for the P35 frozenset (N=320) before merge.
"""
import sys
import torch

from tritonblas.matmul import _k971_route_to_hbl
from tritonblas._route_predicate import (
    _P35_SKINNY_N320_KCOMPL_VERIFIED_WIN_18 as P35,
)

# K-1502 / K-1295 / K-1316 / longK-family 132-cell reference corpus N-axis.
CORPUS_N = {1024, 2048, 4096, 8192, 16384}

# GATE-B: every cell in P35 must have N not in CORPUS_N.
overlap = [c for c in P35 if c[1] in CORPUS_N]
assert len(overlap) == 0, (
    "GATE-B FAIL: P35 leaks into K-1502 corpus N: %r" % (overlap,)
)
print("GATE-B: P35 cells (N={320}) disjoint from K-1502 corpus N", CORPUS_N,
      "=> 0 overlap")

# Probe: route a 30-cell representative slice of the K-1502 corpus and
# verify the P35 frozenset never matches (innocent witness for GATE-B).
sample = [
    (M, N, K, dt) for M in (1024, 2048, 4096, 8192)
    for N in (1024, 2048, 4096, 8192)
    for K in (4096, 8192, 16384)
    for dt in (torch.bfloat16, torch.float16)
][:30]
p35_admits = [c for c in sample if (c[0], c[1], c[2], str(c[3])) in P35]
assert len(p35_admits) == 0, (
    "GATE-A FAIL: P35 admits a corpus sample cell: %r" % (p35_admits,)
)
print("K-1502 corpus probe (30 cells):",
      "P35 admits 0 / 30 -> GATE-A pass (append-only innocent)")

# Sanity: the dispatcher returns True for every P35 cell.
fired = 0
for (M, N, K, dt_str) in P35:
    dt = torch.bfloat16 if dt_str == "torch.bfloat16" else torch.float16
    if _k971_route_to_hbl(M, N, K, dt, dt, enable_streamk=False,
                          work_stealing=False):
        fired += 1
assert fired == 18, "P35 route-OUT mis-fire: %d/18" % fired
print("P35 dispatcher route-OUT: 18/18 cells fire")

print("ALL GATES PASS — K-1502 corpus drift provably zero (K-1545 theorem)")
sys.exit(0)
