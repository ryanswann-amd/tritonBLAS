# tritonblas / _k1465_p19_skinny_n8192_kcompl.py
#
# K-1480 P19 — N=8192 K-COMPLEMENT route-OUT productionisation.
# 12th-position frozenset in the `_k971_route_to_hbl` predicate chain
# (lineage: K-971 P0 anchor → K-1295 P12 PMC square mid → K-1417 P15
#  N=512 K-COMPL → K-1433 P16 N=1024 BASE K-COMPL → K-1442 P17 N=2048
#  K-COMPL → K-1458 P18 N=4096 K-COMPL → K-1480 P19 N=8192 K-COMPL).
#
# Source: K-1465 N=8192 K-COMPLEMENT sweep, re-paired in K-1468 on radha
#         rad-mi300x-splinter1 (gfx942, mi300x-es; c42 down).
# Sweep:  M ∈ {2048, 4096, 8192} × N=8192 × K ∈ {2048, 4096, 8192,
#         16384, 32768} × dtype ∈ {bf16, fp16}, 30 cells total.
# Method: paired n=30 HIP-graph hot-cache, 50 replays/iter, B=10000
#         vectorised paired bootstrap CI99.
# Admit:  strict-equality at >=1.05x:
#           ratio_median   >= 1.05  AND
#           ratio_ci99_lo  >= 1.05
# Result: 11 / 30 cells admit; admit-set geomean(tb/hbl) = 1.174×.
#
# Disjointness (K-1175 / R-1329 invariant): the 11-cell admit set sits
# strictly on the N=8192 column.  No prior frozenset (P0..P18) contains
# any (M, 8192, K, dtype) cell with M ∈ {2048, 4096, 8192} and
# K ∈ {2048, 4096, 8192, 16384, 32768}, so pairwise-disjointness is
# preserved (verified in tests/test_k1465_p19_skinny_n8192_kcompl.py).
#
# Keys are (M, N, K, str(dtype)) — torch-free host module convention so
# the predicate can be evaluated without importing torch (matches the
# K-1175 host-key tuple convention used by the rest of the predicate
# chain).
from __future__ import annotations

_K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT = frozenset({
    # M=2048 row × N=8192 × K × dtype
    (2048, 8192,  2048, "torch.float16"),    # r=1.139  CI99-lo=1.137
    (2048, 8192,  4096, "torch.float16"),    # r=1.135  CI99-lo=1.130
    (2048, 8192,  8192, "torch.bfloat16"),   # r=1.102  CI99-lo=1.084
    (2048, 8192,  8192, "torch.float16"),    # r=1.410  CI99-lo=1.396
    (2048, 8192, 16384, "torch.float16"),    # r=1.383  CI99-lo=1.383
    (2048, 8192, 32768, "torch.float16"),    # r=1.406  CI99-lo=1.405
    # M=4096 row × N=8192 × K × dtype
    (4096, 8192,  2048, "torch.float16"),    # r=1.102  CI99-lo=1.100
    (4096, 8192,  4096, "torch.float16"),    # r=1.107  CI99-lo=1.103
    (4096, 8192, 16384, "torch.float16"),    # r=1.056  CI99-lo=1.055
    (4096, 8192, 32768, "torch.float16"),    # r=1.060  CI99-lo=1.059
    # M=8192 row × N=8192 × K × dtype
    (8192, 8192,  2048, "torch.float16"),    # r=1.098  CI99-lo=1.091
})

assert len(_K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT) == 11, (
    "K-1480 P19 admit cardinality pinned at 11 cells (>=1.05x strict-equality "
    "gate over the K-1465/K-1468 N=8192 K-COMPLEMENT sweep)."
)


def _k1465_p19_route_to_hbl(M: int, N: int, K: int, dtype) -> bool:
    """12th-position predicate (K-1480 P19).

    Returns True iff the (M, N, K, dtype) cell is in the K-1480 admit set
    and should be routed OUT to hipBLASLt (`torch.matmul`) instead of the
    tritonblas routed kernel.

    `dtype` may be a torch dtype object or its string repr — the predicate
    canonicalises to `str(dtype)` for the host-key tuple lookup.
    """
    key = (M, N, K, str(dtype))
    return key in _K1465_P19_SKINNY_N8192_KCOMPL_ROUTEOUT
