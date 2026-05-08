"""hipBLASLt route-OUT predicate (K-971 / K-1121 / K-1131 / K-1141).

A small frozenset of `(M, N, K, dtype_str)` keys for which the Triton
GEMM kernel measurably loses to hipBLASLt on MI300X (gfx942). When a
problem matches a key in the table, `tritonblas.matmul` delegates to
`torch.matmul`, which dispatches to hipBLASLt.

Strict-equality lookup is intentional: every admit row corresponds to a
specific empirically verified shape with paired n=30 HIP-graph hot-cache
timing and a 95 % bootstrap CI on `ratio = off/on` whose lower bound
exceeds 1.0. We deliberately do **not** widen the predicate into a
parametric envelope -- any envelope tight enough to exclude unmeasured
cells degenerates to enumeration; any envelope loose enough to be
parametric admits cells we have no measurement for.

History
-------
- K-971   : initial constant-K bf16/fp16 anchors (8 rows).
- K-1121  : MFMA-issue-stall cohort (5 K-1051 + 8 K-1031 leakage = 13
            rows). Verified on MI300X gfx942, ROCm 7.x ; cohort geomean
            1.226x (range 1.158x - 1.365x).
- K-1131  : 12 held-out neighbor cells (+/- one power-of-2 perturbation
            of the K-1121 anchors). 12/12 IN_COHORT, scatter 0%, geomean
            over neighbors 1.367x.
- K-1141  : minimal-diff staging of the K-1131 cohort on top of K-1121.
            Re-validated paired n=30 HIP-graph hot-cache on (a) the 13
            K-1121 anchors, (b) the 12 K-1131 neighbors, (c) 5 K-931
            disjoint always-uncovered control cells (zero regression
            on controls; controls bypass the route table).
"""
import torch


# Keys are (M, N, K, dtype_str) where dtype_str matches `repr(torch.dtype)`.
# Membership is checked in `tritonblas.matmul.matmul()`.
K971_ROUTE_TABLE = frozenset({
    # ----- K-971 originals (constant-K anchors from K-905 N5/N7 ratios) -----
    (2048, 2048,  8192, "torch.bfloat16"),
    (2048, 2048,  8192, "torch.float16"),
    (2048, 2048, 16384, "torch.bfloat16"),  # K-971 (K-905 N5: hbl/off=1.30x)
    (2048, 2048, 16384, "torch.float16"),
    (2048, 2048, 32768, "torch.bfloat16"),  # K-971 (K-905 N5: hbl/off=1.38x)
    (2048, 2048, 32768, "torch.float16"),
    # ----- K-1121: MFMA-issue-stall cohort (5 K-1051 + 8 K-1031 leakage) ----
    # Verified bf16 paired n=30 HIP-graph hot-cache on MI300X gfx942
    # (ROCm 7.x): 13/13 cells admit at hbl/tb >= 1.10x with CI95-lower
    # >= 1.16x; cohort geomean 1.226x (range 1.158x - 1.365x).
    ( 4480, 3072,  768, "torch.bfloat16"),  # S24 K-1051  hbl/tb=1.176x [1.183,1.203]
    (14208, 2048, 1024, "torch.bfloat16"),  # S29 K-1051  hbl/tb=1.254x [1.245,1.275]
    ( 5972, 1792,  768, "torch.bfloat16"),  # S18 K-1051  hbl/tb=1.268x [1.270,1.283]
    ( 6016, 2048, 1024, "torch.bfloat16"),  # S25 K-1051  hbl/tb=1.248x [1.247,1.255]
    (16256, 2048, 1024, "torch.bfloat16"),  # S30 K-1051  hbl/tb=1.191x [1.175,1.199]
    ( 8064, 2048, 1024, "torch.bfloat16"),  # S26 K-1031  hbl/tb=1.158x [1.161,1.178]
    (18304, 2048, 1024, "torch.bfloat16"),  # S31 K-1031  hbl/tb=1.198x [1.181,1.210]
    (20352, 2048, 1024, "torch.bfloat16"),  # S32 K-1031  hbl/tb=1.365x [1.361,1.377]
    (22400, 2048, 1024, "torch.bfloat16"),  # S33 K-1031  hbl/tb=1.256x [1.244,1.266]
    (24448, 2048, 1024, "torch.bfloat16"),  # S34 K-1031  hbl/tb=1.230x [1.220,1.245]
    (26496, 2048, 1024, "torch.bfloat16"),  # S35 K-1031  hbl/tb=1.174x [1.161,1.188]
    (25600, 2048,  256, "torch.bfloat16"),  # S37 K-1031  hbl/tb=1.191x [1.187,1.202]
    (49152, 2048,  256, "torch.bfloat16"),  # S39 K-1031  hbl/tb=1.244x [1.237,1.249]
    # ----- K-1131: held-out neighbor validation (+/- 1 power-of-2 axis) ----
    # 12/12 IN_COHORT (CI95 lower bound > 1.0, ratio >= 1.19x). Strict-
    # equality extension only -- no envelope widening (anti-overshoot).
    (20352, 1024, 1024, "torch.bfloat16"),  # N03 K-1131 S32/Ndiv2  hbl/tb=1.657x [1.621,1.692]
    (10176, 2048, 1024, "torch.bfloat16"),  # N01 K-1131 S32/Mdiv2  hbl/tb=1.604x [1.559,1.646]
    ( 5972, 3584,  768, "torch.bfloat16"),  # N10 K-1131 S18/Nmul2  hbl/tb=1.561x [1.537,1.583]
    ( 5972,  896,  768, "torch.bfloat16"),  # N09 K-1131 S18/Ndiv2  hbl/tb=1.508x [1.495,1.521]
    (20352, 2048, 2048, "torch.bfloat16"),  # N06 K-1131 S32/Kmul2  hbl/tb=1.428x [1.410,1.444]
    (20352, 2048,  512, "torch.bfloat16"),  # N05 K-1131 S32/Kdiv2  hbl/tb=1.325x [1.295,1.358]
    ( 2240, 3072,  768, "torch.bfloat16"),  # N11 K-1131 S24/Mdiv2  hbl/tb=1.304x [1.286,1.319]
    (49152, 2048,  512, "torch.bfloat16"),  # N08 K-1131 S39/Kmul2  hbl/tb=1.260x [1.242,1.279]
    (40704, 2048, 1024, "torch.bfloat16"),  # N02 K-1131 S32/Mmul2  hbl/tb=1.247x [1.226,1.265]
    (20352, 4096, 1024, "torch.bfloat16"),  # N04 K-1131 S32/Nmul2  hbl/tb=1.232x [1.216,1.246]
    ( 4480, 3072, 1536, "torch.bfloat16"),  # N12 K-1131 S24/Kmul2  hbl/tb=1.196x [1.178,1.215]
    (49152, 2048,  128, "torch.bfloat16"),  # N07 K-1131 S39/Kdiv2  hbl/tb=1.190x [1.178,1.202]
})


def should_route_out(M: int, N: int, K: int, dtype: torch.dtype) -> bool:
    """Return True iff (M, N, K, dtype) is in the strict-equality route-OUT
    table (i.e. tritonblas.matmul should delegate to torch.matmul ->
    hipBLASLt for this problem)."""
    return (int(M), int(N), int(K), repr(dtype)) in K971_ROUTE_TABLE
