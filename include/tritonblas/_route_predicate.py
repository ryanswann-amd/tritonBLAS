"""hipBLASLt route-OUT predicate (K-971 / K-1121 / K-1131 / K-1141 / K-1148).

A small frozenset of `(M, N, K, dtype_str)` keys for which the Triton
GEMM kernel measurably loses to hipBLASLt on MI300X (gfx942), plus an
axis-aligned structural envelope (E1) over the K-1121 MFMA-issue-stall
cohort (K-1148). When a problem matches the table OR falls inside E1,
`tritonblas.matmul` delegates to `torch.matmul`, which dispatches to
hipBLASLt.

Strict-equality lookup is intentional for the K-971 / K-1131 entries:
every admit row corresponds to a specific empirically verified shape
with paired n=30 HIP-graph hot-cache timing and a 95 % bootstrap CI on
`ratio = off/on` whose lower bound exceeds 1.0.

The K-1121 cohort is additionally generalised in K-1148 to its natural
axis-aligned structural envelope E1 (the smallest two-axis box that
covers all 13 anchors):

    E1 := { (M, N, K, bf16) : N in {1792, 2048, 3072}
                              AND K in {256, 768, 1024} }

K-1142's inverse-predicate audit measured every K-931 top-40 cell
inside E1 at paired n=30 and confirmed exactly two false positives:

    ( 256, 2048, 256, bf16)  hbl/tb = 0.973x  CI95 = [0.966, 0.980]
    (2048, 1792, 256, bf16)  hbl/tb = 1.015x  CI95 = [1.010, 1.020]

Both FPs admit a TB-WIN / TB-TIE (CI95-upper strictly below the 1.05
admission floor) and are routed OUT under E1 as **accepted residual**.
The catalog-wide trade quantified by K-1148 (paired n=30 backtest
across (a) 13 K-1121 anchors, (b) 12 K-1131 neighbors, (c) the 2 FPs)
shows the structural coverage gain dominates the small loss on FP
cells.

History
-------
- K-971   : initial constant-K bf16/fp16 anchors (6 rows).
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
- K-1148  : structural extension of the K-1121 cohort into the
            axis-aligned envelope E1 (N in {1792,2048,3072}, K in
            {256,768,1024}, bf16). Two K-1142 inverse-predicate FPs
            accepted as residual. Three-cohort paired n=30 backtest on
            MI300X confirms (a) anchors retained, (b) neighbors
            unchanged, (c) FP loss bounded.
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


# ----------------------------------------------------------------------
# K-1148: axis-aligned structural envelope E1 over the K-1121 anchor
# cohort. Every K-1121 anchor is inside E1 by construction (the envelope
# is the closure of the anchor set under axis-aligned bounding box).
# Two confirmed K-1142 inverse-predicate false positives lie inside E1
# and are accepted as residual:
#     (M=256,  N=2048, K=256, bf16)  hbl/tb = 0.973x  TB-WIN
#     (M=2048, N=1792, K=256, bf16)  hbl/tb = 1.015x  TB-TIE
# Both have CI95-upper strictly below the 1.05 admission floor; routing
# them OUT incurs at most a few percent slowdown on those exact cells.
# K-1148's three-cohort backtest (paired n=30 HIP-graph hot-cache on
# MI300X) confirms the catalog-wide trade is favourable.
# ----------------------------------------------------------------------
_K1148_E1_N_SET = frozenset({1792, 2048, 3072})
_K1148_E1_K_SET = frozenset({256, 768, 1024})
_K1148_E1_DTYPE = "torch.bfloat16"


def _in_k1148_e1_envelope(M: int, N: int, K: int, dtype_str: str) -> bool:
    """K-1148 structural extrapolation of the K-1121 cohort.

    Returns True iff (M, N, K, dtype) falls inside the axis-aligned
    envelope E1 := {N in {1792,2048,3072} AND K in {256,768,1024} AND
    dtype == bfloat16}.  Two K-1142 inverse-predicate FPs are inside
    this envelope and are routed OUT as accepted residual; see module
    docstring for the cell-level measurement."""
    return (
        dtype_str == _K1148_E1_DTYPE
        and N in _K1148_E1_N_SET
        and K in _K1148_E1_K_SET
    )


def should_route_out(M: int, N: int, K: int, dtype: torch.dtype) -> bool:
    """Return True iff (M, N, K, dtype) should delegate to hipBLASLt.

    Routes OUT when EITHER:
      (a) the (M, N, K, dtype_str) tuple matches a strict-equality entry
          in K971_ROUTE_TABLE (K-971 / K-1131 cells), OR
      (b) (M, N, K, dtype) lies inside the K-1148 axis-aligned envelope
          E1 over the K-1121 cohort.
    """
    dt = repr(dtype)
    key = (int(M), int(N), int(K), dt)
    if key in K971_ROUTE_TABLE:
        return True
    return _in_k1148_e1_envelope(int(M), int(N), int(K), dt)
