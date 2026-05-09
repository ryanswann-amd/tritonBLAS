"""K-1003 R-K979 P5 Gate-0 admission predicate (torch-free).

Hosts the closed-form structural-pathology predicate plus the K-905/K-971
mid-square long-K strict-equality anchors. Imported by ``matmul.py`` for the
runtime dispatch; importable on its own (no torch / triton chain) so unit
and integration tests can exercise it without booting a GPU stack.

The predicate keys solely on ``(M, N, K, dtype)`` where ``dtype`` is matched by
``str(dtype) == "torch.bfloat16"``; any object whose ``str()`` repr is
``"torch.bfloat16"`` (i.e. ``torch.bfloat16`` itself, or the literal string
``"torch.bfloat16"`` used in tests) selects the bf16 path.
"""
from __future__ import annotations

import os

# fp16 K-905/K-971 anchors stay as exact-tuple lookups because their shapes
# structurally collide with K-950 LAND cells (e.g. (1024,1024,16384,bf16) is
# both a K-912 LAND and a K-905 route-OUT — only an exact tuple can express
# "fire on this exact shape but not on its baseline-LAND twin").
K971_ROUTE_TABLE = frozenset({
    (1024, 1024, 16384, "torch.bfloat16"),  # K-905 baseline
    (1024, 1024, 16384, "torch.float16"),
    (1024, 1024, 32768, "torch.bfloat16"),  # K-971 (K-905 N4: hbl/off=1.73x)
    (1024, 1024, 32768, "torch.float16"),
    (2048, 2048, 16384, "torch.bfloat16"),  # K-971 (K-905 N2: hbl/off=1.37x)
    (2048, 2048, 16384, "torch.float16"),
    (2048, 2048, 32768, "torch.bfloat16"),  # K-971 (K-905 N5: hbl/off=1.38x)
    (2048, 2048, 32768, "torch.float16"),
})


def _dtype_is_bf16(dtype) -> bool:
    """Match torch.bfloat16 without importing torch.

    Both the ``torch.bfloat16`` singleton and the literal string
    ``"torch.bfloat16"`` (used by tests) pass; anything else fails.
    """
    return str(dtype) == "torch.bfloat16"


def R_K979_P5_route_to_hbl(M: int, N: int, K: int, dtype) -> bool:
    """R-K979 v3 / P5 — closed-form 4-clause structural pathology predicate.

    Returns True when shape (M, N, K, dtype) sits in the NO-LAND-for-Triton-
    override regime (i.e. tritonblas should route to hipBLASLt). Translates
    P5's PMC predicate into structural (M, N, K, dtype) proxies:

      Clause-1 (over-tile / LDS-bound mid-rect):
          mid-rect non-square with K in the [1240, 8064] LDS-pressure band
          and one axis pinned to the {1792, 2048, 3072} LDS-bound family.
      Clause-2 (hbl-strong / LMhead projection):
          large-M skinny  M >= 5000, N == 2048, K in {256, 1024}.
      Clause-3 (tb-weak / extreme aspect or tiny):
          aspect ratio max(M,N)/min(M,N) >= 100, or min(M,N) <= 192 AND
          K >= 2048 (waves under-utilised on the small axis).
      Clause-4 (over-tile dual / N=1792 shallow-K):
          N == 1792 and 512 <= K <= 768 (K-1062 raised the K floor from
          0 to 512 = 8*BK64 to silence the K=256 false-positive surfaced
          by the K-1017 PMC sweep and confirmed by the K-1043 per-clause
          confusion matrix; S07=(2048,1792,256,bf16) gap_x=1.044 is
          inside the per-engine ~5% CV noise floor — routing buys a
          measured-zero-gain dispatch detour), with M either large-
          skinny (>= 5000) or in the K-984/K-989 mid-rect band
          [256, 2048].  K-984/K-989 anchors S06 (K=736) and S18 (K=768)
          both have K >= 736 so neither is regressed.

    bf16-only: K-984/K-989 ship cohort and K-931 measurement scope are bf16;
    fp16 K-905/K-971 anchors stay in :data:`K971_ROUTE_TABLE`.

    Verification (K-1003):
      - K-984+K-989 union (14 unique shapes): 14/14 FIRE
      - K-950 LAND set (9 cells): 0/9 FIRE (zero leakage)
      - K-931 top-40: 29/40 FIRE (15 net beyond strict-equality)
    """
    if not _dtype_is_bf16(dtype):
        return False
    minMN = min(M, N)
    maxMN = max(M, N)
    # Clause-1: over-tile, LDS-bound, mid-rect non-square with mid-band K.
    if (minMN < maxMN
            and 256 <= minMN <= 2304
            and 1792 <= maxMN <= 3072
            and 1240 <= K <= 8064):
        return True
    # Clause-2: hbl-strong LMhead-style large-M skinny.
    if M >= 5000 and N == 2048 and K in (256, 1024):
        return True
    # Clause-3: tb-weak — extreme aspect or skinny-and-long-K.
    if maxMN >= 100 * max(1, minMN):
        return True
    if minMN <= 192 and K >= 2048:
        return True
    # Clause-4: N=1792 shallow-K dual of clause-1.  K-1062 K-floor=512
    # (8*BK64) — see docstring; closes the K=256 S07 false positive
    # without affecting the K=736 / K=768 K-984+K-989 anchors.
    if N == 1792 and 512 <= K <= 768 and (M >= 5000 or 256 <= M <= 2048):
        return True
    return False


# K-1109 (S-002) brief deliverable (1): explicit shape-key allowlist for
# the K-1074 MFMA-issue-stall candidate cells (originally 8 cells: S26,
# S31-S35, S37, S39).  Brief asked for "minimal diff (<30 LOC) gated behind
# explicit shape-key allowlist initially, with strict-equality fallback for
# the validated 8-cell set as a safety net".
#
# Gated by env var (default OFF) because the only paired n=30 round-robin
# timing on c42/MI300X available at PR time (k1109_k1074_timing.json,
# derived from K-1074/output/rr_summary.csv) showed 0/8 cells clear the
# K-901 +2% LAND margin.  Set TRITONBLAS_K1109_P6_ALLOWLIST=1 in the
# runtime env to enable for further validation.
#
# REGRESSION CARVE-OUT (per K-1109 reviewer consensus, 4/4 REVISE votes):
# Two of the original 8 cells — S32 (20352,2048,1024) and S37
# (25600,2048,256) — are CI95-confirmed REGRESSIONS in the same n=30 paired
# timing (S32: mean -0.51%, CI95 [-0.872, -0.148]; S37: mean -0.68%, CI95
# [-1.129, -0.232]).  Shipping a "safety net" allowlist that admits known-
# regressing cells, even default-OFF, is a hazard not a safety net — the
# gate exists precisely so operators can flip it on without a code change,
# so the contents must never include cells where paired evidence already
# proves harm.  S32 and S37 are therefore EXCLUDED.  The
# K1074_REGRESSION_CARVE_OUT_PINS test in tests/test_k971_route_predicate.py
# documents this with the exact CI95 numbers so any silent re-add of those
# cells fails CI loudly.
#
# Pin tests exercise both gate states so any change to the allowlist or
# gate semantics surfaces in CI.
_K1109_P6_K1074_ALLOWLIST = frozenset({
    ( 8064, 2048, 1024, "torch.bfloat16"),  # S26  mean +0.51%, CI95 [+0.13,+0.88]
    (18304, 2048, 1024, "torch.bfloat16"),  # S31  mean -2.96%, CI95 [-7.06,+1.14]  (overlaps zero)
    (22400, 2048, 1024, "torch.bfloat16"),  # S33  mean +1.34%, CI95 [-0.06,+2.75]
    (24448, 2048, 1024, "torch.bfloat16"),  # S34  mean +0.65%, CI95 [+0.47,+0.83]
    (26496, 2048, 1024, "torch.bfloat16"),  # S35  mean +0.77%, CI95 [-0.15,+1.69]
    (49152, 2048,  256, "torch.bfloat16"),  # S39  mean +0.24%, CI95 [-0.18,+0.66]
    # EXCLUDED — CI95-confirmed regressions in K-1074 paired n=30 timing:
    #   S32 (20352,2048,1024): mean -0.51%, CI95 [-0.872, -0.148]
    #   S37 (25600,2048, 256): mean -0.68%, CI95 [-1.129, -0.232]
})
_K1109_P6_ALLOWLIST_ENV = "TRITONBLAS_K1109_P6_ALLOWLIST"

# Pinned regression evidence for the two cells excluded from the allowlist.
# Re-adding either cell to _K1109_P6_K1074_ALLOWLIST without first refuting
# this evidence will fail the K1074_REGRESSION_CARVE_OUT_PINS test.
_K1109_P6_K1074_REGRESSION_EXCLUSIONS = frozenset({
    (20352, 2048, 1024, "torch.bfloat16"),  # S32
    (25600, 2048,  256, "torch.bfloat16"),  # S37
})


def _k1109_p6_allowlist_admit(M: int, N: int, K: int, dtype) -> bool:
    """K-1109 brief deliverable (1) — strict-equality safety-net allowlist.

    Returns True iff the env gate is set AND (M, N, K, dtype) is one of the
    6 K-1074 MFMA-issue-stall candidate cells whose paired n=30 timing did
    NOT fall entirely below the zero line (S32 and S37 carved out per
    reviewer consensus — see comment block above the allowlist). Default-OFF;
    see module-level docstring above the allowlist for the rationale.
    """
    if os.environ.get(_K1109_P6_ALLOWLIST_ENV) != "1":
        return False
    return (int(M), int(N), int(K), str(dtype)) in _K1109_P6_K1074_ALLOWLIST


def R_K1037_P6_admit_wpeu1(M: int, N: int, K: int, dtype) -> bool:
    """K-1089 — Structural surrogate of K-1037 P6 MFMA-issue-stall classifier.

    Returns True iff the (M, N, K, dtype) shape's structural fingerprint
    matches K-1037's paired-PMC P6 admit condition

        waves_per_CU_ratio (tb/hbl) < 1.70x
        AND MFMA_pct_tb >= 1.0%
        AND LDS_BC_cyc_per_inst_tb < 1.5

    K-1037 verified perfect 18/18 agreement of the 3-clause PMC predicate
    against K-1017's `route_action_table.csv` (TP=2/2 on S24, S29; TN=16/16
    on the remaining 15 occupancy + 1 HBM-L2-stall cells). K-1051 separately
    confirmed via 4-iter PMC research that wpeu=1 wins are predictable from
    2-3 counter clauses on K-1032 cells.

    K-1037's P6 reads paired runtime PMC (`pmc_view.tb` AND `pmc_view.hbl`)
    so it cannot run in a hot dispatch path (R-1037.A-PRIORI-PMC-CLASSIFIERS-
    LAND-AS-OFFLINE-CLASSIFIED-SHAPE-TABLES-NOT-LIVE-PMC-PREDICATES). This
    function regresses each PMC clause onto structural shape features so the
    same admit decision can be taken at dispatch time on the four fields
    (M, N, K, dtype) available without instrumentation.

    Surrogate-C2 (MFMA_pct_tb >= 1.0%):
        Translated to K >= 512. K-1037 P6 positives have MFMA% in
        [3.79, 4.79]; K-1017 cell S38 (K=200, MFMA%=0.07) is the only
        false-positive risk and is excluded by this floor.
    Surrogate-C3 (LDS_BC_cpi_tb < 1.5):
        Translated to "not in P5 LDS-bound regime". Composition: caller
        consults R_K979_P5_route_to_hbl first; this admit only fires on
        shapes NOT already silenced by P5 clauses 1+3 (LDS-port-contention
        and asymmetric-aspect axes).
    Surrogate-C1 (waves_per_CU_ratio < 1.70x — load-bearing):
        Two regression envelopes drawn from K-1017 + K-1032 + K-1044
        paired-PMC corpus:

        Envelope A (S29 wide-rect family):
            N == 2048 AND K == 1024 AND 13000 <= M <= 14999.
            Catches S29 (M=14208, ratio=1.51) only. Bounded above by S30
            (M=16256, ratio=1.75 — OCC) and below by K-984/K-989 LAND
            anchors S25 (M=6016), S27 (M=10112), S28 (M=12160) which all
            route-OUT cleanly under the existing P5 Clause-2 and gain
            5.31x-5.63x speedup; tightening to M>=13000 preserves those
            anchors while admitting only the K-1037-verified positive.
            (K-1044 B3b=(6016, 2048, 1024) collides with S25 anchor — it
            stays route-OUT under K-989's strict-equality table; the
            streamk-LAND verdict K-1044 reports is orthogonal to wpeu=1.)
        Envelope B (S24 + K-1044 B3a moderate-square mid-K family):
            minMN >= 2048 AND maxMN <= 4500 AND 512 <= K <= 1024.
            Catches S24 (4480, 3072, 768) (ratio=1.53). Bounded above by
            maxMN<=4500 to preserve K-984/K-989 LAND anchors (S18 maxMN=5972
            and S25 maxMN=6016 sit clear above).

    bf16-only by design (K-1037 / K-1017 / K-1044 ground truth scope).

    Composition with K-1062-style hardening: like P5 Clause-4's K-floor=512
    architectural bound, both envelopes are bounded BELOW by floors tied to
    the calibration set's lowest M/K positives. The ROUTE_OUT_HBL effect
    is delegated upstream to caller (matmul.py); this function only reports
    the structural P6 admit verdict.

    Calibration verification on K-1017 18-cell set:
        S24 -> True   (P6 positive, ratio=1.532)        ENVELOPE B
        S29 -> True   (P6 positive, ratio=1.514)        ENVELOPE A
        all 16 negatives -> False                       neither envelope

    K-1031 NO-LAND set (K-1017 cells where P5 does not fire) regression:
        S07 (2048, 1792, 256)  -> False (K=256<512 / N=1792)
        S16 (256, 256, 2048)   -> False (minMN=256<2048)
        S38 (384, 128, 200)    -> False (K=200<512)

    K-1104 NULL-RESULT (do not relax these envelopes):
        K-1098 4-iteration RESEARCH (primary-source-verified) showed that
        the 8 K-1074 candidate cells (S26, S31, S32, S33, S34, S35, S37,
        S39) fire P6 on **0/8** under K-1037's paired-PMC classifier --
        all 8 fail load-bearing C1 with waves_per_CU_ratio (tb/hbl) in
        [1.753, 2.740]. The K-1074 paired n=30 round-robin on c42/MI300X
        confirmed: 0/8 cells clear the K-901 +2% LAND margin under wpeu=1,
        2/8 (S32, S37) are confirmed regressions (CI95 hi < 0), cohort
        geomean ~= 1.000x. P5 Clause-2 already routes the K=1024 family
        OUT to hipBLASLt with 2.6x-3.1x measured speedup on K-989 anchors
        (S25, S27, S28), which is the structurally-correct production
        behaviour for those cells.

        Hard separation wall blocks any C1 relaxation: K-1037 NEG cell
        S30 (M=16256, waves_ratio=1.7530) and K-1074 candidate S34
        (M=24448, waves_ratio=1.7530) collide exactly on C1, so no cutoff
        in the (1.5320, 1.7530) feasibility band can admit S34 without
        producing a false-positive on S30.

        Envelope A's M-band [13000, 14999] is therefore architecturally
        bounded -- the upper bound (14999) sits below S30's M=16256, and
        the lower bound (13000) sits above K-984/K-989 LAND anchors
        S25/S27/S28 (M in {6016, 10112, 12160}) which carry verified
        2.6x-3.1x P5 Clause-2 routing speedup.

        References: K-1098 backtest (output/p6_clause_by_clause_k1074.csv),
        K-1074 paired n=30 (rr_summary.csv), K-1049/K-1055/K-1062 adversarial
        gate (FAIL: zero NEW NO-LAND-leak gain).

    K-1109 reproduction (productionizes the K-1066 lock-the-envelope pattern):
        Independently re-derived K-1037 P6 from the primary-source paired-
        PMC CSV for all 8 K-1074 cohort-extension cells + the 4 K-1044
        anchor cells called out in the K-1109 brief + 8 K-1017 held-out
        OCC/HBM negatives.  Result: 0/8 K-1074 FIRE (brief premise of 8/8
        FALSIFIED), 2/2 K-1037 P6 positives still FIRE (S24, S29), 0/8
        K-1017 NEG false positives, S30/S34 C1 collinearity wall confirmed
        at waves_ratio=1.7530.  K-1044 B3a (S18, 5972,1792,768) and B3b
        (S25, 6016,2048,1024) fire exact PMC P6 but the surrogate's
        Envelope-A M>=13000 floor and Envelope-B minMN>=2048 floor
        correctly preserve them as K-989 LAND anchors (5.31x-5.63x
        routing speedup).  Envelopes A and B are LOCKED -- pin tests in
        tests/test_k971_route_predicate.py (P6_NEGATIVES_K1074,
        P6_NEGATIVES_K1044_ANCHORS, test_p6_c1_collinearity_wall_*)
        gate any future relaxation through CI.

        Reproduction artifact: /home/ryaswann/mc2-workspaces/K-1109/output/
        k1109_p6_audit.csv (re-derivable via scripts/k1109_p6_verification.py).
    """
    if not _dtype_is_bf16(dtype):
        return False
    # K-1109 brief deliverable (1): strict-equality safety-net allowlist
    # (env-gated, default OFF). Consulted FIRST so it can admit cells that
    # the structural envelopes reject (the K-1074 cohort fails C1 by design,
    # so it would never fire through the envelope path below).
    if _k1109_p6_allowlist_admit(M, N, K, dtype):
        return True
    if K < 512:
        return False  # Surrogate-C2: insufficient MFMA work density
    minMN, maxMN = min(M, N), max(M, N)
    # Envelope A: K-931 wide-rect (N=2048, K=1024) family at moderate M.
    # K-1017 paired ratios: S26 M=8064 (2.03x — OCC), S29 M=14208 (1.51x —
    # P6+), S30 M=16256 (1.75x — OCC). K-984/K-989 LAND anchors S25 M=6016
    # / S27 M=10112 / S28 M=12160 sit clear below the M>=13000 floor and
    # remain route-OUT under P5 Clause-2 with 5.31x-5.63x routing speedup.
    if N == 2048 and K == 1024 and 13000 <= M <= 14999:
        return True
    # Envelope B: moderate-square mid-K family (S24 fingerprint).
    # Ceiling maxMN<=4500 preserves K-984 anchor S18 (5972,1792,768) and
    # K-989 anchor S25 (6016,2048,1024) which both sit clear above.
    if 2048 <= minMN and maxMN <= 4500 and 512 <= K <= 1024:
        return True
    return False


# ---------------------------------------------------------------------------
# K-1144 P8 — MFMA-issue-stall direct hipBLASLt route-OUT (productionizes
# K-1121 + K-1131).
#
# Brief context: K-1074 paired n=30 LAND audit on c42/MI300X falsified the
# K-1089 wpeu=1 in-kernel admit for the 13-cell MFMA-issue-stall cohort
# (5 K-1051 + 8 K-1031 leakage); 0/13 cells cleared the K-901 +2% margin
# (2/13 — S32, S37 — were CI95-confirmed regressions, see K-1109's carve-
# out documentation).  K-1121 measured paired n=30 HIP-graph hot-cache on
# the same 13 cells with `OFF=tritonblas`/`ON=hipBLASLt` and reported
# 13/13 admit at hbl/tb >= 1.10x (cohort geomean 1.226x, range
# 1.158x-1.365x; CI95-lower clears the K-1007 >=1.05x admission floor on
# every cell).  K-1127 then adversarially backtested the 13-cell envelope
# against the K-931 always-uncovered top-40 production catalog and reported
# 0 false positives at n=20.  K-1131 perturbed the 13 anchors along
# +/-1 power-of-2 in M, N, K and admitted all 12/12 neighbors at the
# stricter generalization gate (ratio >= 1.15x AND CI95-lo > 1.00; geomean
# 1.367x, max 1.657x, scatter 0.0%).  Triple-validated: original cohort
# (K-1121) + adversarial backtest (K-1127) + neighbor envelope (K-1131).
#
# Composition with the existing predicate stack (no overlap, no precedence
# inversion):
#   * P8 fires BEFORE the K-1089 R_K1037_P6_admit_wpeu1 admit-back-to-kernel
#     check, because S24 (4480, 3072, 768) and S29 (14208, 2048, 1024) sit
#     inside K-1089 Envelopes B and A respectively and would otherwise be
#     short-circuited back to in-kernel by P6 -- but K-1074's paired n=30
#     evidence falsified that admit and K-1121's paired n=30 confirms
#     hipBLASLt wins on these exact cells (S24 ratio ~1.20x, S29 ratio
#     ~1.22x).  K-1131 neighbor N11 (2240, 3072, 768) is also inside
#     P6 Envelope B and is similarly reverted to route-OUT by P8.  This
#     is the "no precedence inversion vs K-1089" composition rule
#     codified.
#   * P8 is independent of (composes safely with) K-1062 P5 Clause-4
#     K-floor=512: P8 cells either don't hit Clause-4 at all (none use
#     N=1792 with 512<=K<=768 except S18, see below) or already route-OUT
#     via Clause-4 (S18 = 5972, 1792, 768; K=768 is in-band).  Strict-
#     equality dispatch on P8 returns True regardless, so even on overlap
#     the routing decision is identical -- no double-routing.
#   * P8 is also independent of K-1066 R-K979 P5 generalized clauses.
#     Cells covered by both P5 (Clause-2 large-M skinny) and P8 simply
#     return True via whichever check fires first; the dispatch outcome
#     is unchanged ("route to hipBLASLt").  Strict-equality is a frozenset
#     hash; lookup cost is O(1) per K-1060 round-robin harness analysis.
#   * P8 is also independent of the K-1109 env-gated allowlist.  The
#     K-1109 allowlist (default OFF) admits 6 of the K-1074 cells back
#     to in-kernel; with the env unset (production default) those cells
#     route via P5 Clause-2.  P8 makes the route-OUT explicit and does
#     not depend on the K-1109 env state.  When the K-1109 env IS set,
#     P8's BEFORE-P6 placement still wins -- the brief's "no double-
#     routing" guarantee holds in both env states.
#
# Anti-overshoot discipline (R-1131): the route table is STRICT-EQUALITY
# only.  K-1131 deliberately did NOT widen to a parametric envelope despite
# 12/12 + zero-scatter neighbor admission, because the K-1097->K-1115 PMC
# classifier exhaustion chain forecloses any non-strict-equality predicate
# on this cohort that hasn't survived held-out validation.  Future shape-
# class consolidation (R-1121.SHAPE-CLASS-PREDICATE-CANDIDATE-WHEN-COHORT-
# SHARES-NK-SIGNATURE-WITH-M-SWEEP) is filed as a follow-up; it must
# re-derive thresholds from already-admitted P8 cells (not raw counter
# data) to avoid re-opening the predicate-fit trap.
#
# 25 cells = 13 K-1121 anchors + 12 K-1131 neighbors, all bf16.
# K-1121 admission gate: hbl/tb >= 1.10x mean AND CI95-lo >= 1.05x (K-1007).
# K-1131 generalization gate: ratio >= 1.15x AND CI95-lo > 1.00.
# All cells satisfy both gates per source manifests.
# ---------------------------------------------------------------------------

# K-1121 anchors (5 K-1051 PMC counter-matrix cells + 8 K-1031 leakage
# cohort cells).  Per-cell mean speedup (hbl/tb) annotated from K-1121's
# paired n=30 HIP-graph hot-cache on rad-mi300x-1 / MI300X / ROCm 7.2;
# range 1.158x-1.365x; cohort geomean 1.226x.
_K1121_P8_ANCHORS_13 = frozenset({
    # ----- K-1051 PMC counter-matrix cells (overlap with K-1089 P6) -----
    ( 4480, 3072,  768, "torch.bfloat16"),  # S24  ~1.20x  (P6 Env-B overlap)
    (14208, 2048, 1024, "torch.bfloat16"),  # S29  ~1.22x  (P6 Env-A overlap)
    ( 5972, 1792,  768, "torch.bfloat16"),  # S18  ~1.16x  (P5 C4 K=768)
    ( 6016, 2048, 1024, "torch.bfloat16"),  # S25  ~1.21x  (P5 C2)
    (16256, 2048, 1024, "torch.bfloat16"),  # S30  ~1.24x  (P5 C2)
    # ----- K-1031 leakage MFMA-issue-stall candidates -----
    ( 8064, 2048, 1024, "torch.bfloat16"),  # S26  ~1.22x  (P5 C2)
    (18304, 2048, 1024, "torch.bfloat16"),  # S31  ~1.27x  (P5 C2)
    (20352, 2048, 1024, "torch.bfloat16"),  # S32  ~1.25x  (P5 C2; K-1109 carve-out)
    (22400, 2048, 1024, "torch.bfloat16"),  # S33  ~1.21x  (P5 C2)
    (24448, 2048, 1024, "torch.bfloat16"),  # S34  ~1.19x  (P5 C2)
    (26496, 2048, 1024, "torch.bfloat16"),  # S35  ~1.24x  (P5 C2)
    (25600, 2048,  256, "torch.bfloat16"),  # S37  ~1.30x  (P5 C2; K-1109 carve-out)
    (49152, 2048,  256, "torch.bfloat16"),  # S39  ~1.37x  (P5 C2)
})

# K-1131 held-out neighbors (12 cells perturbed +/-1 power-of-2 from 4
# representative anchors: S32 peak, S39 extreme-M / shallow-K, S18 only-
# non-2K-N, S24 only-N=3072).  Per-cell mean speedup (hbl/tb) annotated
# from K-1131's paired n=30 + 10k-resample paired bootstrap on rad-mi300x-1;
# range 1.15x-1.657x; cohort geomean 1.367x; envelope-edge cells N04/N06.
_K1131_P8_NEIGHBORS_12 = frozenset({
    # ----- S32 (20352, 2048, 1024) family — full 6-axis coverage -----
    (20352, 2048,  512, "torch.bfloat16"),  # N01 S32 K/2  IN_COHORT
    (20352, 2048, 2048, "torch.bfloat16"),  # N02 S32 K*2  IN_COHORT
    (40704, 2048, 1024, "torch.bfloat16"),  # N03 S32 M*2  IN_COHORT
    (10176, 2048, 1024, "torch.bfloat16"),  # N04 S32 M/2  ratio 1.604x (envelope edge)
    (20352, 4096, 1024, "torch.bfloat16"),  # N05 S32 N*2  IN_COHORT
    (20352, 1024, 1024, "torch.bfloat16"),  # N06 S32 N/2  ratio 1.657x (cohort MAX)
    # ----- S39 (49152, 2048, 256) family — K-axis only (extreme-M) -----
    (49152, 2048,  128, "torch.bfloat16"),  # N07 S39 K/2  IN_COHORT
    (49152, 2048,  512, "torch.bfloat16"),  # N08 S39 K*2  IN_COHORT
    # ----- S18 (5972, 1792, 768) family — N-axis only (only non-2K N) -----
    ( 5972,  896,  768, "torch.bfloat16"),  # N09 S18 N/2  IN_COHORT
    ( 5972, 3584,  768, "torch.bfloat16"),  # N10 S18 N*2  IN_COHORT
    # ----- S24 (4480, 3072, 768) family — M/2 + K*2 (only N=3072) -----
    ( 2240, 3072,  768, "torch.bfloat16"),  # N11 S24 M/2  IN_COHORT (P6 Env-B overlap)
    ( 4480, 3072, 1536, "torch.bfloat16"),  # N12 S24 K*2  IN_COHORT
})

# K-1175 / K-1161 E2 axis extension admits.  K-1161 paired n=30 HIP-graph
# hot-cache validation of the K-axis extension envelope
#   E2 := N in {1792, 2048, 3072} AND K in {96, 128, 192, 256, 512, 768, 1024}
# returned a **NEGATIVE_AXIS_PIVOT** verdict overall (3/8 = 37.5% admit,
# below the 50% pivot threshold; well below the 80% landing threshold).
# In particular:
#   - K-floor relaxation (K < 256: E2_K1=512x2048x96, E2_K2=512x2048x192):
#       0/2 admit -- both are TB-favoured (small-M Triton-win tail).
#   - K=512 interior fill (E2_I1=192x2048x512, E2_I2=156x1792x512,
#       E2_I3=160x3072x512): 0/3 admit -- all TB-favoured (small-M tail).
#   - K=736 interior fill (E2_I4=736x1792x736): 1/1 admits at hbl/tb=1.315x
#       CI95=[1.313, 1.317].  Sole K-interior route-OUT-safe cell.
#   - M-axis extension at K=1024 (E2_M1=10112x2048x1024,
#       E2_M2=12160x2048x1024, both at K-1121 anchor K-set): 2/2 admit at
#       hbl/tb 1.759x and 1.499x respectively.  Orthogonal axis, not in
#       K-floor=128 / K=512 scope but K-1161 uncovered.
#
# Per K-1161 R-1161.K-FLOOR-RELAXATION-SURFACES-TB-FAVOURED-SMALL-M-TAIL:
# the K-1121 MFMA-issue-stall cohort's small-M corner (M <= 512) of the
# (M, K) plane is a Triton win region; K-1142's (M >= 4480) carve-out
# already excluded these cells by M-floor regardless of K-floor strictness.
# K-floor=128 is NOT hipBLASLt-favourable for this regime; only the M-axis
# extension and the K=736 single-cell pin are landed here.  The K-floor
# of any future structural extension MUST remain at K >= 256 unless paired
# with a strict M-floor strictly above the small-M Triton-favoured tail.
#
# Strict-equality only (R-1131): no parametric envelope -- each cell was
# directly measured at paired n=30 with hbl/tb >= 1.315x and CI95-lo > 1.05x.
_K1161_E2_ADMITS_3 = frozenset({
    # ----- K-interior single-cell admit (only K-axis cell that survived) -----
    (  736, 1792,  736, "torch.bfloat16"),  # E2_I4 hbl/tb=1.315x CI95=[1.313, 1.317] (K=736 upper edge)
    # ----- M-axis extension at K-1121 anchor K=1024 (orthogonal axis) -----
    (10112, 2048, 1024, "torch.bfloat16"),  # E2_M1 hbl/tb=1.759x CI95=[1.753, 1.770]
    (12160, 2048, 1024, "torch.bfloat16"),  # E2_M2 hbl/tb=1.499x CI95=[1.496, 1.504]
})

# K-1231 / K-1205 E_N3 N-axis extension admits.  Productionised on top of
# K-1216's stacked dispatch chain (fix/K-1209-stacked @ fb9461f).  Paired
# n=30 HIP-graph hot-cache validation on rad-mi300x-1 fallback (c42 SSH
# plane outage continued through K-1216's 10th recurrence and was still
# refused on the K-1231 verification window) of the N-floor relaxation:
#   E_N3 := N = 128 (one power-of-2 tier below the natural sub-1024 N floor)
#           AND K in {256, 768, 1024} (K-1144 K-set held fixed -- NOT a
#               K-axis relaxation; K-1161/K-1176 already established
#               K-floor < 256 is NEGATIVE on this cohort)
#           AND M in K-1121 anchor M-set (mirrored projection per K-1131)
#           AND bf16
# returned a clean POSITIVE landing verdict: 8/9 = 88.9% admit, well above
# the 75% landing threshold, and decisively diverging from K-1161's K-axis
# NEGATIVE (1/6 = 16.7%) on adjacent cells / same hardware / same protocol
# -- empirically REFUTING K-1199's analytical (M<->N) symmetry prediction.
#
# Mechanistic reading: at extreme aspect ratios with M >> N=128, hipBLASLt's
# Tensile schedule (wide unroll + many waves) hides the same MFMA-issue-
# stall bottleneck more aggressively than at K-1121 mid-square N=2048 tiles
# -- per-cell speedups widen 1.14x-3.25x at N=128 vs 1.16x-1.27x at N=2048.
# The single rejection EN_K768_S24 (M=4480 K=768 N=128 bf16 hbl/tb=0.967x
# CI95=[0.94, 1.00]) sits exactly at the K-1142 M-floor; the same small-M
# Triton-favoured tail surfaced by K-1161 along the K-axis is active here
# at the M-floor of the K-1121 anchor M-set (R-1205.SMALL-M-TRITON-
# FAVOURED-TAIL-IS-AXIS-AGNOSTIC-AT-COHORT-M-FLOOR).  The strict-equality
# posture EXCLUDES this false positive by construction (carve-out by
# omission); see K1205_EN3_REJECTED_1_LIST in the pin tests for an
# explicit no-leak regression trap.
#
# Strict-equality only (R-1131 / R-1175): no parametric envelope -- each
# cell was directly measured at paired n=30 with hbl/tb >= 1.144x and
# CI95-lo > 1.05x.  N=128 keeps the four sub-frozensets pairwise disjoint
# by construction (no prior P8 cell has N < 896).
_K1205_EN3_ADMITS_8 = frozenset({
    # ----- N=128 K=1024 cluster (5 admits; speedups 1.37x-3.25x) -----
    ( 6016,  128, 1024, "torch.bfloat16"),  # EN_K1024_S25 hbl/tb=1.370x CI95=[1.362, 1.374]
    ( 8064,  128, 1024, "torch.bfloat16"),  # EN_K1024_S26 hbl/tb=1.402x CI95=[1.396, 1.415]
    (14208,  128, 1024, "torch.bfloat16"),  # EN_K1024_S29 hbl/tb=3.254x CI95=[3.234, 3.267] (cohort MAX)
    (16256,  128, 1024, "torch.bfloat16"),  # EN_K1024_S30 hbl/tb=3.024x CI95=[3.011, 3.056]
    (22400,  128, 1024, "torch.bfloat16"),  # EN_K1024_S33 hbl/tb=2.258x CI95=[2.234, 2.269]
    # ----- N=128 K=256 cluster (2 admits; extreme-M; speedups 1.14x-1.78x) -----
    (25600,  128,  256, "torch.bfloat16"),  # EN_K256_S37  hbl/tb=1.144x CI95=[1.140, 1.149] (cohort MIN admit)
    (49152,  128,  256, "torch.bfloat16"),  # EN_K256_S39  hbl/tb=1.781x CI95=[1.768, 1.803]
    # ----- N=128 K=768 cluster (1 admit; M=4480 carve-out via omission) -----
    ( 5972,  128,  768, "torch.bfloat16"),  # EN_K768_S18  hbl/tb=1.227x CI95=[1.226, 1.227]
    # NOTE: EN_K768_S24 (M=4480 N=128 K=768 bf16 hbl/tb=0.967x) is
    # DELIBERATELY OMITTED -- this is the K-1205 N-axis false-positive
    # carve-out at the K-1142 M-floor.  See K1205_EN3_REJECTED_1_LIST
    # in tests/test_k971_route_predicate.py for the no-leak pin.
})

# ---------------------------------------------------------------------------
# K-1219 / K-1240 -- E3 N-floor=256 anchor-projected admits (7 cells).
# K-1131-style +/-1-power-of-2 N-axis projection of K-1121 PMC anchors
# {S18, S24, S25, S30, S37, S39} + K-1175 E2_M1 admit, projected to
# N=256 (one tier below the K-1131 N09 N=896 natural floor).  Per K-1142
# carve-out, M >= 4480 on every cell.  Validated independently in K-1219
# (commit f110d64 on fix/K-1219-n256, branch productionised pre-K-1282)
# at paired n=30 HIP-graph hot-cache MI300X with cohort geomean
# tb-over-hbl 1.505x; admitted to the unified K-1282 P8 envelope.
#
# K-1303 (S-002) layers this set onto the K-1275 baseline (K-1216 stacked
# dispatch chain @ fb9461f extended by K-1227's _K1205_EN3_ADMITS_8 to
# 36 cells).  N=256 sits on a disjoint N-axis tier below the K-1131
# N09 N=896 natural floor and above K-1227's N=128 tier, so the union
# is pairwise-disjoint with all four prior provenance frozensets by
# construction (asserted below).
# ---------------------------------------------------------------------------
_K1219_E3_NFLOOR256_ADMITS_7 = frozenset({
    # ----- E3 N=256 anchor-projected admits (M >= 4480 per K-1142) -----
    ( 5972,  256,  768, "torch.bfloat16"),  # E3_S18_N256
    ( 4480,  256,  768, "torch.bfloat16"),  # E3_S24_N256
    ( 6016,  256, 1024, "torch.bfloat16"),  # E3_S25_N256  MI300X tb/hbl 1.215 CI95=[1.206,1.224]
    (16256,  256, 1024, "torch.bfloat16"),  # E3_S30_N256  MI300X tb/hbl 2.274 CI95=[2.260,2.291]
    (25600,  256,  256, "torch.bfloat16"),  # E3_S37_N256
    (49152,  256,  256, "torch.bfloat16"),  # E3_S39_N256
    (10112,  256, 1024, "torch.bfloat16"),  # E3_E2M1_N256 MI300X tb/hbl 2.926 CI95=[2.911,2.942]
})

# Composed 43-cell P8 envelope (K-1303 unified).  Five named provenance
# frozensets (K-1121 anchors, K-1131 neighbors, K-1175/K-1161 E2 admits,
# K-1231/K-1205 E_N3 N=128 admits, K-1219 E3 N=256 admits) -- the
# dispatch path consults the union.  Per R-1144.DUAL-FROZENSET-PROVENANCE
# and its K-1175 / K-1205 / K-1219 extensions, source-ticket lineage is
# load-bearing for future reviewers (precedence-inversion debugging,
# PMC re-classifier work, ADR audits) so each measurement campaign keeps
# its own named set with a runtime size + pairwise-disjointness check.
_P8_MFMA_ISSUE_STALL_ROUTEOUT = (
    _K1121_P8_ANCHORS_13
    | _K1131_P8_NEIGHBORS_12
    | _K1161_E2_ADMITS_3
    | _K1205_EN3_ADMITS_8
    | _K1219_E3_NFLOOR256_ADMITS_7
)
assert len(_P8_MFMA_ISSUE_STALL_ROUTEOUT) == 43, (
    "K-1303 unified P8 envelope must be exactly 43 cells (13 K-1121 "
    "anchors + 12 K-1131 neighbors + 3 K-1161 E2 admits + 8 K-1205 E_N3 "
    "N=128 admits + 7 K-1219 E3 N=256 admits); a duplicate or stray "
    "entry has crept in.")
# Cross-check: the five sub-sets must be pairwise disjoint by construction.
# K-1131 perturbed AWAY from K-1121 anchors; K-1161 E2 admits were
# selected from the K-931 always-uncovered top-40 catalog minus all
# K-1121 anchors and minus all K-1131 neighbors; K-1205 E_N3 admits use
# N=128 (no prior P8 cell has N < 896 -- disjointness by N-axis projection);
# K-1219 E3 admits use N=256 (disjoint from N=128 K-1205 by construction
# and below the K-1131 N09 N=896 natural floor).
assert _K1121_P8_ANCHORS_13.isdisjoint(_K1131_P8_NEIGHBORS_12), (
    "K-1144 P8 anchors and neighbors overlap; K-1131 neighbor generation "
    "rules require strict disjointness from the 13 K-1121 anchors.")
assert _K1121_P8_ANCHORS_13.isdisjoint(_K1161_E2_ADMITS_3), (
    "K-1175 K-1161 E2 admits overlap with K-1121 anchors; the K-1161 "
    "candidate generator excluded all K-1121 anchors by construction.")
assert _K1131_P8_NEIGHBORS_12.isdisjoint(_K1161_E2_ADMITS_3), (
    "K-1175 K-1161 E2 admits overlap with K-1131 neighbors; the K-1161 "
    "candidate generator excluded all K-1131 neighbors by construction.")
assert _K1205_EN3_ADMITS_8.isdisjoint(_K1121_P8_ANCHORS_13), (
    "K-1205 E_N3 admits overlap with K-1121 anchors; E_N3 candidates "
    "have N=128 by construction and no K-1121 anchor has N=128.")
assert _K1205_EN3_ADMITS_8.isdisjoint(_K1131_P8_NEIGHBORS_12), (
    "K-1205 E_N3 admits overlap with K-1131 neighbors; E_N3 candidates "
    "have N=128 by construction and no K-1131 neighbor has N=128.")
assert _K1205_EN3_ADMITS_8.isdisjoint(_K1161_E2_ADMITS_3), (
    "K-1205 E_N3 admits overlap with K-1161 E2 admits; E_N3 candidates "
    "have N=128 by construction and no K-1161 E2 admit has N=128.")
# K-1219 (N=256) cross-disjointness with the prior 36-cell envelope.
assert _K1219_E3_NFLOOR256_ADMITS_7.isdisjoint(_K1121_P8_ANCHORS_13), (
    "K-1219 E3 N=256 admit overlaps a K-1121 anchor; the K-1131-style "
    "anchor projection excluded all K-1121 anchors by construction.")
assert _K1219_E3_NFLOOR256_ADMITS_7.isdisjoint(_K1131_P8_NEIGHBORS_12), (
    "K-1219 E3 N=256 admit overlaps a K-1131 neighbor; the K-1131 "
    "perturbation set's smallest N is 896, well above the N=256 tier.")
assert _K1219_E3_NFLOOR256_ADMITS_7.isdisjoint(_K1161_E2_ADMITS_3), (
    "K-1219 E3 N=256 admit overlaps a K-1161 E2 admit; the K-1161 "
    "K-axis admits all sit at N in {1792, 2048}, not N=256.")
# K-1303 cross-band disjointness: K-1227 N=128 vs K-1219 N=256.
assert _K1219_E3_NFLOOR256_ADMITS_7.isdisjoint(_K1205_EN3_ADMITS_8), (
    "K-1303 composition error: K-1219 N=256 admits overlap K-1205 E_N3 "
    "N=128 admits.  The two campaigns operate on disjoint N axes by "
    "construction; an overlap indicates an authoring typo in one of "
    "the two frozensets.")


# ---------------------------------------------------------------------------
# K-1209-stacked / K-1216 (S-002): E1 axis-aligned envelope productionised
# from K-1151 / K-1142 stacked AFTER the P8 28-cell strict-equality route.
# K-1209 ablation on K-931 always-uncovered top-40 confirmed E1 is fully
# dominated by P8+K-1175 on this cohort (0 marginal cells, 0 marginal FPs);
# E1 is retained as defense-in-depth for cohorts beyond K-931 where the
# audit chain has not yet enumerated every K-1142-envelope-admittable cell.
# E1 ships with the K-1142 (M >= 4480) ∧ (K >= 256) carve-out which holds
# at 0 FPs across all 4 stacked configurations on K-931 top-40 (K-1209).
# E2 K-floor=128 is EXCLUDED per K-1176 cross-arch failure (0/8 cells on
# MI325X/MI355X) — no K-axis relaxation past K=256.
# ---------------------------------------------------------------------------
K1142_E1_NS = frozenset({1792, 2048, 3072})
K1142_E1_KS = frozenset({256, 768, 1024})
K1142_E1_M_FLOOR = 4480  # K-1121 anchor S24's M (margin 0; excludes FP1=256x2048x256)
K1142_E1_K_FLOOR = 256   # K-1161 NEGATIVE_AXIS_PIVOT pin; do NOT relax to 128


def R_K1142_E1_route_to_hbl(M: int, N: int, K: int, dtype) -> bool:
    """K-1151 E1 envelope route-OUT — bf16 ∧ N∈{1792,2048,3072} ∧
    K∈{256,768,1024} ∧ M≥4480 ∧ K≥256. Stacked AFTER P8 in K-1209-stacked
    so P8 wins on overlap (no double-routing); E1 only fires on cells P8
    does not already enumerate. K-1209 confirmed 0 marginal coverage on
    K-931 top-40 beyond P8+K-1175 — E1 is defense-in-depth for non-K-931
    catalogs where the audit chain has not converged.
    """
    if not _dtype_is_bf16(dtype):
        return False
    if N not in K1142_E1_NS or K not in K1142_E1_KS:
        return False
    if M < K1142_E1_M_FLOOR or K < K1142_E1_K_FLOOR:
        return False
    return True


def _p8_mfma_issue_stall_routeout(M: int, N: int, K: int, dtype) -> bool:
    """K-1144 P8 (extended by K-1175 / K-1231-K-1205 / K-1219 -- unified
    in K-1303) — direct hipBLASLt route-OUT for the quintuply-validated
    MFMA-issue-stall cohort (K-1121 anchors + K-1131 neighbors +
    K-1175/K-1161 E2 admits + K-1231/K-1205 E_N3 N=128 admits +
    K-1219 E3 N=256 admits).

    Returns True iff (M, N, K, dtype) matches one of the 43 strict-equality
    keys in :data:`_P8_MFMA_ISSUE_STALL_ROUTEOUT`.  bf16-only by design
    (the entire K-1121 / K-1131 source measurement scope is bf16; fp16
    parity is tracked separately on the K-1093 / K-1125 line).

    This predicate is consulted **before** the K-1089 P6 admit-back-to-
    kernel check inside ``_k971_route_to_hbl`` so K-1121's measurement
    evidence (hipBLASLt wins on S24, S29 at ~1.20x, ~1.22x respectively)
    overrides the K-1089 envelope admit for the small subset of cells
    where the two envelopes overlap (S24 and S29 inside P6 Env-B and
    Env-A; K-1131 N11 inside P6 Env-B).  K-1074's paired n=30 LAND audit
    plus K-1098's clause-by-clause backtest established that the K-1089
    admit was falsified for these cells -- P8 codifies the correction.

    For cells that are NOT in any P6 envelope, P8's strict-equality match
    is functionally identical to the existing P5 Clause-2 route-OUT
    decision (both return True -> route to hipBLASLt).  The dispatch
    outcome is unchanged for those cells; P8 just makes the routing
    rationale source-of-truth attributable to the K-1121 + K-1131
    measurement campaign rather than P5's structural pathology heuristic.
    """
    if not _dtype_is_bf16(dtype):
        return False
    return (int(M), int(N), int(K), str(dtype)) in _P8_MFMA_ISSUE_STALL_ROUTEOUT


def k971_route_decision(M, N, K, a_dtype, b_dtype, enable_streamk,
                        work_stealing, disable_env_set: bool = False) -> bool:
    """Pure routing decision — same logic as ``matmul._k971_route_to_hbl``
    but without reading the environment (caller passes ``disable_env_set``).

    Useful for tests that want to exercise the *full* dispatch decision
    (predicate + strict-equality + streamk/work-stealing/dtype carve-outs)
    without monkey-patching ``os.environ``.

    Precedence (top-down, first match wins):
      1. K-1144 P8 strict-equality MFMA-issue-stall route-OUT (overrides
         K-1089 P6 admit for S24/S29/N11; identity for cells already
         routed by P5; see _p8_mfma_issue_stall_routeout docstring).
      2. K-1089 P6 admit gate -> in-kernel wpeu=1 dispatch (returns False
         from this function so caller falls through to in-kernel).
      3. K-1066 R-K979 P5 closed-form structural pathology -> hipBLASLt.
      4. K-905 / K-971 strict-equality anchor table -> hipBLASLt.
    """
    if disable_env_set:
        return False
    if enable_streamk or work_stealing or str(a_dtype) != str(b_dtype):
        return False
    # K-1144 + K-1175 + K-1231/K-1205 + K-1219 (K-1303 unified): P8
    # 43-cell strict-equality (13 K-1121 anchors + 12 K-1131 neighbors +
    # 3 K-1161 E2 admits + 8 K-1205 E_N3 N=128 admits + 7 K-1219 E3
    # N=256 admits) takes precedence over P6 admit so K-1121's paired
    # n=30 evidence overrides K-1089 envelope admit for the S24/S29/N11
    # overlap.  K-1205's 8 N=128 and K-1219's 7 N=256 additions are
    # disjoint from K-1089 P6 envelope (P6 has no N<896 cells), so the
    # precedence-inversion question is unchanged.
    if _p8_mfma_issue_stall_routeout(int(M), int(N), int(K), a_dtype):
        return True
    # K-1209-stacked / K-1216: E1 axis-aligned envelope as defense-in-depth
    # AFTER P8. K-1209 ablation showed E1 is fully dominated by P8+K-1175 on
    # K-931 top-40 (0 marginal cells / 0 marginal FPs); E1 only fires on
    # non-K-931 cohort cells the K-1142->K-1161->K-1175 audit chain has not
    # yet enumerated. E2 K-floor=128 EXCLUDED (K-1176 cross-arch failure).
    if R_K1142_E1_route_to_hbl(int(M), int(N), int(K), a_dtype):
        return True
    # K-1089: P6 admit overrides P5 route-OUT for MFMA-stall structural
    # signature; cells fall through to in-kernel dispatch.
    if R_K1037_P6_admit_wpeu1(int(M), int(N), int(K), a_dtype):
        return False
    if R_K979_P5_route_to_hbl(int(M), int(N), int(K), a_dtype):
        return True
    return (int(M), int(N), int(K), str(a_dtype)) in K971_ROUTE_TABLE
