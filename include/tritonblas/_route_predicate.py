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
_K905_K971_LDS_BC_ANCHORS_8 = frozenset({
    (1024, 1024, 16384, "torch.bfloat16"),  # K-905 baseline
    (1024, 1024, 16384, "torch.float16"),
    (1024, 1024, 32768, "torch.bfloat16"),  # K-971 (K-905 N4: hbl/off=1.73x)
    (1024, 1024, 32768, "torch.float16"),
    (2048, 2048, 16384, "torch.bfloat16"),  # K-971 (K-905 N2: hbl/off=1.37x)
    (2048, 2048, 16384, "torch.float16"),
    (2048, 2048, 32768, "torch.bfloat16"),  # K-971 (K-905 N5: hbl/off=1.38x)
    (2048, 2048, 32768, "torch.float16"),
})

# ---------------------------------------------------------------------------
# K-1335 / K-1326 (S-002) — longK_smallSquare bf16 route-OUT extension
# (the 5th-position dispatch predicate; K971_ROUTE_TABLE).
#
# K-1335 productionizes K-1326's `P9_longK_smallSquare_routeOUT` proposal as
# a 4-cell strict-equality extension to the K-905/K-971 LDS-bank-conflict
# (LDS-BC) anchor family.  Cells were identified by K-1308's per-shape
# residual decomposition of the K-1322 51-cell P8 envelope sweep:
# `longK_smallSquare` (M=N∈{1024, 2048}, K∈{4096, 8192}, bf16; 4 cells,
# residual mean 0.695, aggregate gap 2.779) is the worst residual bucket
# AND zero admit predicate fires on any of these cells (any-pred-fired = N
# for the bucket).  K-1326 then attributed the bottleneck via PMC features
# carried over from the K-913 baseline triage:
#
#   LDS_BC      1.78 cyc/inst  (vs 0.00 on hbl baseline)
#   MFMA%       9.33%          (vs 15.85% on hbl baseline)
#   SQ_INSTS_LDS  4.00x inflated vs hbl
#
# This is the same LDS-bank-conflict fingerprint as K-905 / K-930 / K-971
# (mechanism-shape-invariant per K-913 §3, ±19% across the 4-cell
# neighborhood) — NOT the K-1144 P8 MFMA-issue-stall pathology (different
# counter signature).  Per R-1329.MECHANISTIC-DISAMBIGUATION-OF-PREDICATE-
# ATTACHMENT-POINT, the attachment site must match the mechanism, so the 4
# K-1335 admits attach to the K-905/K-971 LDS-BC envelope (the 5th-position
# dispatch predicate `K971_ROUTE_TABLE`), NOT to the K-1322 51-cell P8
# `_P8_MFMA_ISSUE_STALL_ROUTEOUT` envelope.  The K-1322 P8 envelope and its
# 6 named sub-frozensets remain byte-identical; the dispatch ladder
# precedence is unchanged.
#
# The autotune in-kernel path has converged on a single config that is
# structurally LDS-bank-conflict-bound — K-1308's cross-branch ratio
# invariant (4-pred ratio == main ratio within paired-n30 ±3% CV) holds
# on these 4 cells, ruling out any (BM, BN, BK, WPEU, NS, NW) catalog
# perturbation that could close the gap.  Route-OUT is the only remaining
# mechanism (R-1329.CROSS-BRANCH-RATIO-INVARIANT-IMPLIES-ROUTE-OUT-ONLY).
#
# Verification (paired n=30 HIP-graph hot-cache, B=10000 paired bootstrap,
# MI300X / gfx942 dedicated, 4 admit cells + 4 K-axis-neighbor controls):
#   - 4-cell admit-set geomean: ~1.30x route-OUT win (CI95 strictly > 1.0)
#   - 4-cell neighbor controls: no significant regression (CI95-hi >= 0)
# Per R-1329.ADMIT-ON-DEDICATED-GPU-NOT-COTENANT, paired n=30 selection
# uses dedicated-GPU measurement; co-tenant measurements admissible only
# as the regression-firewall lower bound.
#
# Disjointness (K-axis projection per R-1329.K-AXIS-PROJECTION-DISJOINTNESS-
# ASSERTS-ARE-CHEAP-INSURANCE): K-1335 K∈{4096, 8192}; K-905/K-971
# K∈{16384, 32768} — natural K-axis separator, asserted at module load.
# bf16-only by design (K-913 anchor + K-1326 bucket are bf16; fp16 parity
# tracked separately on the K-1093 / K-1125 line).
# ---------------------------------------------------------------------------
_K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4 = frozenset({
    (1024, 1024,  4096, "torch.bfloat16"),  # K-1326 A1; K-913 PMC anchor neighborhood (LDS_BC 1.78 cyc/inst)
    (1024, 1024,  8192, "torch.bfloat16"),  # K-1326 A2; K-913 PMC anchor (1024,1024,8192) bf16
    (2048, 2048,  4096, "torch.bfloat16"),  # K-1326 A3; same LDS-BC fingerprint per K-913 §3 ±19%
    (2048, 2048,  8192, "torch.bfloat16"),  # K-1326 A4; same LDS-BC fingerprint per K-913 §3 ±19%
})

# Composed K-905/K-971/K-1335 LDS-BC route-OUT table (the 5th-position
# dispatch predicate).  Two named provenance frozensets; consulted as a
# strict-equality union by `_k971_route_to_hbl` after P8 / E1 / P6 / P5.
# Per R-1144.DUAL-FROZENSET-PROVENANCE (lifted to the LDS-BC envelope per
# R-1329.MECHANISTIC-DISAMBIGUATION-OF-PREDICATE-ATTACHMENT-POINT), each
# measurement campaign keeps its own named set with a runtime cardinality
# pin and a K-axis-projection disjointness assert.
K971_ROUTE_TABLE = (
    _K905_K971_LDS_BC_ANCHORS_8
    | _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4
)
assert len(K971_ROUTE_TABLE) == 12, (
    "K-1335 unified LDS-BC route-OUT table must be exactly 12 cells "
    "(8 K-905/K-971 anchors + 4 K-1335 longK_smallSquare admits); a "
    "duplicate or stray entry has crept in.")
assert len(_K905_K971_LDS_BC_ANCHORS_8) == 8
assert len(_K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4) == 4
# K-axis-projection disjointness: K-1335 K∈{4096, 8192}; K-905/K-971
# K∈{16384, 32768}.  Cheap module-load-time assert per R-1329.K-AXIS-
# PROJECTION-DISJOINTNESS-ASSERTS-ARE-CHEAP-INSURANCE.
assert _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4.isdisjoint(
    _K905_K971_LDS_BC_ANCHORS_8), (
    "K-1335 longK_smallSquare admits overlap K-905/K-971 LDS-BC anchors; "
    "K-axis-projection disjointness (K-1335 K∈{4096, 8192} vs K-905/K-971 "
    "K∈{16384, 32768}) is the structural separator and a violation "
    "indicates an authoring typo in one of the two frozensets.")


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

# ---------------------------------------------------------------------------
# K-1283 / K-1297 (S-002) -- A1 sub-cohort targeted P8 admit-cell extension
# (8 cells).  K-1283 classified the K-1132 wpeu=1 13-cell residual into A1
# (LDS-wait-starvation, MFMA-issue-stall dominated) and A2 (occupancy-bound,
# paired waves ratio >= 1.7) sub-cohorts.  All five A1 *parent* cells (S18,
# S24, S25, S29 in `_K1121_P8_ANCHORS_13`; N11 in `_K1131_P8_NEIGHBORS_12`)
# are already routed.  K-1131 only perturbed S32 (A2), S39 (A2), S18 (A1),
# and S24 (A1) -- leaving S25 and S29 (A1 anchors) with **zero**
# +/-1-power-of-2 perturbation coverage, plus several un-covered axes on
# S18 / S24.
#
# K-1297 closes the A1-only perturbation gap with a strict-equality
# extension validated under paired n=30 HIP-graph hot-cache (V2 protocol;
# see K-1297 PR / fix/K-1283-A1 @ f4cc5cf for the V1-vs-V2 audit).  V2
# admits 8 cells at CI95-lo > 1.0x on rad-mi300x-1; cohort geomean =
# 1.389x.  All admits respect the K-1142 M-floor (M >= 4480) and the
# K-1161 K-floor (K >= 256).
#
# K-1322 (S-002) layers this set onto the K-1303 unified envelope (which
# already contains K-1121 + K-1131 + K-1161 E2 + K-1205 N=128 + K-1219
# N=256 = 43 cells), growing the unified envelope to 51 cells.  K-1283
# A1 perturbations are K-1131-style perturbations of A1 anchors that
# K-1131 itself did NOT enumerate (S25, S29, plus un-covered M/K axes of
# S18 / S24).  Disjointness with all five prior sub-frozensets is by
# construction (asserted at module load).
# ---------------------------------------------------------------------------
_K1283_A1_PERTURBATIONS_8 = frozenset({
    # ----- S25 (6016, 2048, 1024) family -- A1 anchor, 5/5 axes admit -----
    (12032, 2048, 1024, "torch.bfloat16"),  # A1_S25_Mx2
    ( 6016, 4096, 1024, "torch.bfloat16"),  # A1_S25_Nx2
    ( 6016, 1024, 1024, "torch.bfloat16"),  # A1_S25_N/2
    ( 6016, 2048, 2048, "torch.bfloat16"),  # A1_S25_Kx2
    ( 6016, 2048,  512, "torch.bfloat16"),  # A1_S25_K/2
    # ----- S29 (14208, 2048, 1024) family -- A1 anchor, 3/6 axes admit -----
    (28416, 2048, 1024, "torch.bfloat16"),  # A1_S29_Mx2
    (14208, 4096, 1024, "torch.bfloat16"),  # A1_S29_Nx2
    (14208, 2048,  512, "torch.bfloat16"),  # A1_S29_K/2
    # NOTE: 7 K-1297 candidates were rejected (CI95-lo <= 1.0x) under V2
    # and are DELIBERATELY OMITTED here (S29 boundary rejects, S18_Mx2
    # noise-floor straddler, S18/S24 K-axis rejects).  See K-1297 PR
    # body and tests/test_k971_route_predicate.py K-1283 no-leak pins.
})

# Composed 51-cell P8 envelope (K-1322 unified).  Six named provenance
# frozensets (K-1121 anchors, K-1131 neighbors, K-1175/K-1161 E2 admits,
# K-1231/K-1205 E_N3 N=128 admits, K-1219 E3 N=256 admits, K-1283/K-1297
# A1 perturbations) -- the dispatch path consults the union.  Per
# R-1144.DUAL-FROZENSET-PROVENANCE and its K-1175 / K-1205 / K-1219 /
# K-1297 extensions, source-ticket lineage is load-bearing for future
# reviewers (precedence-inversion debugging, PMC re-classifier work, ADR
# audits) so each measurement campaign keeps its own named set with a
# runtime size + pairwise-disjointness check.
_P8_MFMA_ISSUE_STALL_ROUTEOUT = (
    _K1121_P8_ANCHORS_13
    | _K1131_P8_NEIGHBORS_12
    | _K1161_E2_ADMITS_3
    | _K1205_EN3_ADMITS_8
    | _K1219_E3_NFLOOR256_ADMITS_7
    | _K1283_A1_PERTURBATIONS_8
)
assert len(_P8_MFMA_ISSUE_STALL_ROUTEOUT) == 51, (
    "K-1322 unified P8 envelope must be exactly 51 cells (13 K-1121 "
    "anchors + 12 K-1131 neighbors + 3 K-1161 E2 admits + 8 K-1205 E_N3 "
    "N=128 admits + 7 K-1219 E3 N=256 admits + 8 K-1283/K-1297 A1 "
    "perturbations); a duplicate or stray entry has crept in.")
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
# K-1322 / K-1297 A1 cross-disjointness with the prior five sub-frozensets.
# K-1283 A1 perturbations are perturbations of A1 anchors that K-1131 did
# not enumerate; their N values are in {1024, 2048, 4096} (S25/S29 family),
# overlapping the K-1131 / K-1121 N axis (N in {896, 1024, 2048}) on the N
# coordinate but always differing on (M, K) by construction.  N=128 / N=256
# tiers (K-1205 / K-1219) are trivially disjoint.
assert _K1283_A1_PERTURBATIONS_8.isdisjoint(_K1121_P8_ANCHORS_13), (
    "K-1283 A1 perturbation overlaps a K-1121 anchor; the K-1131-style "
    "+/-1-power-of-2 perturbation generator excludes the parent anchor "
    "by construction.")
assert _K1283_A1_PERTURBATIONS_8.isdisjoint(_K1131_P8_NEIGHBORS_12), (
    "K-1283 A1 perturbation overlaps a K-1131 neighbor; K-1297 selected "
    "only A1-anchor perturbations on axes K-1131 did NOT enumerate (S25, "
    "S29, plus un-covered M/K axes of S18/S24).")
assert _K1283_A1_PERTURBATIONS_8.isdisjoint(_K1161_E2_ADMITS_3), (
    "K-1283 A1 perturbation overlaps a K-1161 E2 admit; the three E2 "
    "admits ({(736,1792,736), (10112,2048,1024), (12160,2048,1024)}) are "
    "structurally distinct from the K-1297 A1 perturbation cells.")
assert _K1283_A1_PERTURBATIONS_8.isdisjoint(_K1205_EN3_ADMITS_8), (
    "K-1283 A1 perturbation overlaps a K-1205 E_N3 admit; K-1297 cells "
    "all have N >= 1024; K-1205 cells all have N=128.")
assert _K1283_A1_PERTURBATIONS_8.isdisjoint(_K1219_E3_NFLOOR256_ADMITS_7), (
    "K-1283 A1 perturbation overlaps a K-1219 E3 N=256 admit; K-1297 "
    "cells all have N >= 1024; K-1219 cells all have N=256.")
# K-1335 cross-predicate disjointness: K-1335 longK_smallSquare admits live
# on the LDS-BC `K971_ROUTE_TABLE` (5th-position dispatch predicate) and
# MUST NOT collide with any K-1322 P8 sub-frozenset.  K-1335 cells have
# M=N∈{1024,2048} and K∈{4096,8192}; every K-1322 P8 sub-frozenset's M-axis
# floor or N-axis selector excludes this region by construction (K-1121 /
# K-1131 / K-1161 / K-1283 use M >= 4480 with shape constraints; K-1205
# uses N=128; K-1219 uses N=256).  Disjointness is asserted on the
# (M,N,K,dtype) tuple after stripping the dtype back to a 4-tuple
# representation since both frozensets share the same key shape.
_K1335_VS_P8_DISJOINT = _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4.isdisjoint(
    _P8_MFMA_ISSUE_STALL_ROUTEOUT)
assert _K1335_VS_P8_DISJOINT, (
    "K-1335 longK_smallSquare admit overlaps the K-1322 51-cell P8 "
    "MFMA-issue-stall envelope; the two predicates target distinct "
    "hardware bottlenecks (LDS-BC vs MFMA-issue-stall per K-913 / "
    "K-1326 / R-1329) and must remain disjoint by construction.")


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
    """K-1144 P8 (extended by K-1175 / K-1231-K-1205 / K-1219 / K-1297 --
    unified in K-1303 then K-1322) — direct hipBLASLt route-OUT for the
    sextuply-validated MFMA-issue-stall cohort (K-1121 anchors + K-1131
    neighbors + K-1175/K-1161 E2 admits + K-1231/K-1205 E_N3 N=128 admits +
    K-1219 E3 N=256 admits + K-1283/K-1297 A1 perturbations).

    Returns True iff (M, N, K, dtype) matches one of the 51 strict-equality
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


# ---------------------------------------------------------------------------
# K-1361 (S-002) — P12 `square_mid` PMC-driven route-OUT (6th-position envelope).
#
# K-1361 productionises K-1345's Predicate-Q (square_mid 2048³) and Predicate-R
# (square_mid 4096³) as a 4-cell strict-equality frozenset
# `_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4`, layered as the 6th-position envelope
# in the dispatch precedence chain ABOVE the K971 LDS-BC table (5th-position)
# and BELOW the new K-1367 P13 skinny_N128 K-COMPLEMENT (7th-position).
#
# Source measurement chain:
#   * S1 (2048³ bf16): K-1295 paired n=30 + K-877 S1b direct PMC anchor
#                      (LDS_BC 0.889 cyc/inst, MFMA% gap 1.25 pp).
#   * S2 (4096³ bf16): K-1295 paired n=30 + K-913 §3 5-anchor PMC triangulation
#                      (K-655 / K-818 C2 / K-837 N1a / N1b / K-879).
#   * S1f (2048³ fp16): K-877 S1a direct fp16 PMC anchor (LDS_BC 0.889,
#                       MFMA% gap 1.27 pp) + K-1295 inheritance via K-913 §3
#                       dtype invariance.
#   * S2f (4096³ fp16): K-913 §3 dtype invariance from S1f extended to 4096³
#                       (INFERRED-DTYPE-PAIR; no direct 4096³ fp16 anchor).
# 8192³ bf16 is DROP'd (K-1308 F4 HBM-bound, K-879 N2a confirms zero LDS_BC
# on both engines; out of route-OUT scope per
# R-1345.HBM-ASYMPTOTE-BOUND-IS-NOT-PREDICATE-ADDRESSABLE).
#
# Bucket-rule scoping (per K-1345 RANKED_BUCKETS_v2 authoritative diagonal-only
# rule, NOT the PRD textual 4-combo enumeration): square_mid = M==N==K diagonal
# in {2048,4096}; off-diagonal combos (e.g. M=N=2048, K=4096) belong to K-1313
# WIDE bucket already P10-covered — admitting them would violate A4
# no-double-admit.  RETRY iteration=2 corrected the PRD ambiguity per
# R-1361.PRD-TEXTUAL-ENUMERATION-MUST-RESOLVE-AGAINST-AUTHORITATIVE-BUCKET-RULE.
#
# Verification (paired n=30 inherited from K-1295; PMC anchors direct + 5-anchor
# triangulation; B=10000 paired bootstrap CI95):
#   S1  bf16 paired tb/hbl med 0.258 [0.257, 0.260]  inv 3.87×  ROUTE-OUT
#   S2  bf16 paired tb/hbl med 0.550 [0.544, 0.552]  inv 1.82×  ROUTE-OUT
#   S1f fp16 paired tb/hbl med 0.258 (n/a-INFERRED)  inv 3.87×  ROUTE-OUT-INFERRED
#   S2f fp16 paired tb/hbl med 0.550 (n/a-INFERRED)  inv 1.82×  ROUTE-OUT-INFERRED
#
# Disjointness: P12 cells (M=N=K∈{2048,4096}, both dtypes) are pairwise disjoint
# with all P8 sub-frozensets (P8 uses M≥4480 for K-1121/K-1131/K-1161/K-1283 or
# fixed N∈{128,256} for K-1205/K-1219), with K971_ROUTE_TABLE (which uses
# K∈{16384,32768}, while P12 uses K∈{2048,4096}), and with K-1335 (K-1335 uses
# K∈{4096,8192} with M=N∈{1024,2048}; the only shape-collision candidate is
# (2048,2048,4096,bf16) which IS in K-1335 — by construction P12 omits this
# diagonal anchor since it is already route-OUT'd at the 5th position).
# ---------------------------------------------------------------------------
_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4 = frozenset({
    # S1: 2048³ bf16 — K-877 S1b direct PMC anchor + K-1295 paired n=30
    (2048, 2048, 2048, "torch.bfloat16"),
    # S2: 4096³ bf16 — K-913 §3 5-anchor triangulation + K-1295 paired n=30
    (4096, 4096, 4096, "torch.bfloat16"),
    # S1f: 2048³ fp16 — K-877 S1a direct fp16 anchor + K-913 §3 inheritance
    (2048, 2048, 2048, "torch.float16"),
    # S2f: 4096³ fp16 — INFERRED-DTYPE-PAIR (K-913 §3 from S1f, no 4096³ fp16 anchor)
    (4096, 4096, 4096, "torch.float16"),
})
assert len(_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4) == 4, (
    "K-1361 P12 square_mid frozenset must be exactly 4 cells (S1, S2, S1f, S2f); "
    "any deviation indicates an authoring typo against the K-1345 "
    "RANKED_BUCKETS_v2 diagonal-only authoritative bucket rule.")
# Cross-frozenset disjointness — P12 vs prior envelopes.
_K1361_P12_VS_P8_DISJOINT = (
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4.isdisjoint(_P8_MFMA_ISSUE_STALL_ROUTEOUT))
assert _K1361_P12_VS_P8_DISJOINT, (
    "K-1361 P12 square_mid cell overlaps the K-1322 51-cell P8 envelope; "
    "P8 sub-frozensets gate on M >= 4480 (K-1121/K-1131/K-1161/K-1283) or "
    "fixed N ∈ {128, 256} (K-1205/K-1219), neither of which can match "
    "M=N=K ∈ {2048, 4096}. Investigate before relaxing this assert.")
_K1361_P12_VS_K971_DISJOINT = (
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4.isdisjoint(K971_ROUTE_TABLE))
assert _K1361_P12_VS_K971_DISJOINT, (
    "K-1361 P12 square_mid cell overlaps K971_ROUTE_TABLE; K971 (K-905/K-971 "
    "anchors + K-1335 longK_smallSquare) covers K ∈ {4096, 8192, 16384, 32768} "
    "with M=N ∈ {1024, 2048}; P12's M=N=K ∈ {2048, 4096} diagonal does not "
    "intersect that K-axis range with M=N=4096 and excludes the K-1335 "
    "(2048,2048,4096) cell by construction (already 5th-position route-OUT).")


def _k1361_p12_square_mid_routeout(M: int, N: int, K: int, dtype) -> bool:
    """K-1361 P12 — direct hipBLASLt route-OUT for the 4-cell square_mid
    PMC-driven cohort (`_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4`).

    Returns True iff (M, N, K, dtype) matches one of the 4 strict-equality
    keys: M=N=K ∈ {2048, 4096} × dtype ∈ {bf16, fp16}.

    Source measurement quality varies per cell (MEASURED-DIRECT for S1 + S1f
    via K-877; MEASURED-VIA-TRIANGULATION for S2 via K-913 §3 5-anchor;
    INFERRED-DTYPE-PAIR for S2f via K-913 §3 dtype invariance from S1f).
    The runtime predicate behaviour is identical for all 4 cells; the
    quality grading is preserved as comments + the K-1361-FOLLOW-A0 fresh
    capture remediation.

    Stacked AFTER K971_ROUTE_TABLE (5th-position) and BEFORE K-1367 P13
    skinny_N128 K-COMPLEMENT (7th-position) per K-1175 stacked-predicate
    convention.  Disjoint by construction with all P1–P11 sub-frozensets
    via the cross-frozenset asserts above.
    """
    return (int(M), int(N), int(K), str(dtype)) in _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4


# ---------------------------------------------------------------------------
# K-1367 (S-002) — P13 `skinny_N128` K-COMPLEMENT route-OUT (7th-position).
#
# K-1379 productionises the K-1367 RETRY-winning 18-cell cohort as
# `_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18`, layered as the
# 7th-position envelope in the dispatch precedence chain on top of the
# K-1361 P12 55-cell baseline (envelope grows 55 → 73 cells; +18 admits).
#
# Bucket scope: K-1308's 4-bucket residual decomposition surfaced
# `skinny_N128` (raw aggregate gap 3.294) as the rank-#1 predicate-addressable
# residual; K-1345 demoted it to "wrapper-bound, already covered by K-1227's
# N=128 envelope".  K-1367 then proved that the demotion is K-bounded: at
# K ≥ 4096, TB compute time dominates the ~30 µs `triton_op` wrapper-overhead
# ceiling and the wrapper-bound argument no longer applies (codified as
# R-1367.WRAPPER-OVERHEAD-CEILING-DISAPPEARS-AT-LARGE-K-WHERE-TB-TIME-DOMINATES-30US-TRITONOP).
# The 18-cell K-COMPLEMENT region is exactly that K ≥ 4096 sub-region of the
# `skinny_N128` bucket K-1227 does NOT already cover.
#
# K-COMPLEMENT region:
#   M ∈ {2048, 4096, 8192}  (3 anchors)
#   N = 128                  (column-narrow regime)
#   K ∈ {4096, 8192, 16384}  (above the 30 µs wrapper-overhead ceiling)
#   dtype ∈ {bf16, fp16}     (K-913 §3 dtype invariance for both)
# = 3 × 1 × 3 × 2 = 18 cells.
#
# PMC mechanism (rocprofv3 4-pass capture on 7 representative cells):
#   TB persistent_matmul SQ_LDS_BANK_CONFLICT/inst ≈ 1.45
#   HBL Cijk_*           SQ_LDS_BANK_CONFLICT/inst = 0     (sharp discriminator)
#   n_dispatches = 22 per (cell, engine)  (positive-filter-check passed; the
#                                          HBL LDS_BC=0 is real measurement,
#                                          not a silent kernel-name-filter miss)
# This is the SAME K-913 §3 LDS-bank-conflict mechanism that anchored K-1335
# / K-1338 / K-1361, specialised to the N=128 column-narrow regime.  The
# persistent_matmul tile pattern systematically conflicts on bf16/fp16 N=128
# LDS layouts; hipBLASLt's tile selection avoids the bank-conflict geometry
# entirely.  Codified as
# R-1367.K913-LDS-BC-MECHANISM-IS-SHARPLY-DISCRIMINATING-FOR-SKINNY-N128.
#
# Verification (paired n=30 HIP-graph hot-cache, B=10000 vectorised paired
# bootstrap CI95 — vectorised per R-1367.VECTORIZED-BOOTSTRAP-IS-DEFAULT-VALIDATION-GATE):
#   18/18 ROUTE-OUT  (every cell CI95-lo > 1.0)
#   cohort geomean tb/hbl = 1.678×        (HBL is 1.678× faster ⇒ route-OUT win)
#   min CI95-lo            = 1.124         (worst cell still strictly route-OUT)
#   no regressions on the 55-cell K-1361 P12 envelope (paired-arm stable)
#
# Projected K-1247 cohort lift: the 18-cell K-COMPLEMENT cohort is fully
# inside the K-1247 61-cell backtest corpus; cohort geomean 1.678× contributes
# directly to the realisable cohort lift; measurement deferred to
# K-1367-FOLLOW-B once the productionised branch is on dedicated MI300X.
#
# Why N=128 is hipBLASLt-favoured (mechanistic — answers task brief Q3):
#   The MFMA tile efficiency threshold for the persistent_matmul kernel sits
#   at N >= 256 for waves_per_CU >= 2; at N=128 the column-narrow tile under-
#   utilises the MFMA throughput and incurs systematic LDS-bank-conflicts on
#   the persistent tile schedule (TB SQ_LDS_BC/inst ≈ 1.45 vs HBL = 0).  At
#   K >= 4096 the compute time amortises the wrapper-overhead floor, so the
#   raw HBL win (1.678× cohort geomean) is realisable end-to-end.
#
# Disjointness: P13 cells use N=128, M ∈ {2048, 4096, 8192}, K ∈ {4096, 8192,
# 16384}.  Disjoint with:
#   * P8 sub-frozensets — K-1121/K-1131/K-1161/K-1283 use M ≥ 4480 (P13
#     allows M ≥ 2048; would collide on M ∈ {4096, 8192} but those P8 cohorts
#     never use N=128); K-1205 uses N=128 BUT K ∈ {2048, 8192} with M ≥ 4480;
#     the only on-paper collision candidates are M=8192,N=128,K=8192 cells;
#     verified disjoint by frozenset.isdisjoint at module load.
#   * K971_ROUTE_TABLE — K-905/K-971 anchors use M=N ∈ {1024, 2048};
#     K-1335 uses M=N ∈ {1024, 2048} with K ∈ {4096, 8192}; neither has
#     N=128, so no collision.
#   * K-1361 P12 — uses M=N=K ∈ {2048, 4096} with N != 128; no collision.
# All asserted at module load.
# ---------------------------------------------------------------------------
_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 = frozenset({
    # M=2048 row × K ∈ {4096, 8192, 16384} × {bf16, fp16}
    (2048, 128,  4096, "torch.bfloat16"),
    (2048, 128,  4096, "torch.float16"),
    (2048, 128,  8192, "torch.bfloat16"),
    (2048, 128,  8192, "torch.float16"),
    (2048, 128, 16384, "torch.bfloat16"),
    (2048, 128, 16384, "torch.float16"),
    # M=4096 row × K ∈ {4096, 8192, 16384} × {bf16, fp16}
    (4096, 128,  4096, "torch.bfloat16"),
    (4096, 128,  4096, "torch.float16"),
    (4096, 128,  8192, "torch.bfloat16"),
    (4096, 128,  8192, "torch.float16"),
    (4096, 128, 16384, "torch.bfloat16"),
    (4096, 128, 16384, "torch.float16"),
    # M=8192 row × K ∈ {4096, 8192, 16384} × {bf16, fp16}
    (8192, 128,  4096, "torch.bfloat16"),
    (8192, 128,  4096, "torch.float16"),
    (8192, 128,  8192, "torch.bfloat16"),
    (8192, 128,  8192, "torch.float16"),
    (8192, 128, 16384, "torch.bfloat16"),
    (8192, 128, 16384, "torch.float16"),
})
assert len(_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18) == 18, (
    "K-1367 P13 skinny_N128 K-COMPLEMENT frozenset must be exactly 18 cells "
    "(M ∈ {2048,4096,8192} × N=128 × K ∈ {4096,8192,16384} × {bf16,fp16}); "
    "any deviation indicates an authoring typo against the K-1308 bucket rule "
    "or the K-1227 wrapper-bound K-COMPLEMENT scoping.")
# Cross-frozenset disjointness — P13 vs prior envelopes.
_K1367_P13_VS_P8_DISJOINT = (
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18.isdisjoint(_P8_MFMA_ISSUE_STALL_ROUTEOUT))
assert _K1367_P13_VS_P8_DISJOINT, (
    "K-1367 P13 skinny_N128 K-COMPLEMENT cell overlaps the K-1322 51-cell P8 "
    "envelope; P8's K-1205 N=128 sub-frozenset uses K ∈ {2048, 8192} with "
    "M ≥ 4480 — no admitted P13 cell satisfies BOTH selectors simultaneously "
    "(P13 K=8192 cells exist at M ∈ {2048, 4096, 8192} but K-1205 specific "
    "tuples must be enumerated to overlap; disjointness verified by frozenset "
    "intersection at module load).")
_K1367_P13_VS_K971_DISJOINT = (
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18.isdisjoint(K971_ROUTE_TABLE))
assert _K1367_P13_VS_K971_DISJOINT, (
    "K-1367 P13 skinny_N128 K-COMPLEMENT cell overlaps K971_ROUTE_TABLE; "
    "K971_ROUTE_TABLE (K-905/K-971 + K-1335) uses M=N ∈ {1024, 2048}; "
    "P13 cells all use N=128 — natural disjointness, but assert as cheap "
    "insurance per R-1329.K-AXIS-PROJECTION-DISJOINTNESS-ASSERTS-ARE-CHEAP-INSURANCE.")
_K1367_P13_VS_P12_DISJOINT = (
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18.isdisjoint(_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4))
assert _K1367_P13_VS_P12_DISJOINT, (
    "K-1367 P13 skinny_N128 K-COMPLEMENT cell overlaps K-1361 P12 square_mid; "
    "P12 cells use M=N=K ∈ {2048, 4096}; P13 cells all use N=128 — natural "
    "disjointness, asserted for completeness (A4 no-double-admit).")


def _k1367_p13_skinny_n128_routeout(M: int, N: int, K: int, dtype) -> bool:
    """K-1367 P13 — direct hipBLASLt route-OUT for the 18-cell skinny_N128
    K-COMPLEMENT cohort (`_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18`).

    Returns True iff (M, N, K, dtype) matches one of the 18 strict-equality
    keys: M ∈ {2048, 4096, 8192} × N = 128 × K ∈ {4096, 8192, 16384} ×
    dtype ∈ {torch.bfloat16, torch.float16}.

    Source measurement: paired n=30 HIP-graph hot-cache benchmarks on
    MI300X / gfx942 with B=10000 vectorised paired bootstrap (numpy
    advanced-indexing per R-1298 / R-1367); 18/18 ROUTE-OUT, cohort geomean
    tb/hbl = 1.678×, min CI95-lo = 1.124.  PMC mechanism confirmed via
    rocprofv3 4-pass capture on 7 representative cells (TB SQ_LDS_BC/inst
    ≈ 1.45 vs HBL = 0; positive-filter-check + fail-loud sweep verified
    HBL LDS_BC = 0 is real measurement).

    Stacked at 7th-position per K-1175 stacked-predicate convention;
    disjoint by construction with all P1–P12 sub-frozensets via the
    cross-frozenset asserts above.
    """
    return (
        (int(M), int(N), int(K), str(dtype))
        in _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18
    )


# ---------------------------------------------------------------------------
# K-1397 (S-002) — P13 `skinny_N256` K-COMPLEMENT route-OUT (8th-position).
#
# K-1402 productionises the K-1397-winning 12-cell `skinny_N256` cohort as
# `_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12`, layered as the 8th-position
# envelope in the dispatch precedence chain on top of the K-1389 P13 73-cell
# baseline (envelope grows 73 → 85 cells; +12 admits).
#
# Bucket scope: K-1365's post-P12 4-bucket decomposition surfaced
# `skinny_N256` as the last residual-addressable bucket after P12/P13/N128
# closed `skinny_N128`.  The K-1397 measurement campaign demonstrated that
# the same `wrapper-overhead-ceiling-disappears-at-large-K` argument
# (R-1367.WRAPPER-OVERHEAD-CEILING-DISAPPEARS-AT-LARGE-K-WHERE-TB-TIME-DOMINATES-30US-TRITONOP)
# generalises to N=256 column-narrow GEMMs, but the K-axis admit set sits
# at the EXTREMES of the K-1308 sweep grid (K=2048 and K=32768) where the
# aspect ratio either (a) starves persistent_matmul's tile parallelism with
# too few K-tiles per CU (K=2048 small-K starvation) or (b) overflows LDS
# bank-conflict capacity at the persistent N=256 tile width (K=32768 large-K
# saturation).  The complementary mid-K band (K ∈ {4096, 8192, 16384}) is
# already routed correctly by tritonblas persistent_matmul; the K-COMPLEMENT
# is exactly the {K=2048, K=32768} pair (codified as
# R-1397.SKINNY-N256-K-COMPLEMENT-IS-EXTREMES-NOT-MID-BAND).
#
# K-COMPLEMENT region:
#   M ∈ {2048, 4096, 8192}   (3 anchors, same as K-1367 N=128 row)
#   N = 256                   (column-narrow regime, +1 column step from N=128)
#   K ∈ {2048, 32768}         (extremes — small-K starvation + large-K LDS-BC)
#   dtype ∈ {bf16, fp16}      (K-913 §3 dtype invariance preserved)
# = 3 × 1 × 2 × 2 = 12 cells.
#
# PMC mechanism (consistent with K-913 longK_smallSquare findings):
#   hipBLASLt's split-K kernel selection wins over tritonblas
#   persistent_matmul at extreme aspect ratios where LDS bank conflicts
#   dominate the persistent N=256 tile layout (TB SQ_LDS_BC/inst remains
#   high at N=256, HBL split-K trades LDS pressure for atomic-reduction).
#
# Disjointness rationale (verified by frozenset.isdisjoint at module load):
#   * P8 sub-frozensets — K-1121/K-1131/K-1161/K-1283 use M >= 4480 (P13
#     uses M ∈ {2048, 4096} below threshold; M=8192 cells are admitted by
#     P8 only at specific (N,K) tuples enumerated in K-1121 anchors which
#     do not include N=256, K ∈ {2048, 32768}).  K-1219 N=256 sub-frozenset
#     uses K ∈ {1024, 2048} with M ∈ {1024, 2048} — partial overlap candidate
#     at (2048, 256, 2048, *); verified disjoint by frozenset.isdisjoint at
#     module load (K-1219 specific tuples must be enumerated to overlap).
#   * K971_ROUTE_TABLE — K-905/K-971 anchors use M=N ∈ {1024, 2048};
#     K-1335 uses M=N ∈ {1024, 2048}; neither has N=256, no collision.
#   * K-1361 P12 — uses M=N=K ∈ {2048, 4096} with N != 256; no collision.
#   * K-1367 P13 — uses N=128; K-1397 uses N=256; natural disjointness.
# All asserted at module load.
# ---------------------------------------------------------------------------
_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 = frozenset({
    # M=2048 row × K ∈ {2048, 32768} × {bf16, fp16}
    (2048, 256,  2048, "torch.bfloat16"),
    (2048, 256,  2048, "torch.float16"),
    (2048, 256, 32768, "torch.bfloat16"),
    (2048, 256, 32768, "torch.float16"),
    # M=4096 row × K ∈ {2048, 32768} × {bf16, fp16}
    (4096, 256,  2048, "torch.bfloat16"),
    (4096, 256,  2048, "torch.float16"),
    (4096, 256, 32768, "torch.bfloat16"),
    (4096, 256, 32768, "torch.float16"),
    # M=8192 row × K ∈ {2048, 32768} × {bf16, fp16}
    (8192, 256,  2048, "torch.bfloat16"),
    (8192, 256,  2048, "torch.float16"),
    (8192, 256, 32768, "torch.bfloat16"),
    (8192, 256, 32768, "torch.float16"),
})
assert len(_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12) == 12, (
    "K-1397 P13 skinny_N256 K-COMPLEMENT frozenset must be exactly 12 cells "
    "(M ∈ {2048,4096,8192} × N=256 × K ∈ {2048,32768} × {bf16,fp16}); "
    "any deviation indicates an authoring typo against the K-1365 post-P12 "
    "4-bucket decomposition or the K-1397 K-COMPLEMENT extremes scoping.")
# Cross-frozenset disjointness — K-1397 P13 vs prior envelopes.
_K1397_P13_VS_P8_DISJOINT = (
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12.isdisjoint(_P8_MFMA_ISSUE_STALL_ROUTEOUT))
assert _K1397_P13_VS_P8_DISJOINT, (
    "K-1397 P13 skinny_N256 K-COMPLEMENT cell overlaps the K-1322 51-cell P8 "
    "envelope; K-1219's N=256 sub-frozenset uses K ∈ {1024, 2048} with "
    "M ∈ {1024, 2048} — only (2048, 256, 2048, *) is an on-paper collision "
    "candidate; specific K-1219 tuples must be enumerated to overlap.  "
    "Disjointness verified by frozenset intersection at module load.")
_K1397_P13_VS_K971_DISJOINT = (
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12.isdisjoint(K971_ROUTE_TABLE))
assert _K1397_P13_VS_K971_DISJOINT, (
    "K-1397 P13 skinny_N256 K-COMPLEMENT cell overlaps K971_ROUTE_TABLE; "
    "K971_ROUTE_TABLE (K-905/K-971 + K-1335) uses M=N ∈ {1024, 2048}; "
    "K-1397 cells all use N=256 — natural disjointness, but assert as cheap "
    "insurance per R-1329.K-AXIS-PROJECTION-DISJOINTNESS-ASSERTS-ARE-CHEAP-INSURANCE.")
_K1397_P13_VS_P12_DISJOINT = (
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12.isdisjoint(_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4))
assert _K1397_P13_VS_P12_DISJOINT, (
    "K-1397 P13 skinny_N256 K-COMPLEMENT cell overlaps K-1361 P12 square_mid; "
    "P12 cells use M=N=K ∈ {2048, 4096}; K-1397 cells all use N=256 — natural "
    "disjointness, asserted for completeness (A4 no-double-admit).")
_K1397_P13_VS_K1367_P13_DISJOINT = (
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12.isdisjoint(
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18))
assert _K1397_P13_VS_K1367_P13_DISJOINT, (
    "K-1397 P13 skinny_N256 cell overlaps K-1367 P13 skinny_N128; "
    "K-1367 cells use N=128, K-1397 cells use N=256 — natural disjointness, "
    "asserted for completeness (A4 no-double-admit; the two K-COMPLEMENT "
    "predicates partition the skinny-N column-narrow regime by N-axis).")


def _k1397_p13_skinny_n256_routeout(M: int, N: int, K: int, dtype) -> bool:
    """K-1397 P13 — direct hipBLASLt route-OUT for the 12-cell skinny_N256
    K-COMPLEMENT cohort (`_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12`).

    Returns True iff (M, N, K, dtype) matches one of the 12 strict-equality
    keys: M ∈ {2048, 4096, 8192} × N = 256 × K ∈ {2048, 32768} ×
    dtype ∈ {torch.bfloat16, torch.float16}.

    Source measurement: K-1397 paired n=30 HIP-graph hot-cache benchmarks
    on MI300X / gfx942 with B=10000 vectorised paired bootstrap CI95;
    12/12 ROUTE-OUT, per-cell speedups span 1.04× to 1.12×.  Mechanism is
    hipBLASLt's split-K kernel selection winning over tritonblas
    persistent_matmul at extreme aspect ratios where LDS bank conflicts
    dominate the persistent N=256 tile layout (consistent with K-913
    longK_smallSquare PMC findings).

    Stacked at 8th-position per K-1175 stacked-predicate convention;
    disjoint by construction with all P1–P13(N=128) sub-frozensets via the
    cross-frozenset asserts above.
    """
    return (
        (int(M), int(N), int(K), str(dtype))
        in _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12
    )


# ---------------------------------------------------------------------------
# K-1417 (S-002) — P15 `skinny_N512` K-COMPLEMENT EXTENSION route-OUT
# (9th-position).
#
# K-1417 productionises the K-1409-derived 12-cell `skinny_N512` K-COMPLEMENT
# EXTENSION cohort as `_K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT`, layered
# as the 9th-position envelope in the dispatch precedence chain on top of
# the K-1406 P14 8-predicate baseline (envelope grows by +12 admits at the
# K-axis EXTREMES of the N=512 column-narrow regime).
#
# Bucket scope: third-N successive sibling extension after K-1397 (N=256)
# along the same K-COMPLEMENT EXTREMES axis (K ∈ {2048, 32768}).  The
# `wrapper-overhead-ceiling-disappears-at-large-K` argument
# (R-1367.WRAPPER-OVERHEAD-CEILING-DISAPPEARS-AT-LARGE-K-WHERE-TB-TIME-DOMINATES-30US-TRITONOP)
# generalises to N=512 column-narrow GEMMs; the K-axis admit set sits at
# the EXTREMES of the K-1308 sweep grid (K=2048 small-K starvation +
# K=32768 large-K LDS-BC saturation).  The complementary mid-K band
# (K ∈ {4096, 8192, 16384}) is the K-1409 BASE region (separately
# productionised as the 18-cell N=512 BASE envelope; this K-1417
# productionisation covers only the EXTENSION extremes).  Codified as
# R-1417.SKINNY-N512-K-COMPLEMENT-EXTENSION-IS-EXTREMES (mirroring R-1397
# at N=512).
#
# K-COMPLEMENT EXTENSION region:
#   M ∈ {2048, 4096, 8192}    (3 anchors, sibling row to K-1397 N=256)
#   N = 512                    (column-narrow regime, +1 column step from N=256)
#   K ∈ {2048, 32768}          (extremes — small-K starvation + large-K LDS-BC)
#   dtype ∈ {bf16, fp16}       (K-913 §3 dtype invariance preserved)
# = 3 × 1 × 2 × 2 = 12 cells.
#
# PMC mechanism (consistent with K-913 / K-1397 findings):
#   At K=2048 (small-K starvation), tritonblas persistent_matmul has too
#   few K-tiles per CU for adequate tile parallelism — TB compute time
#   dominates the wrapper-overhead ceiling and the per-cell ratio blows
#   out (TB ≈ 280-290 µs vs HBL ≈ 19-50 µs; ratio 5.6×-14.0×).  At
#   K=32768 (large-K LDS-BC saturation), hipBLASLt's split-K kernel
#   selection wins over tritonblas persistent_matmul at the persistent
#   N=512 tile layout where LDS bank conflicts dominate (TB ≈ 510-780 µs
#   vs HBL ≈ 280-570 µs; ratio 1.36×-2.05×).  Both regimes share the
#   K-913 §3 LDS-BC mechanism specialised to the N=512 column-narrow
#   regime.
#
# Verification (paired n=30 HIP-graph hot-cache, B=10000 vectorised paired
# bootstrap CI95 on MI300X gfx942 / OCI MI300X fallback — c42 down):
#   12/12 ROUTE-OUT  (every cell ratio_median ≥ 1.05 AND CI95-lo > 1.0)
#   cohort geomean tb/hbl = **3.43×** (range 1.36×–14.06×)
#   min CI95-lo            = **1.359** at (8192, 512, 32768, fp16)
#   max ratio_median       = **14.06×** at (2048, 512, 2048, fp16)
#   envelope grows         85 → 97 cells (8 → 9 predicate stack)
#
# Disjointness rationale (verified by frozenset.isdisjoint at module load):
#   * P8 sub-frozensets — K-1219 E3 N=256 sub-frozenset uses N=256;
#     K-1417 cells all use N=512, no collision.  Other P8 anchors use
#     M=N for square cells; no N=512 cells anywhere.
#   * K971_ROUTE_TABLE — K-905/K-971/K-1335 anchors use M=N ∈ {1024, 2048};
#     no N=512.
#   * K-1361 P12 — uses M=N=K ∈ {2048, 4096}; N != 512.
#   * K-1367 P13 — uses N=128.
#   * K-1397 P13 — uses N=256.
# All asserted at module load.
# ---------------------------------------------------------------------------
_K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT = frozenset({
    # M=2048 row × K ∈ {2048, 32768} × {bf16, fp16}
    (2048, 512,  2048, "torch.bfloat16"),
    (2048, 512,  2048, "torch.float16"),
    (2048, 512, 32768, "torch.bfloat16"),
    (2048, 512, 32768, "torch.float16"),
    # M=4096 row × K ∈ {2048, 32768} × {bf16, fp16}
    (4096, 512,  2048, "torch.bfloat16"),
    (4096, 512,  2048, "torch.float16"),
    (4096, 512, 32768, "torch.bfloat16"),
    (4096, 512, 32768, "torch.float16"),
    # M=8192 row × K ∈ {2048, 32768} × {bf16, fp16}
    (8192, 512,  2048, "torch.bfloat16"),
    (8192, 512,  2048, "torch.float16"),
    (8192, 512, 32768, "torch.bfloat16"),
    (8192, 512, 32768, "torch.float16"),
})
assert len(_K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT) == 12, (
    "K-1417 P15 skinny_N512 K-COMPLEMENT EXTENSION frozenset must be exactly "
    "12 cells (M ∈ {2048,4096,8192} × N=512 × K ∈ {2048,32768} × {bf16,fp16}); "
    "any deviation indicates an authoring typo against the K-1397 K-COMPLEMENT "
    "EXTREMES scoping at N=512 (R-1417.SKINNY-N512-K-COMPLEMENT-EXTENSION-IS-EXTREMES).")
# Cross-frozenset disjointness — K-1417 P15 vs prior 8-predicate stack.
_K1409_P15_VS_P8_DISJOINT = (
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT.isdisjoint(_P8_MFMA_ISSUE_STALL_ROUTEOUT))
assert _K1409_P15_VS_P8_DISJOINT, (
    "K-1417 P15 skinny_N512 K-COMPLEMENT EXTENSION cell overlaps the K-1322 "
    "51-cell P8 envelope; P8's K-1219 E3 N=256 sub-frozenset uses N=256 — "
    "P15 uses N=512, natural disjointness, asserted as cheap insurance per "
    "R-1329.K-AXIS-PROJECTION-DISJOINTNESS-ASSERTS-ARE-CHEAP-INSURANCE.")
_K1409_P15_VS_K971_DISJOINT = (
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT.isdisjoint(K971_ROUTE_TABLE))
assert _K1409_P15_VS_K971_DISJOINT, (
    "K-1417 P15 skinny_N512 K-COMPLEMENT EXTENSION cell overlaps "
    "K971_ROUTE_TABLE; K971_ROUTE_TABLE (K-905/K-971 + K-1335) uses M=N ∈ "
    "{1024, 2048}; P15 cells all use N=512 — natural disjointness, asserted "
    "insurance.")
_K1409_P15_VS_P12_DISJOINT = (
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT.isdisjoint(_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4))
assert _K1409_P15_VS_P12_DISJOINT, (
    "K-1417 P15 skinny_N512 K-COMPLEMENT EXTENSION cell overlaps K-1361 P12 "
    "square_mid; P12 cells use M=N=K ∈ {2048, 4096}; P15 cells all use N=512 "
    "— natural disjointness, asserted for completeness (A4 no-double-admit).")
_K1409_P15_VS_K1367_P13_DISJOINT = (
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT.isdisjoint(
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18))
assert _K1409_P15_VS_K1367_P13_DISJOINT, (
    "K-1417 P15 skinny_N512 cell overlaps K-1367 P13 skinny_N128; P13(N=128) "
    "cells use N=128, P15 cells use N=512 — natural disjointness, asserted "
    "for completeness (A4 sibling-N firewall).")
_K1409_P15_VS_K1397_P13_DISJOINT = (
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT.isdisjoint(
        _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12))
assert _K1409_P15_VS_K1397_P13_DISJOINT, (
    "K-1417 P15 skinny_N512 cell overlaps K-1397 P13 skinny_N256; P13(N=256) "
    "cells use N=256, P15 cells use N=512 — natural disjointness, asserted "
    "for completeness (A4 sibling-N firewall; the three K-COMPLEMENT EXTREMES "
    "predicates partition the skinny-N column-narrow regime by N-axis at "
    "{128, 256, 512}).")


def _k1409_p15_skinny_n512_routeout(M: int, N: int, K: int, dtype) -> bool:
    """K-1417 P15 — direct hipBLASLt route-OUT for the 12-cell skinny_N512
    K-COMPLEMENT EXTENSION cohort (`_K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT`).

    Returns True iff (M, N, K, dtype) matches one of the 12 strict-equality
    keys: M ∈ {2048, 4096, 8192} × N = 512 × K ∈ {2048, 32768} ×
    dtype ∈ {torch.bfloat16, torch.float16}.

    Source measurement: K-1417 paired n=30 HIP-graph hot-cache benchmarks
    on MI300X / gfx942 (OCI MI300X fallback; c42 down) with
    B=10000 vectorised paired bootstrap CI95 (numpy advanced-indexing per
    R-1298 / R-1367); 12/12 ROUTE-OUT, cohort geomean tb/hbl = 3.43×,
    min CI95-lo = 1.359, range 1.36×–14.06×.  Mechanism: at K=2048 (small-K
    starvation), tritonblas persistent_matmul's persistent_matmul tile
    parallelism is starved (TB ≈ 280 µs vs HBL ≈ 19-50 µs); at K=32768
    (large-K LDS-BC saturation), hipBLASLt's split-K kernel selection wins
    over tritonblas at the persistent N=512 tile layout where LDS bank
    conflicts dominate (consistent with K-913 longK_smallSquare PMC
    findings, K-1397 N=256 sibling, K-1409 N=512 BASE-region sibling).

    Stacked at 9th-position per K-1175 stacked-predicate convention;
    disjoint by construction with all P1–P14 sub-frozensets via the
    cross-frozenset asserts above.
    """
    return (
        (int(M), int(N), int(K), str(dtype))
        in _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT
    )


# ---------------------------------------------------------------------------
# K-1429 (S-002) — P16 `skinny_N1024` K-COMPLEMENT route-OUT (10th-position).
#
# K-1429 productionises the 29-cell `skinny_N1024` K-COMPLEMENT cohort
# (BASE 18 + EXTENSION 12 swept, 1 BASE cell rejected by the strict 1.05
# admit gate) as `_K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29`, layered as
# the 10th-position envelope in the dispatch precedence chain on top of
# the K-1417 P15 9-predicate baseline (envelope grows by +29 admits at
# the full K-axis sweep of the N=1024 column-narrow regime).
#
# Bucket scope: fourth-N successive sibling extension after K-1417 (N=512)
# along the K-COMPLEMENT axis.  Unlike K-1417 (which limited itself to the
# K-EXTREMES K ∈ {2048, 32768}), K-1429 sweeps the full K-axis
# K ∈ {2048, 4096, 8192, 16384, 32768} and admits 29/30 cells under the
# strict per-cell gate (median ≥ 1.05 ∧ CI95-lo > 1.0).  The single
# rejected cell is (2048, 1024, 4096, bf16) at r=1.0154 (CI95-lo=1.008
# clears > 1.0 but ratio_median misses the 1.05 floor by 3.5 pp).
#
# Codified as R-1429.SKINNY-N1024-K-COMPLEMENT-DISCRIMINATOR-REVERSES-ATTENUATION-AT-N1024
# — the R-1409 monotonic N-axis attenuation (1.678 → 1.471 → 1.372 across
# N=128/256/512) DOES NOT extend to N=1024.  Cohort geomean tb/hbl jumps
# back to **2.23×** (BASE 2.08×, EXTREMES 2.48×), exceeding every prior
# N bucket including N=128.  The R-1382 weak-form prediction "per-cell
# admission collapses around N=1024" is FALSIFIED — 29/30 admit per-cell
# unanimously and the K-1131 A2 1.40× cohort floor is comfortably cleared
# (2.23× vs 1.40× — A2 PASS at N=1024 after MISS at N=512).  Mechanism:
# at N=1024 the persistent_matmul tile aspect (BLOCK_M=128, BLOCK_N
# heuristic) misaligns the M-axis tile against the M ∈ {2048, 4096, 8192}
# anchors, accumulating LDS-bank conflicts that exceed the K-913 §3
# attenuation seen at N=512.  hipBLASLt's split-K kernel re-selects at
# N=1024 to a pattern that better matches the M anchors.
#
# K-COMPLEMENT region (N=1024, K full sweep):
#   M ∈ {2048, 4096, 8192}                   (3 anchors, sibling row to K-1417)
#   N = 1024                                  (column-narrow regime, +1 step from N=512)
#   K ∈ {2048, 4096, 8192, 16384, 32768}      (full K-axis — BASE + EXTREMES)
#   dtype ∈ {bf16, fp16}                      (K-913 §3 dtype invariance preserved)
# = 3 × 1 × 5 × 2 = 30 sweep cells; 29 admit (1 rejected: 2048×1024×4096 bf16).
#
# PMC mechanism (consistent with K-913 / K-1397 / K-1417 findings;
# verified by paired n=30 timing signature, PMC capture deferred to
# K-1429-FOLLOW-D):
#   The K-1417 small-K (K=2048) starvation pattern at N=512 (TB ≈ 280-290 µs
#   vs HBL ≈ 19-50 µs) recurs at N=1024 with similar magnitude (TB ≈ 280-290 µs
#   vs HBL ≈ 31-78 µs; ratio 3.5×-8.85× for K=2048 cells).  The K=32768
#   large-K LDS-BC saturation is also reproduced (TB ≈ 460-1620 µs vs
#   HBL ≈ 285-1316 µs; ratio 1.26×-1.62×).  Crucially the BASE
#   (K ∈ {4096, 8192, 16384}) cells which would be in the BASE region also
#   admit uniformly (17/18 admit) — at N=1024 the BASE geomean (2.08×) and
#   the EXTREMES geomean (2.48×) are within ~20% of each other, in
#   contrast to the prior N buckets where the BASE region had not been
#   productionised.  This justifies merging both K-COMPLEMENT regions
#   into a single 29-cell P16 frozenset rather than the 18+12 split
#   used at N=128/256/512.
#
# Verification (paired n=30 HIP-graph hot-cache, B=10000 vectorised paired
# bootstrap CI95 on MI300X gfx942 / OCI amd-arad fallback — c42 down):
#   29/30 ROUTE-OUT  (every admit cell ratio_median ≥ 1.05 AND CI95-lo > 1.0)
#   cohort geomean tb/hbl = **2.23×** (range 1.26×–8.85×)
#   min CI95-lo            = **1.267** at (8192, 1024, 32768, fp16)
#   max ratio_median       = **8.85×**  at (2048, 1024,  2048, fp16)
#   envelope grows         97 → 126 cells (9 → 10 predicate stack)
#   BASE     geomean 2.08× (17/18 admit; reject at (2048,1024,4096,bf16) r=1.015)
#   EXTREMES geomean 2.48× (12/12 admit)
#
# Disjointness rationale (verified by frozenset.isdisjoint at module load):
#   * P8 sub-frozensets — K-1219 E3 N=256 sub-frozenset uses N=256;
#     K-1429 cells all use N=1024, no collision.  Other P8 anchors use
#     M=N for square cells; no N=1024 cells anywhere.
#   * K971_ROUTE_TABLE — K-905/K-971/K-1335 anchors use M=N ∈ {1024, 2048};
#     K-1429 cells have N=1024 with M ∈ {2048, 4096, 8192} — only M=2048
#     could collide if N=1024 ∧ M=2048 ∧ K were ever in K971_ROUTE_TABLE,
#     which it is not (K971 uses N=1024 only at M=1024).
#   * K-1361 P12 — uses M=N=K ∈ {2048, 4096}; N=1024 never matches.
#   * K-1367 P13 — uses N=128.
#   * K-1397 P13 — uses N=256.
#   * K-1409 P15 (K-1417) — uses N=512.
# All asserted at module load.
# ---------------------------------------------------------------------------
_K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29 = frozenset({
    # === BASE region (K ∈ {4096, 8192, 16384}) — 17 admit cells (1 reject)
    # M=2048 row × K ∈ {4096, 8192, 16384} × {bf16, fp16} (5 of 6 admit;
    # (2048, 1024, 4096, bf16) rejected at r=1.015 < strict 1.05 floor)
    (2048, 1024,  4096, "torch.float16"),     # r=5.062 ci_lo=5.044
    (2048, 1024,  8192, "torch.bfloat16"),    # r=3.336 ci_lo=3.269
    (2048, 1024,  8192, "torch.float16"),     # r=3.145 ci_lo=3.081
    (2048, 1024, 16384, "torch.bfloat16"),    # r=1.997 ci_lo=1.973
    (2048, 1024, 16384, "torch.float16"),     # r=1.871 ci_lo=1.841
    # M=4096 row × K ∈ {4096, 8192, 16384} × {bf16, fp16}
    (4096, 1024,  4096, "torch.bfloat16"),    # r=3.581 ci_lo=3.566
    (4096, 1024,  4096, "torch.float16"),     # r=3.358 ci_lo=3.337
    (4096, 1024,  8192, "torch.bfloat16"),    # r=1.969 ci_lo=1.966
    (4096, 1024,  8192, "torch.float16"),     # r=1.871 ci_lo=1.865
    (4096, 1024, 16384, "torch.bfloat16"),    # r=1.463 ci_lo=1.460
    (4096, 1024, 16384, "torch.float16"),     # r=1.448 ci_lo=1.446
    # M=8192 row × K ∈ {4096, 8192, 16384} × {bf16, fp16}
    (8192, 1024,  4096, "torch.bfloat16"),    # r=2.198 ci_lo=2.187
    (8192, 1024,  4096, "torch.float16"),     # r=2.019 ci_lo=2.011
    (8192, 1024,  8192, "torch.bfloat16"),    # r=1.568 ci_lo=1.548
    (8192, 1024,  8192, "torch.float16"),     # r=1.529 ci_lo=1.522
    (8192, 1024, 16384, "torch.bfloat16"),    # r=1.635 ci_lo=1.626
    (8192, 1024, 16384, "torch.float16"),     # r=1.616 ci_lo=1.594
    # === EXTENSION region (K ∈ {2048, 32768}) — 12/12 admit
    # M=2048 row × K ∈ {2048, 32768} × {bf16, fp16}
    (2048, 1024,  2048, "torch.bfloat16"),    # r=1.519 ci_lo=1.527
    (2048, 1024,  2048, "torch.float16"),     # r=8.849 ci_lo=8.795
    (2048, 1024, 32768, "torch.bfloat16"),    # r=1.616 ci_lo=1.610
    (2048, 1024, 32768, "torch.float16"),     # r=1.591 ci_lo=1.588
    # M=4096 row × K ∈ {2048, 32768} × {bf16, fp16}
    (4096, 1024,  2048, "torch.bfloat16"),    # r=6.028 ci_lo=6.021
    (4096, 1024,  2048, "torch.float16"),     # r=5.667 ci_lo=5.687
    (4096, 1024, 32768, "torch.bfloat16"),    # r=1.495 ci_lo=1.489
    (4096, 1024, 32768, "torch.float16"),     # r=1.456 ci_lo=1.451
    # M=8192 row × K ∈ {2048, 32768} × {bf16, fp16}
    (8192, 1024,  2048, "torch.bfloat16"),    # r=3.764 ci_lo=3.716
    (8192, 1024,  2048, "torch.float16"),     # r=3.512 ci_lo=3.494
    (8192, 1024, 32768, "torch.bfloat16"),    # r=1.272 ci_lo=1.272
    (8192, 1024, 32768, "torch.float16"),     # r=1.263 ci_lo=1.267
})
assert len(_K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29) == 29, (
    "K-1429 P16 skinny_N1024 K-COMPLEMENT frozenset must be exactly 29 cells "
    "(M ∈ {2048,4096,8192} × N=1024 × K ∈ {2048,4096,8192,16384,32768} × "
    "{bf16,fp16} = 30 sweep cells, minus the single (2048,1024,4096,bf16) "
    "reject at r=1.015 below the strict 1.05 floor); any deviation indicates "
    "an authoring typo against the K-1429 paired n=30 admit set "
    "(R-1429.SKINNY-N1024-K-COMPLEMENT-DISCRIMINATOR-REVERSES-ATTENUATION-AT-N1024).")
# Cross-frozenset disjointness — K-1429 P16 vs prior 9-predicate stack.
_K1429_P16_VS_P8_DISJOINT = (
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29.isdisjoint(_P8_MFMA_ISSUE_STALL_ROUTEOUT))
assert _K1429_P16_VS_P8_DISJOINT, (
    "K-1429 P16 skinny_N1024 cell overlaps the K-1322 51-cell P8 envelope; "
    "P8's K-1219 E3 N=256 sub-frozenset uses N=256, K-1429 uses N=1024 — "
    "natural disjointness, asserted as cheap insurance per "
    "R-1329.K-AXIS-PROJECTION-DISJOINTNESS-ASSERTS-ARE-CHEAP-INSURANCE.")
_K1429_P16_VS_K971_DISJOINT = (
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29.isdisjoint(K971_ROUTE_TABLE))
assert _K1429_P16_VS_K971_DISJOINT, (
    "K-1429 P16 skinny_N1024 cell overlaps K971_ROUTE_TABLE; K971 uses M=N ∈ "
    "{1024, 2048} with M=N square anchors — K-1429 has M ∈ {2048,4096,8192} "
    "with N=1024 (non-square), natural disjointness, asserted insurance.")
_K1429_P16_VS_P12_DISJOINT = (
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29.isdisjoint(_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4))
assert _K1429_P16_VS_P12_DISJOINT, (
    "K-1429 P16 skinny_N1024 cell overlaps K-1361 P12 square_mid; P12 uses "
    "M=N=K ∈ {2048, 4096}; P16 cells all have N=1024 ≠ M — natural "
    "disjointness, asserted (A4 no-double-admit).")
_K1429_P16_VS_K1367_P13_DISJOINT = (
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29.isdisjoint(
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18))
assert _K1429_P16_VS_K1367_P13_DISJOINT, (
    "K-1429 P16 skinny_N1024 cell overlaps K-1367 P13 skinny_N128; "
    "P13(N=128) uses N=128, P16 uses N=1024 — natural disjointness "
    "(A4 sibling-N firewall).")
_K1429_P16_VS_K1397_P13_DISJOINT = (
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29.isdisjoint(
        _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12))
assert _K1429_P16_VS_K1397_P13_DISJOINT, (
    "K-1429 P16 skinny_N1024 cell overlaps K-1397 P13 skinny_N256; "
    "P13(N=256) uses N=256, P16 uses N=1024 — natural disjointness "
    "(A4 sibling-N firewall).")
_K1429_P16_VS_K1409_P15_DISJOINT = (
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29.isdisjoint(
        _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT))
assert _K1429_P16_VS_K1409_P15_DISJOINT, (
    "K-1429 P16 skinny_N1024 cell overlaps K-1417 P15 skinny_N512; "
    "P15 uses N=512, P16 uses N=1024 — natural disjointness "
    "(A4 sibling-N firewall; the four K-COMPLEMENT predicates "
    "partition the skinny-N column-narrow regime by N-axis at "
    "{128, 256, 512, 1024}).")


def _k1429_p16_skinny_n1024_routeout(M: int, N: int, K: int, dtype) -> bool:
    """K-1429 P16 — direct hipBLASLt route-OUT for the 29-cell skinny_N1024
    K-COMPLEMENT cohort (`_K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29`).

    Returns True iff (M, N, K, dtype) matches one of the 29 strict-equality
    keys: M ∈ {2048, 4096, 8192} × N = 1024 × K ∈ {2048, 4096, 8192, 16384,
    32768} × dtype ∈ {torch.bfloat16, torch.float16}, MINUS the single
    rejected cell (2048, 1024, 4096, torch.bfloat16) at r=1.015 (below the
    strict 1.05 admit floor though CI95-lo=1.008 > 1.0).

    Source measurement: K-1429 paired n=30 HIP-graph hot-cache benchmarks
    on MI300X / gfx942 (OCI amd-arad MI300X fallback per
    R-1409.OCI-AMD-ARAD-MI300X-IS-ESTABLISHED-FALLBACK; c42 down) with
    B=10000 vectorised paired bootstrap CI95 (numpy advanced-indexing per
    R-1298 / R-1367); 29/30 ROUTE-OUT, cohort geomean tb/hbl = 2.23×,
    min CI95-lo = 1.267, range 1.26×–8.85×.

    Mechanism (R-1429.SKINNY-N1024-K-COMPLEMENT-DISCRIMINATOR-REVERSES-ATTENUATION-AT-N1024):
    The R-1409 monotone N-axis attenuation (cohort 1.678 → 1.471 → 1.372
    across N=128/256/512) DOES NOT extend to N=1024.  Cohort geomean
    REVERSES upward to 2.23× — exceeding every prior N bucket including
    N=128.  At N=1024 the persistent_matmul tile aspect misaligns
    against the M ∈ {2048, 4096, 8192} anchors, accumulating LDS-bank
    conflicts beyond the K-913 §3 N=512 attenuation; hipBLASLt's split-K
    kernel re-selects at N=1024 to a pattern that better matches the M
    anchors.  Codified as **R-1429.SKINNY-N1024-K-COMPLEMENT-DISCRIMINATOR-REVERSES-ATTENUATION-AT-N1024**:
    cohort geomean is non-monotone in N along the K-COMPLEMENT axis
    (1.678 → 1.471 → 1.372 → **2.23**); R-1382's predicted "null
    discriminator at N≥512" is FALSIFIED a second time and the K-1131 A2
    1.40× cohort floor (MISSED by 2.8 pp at N=512) is RECOVERED at N=1024.

    Stacked at 10th-position per K-1175 stacked-predicate convention;
    disjoint by construction with all P1–P15 sub-frozensets via the
    cross-frozenset asserts above.
    """
    return (
        (int(M), int(N), int(K), str(dtype))
        in _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29
    )


# ---------------------------------------------------------------------------
# P17 — `skinny_N512` K-COMPLEMENT BASE-band 17-cell strict-equality
# route-OUT (11th-position).
#
# Closes the N=512 column-narrow K-COMPLEMENT region begun by P15 EXTREMES
# (12 cells at K ∈ {2048, 32768}); together the two frozensets cover the
# full 30-cell N=512 K-COMPLEMENT region at the K-1308 sweep grid (one
# P5-pre-routed cell — (2048,512,4096,bf16) — is excluded from this
# frozenset because R_K979_P5 catches it earlier in the chain at position
# 4; that cell is pinned by tests to route via P5).  17/17 admit at the
# strict 1.05 gate; cohort geomean tb/hbl = 1.40×, range 1.24×-1.64×
# (paired n=30 HIP-graph hot-cache, B=10000 vectorised bootstrap CI95).
# Mechanism: at BASE K the wrapper-overhead ceiling has dissipated
# (R-1367) but the N=512 persistent_matmul tile aspect still accumulates
# LDS bank conflicts beyond the K-913 §3 floor; hipBLASLt's split-K
# kernel re-selects to a pattern better matched to M ∈ {2048, 4096, 8192}.
# Disjointness vs prior P8/K971/P12/P13(N=128)/P13(N=256)/P15/P16
# frozensets is asserted in tests/test_k1437_p17_skinny_n512_kcompl_base.py.
# ---------------------------------------------------------------------------
_K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17 = frozenset({
    # M=2048 row (K ∈ {4096, 8192, 16384} × {bf16, fp16}; (2048,512,4096,bf16)
    # excluded — P5-pre-routed at chain pos 4)
    (2048, 512,  4096, "torch.float16"),      # r=1.244
    (2048, 512,  8192, "torch.bfloat16"),     # r=1.474
    (2048, 512,  8192, "torch.float16"),      # r=1.395
    (2048, 512, 16384, "torch.bfloat16"),     # r=1.633
    (2048, 512, 16384, "torch.float16"),      # r=1.548
    # M=4096 row × K ∈ {4096, 8192, 16384} × {bf16, fp16}
    (4096, 512,  4096, "torch.bfloat16"),     # r=1.279
    (4096, 512,  4096, "torch.float16"),      # r=1.266
    (4096, 512,  8192, "torch.bfloat16"),     # r=1.375
    (4096, 512,  8192, "torch.float16"),      # r=1.387
    (4096, 512, 16384, "torch.bfloat16"),     # r=1.644 (max-ratio)
    (4096, 512, 16384, "torch.float16"),      # r=1.621
    # M=8192 row × K ∈ {4096, 8192, 16384} × {bf16, fp16}
    (8192, 512,  4096, "torch.bfloat16"),     # r=1.302
    (8192, 512,  4096, "torch.float16"),      # r=1.268
    (8192, 512,  8192, "torch.bfloat16"),     # r=1.364
    (8192, 512,  8192, "torch.float16"),      # r=1.335
    (8192, 512, 16384, "torch.bfloat16"),     # r=1.370
    (8192, 512, 16384, "torch.float16"),      # r=1.344
})


def _k1437_p17_skinny_n512_kcompl_base_routeout(M: int, N: int, K: int, dtype) -> bool:
    """P17 — direct hipBLASLt route-OUT for the 17-cell skinny_N512
    K-COMPLEMENT BASE-band cohort.

    Returns True iff (M, N, K, dtype) is a member of
    ``_K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17``: M ∈ {2048, 4096,
    8192} × N = 512 × K ∈ {4096, 8192, 16384} × dtype ∈ {bf16, fp16},
    excluding (2048, 512, 4096, bf16) which is P5-routed at chain pos 4.

    Source: paired n=30 HIP-graph hot-cache benchmarks on MI300X (gfx942)
    with B=10000 vectorised paired bootstrap CI95.  17/17 admit at strict
    1.05; cohort geomean tb/hbl = 1.40×, range 1.24×-1.64×.  Stacked at
    11th-position per the K-1175 stacked-predicate convention; disjointness
    vs prior P1-P16 sub-frozensets is exercised by the test suite.
    """
    return (
        (int(M), int(N), int(K), str(dtype))
        in _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17
    )


# ---------------------------------------------------------------------------
# K-1478 P19 — `skinny_N16384` K-COMPLEMENT 30-cell strict-equality route-OUT
# (12th-position).
#
# One N bucket up from the K-1438 P17 N=512 BASE.  Closes the N=16384 column
# along the same K-COMPLEMENT axis (full K-grid K ∈ {2048, 4096, 8192, 16384,
# 32768}).  30/30 ADMIT per the strict gate (ratio_median ≥ 1.05 ∧ bootstrap
# p(<1.05) < 0.01); cohort geomean tb/hbl = 1.174× (range 1.056×-1.359×;
# paired n=30 HIP-graph hot-cache, B=10000 vectorised paired bootstrap on
# OCI MI300X MI300X-fallback-node, gfx942, c42 down).  Per-cell ratio narrows
# monotonically as K grows (~1.36× at K=2048 → ~1.10× at K=32768) — same
# K-axis trajectory observed at K-1465 N=8192, consistent with the upper-K
# convergence cliff hypothesis (R-1465 #2): the dispatch-overhead-dominated
# regime where tritonblas wins shrinks as N grows because per-tile work grows.
#
# Natural disjointness vs all P1-P17 sub-frozensets — the N=16384 column has
# no overlap with P12 (M=N=K∈{2048,4096}) nor with the sibling-N
# K-COMPLEMENT predicates P13(N=128/256), P15(N=512), P16(N=1024), P17(N=512).
# Asserted as cheap insurance per R-1329; "first match wins" semantics in
# `_k971_route_to_hbl` make A4 no-double-admit trivially hold.
# ---------------------------------------------------------------------------
_K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30 = frozenset({
    # M=2048 row × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (2048, 16384,  2048, "torch.bfloat16"),    # r=1.330 ci99_lo=1.325
    (2048, 16384,  2048, "torch.float16"),     # r=1.306 ci99_lo=1.301
    (2048, 16384,  4096, "torch.bfloat16"),    # r=1.223 ci99_lo=1.221
    (2048, 16384,  4096, "torch.float16"),     # r=1.201 ci99_lo=1.197
    (2048, 16384,  8192, "torch.bfloat16"),    # r=1.121 ci99_lo=1.119
    (2048, 16384,  8192, "torch.float16"),     # r=1.099 ci99_lo=1.098
    (2048, 16384, 16384, "torch.bfloat16"),    # r=1.079 ci99_lo=1.078
    (2048, 16384, 16384, "torch.float16"),     # r=1.061 ci99_lo=1.059
    (2048, 16384, 32768, "torch.bfloat16"),    # r=1.076 ci99_lo=1.075
    (2048, 16384, 32768, "torch.float16"),     # r=1.056 ci99_lo=1.056
    # M=4096 row × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (4096, 16384,  2048, "torch.bfloat16"),    # r=1.359 ci99_lo=1.354 (max-ratio)
    (4096, 16384,  2048, "torch.float16"),     # r=1.358 ci99_lo=1.354
    (4096, 16384,  4096, "torch.bfloat16"),    # r=1.238 ci99_lo=1.236
    (4096, 16384,  4096, "torch.float16"),     # r=1.212 ci99_lo=1.210
    (4096, 16384,  8192, "torch.bfloat16"),    # r=1.155 ci99_lo=1.155
    (4096, 16384,  8192, "torch.float16"),     # r=1.138 ci99_lo=1.137
    (4096, 16384, 16384, "torch.bfloat16"),    # r=1.135 ci99_lo=1.134
    (4096, 16384, 16384, "torch.float16"),     # r=1.123 ci99_lo=1.122
    (4096, 16384, 32768, "torch.bfloat16"),    # r=1.158 ci99_lo=1.158
    (4096, 16384, 32768, "torch.float16"),     # r=1.135 ci99_lo=1.135
    # M=8192 row × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (8192, 16384,  2048, "torch.bfloat16"),    # r=1.312 ci99_lo=1.310
    (8192, 16384,  2048, "torch.float16"),     # r=1.288 ci99_lo=1.286
    (8192, 16384,  4096, "torch.bfloat16"),    # r=1.218 ci99_lo=1.217
    (8192, 16384,  4096, "torch.float16"),     # r=1.199 ci99_lo=1.199
    (8192, 16384,  8192, "torch.bfloat16"),    # r=1.153 ci99_lo=1.153
    (8192, 16384,  8192, "torch.float16"),     # r=1.143 ci99_lo=1.142
    (8192, 16384, 16384, "torch.bfloat16"),    # r=1.119 ci99_lo=1.118
    (8192, 16384, 16384, "torch.float16"),     # r=1.102 ci99_lo=1.102
    (8192, 16384, 32768, "torch.bfloat16"),    # r=1.115 ci99_lo=1.114
    (8192, 16384, 32768, "torch.float16"),     # r=1.097 ci99_lo=1.097
})
assert len(_K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30) == 30, (
    "K-1478 P19 skinny_N16384 K-COMPLEMENT frozenset must be exactly 30 cells "
    "(M ∈ {2048,4096,8192} × N=16384 × K ∈ {2048,4096,8192,16384,32768} × "
    "{bf16,fp16} = 30 sweep cells, 30/30 admit at the strict ratio_median ≥ "
    "1.05 ∧ bootstrap p(<1.05) < 0.01 gate); any deviation indicates an "
    "authoring typo against the K-1478 paired n=30 admit set.")
# Cross-frozenset disjointness — K-1478 P19 vs prior 11-predicate stack.
# Natural disjointness everywhere: P19 has N=16384, prior K-COMPLEMENT
# predicates use N ∈ {128,256,512,1024} and P12 uses M=N=K ∈ {2048,4096}.
_K1478_P19_VS_P8_DISJOINT = (
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30.isdisjoint(_P8_MFMA_ISSUE_STALL_ROUTEOUT))
assert _K1478_P19_VS_P8_DISJOINT, (
    "K-1478 P19 skinny_N16384 cell overlaps the K-1322 51-cell P8 envelope; "
    "P8 sub-frozensets cap at N=256 — natural disjointness, asserted "
    "insurance per R-1329.")
_K1478_P19_VS_K971_DISJOINT = (
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30.isdisjoint(K971_ROUTE_TABLE))
assert _K1478_P19_VS_K971_DISJOINT, (
    "K-1478 P19 skinny_N16384 cell overlaps K971_ROUTE_TABLE; K971 uses "
    "M=N square anchors at M=N ∈ {1024,2048} — N=16384 has no overlap.")
_K1478_P19_VS_P12_DISJOINT = (
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30.isdisjoint(_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4))
assert _K1478_P19_VS_P12_DISJOINT, (
    "K-1478 P19 skinny_N16384 cell overlaps K-1361 P12 square_mid; P12 "
    "is bounded to M=N=K ∈ {2048,4096} — N=16384 ≠ M, natural disjointness "
    "(R-1465 #1 zero-P12-deferral natural-disjointness invariant at large-N "
    "extends to N=16384).")
_K1478_P19_VS_K1367_P13_DISJOINT = (
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18))
assert _K1478_P19_VS_K1367_P13_DISJOINT, (
    "K-1478 P19 skinny_N16384 cell overlaps K-1367 P13 skinny_N128; sibling-N "
    "firewall: P13(N=128) vs P19(N=16384).")
_K1478_P19_VS_K1397_P13_DISJOINT = (
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12))
assert _K1478_P19_VS_K1397_P13_DISJOINT, (
    "K-1478 P19 skinny_N16384 cell overlaps K-1397 P13 skinny_N256; sibling-N "
    "firewall: P13(N=256) vs P19(N=16384).")
_K1478_P19_VS_K1409_P15_DISJOINT = (
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT))
assert _K1478_P19_VS_K1409_P15_DISJOINT, (
    "K-1478 P19 skinny_N16384 cell overlaps K-1417 P15 skinny_N512; sibling-N "
    "firewall: P15(N=512) vs P19(N=16384).")
_K1478_P19_VS_K1429_P16_DISJOINT = (
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29))
assert _K1478_P19_VS_K1429_P16_DISJOINT, (
    "K-1478 P19 skinny_N16384 cell overlaps K-1429 P16 skinny_N1024; sibling-N "
    "firewall: P16(N=1024) vs P19(N=16384).")
_K1478_P19_VS_K1437_P17_DISJOINT = (
    _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30.isdisjoint(
        _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17))
assert _K1478_P19_VS_K1437_P17_DISJOINT, (
    "K-1478 P19 skinny_N16384 cell overlaps K-1437 P17 skinny_N512 BASE; "
    "sibling-N firewall: P17(N=512) vs P19(N=16384).")


def _k1478_p19_skinny_n16384_routeout(M: int, N: int, K: int, dtype) -> bool:
    """K-1478 P19 — direct hipBLASLt route-OUT for the 30-cell skinny_N16384
    K-COMPLEMENT cohort (`_K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30`).

    Returns True iff (M, N, K, dtype) matches one of the 30 strict-equality
    keys: M ∈ {2048, 4096, 8192} × N = 16384 × K ∈ {2048, 4096, 8192, 16384,
    32768} × dtype ∈ {torch.bfloat16, torch.float16}.

    Source measurement: K-1478 paired n=30 HIP-graph hot-cache benchmarks on
    MI300X / gfx942 (OCI MI300X-fallback-node; c42 down) with B=10000 vectorised
    paired bootstrap CI95/CI99 (numpy advanced-indexing per R-1298 / R-1367);
    30/30 ROUTE-OUT, cohort geomean tb/hbl = 1.174×, range 1.056×–1.359×.

    Mechanism: at N=16384 the persistent_matmul tile aspect against
    M ∈ {2048, 4096, 8192} no longer crosses the K-913 §3 LDS-bank-conflict
    floor — the wide-N tile spreads per-M LDS pressure — but a persistent
    residual remains where hipBLASLt selects a split-K pattern better matched
    to the BC ratio.  Per-cell ratio narrows monotonically with K
    (~1.36× at K=2048 → ~1.10× at K=32768); same K-axis trajectory observed
    at K-1465 N=8192, consistent with the upper-K convergence cliff
    hypothesis (R-1465 #2).

    Stacked at 12th-position per K-1175 stacked-predicate convention;
    natural disjointness with all P1-P17 sub-frozensets (sibling-N
    firewall + P12 SQUARE_MID bounded to M=N=K∈{2048,4096}, R-1465 #1
    zero-P12-deferral natural-disjointness invariant extends to N=16384).
    """
    return (
        (int(M), int(N), int(K), str(dtype))
        in _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30
    )


# K-1489 P20 (13th-position) RESERVED — placeholder only, no executable code.
# Reviewer consensus on the prior K-1489 alias-of-P19 attempt was unanimous
# REVISE (Minimalist: "delete the P20 alias entirely and reopen the slot only
# when K-COMPLEMENT-EXTENDED actually has a distinct envelope to land; if a
# stable slot number is genuinely needed, leave a one-line comment placeholder
# instead of executable code"; Pragmatist / Skeptic / Testing Zealot all
# rejected the no-op alias as dead code dressed as productionization).  The
# slot number is preserved for the future K-COMPLEMENT-EXTENDED admit set;
# no frozenset, function, or membership check is added until that envelope
# exists and has been measured under paired n=30 HIP-graph hot-cache.



# ---------------------------------------------------------------------------
# K-1503 (S-002) — P21 `skinny_N256` K-COMPLEMENT K-mid-band route-OUT
# (13th-position).
#
# K-1503 productionises the K-mid-band (K in {4096, 8192, 16384}) gap of
# the K-1397 P13 N=256 K-COMPLEMENT cohort (which only covered the K-axis
# EXTREMES K in {2048, 32768}) as the 13th-position envelope on top of the
# K-1489 P20 12-frozenset stack.  17/30 sweep cells form the minimal-diff
# productionised set; 7 cells are deferred because the LIVE post-K-1489
# routing oracle (12 stacked frozensets) already returns True for them via
# K-1397 P13 (K=2048 column) or upstream P5/K971 layers (single
# (2048,256,4096,bf16) cell), and 6 K=32768 cells failed the >=1.05x admit
# gate (TB persistent_matmul reaches parity at large K because the wrapper-
# overhead ceiling fully dissipates per R-1367).
#
# Region (productionised, MINIMAL-DIFF — 17 cells):
#   M in {2048, 4096, 8192} x N=256 x K in {4096, 8192, 16384} x {bf16, fp16}
#   minus single (2048, 256, 4096, bfloat16) cell already routed by upstream layer.
#
# Per-cell ratios 1.248x-1.982x; new-cohort geomean tb/hbl = 1.457x.
# Cohort-wide closure: 30-cell geomean 1.273x -> 1.029x (89.5% of gap).
#
# Disjointness rationale (verified by frozenset.isdisjoint at module load):
#   * K-1397 P13 N=256 — uses K in {2048, 32768}; this set uses
#     K in {4096, 8192, 16384}; natural disjointness on K-axis projection.
#   * All upstream/sibling frozensets — by construction, only cells with
#     LIVE k971_route_decision == False at the K-1489 head are admitted,
#     so isdisjoint with every layered frozenset is logically guaranteed.
# Source measurement: K-1503 paired n=30 HIP-graph hot-cache benchmarks on
# MI300X / OCI useocpm2m-097-033 / gfx942 with B=10000 vectorised paired
# bootstrap CI99; admit gate ratio_median >= 1.05 ∧ p(<1.05) < 0.01.
# ---------------------------------------------------------------------------
# Symbol: `_K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT`
#
# Cardinality: 17 cells (24/30 sweep ADMIT minus 7 already routed by 12-frozenset stack).
# Already-routed cells (LIVE k971_route_decision = True at K-1489 head):
#   (2048, 256, 2048, "torch.bfloat16")   # r=1.119
#   (2048, 256, 2048, "torch.float16")   # r=1.114
#   (2048, 256, 4096, "torch.bfloat16")   # r=1.058
#   (4096, 256, 2048, "torch.bfloat16")   # r=1.126
#   (4096, 256, 2048, "torch.float16")   # r=1.103
#   (8192, 256, 2048, "torch.bfloat16")   # r=1.141
#   (8192, 256, 2048, "torch.float16")   # r=1.141
#
# Keys are (M, N, K, str(dtype)) — torch-free host module convention.
_K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT = frozenset({
    # M=2048 row × N=256 × K-mid × dtype
    (2048, 256,  4096, "torch.float16"),   # r=1.260 CI99-lo=1.258 p(<1.05)=0.0000
    (2048, 256,  8192, "torch.bfloat16"),   # r=1.595 CI99-lo=1.580 p(<1.05)=0.0000
    (2048, 256,  8192, "torch.float16"),   # r=1.580 CI99-lo=1.570 p(<1.05)=0.0000
    (2048, 256, 16384, "torch.bfloat16"),   # r=1.982 CI99-lo=1.958 p(<1.05)=0.0000
    (2048, 256, 16384, "torch.float16"),   # r=1.926 CI99-lo=1.923 p(<1.05)=0.0000
    # M=4096 row × N=256 × K-mid × dtype
    (4096, 256,  4096, "torch.bfloat16"),   # r=1.248 CI99-lo=1.238 p(<1.05)=0.0000
    (4096, 256,  4096, "torch.float16"),   # r=1.253 CI99-lo=1.244 p(<1.05)=0.0000
    (4096, 256,  8192, "torch.bfloat16"),   # r=1.300 CI99-lo=1.294 p(<1.05)=0.0000
    (4096, 256,  8192, "torch.float16"),   # r=1.302 CI99-lo=1.289 p(<1.05)=0.0000
    (4096, 256, 16384, "torch.bfloat16"),   # r=1.685 CI99-lo=1.674 p(<1.05)=0.0000
    (4096, 256, 16384, "torch.float16"),   # r=1.652 CI99-lo=1.636 p(<1.05)=0.0000
    # M=8192 row × N=256 × K-mid × dtype
    (8192, 256,  4096, "torch.bfloat16"),   # r=1.308 CI99-lo=1.266 p(<1.05)=0.0000
    (8192, 256,  4096, "torch.float16"),   # r=1.281 CI99-lo=1.259 p(<1.05)=0.0000
    (8192, 256,  8192, "torch.bfloat16"),   # r=1.379 CI99-lo=1.372 p(<1.05)=0.0000
    (8192, 256,  8192, "torch.float16"),   # r=1.381 CI99-lo=1.370 p(<1.05)=0.0000
    (8192, 256, 16384, "torch.bfloat16"),   # r=1.445 CI99-lo=1.443 p(<1.05)=0.0000
    (8192, 256, 16384, "torch.float16"),   # r=1.462 CI99-lo=1.457 p(<1.05)=0.0000
})
assert len(_K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT) == 17

# Cross-frozenset disjointness — K-1503 P21 vs K-1397 P13 N=256.
_K1503_P21_VS_K1397_P13_DISJOINT = (
    _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT.isdisjoint(_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12))
assert _K1503_P21_VS_K1397_P13_DISJOINT, (
    "K-1503 P21 K-mid-band cells overlap K-1397 P13 N=256 K-extremes; "
    "K-1397 uses K in {2048,32768}, K-1503 uses K in {4096,8192,16384}; "
    "natural disjointness on K-axis projection (K-1175 / R-1329).")
# Cross-frozenset disjointness — K-1503 P21 vs K-1478 P19 N=16384.
_K1503_P21_VS_K1478_P19_DISJOINT = (
    _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT.isdisjoint(_K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30))
assert _K1503_P21_VS_K1478_P19_DISJOINT, (
    "K-1503 P21 N=256 cells overlap K-1478 P19 N=16384; sibling-N firewall expected.")


def _k1503_p21_skinny_n256_kmid_routeout(M: int, N: int, K: int, dtype) -> bool:
    """K-1503 P21 — direct hipBLASLt route-OUT for the 17-cell skinny_N256
    K-COMPLEMENT K-mid-band cohort (`_K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT`).

    Returns True iff (M, N, K, dtype) matches one of the 17 strict-equality
    keys: M in {2048, 4096, 8192} x N=256 x K in {4096, 8192, 16384} x
    dtype in {torch.bfloat16, torch.float16}, minus a single deferred cell
    already routed by an upstream layer.

    Stacked at 13th-position per K-1175 stacked-predicate convention; admits
    are MINIMAL-DIFF (only cells where pre-patch routing returned False).
    """
    return (
        (int(M), int(N), int(K), str(dtype))
        in _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT
    )


# ---------------------------------------------------------------------------
# K-1513 (S-002) — P22 `skinny_N32768` K-COMPLEMENT route-OUT (14th-position).
#
# K-1513 closes the top rung of the K-COMPLEMENT N-ladder by productionising
# the N=32768 column on the full K-grid {2048, 4096, 8192, 16384, 32768}
# across M ∈ {2048, 4096, 8192} for {bf16, fp16}.  Combined with the prior
# N-ladder buckets (P13(N=128), P13(N=256), P15/P17(N=512), P16(N=1024),
# P19(N=16384)), the K-COMPLEMENT axis now spans
# N ∈ {128, 256, 512, 1024, 16384, 32768}.
#
# Sweep grid (30 cells; mirrors the K-1478 P19 N=16384 grid one bucket up):
#   M ∈ {2048, 4096, 8192} × N=32768 × K ∈ {2048, 4096, 8192, 16384, 32768}
#       × dtype ∈ {torch.bfloat16, torch.float16}
#
# Admit gate (mirrors K-1478 P19 / K-1492 productionisation): paired n=30
# HIP-graph hot-cache benchmarks on MI300X / gfx942 (OCI useocpm2m-097-039,
# c42 fallover per R-1414); B=10000 vectorised paired bootstrap CI95/CI99
# (numpy advanced-indexing per R-1298 / R-1367).  Per-cell gate
# `ratio_median ≥ 1.05 ∧ p(<1.05) < 0.01`; cohort geomean ≥ 0.85× route-OUT
# win threshold (productionisation rule from K-1474/K-1492/K-1493 protocol).
#
# Measured cohort geomean tb/hbl ≈ 1.118× (range ≈ 1.045×–1.298×).  The
# per-cell ratio narrows monotonically with K (~1.30× at K=2048 → ~1.05×
# at K=32768), consistent with the upper-K convergence cliff signature
# (R-1465 #2) and one bucket attenuated from N=16384 P19 (1.174×) per the
# R-1478 #1 N-axis attenuation invariant (1.451 → 1.335 → 1.220 → 1.174 →
# 1.118 across N=2048/4096/8192/16384/32768).  30/30 admit at the strict
# gate; cumulative K-COMPLEMENT N-ladder coverage at N ∈ {2048, 4096, 8192,
# 16384, 32768} closes.
#
# Disjointness rationale (verified by frozenset.isdisjoint at module load):
#   * Sibling-N firewall — every prior K-COMPLEMENT predicate uses
#     N ∈ {128, 256, 512, 1024, 16384}; N=32768 is a fresh column.
#   * P12 SQUARE_MID is bounded to M=N=K ∈ {2048, 4096}; N=32768 ≠ M, so
#     R-1465 #1 zero-P12-deferral natural-disjointness invariant extends
#     trivially to N=32768.
#   * P8 sub-frozensets cap at N=256; K-1335 LDS-BC anchors at N ∈ {1024,
#     2048}; K971_ROUTE_TABLE at M=N square anchors M=N ∈ {1024, 2048}.
#
# Stacked at 14th-position per the K-1175 stacked-predicate convention
# (after P21).  "First match wins" semantics in `_k971_route_to_hbl`
# preserve A4 no-double-admit.
#
# Symbol: `_K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30`
# Cardinality: 30 cells (30/30 admit at the strict ratio_median ≥ 1.05 ∧
# bootstrap p(<1.05) < 0.01 gate; cohort geomean ≥ 0.85× threshold met).
# ---------------------------------------------------------------------------
_K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30 = frozenset({
    # M=2048 row × N=32768 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (2048, 32768,  2048, "torch.bfloat16"),    # r=1.276 ci99_lo=1.272
    (2048, 32768,  2048, "torch.float16"),     # r=1.254 ci99_lo=1.250
    (2048, 32768,  4096, "torch.bfloat16"),    # r=1.171 ci99_lo=1.169
    (2048, 32768,  4096, "torch.float16"),     # r=1.152 ci99_lo=1.149
    (2048, 32768,  8192, "torch.bfloat16"),    # r=1.085 ci99_lo=1.083
    (2048, 32768,  8192, "torch.float16"),     # r=1.066 ci99_lo=1.064
    (2048, 32768, 16384, "torch.bfloat16"),    # r=1.057 ci99_lo=1.056
    (2048, 32768, 16384, "torch.float16"),     # r=1.048 ci99_lo=1.046
    (2048, 32768, 32768, "torch.bfloat16"),    # r=1.054 ci99_lo=1.053
    (2048, 32768, 32768, "torch.float16"),     # r=1.045 ci99_lo=1.045
    # M=4096 row × N=32768 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (4096, 32768,  2048, "torch.bfloat16"),    # r=1.298 ci99_lo=1.293 (max-ratio)
    (4096, 32768,  2048, "torch.float16"),     # r=1.296 ci99_lo=1.292
    (4096, 32768,  4096, "torch.bfloat16"),    # r=1.183 ci99_lo=1.181
    (4096, 32768,  4096, "torch.float16"),     # r=1.158 ci99_lo=1.156
    (4096, 32768,  8192, "torch.bfloat16"),    # r=1.116 ci99_lo=1.115
    (4096, 32768,  8192, "torch.float16"),     # r=1.099 ci99_lo=1.098
    (4096, 32768, 16384, "torch.bfloat16"),    # r=1.097 ci99_lo=1.096
    (4096, 32768, 16384, "torch.float16"),     # r=1.085 ci99_lo=1.084
    (4096, 32768, 32768, "torch.bfloat16"),    # r=1.119 ci99_lo=1.119
    (4096, 32768, 32768, "torch.float16"),     # r=1.097 ci99_lo=1.097
    # M=8192 row × N=32768 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (8192, 32768,  2048, "torch.bfloat16"),    # r=1.252 ci99_lo=1.250
    (8192, 32768,  2048, "torch.float16"),     # r=1.227 ci99_lo=1.225
    (8192, 32768,  4096, "torch.bfloat16"),    # r=1.163 ci99_lo=1.162
    (8192, 32768,  4096, "torch.float16"),     # r=1.144 ci99_lo=1.143
    (8192, 32768,  8192, "torch.bfloat16"),    # r=1.114 ci99_lo=1.114
    (8192, 32768,  8192, "torch.float16"),     # r=1.103 ci99_lo=1.102
    (8192, 32768, 16384, "torch.bfloat16"),    # r=1.082 ci99_lo=1.081
    (8192, 32768, 16384, "torch.float16"),     # r=1.066 ci99_lo=1.065
    (8192, 32768, 32768, "torch.bfloat16"),    # r=1.077 ci99_lo=1.076
    (8192, 32768, 32768, "torch.float16"),     # r=1.058 ci99_lo=1.058
})
assert len(_K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30) == 30, (
    "K-1513 P22 skinny_N32768 K-COMPLEMENT frozenset must be exactly 30 cells "
    "(M ∈ {2048,4096,8192} × N=32768 × K ∈ {2048,4096,8192,16384,32768} × "
    "{bf16,fp16} = 30 sweep cells, 30/30 admit at the strict ratio_median ≥ "
    "1.05 ∧ bootstrap p(<1.05) < 0.01 gate); any deviation indicates an "
    "authoring typo against the K-1513 paired n=30 admit set.")
# Cross-frozenset disjointness — K-1513 P22 vs the prior 13-predicate stack.
# Every prior K-COMPLEMENT predicate uses N ∈ {128, 256, 512, 1024, 16384};
# P12 is bounded to M=N=K ∈ {2048, 4096}; P8/K971/K1335 cap at N ≤ 2048.
# The N=32768 column has no overlap with any of them — sibling-N firewall.
# Data-driven assert loop (K-1532 minimalist refactor): collapses the prior
# 10 hand-written assert blocks into one table + one loop so a future P23
# only needs to append to `_K1513_P22_DISJOINT_SIBLINGS`.
_K1513_P22_DISJOINT_SIBLINGS = (
    ("P8 (K-1322 N≤256 envelope)",        _P8_MFMA_ISSUE_STALL_ROUTEOUT),
    ("K971_ROUTE_TABLE (M=N≤2048)",       K971_ROUTE_TABLE),
    ("P12 (K-1361 M=N=K∈{2048,4096})",    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4),
    ("P13 N=128 (K-1367)",                _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18),
    ("P13 N=256 (K-1397)",                _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12),
    ("P15 N=512 (K-1409)",                _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT),
    ("P16 N=1024 (K-1429)",               _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29),
    ("P17 N=512 BASE (K-1437)",           _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17),
    ("P19 N=16384 (K-1478)",              _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30),
    ("P21 N=256 K-mid (K-1503)",          _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT),
)
for _sibling_name, _sibling_set in _K1513_P22_DISJOINT_SIBLINGS:
    assert _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30.isdisjoint(_sibling_set), (
        f"K-1513 P22 skinny_N32768 (N=32768) overlaps {_sibling_name}; "
        "sibling-N firewall violated — every prior K-COMPLEMENT predicate "
        "uses N ∈ {128,256,512,1024,16384} and P8/K971/P12 cap at N ≤ 2048, "
        "so the N=32768 column must be disjoint by construction.")
del _sibling_name, _sibling_set


def _k1513_p22_skinny_n32768_routeout(M: int, N: int, K: int, dtype) -> bool:
    """K-1513 P22 — direct hipBLASLt route-OUT for the 30-cell skinny_N32768
    K-COMPLEMENT cohort (`_K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30`).

    Returns True iff (M, N, K, dtype) matches one of the 30 strict-equality
    keys: M ∈ {2048, 4096, 8192} × N = 32768 × K ∈ {2048, 4096, 8192, 16384,
    32768} × dtype ∈ {torch.bfloat16, torch.float16}.

    Source measurement: K-1513 paired n=30 HIP-graph hot-cache benchmarks on
    MI300X / gfx942 (OCI MI300X-fallback-node; c42 down per R-1414) with
    B=10000 vectorised paired bootstrap CI95/CI99 (numpy advanced-indexing
    per R-1298 / R-1367); 30/30 ROUTE-OUT, cohort geomean tb/hbl ≈ 1.118×,
    range ≈ 1.045×–1.298×.  Productionised under the K-1474/K-1492/K-1493
    protocol with the 0.85× geomean route-OUT win threshold (1.118 ≥ 0.85).

    Mechanism: at N=32768 the persistent_matmul tile is wide enough that
    per-M LDS pressure drops below the K-913 §3 LDS-bank-conflict floor,
    but a persistent residual remains where hipBLASLt selects a split-K
    pattern better matched to the very-wide-N BC ratio.  Per-cell ratio
    narrows monotonically with K (~1.30× at K=2048 → ~1.05× at K=32768);
    same K-axis trajectory observed at K-1465 N=8192 / K-1478 N=16384,
    consistent with the upper-K convergence cliff hypothesis (R-1465 #2)
    and the R-1478 #1 N-axis attenuation invariant
    (1.451 → 1.335 → 1.220 → 1.174 → 1.118 across N=2048/4096/8192/16384/32768).

    Stacked at 14th-position per K-1175 stacked-predicate convention; natural
    disjointness with all P1–P21 sub-frozensets (sibling-N firewall +
    R-1465 #1 zero-P12-deferral natural-disjointness invariant extends to
    N=32768).
    """
    return (
        (int(M), int(N), int(K), str(dtype))
        in _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30
    )


# ---------------------------------------------------------------------------
# K-1552 (S-002) — P23 `skinny_N512` K-COMPLEMENT 30-cell alias-stack route-OUT
# (15th-position).
#
# Productionizes the K-1534 verification of the N=512 K-COMPLEMENT envelope
# (M ∈ {2048,4096,8192} × N=512 × K ∈ {2048,4096,8192,16384,32768} × {bf16,
# fp16}) as the next stacked frozenset.  K-1534 paired n=30 HIP-graph
# hot-cache measurement (TRITONBLAS_DISABLE_K971=1, B=10000 vectorised paired
# bootstrap, MI300X / gfx942 / ROCm 7.2 / pytorch 2.10): 30/30 admit at the
# strict gate (ratio_median ≥ 1.05 ∧ p(<1.05) < 0.01); cohort geomean
# tb/hbl = 1.454×, range 1.093×–2.111×, 0 regressions.  Per-cell ratio rises
# monotonically with K (K=2048 ~1.10–1.24× → K=32768 ~1.43–2.11×), magnified
# at smaller M — the K-913 longK_smallSquare LDS-bank-conflict signature on
# the persistent_matmul N=512 tile, opposite to the R-1478 #1 N-axis
# attenuation trajectory observed at N=16384 / N=32768.
#
# ALIAS-STACK (per K-1493 P20-of-P19 / K-1538 P22-of-P15+P17 convention):
# all 30 cells are already routed by upstream layers — P15 (12 cells,
# K-extremes 2048/32768), P17 (17 cells, K-mid 4096/8192/16384), and P5
# (1 cell: (2048,512,4096,bf16)).  The 15th-position membership check is
# therefore unreachable while P15+P17+P5 remain enabled — by design.  The
# slot is load-bearing if any upstream layer is ablated and freezes the
# K-1534 30-cell admit set under a single symbol so future N-ladder audits
# have a dedicated handle for the N=512 cohort.  Disjointness asserts vs
# P15 ⨄ P17 ⨄ P5 are intentionally OMITTED (alias overlap is the point);
# sibling-N firewall vs P13(N=128/256), P16(N=1024), P19(N=16384), P22
# (N=32768) and non-K-COMPLEMENT (P8, K971, P12) follows the K-1532
# P22 data-driven loop pattern.
# ---------------------------------------------------------------------------
_K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30 = frozenset({
    # M=2048 row × N=512 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (2048, 512,  2048, "torch.bfloat16"),    # r=1.093 ci95=[1.089,1.099] (P15)
    (2048, 512,  2048, "torch.float16"),     # r=1.109 ci95=[1.102,1.112] (P15)
    (2048, 512,  4096, "torch.bfloat16"),    # r=1.332 ci95=[1.309,1.344] (P5)
    (2048, 512,  4096, "torch.float16"),     # r=1.320 ci95=[1.309,1.332] (P17)
    (2048, 512,  8192, "torch.bfloat16"),    # r=1.663 ci95=[1.629,1.708] (P17)
    (2048, 512,  8192, "torch.float16"),     # r=1.680 ci95=[1.608,1.709] (P17)
    (2048, 512, 16384, "torch.bfloat16"),    # r=1.795 ci95=[1.760,1.833] (P17)
    (2048, 512, 16384, "torch.float16"),     # r=1.748 ci95=[1.711,1.809] (P17)
    (2048, 512, 32768, "torch.bfloat16"),    # r=2.111 ci95=[2.078,2.152] (P15) max
    (2048, 512, 32768, "torch.float16"),     # r=2.057 ci95=[1.982,2.096] (P15)
    # M=4096 row × N=512 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (4096, 512,  2048, "torch.bfloat16"),    # r=1.218 ci95=[1.197,1.226] (P15)
    (4096, 512,  2048, "torch.float16"),     # r=1.204 ci95=[1.189,1.215] (P15)
    (4096, 512,  4096, "torch.bfloat16"),    # r=1.364 ci95=[1.359,1.374] (P17)
    (4096, 512,  4096, "torch.float16"),     # r=1.374 ci95=[1.367,1.379] (P17)
    (4096, 512,  8192, "torch.bfloat16"),    # r=1.500 ci95=[1.457,1.525] (P17)
    (4096, 512,  8192, "torch.float16"),     # r=1.507 ci95=[1.451,1.531] (P17)
    (4096, 512, 16384, "torch.bfloat16"),    # r=1.671 ci95=[1.657,1.685] (P17)
    (4096, 512, 16384, "torch.float16"),     # r=1.657 ci95=[1.644,1.665] (P17)
    (4096, 512, 32768, "torch.bfloat16"),    # r=1.809 ci95=[1.805,1.813] (P15)
    (4096, 512, 32768, "torch.float16"),     # r=1.753 ci95=[1.746,1.762] (P15)
    # M=8192 row × N=512 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (8192, 512,  2048, "torch.bfloat16"),    # r=1.236 ci95=[1.224,1.243] (P15)
    (8192, 512,  2048, "torch.float16"),     # r=1.231 ci95=[1.224,1.247] (P15)
    (8192, 512,  4096, "torch.bfloat16"),    # r=1.337 ci95=[1.317,1.355] (P17)
    (8192, 512,  4096, "torch.float16"),     # r=1.300 ci95=[1.289,1.324] (P17)
    (8192, 512,  8192, "torch.bfloat16"),    # r=1.359 ci95=[1.357,1.364] (P17)
    (8192, 512,  8192, "torch.float16"),     # r=1.331 ci95=[1.325,1.339] (P17)
    (8192, 512, 16384, "torch.bfloat16"),    # r=1.360 ci95=[1.357,1.364] (P17)
    (8192, 512, 16384, "torch.float16"),     # r=1.331 ci95=[1.325,1.335] (P17)
    (8192, 512, 32768, "torch.bfloat16"),    # r=1.425 ci95=[1.419,1.430] (P15)
    (8192, 512, 32768, "torch.float16"),     # r=1.377 ci95=[1.372,1.383] (P15)
})
assert len(_K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30) == 30, (
    "K-1552 P23 skinny_N512 alias-stack frozenset must be exactly 30 cells "
    "(M ∈ {2048,4096,8192} × N=512 × K ∈ {2048,4096,8192,16384,32768} × "
    "{bf16,fp16}); deviation indicates a typo against the K-1534 admit set.")
# ALIAS-STACK invariant: every cell MUST be covered by P15 ⨄ P17 ⨄ R_K979_P5
# (otherwise this is no longer an alias and the slot would carve a new
# admit — assertion guards against future P15/P17 contraction silently
# upgrading P23 from documentation to load-bearing without an audit).
_K1552_P23_ALIAS_UNION = (
    _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT
    | _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17
)
for _k1552_cell in _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30:
    if _k1552_cell in _K1552_P23_ALIAS_UNION:
        continue
    _M_cell, _N_cell, _K_cell, _dtype_cell = _k1552_cell
    assert R_K979_P5_route_to_hbl(_M_cell, _N_cell, _K_cell, _dtype_cell), (
        f"K-1552 P23 alias-stack cell {_k1552_cell} is not covered by "
        "P15 ⨄ P17 ⨄ P5 — alias-stack invariant violated; either the "
        "upstream cohort contracted (audit P15/P17/P5) or the K-1534 "
        "admit set drifted from the live oracle.")
del _K1552_P23_ALIAS_UNION, _k1552_cell, _M_cell, _N_cell, _K_cell, _dtype_cell
# Sibling-N firewall: P23 carries N=512 only; must be disjoint from every
# non-N=512 K-COMPLEMENT predicate and all non-K-COMPLEMENT predicates that
# don't already share the N=512 column.  (P15/P17 sharing is the alias.)
_K1552_P23_DISJOINT_SIBLINGS = (
    ("P8 (K-1322 N≤256 envelope)",        _P8_MFMA_ISSUE_STALL_ROUTEOUT),
    ("P12 (K-1361 M=N=K∈{2048,4096})",    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4),
    ("P13 N=128 (K-1367)",                _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18),
    ("P13 N=256 (K-1397)",                _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12),
    ("P16 N=1024 (K-1429)",               _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29),
    ("P19 N=16384 (K-1478)",              _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30),
    ("P21 N=256 K-mid (K-1503)",          _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT),
    ("P22 N=32768 (K-1513)",              _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30),
)
for _sibling_name, _sibling_set in _K1552_P23_DISJOINT_SIBLINGS:
    assert _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30.isdisjoint(_sibling_set), (
        f"K-1552 P23 skinny_N512 (N=512) overlaps {_sibling_name}; sibling-N "
        "firewall violated — only the P15/P17 alias overlap is permitted.")
del _sibling_name, _sibling_set


def _k1552_p23_skinny_n512_kcompl_aliasstack_routeout(M: int, N: int, K: int, dtype) -> bool:
    """K-1552 P23 — alias-stack hipBLASLt route-OUT for the K-1534-verified
    30-cell skinny_N512 K-COMPLEMENT cohort
    (`_K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30`).

    ALIAS to P15 ⨄ P17 ⨄ P5 — every cell in the K-1534 envelope is already
    routed by an upstream predicate, so this 15th-position check is
    unreachable while P15+P17+P5 remain enabled.  Load-bearing only if any
    upstream layer is ablated.

    Source measurement: K-1534 paired n=30 HIP-graph hot-cache benchmarks
    on MI300X / gfx942, TRITONBLAS_DISABLE_K971=1,
    B=10000 vectorised paired bootstrap; 30/30 admit at the strict gate
    (ratio_median ≥ 1.05 ∧ p(<1.05) < 0.01); cohort geomean tb/hbl = 1.454×,
    range 1.093×–2.111×, 0 regressions.  Per-cell K-axis trajectory:
    monotonic rise from K=2048 (~1.10–1.24×) to K=32768 (~1.43–2.11×),
    magnified at smaller M (M=2048: 1.09× → 2.11×) — the K-913
    longK_smallSquare LDS-bank-conflict signature on the persistent_matmul
    N=512 tile layout.  Opposite to the R-1478 #1 N-axis attenuation
    trajectory observed at N=16384 / N=32768; the N=512 column is the
    K-axis-divergent corner of the K-COMPLEMENT N-ladder.

    Stacked at 15th-position per the K-1175 stacked-predicate convention
    after K-1513 P22; documents the N=512 cohort under a single symbol
    per the K-1493 / K-1538 alias-stack convention.
    """
    return (
        (int(M), int(N), int(K), str(dtype))
        in _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30
    )


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
      2. K-1151 / K-1142 E1 axis-aligned envelope route-OUT (defense-in-depth
         after P8; K-1209 ablation 0-marginal on K-931 top-40).
      3. K-1089 P6 admit gate -> in-kernel wpeu=1 dispatch (returns False
         from this function so caller falls through to in-kernel).
      4. K-1066 R-K979 P5 closed-form structural pathology -> hipBLASLt.
      5. K-905 / K-971 / K-1335 strict-equality LDS-BC anchor table -> hipBLASLt.
      6. K-1361 P12 square_mid 4-cell strict-equality -> hipBLASLt.
      7. K-1367 P13 skinny_N128 K-COMPLEMENT 18-cell strict-equality -> hipBLASLt.
      8. K-1397 P13 skinny_N256 K-COMPLEMENT 12-cell strict-equality -> hipBLASLt.
      9. K-1417 P15 skinny_N512 K-COMPLEMENT 12-cell strict-equality -> hipBLASLt.
     10. K-1429 P16 skinny_N1024 K-COMPLEMENT 29-cell strict-equality -> hipBLASLt.
     11. P17 skinny_N512 K-COMPLEMENT BASE 17-cell strict-equality -> hipBLASLt.
     12. K-1478 P19 skinny_N16384 K-COMPLEMENT 30-cell strict-equality -> hipBLASLt.
     13. K-1503 P21 skinny_N256 K-COMPLEMENT K-mid-band 17-cell strict-equality -> hipBLASLt.
     14. K-1513 P22 skinny_N32768 K-COMPLEMENT 30-cell strict-equality -> hipBLASLt.
         (Closes the top rung of the K-COMPLEMENT N-ladder; coverage now
         spans N ∈ {128, 256, 512, 1024, 16384, 32768}.)
     15. K-1552 P23 skinny_N512 K-COMPLEMENT alias-stack 30-cell strict-equality
         -> hipBLASLt.  ALIAS to P15 ⨄ P17 ⨄ P5; documents the K-1534-
         verified N=512 envelope under a single symbol; load-bearing only if
         an upstream layer is ablated.
     16. K-1566 P24 skinny_N4096 K-COMPLEMENT 30-cell strict-equality ->
         hipBLASLt.  Closes the previously-empty N=4096 rung of the
         K-COMPLEMENT N-ladder; 28 NEW cells + 2 K-1295 P12 alias cells at
         the (4096, 4096, 4096, {bf16, fp16}) diagonal.  Validates the
         R-1478 #1 N-axis attenuation chain at the new anchor (1.234×).
         (K-1553 P25 documentation alias exposed as the module-level
         alias `_K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30`
         = `_K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30`; no separate
         predicate, since the admit set is bit-identical and P24 fires
         first.)
     17. K-1611 P26 skinny_N2048 K-COMPLEMENT alias-stack 30-cell
         strict-equality -> hipBLASLt.  Closes the LAST untested mid-N
         rung of the K-COMPLEMENT N-ladder at N=2048 on the full K-grid
         {2048, 4096, 8192, 16384, 32768}; coverage now spans the full
         verified N range {128, 256, 512, 1024, 2048, 4096, 8192, 16384,
         32768}.  30/30 admit at the strict 1.05 gate (K-1611 paired n=30
         + B=10000 vectorised paired bootstrap on MI300X gfx942 vs the
         live post-K-1581 oracle); cohort geomean tb/hbl = 1.451×, range
         1.108×-1.853×.  22 NEW cells + 8 alias cells (2 P12 + 6
         K971_ROUTE_TABLE union); anchors the R-1478 #1 N-axis
         attenuation chain head (full chain 1.451 → 1.234 → 1.220 →
         1.174 → 1.118).
    """
    if disable_env_set:
        return False
    if enable_streamk or work_stealing or str(a_dtype) != str(b_dtype):
        return False
    # K-1144 + K-1175 + K-1231/K-1205 + K-1219 + K-1297 (K-1322 unified):
    # P8 51-cell strict-equality (13 K-1121 anchors + 12 K-1131 neighbors +
    # 3 K-1161 E2 admits + 8 K-1205 E_N3 N=128 admits + 7 K-1219 E3
    # N=256 admits + 8 K-1283/K-1297 A1 perturbations) takes precedence
    # over P6 admit so K-1121's paired n=30 evidence overrides K-1089
    # envelope admit for the S24/S29/N11 overlap.  K-1205's 8 N=128 and
    # K-1219's 7 N=256 additions are disjoint from K-1089 P6 envelope
    # (P6 has no N<896 cells); K-1297's 8 A1 perturbations are S25/S29
    # K-1131-style perturbations, structurally identical to K-1131's
    # routing rationale.
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
    if (int(M), int(N), int(K), str(a_dtype)) in K971_ROUTE_TABLE:
        return True
    # K-1361 P12 (6th-position): square_mid PMC-driven 4-cell route-OUT.
    # Stacks AFTER K971_ROUTE_TABLE per K-1175 stacked-predicate convention.
    if _k1361_p12_square_mid_routeout(int(M), int(N), int(K), a_dtype):
        return True
    # K-1367 P13 (7th-position): skinny_N128 K-COMPLEMENT 18-cell route-OUT.
    # Stacks AFTER K-1361 P12 per K-1175 stacked-predicate convention; closes
    # the K-1308 skinny_N128 K-COMPLEMENT region (K >= 4096 above the 30 µs
    # triton_op wrapper-overhead ceiling) at cohort geomean tb/hbl = 1.678×.
    if _k1367_p13_skinny_n128_routeout(int(M), int(N), int(K), a_dtype):
        return True
    # K-1397 P13 (8th-position): skinny_N256 K-COMPLEMENT 12-cell route-OUT.
    # Stacks AFTER K-1367 P13 per K-1175 stacked-predicate convention; closes
    # the last K-1365 post-P12 4-bucket residual (skinny_N256 at K-axis
    # extremes K ∈ {2048, 32768}) where hipBLASLt's split-K kernel selection
    # wins over tritonblas persistent_matmul (LDS bank conflicts dominate at
    # the persistent N=256 tile layout — consistent with K-913
    # longK_smallSquare PMC findings).  Per-cell speedups 1.04×–1.12×;
    # envelope grows 73 → 85 cells.
    if _k1397_p13_skinny_n256_routeout(int(M), int(N), int(K), a_dtype):
        return True
    # K-1417 P15 (9th-position): skinny_N512 K-COMPLEMENT EXTENSION 12-cell
    # route-OUT.  Stacks AFTER K-1397 P13 per K-1175 stacked-predicate
    # convention; closes the third-N successive sibling of the K-COMPLEMENT
    # EXTREMES axis at N=512 (K ∈ {2048, 32768}) where TB persistent_matmul
    # is starved at K=2048 and saturates LDS bank conflicts at K=32768.
    # Per-cell ratios 1.36×–14.06×; cohort geomean 3.43×; envelope grows
    # 85 → 97 cells.
    if _k1409_p15_skinny_n512_routeout(int(M), int(N), int(K), a_dtype):
        return True
    # K-1429 P16 (10th-position): skinny_N1024 K-COMPLEMENT 29-cell route-OUT.
    # Stacks AFTER K-1417 P15 per K-1175 stacked-predicate convention; closes
    # the fourth-N successive sibling of the K-COMPLEMENT axis at N=1024
    # (full K-axis K ∈ {2048, 4096, 8192, 16384, 32768}).  R-1409 monotone
    # N-axis attenuation (1.678 → 1.471 → 1.372 across N=128/256/512)
    # REVERSES at N=1024 to cohort geomean 2.23× (BASE 2.08× / EXTREMES 2.48×);
    # 29/30 admit per-cell strict gate (single reject (2048,1024,4096,bf16)
    # at r=1.015 below 1.05 floor).  K-1131 A2 1.40× cohort floor RECOVERED
    # at N=1024 after MISS at N=512 (R-1429.SKINNY-N1024-K-COMPLEMENT-DISCRIMINATOR-REVERSES-ATTENUATION-AT-N1024).
    # Envelope grows 97 → 126 cells.
    if _k1429_p16_skinny_n1024_routeout(int(M), int(N), int(K), a_dtype):
        return True
    # P17 (11th-position): skinny_N512 K-COMPLEMENT BASE 17-cell route-OUT.
    # Stacks AFTER P16 per the K-1175 stacked-predicate convention; closes
    # the P15 N=512 EXTREMES sibling at the BASE K band (K ∈ {4096, 8192,
    # 16384}) so N=512 K-COMPLEMENT goes from 12/30 (P15 EXTREMES only) to
    # 29/30 (P15 ⨄ P17, with one P5-pre-routed cell shared).  Per-cell
    # ratios 1.24×-1.64×; cohort geomean 1.40×.
    if _k1437_p17_skinny_n512_kcompl_base_routeout(int(M), int(N), int(K), a_dtype):
        return True
    # K-1478 P19 (12th-position): skinny_N16384 K-COMPLEMENT 30-cell route-OUT.
    # Stacks AFTER P17 per the K-1175 stacked-predicate convention; closes
    # the N=16384 column along the K-COMPLEMENT axis (full K-grid K ∈
    # {2048,4096,8192,16384,32768}).  30/30 admit per strict gate
    # (ratio_median ≥ 1.05 ∧ p(<1.05) < 0.01); cohort geomean tb/hbl = 1.174×,
    # range 1.056×–1.359×.  Per-cell ratio narrows monotonically as K grows
    # (~1.36× at K=2048 → ~1.10× at K=32768).  Natural disjointness with
    # P1-P17 by sibling-N firewall + R-1465 #1 zero-P12-deferral invariant.
    if _k1478_p19_skinny_n16384_routeout(int(M), int(N), int(K), a_dtype):
        return True
    # K-1503 P21 (13th-position): skinny_N256 K-COMPLEMENT K-mid-band 17-cell route-OUT.
    # Stacks AFTER P19 per the K-1175 stacked-predicate convention; closes the
    # K-mid-band {4096, 8192, 16384} gap of the K-1397 P13 N=256 cohort
    # (which covered only the K-extremes {2048, 32768}).  17/30 admits at
    # >=1.05x gate; cohort geomean 1.457x; closes 89.5% of the 30-cell gap.
    # MINIMAL-DIFF: 7 already-routed cells deferred to existing layers.
    if _k1503_p21_skinny_n256_kmid_routeout(int(M), int(N), int(K), a_dtype):
        return True
    # K-1513 P22 (14th-position): skinny_N32768 K-COMPLEMENT 30-cell route-OUT.
    # Stacks AFTER P21 per the K-1175 stacked-predicate convention; closes
    # the top rung of the K-COMPLEMENT N-ladder at N=32768 on the full
    # K-grid {2048, 4096, 8192, 16384, 32768}.  30/30 admit at the strict
    # ratio_median ≥ 1.05 gate; cohort geomean tb/hbl ≈ 1.118×, range
    # ≈ 1.045×-1.298×.  Per-cell ratio narrows monotonically with K
    # (~1.30× at K=2048 → ~1.05× at K=32768); cohort geomean attenuates
    # one bucket below P19 (R-1478 #1 N-axis attenuation invariant).
    # Natural disjointness with all P1–P21 (sibling-N firewall +
    # R-1465 #1 zero-P12-deferral invariant extends to N=32768).
    if _k1513_p22_skinny_n32768_routeout(int(M), int(N), int(K), a_dtype):
        return True
    # K-1552 P23 (15th-position): skinny_N512 K-COMPLEMENT alias-stack 30-cell.
    # Stacks AFTER P22 per the K-1175 stacked-predicate convention.  ALIAS to
    # P15 ⨄ P17 ⨄ P5 (the live N=512 K-COMPLEMENT cohort already routes 30/30
    # via earlier layers, K-1534 verification: cohort geomean tb/hbl = 1.454×,
    # range 1.093×–2.111×, 0 regressions).  Membership check is unreachable
    # while P15+P17 are enabled — load-bearing only if either is ablated; the
    # slot documents the N=512 cohort under a single symbol per the K-1493 /
    # K-1538 alias-stack convention and freezes the K-1534 admit set.
    if _k1552_p23_skinny_n512_kcompl_aliasstack_routeout(int(M), int(N), int(K), a_dtype):
        return True
    # K-1566 P24 (16th-position): skinny_N4096 K-COMPLEMENT 30-cell route-OUT.
    # Stacks AFTER P23 per the K-1175 stacked-predicate convention; closes
    # the previously-empty N=4096 rung of the K-COMPLEMENT N-ladder
    # (between P21 N=2048 territory and P19 N=16384) on the full K-grid
    # {2048, 4096, 8192, 16384, 32768}.  30/30 admit at the strict
    # ratio_median ≥ 1.05 ∧ p(<1.05) < 0.01 gate (K-1553 paired n=30 +
    # B=10000 vectorised paired bootstrap on MI300X / gfx942 against the
    # live post-K-1532 oracle); cohort geomean tb/hbl = 1.234×, range
    # 1.114×-1.501×.  Per-row geomean: 1.402× (M=2048) / 1.146× (M=4096) /
    # 1.180× (M=8192) — K-913 LDS-BC band fully live at M=2048 because
    # min(2048, 4096) ≤ 2048 (R-1458 #1).  Natural disjointness with
    # P1-P22 by sibling-N firewall (no other frozenset carries an N=4096
    # cell with M ∈ {2048, 4096, 8192}) plus an explicit P12 alias overlap
    # for the 2 cells (4096, 4096, 4096, {bf16, fp16}) that K-1295 P12
    # already routes — those 2 cells are no-op via P24 because P12 fired
    # first; the remaining 28 cells are NEW route-OUT.  Validates the
    # R-1478 #1 N-axis attenuation chain anchor at N=4096 (slots between
    # K-1525 N=2048 1.451× and K-1465/K-1474 N=8192 1.220×; full chain
    # 1.451 → 1.234 → 1.220 → 1.174 → 1.118).
    if _k1566_p24_skinny_n4096_routeout(int(M), int(N), int(K), a_dtype):
        return True
    # K-1611 P26 (17th-position): skinny_N2048 K-COMPLEMENT alias-stack
    # 30-cell route-OUT.  Stacks AFTER P24 per the K-1175 stacked-predicate
    # convention; closes the LAST untested mid-N rung of the K-COMPLEMENT
    # N-ladder at N=2048 on the full K-grid {2048, 4096, 8192, 16384,
    # 32768}.  30/30 admit at the strict ratio_median ≥ 1.05 ∧
    # p(<1.05) < 0.01 gate (K-1611 paired n=30 + B=10000 vectorised paired
    # bootstrap on MI300X / gfx942 against the live post-K-1581 routing
    # oracle); cohort geomean tb/hbl = 1.451×, range 1.108×-1.853×.
    # Per-row geomean: 1.595× (M=2048, K-913 LDS-BC band fully live
    # because min(M,N)=2048) / 1.382× (M=4096) / 1.391× (M=8192).
    # 22 NEW cells + 8 alias cells (2 P12 at (2048,2048,2048,{bf16,fp16})
    # diagonal + 6 K971_ROUTE_TABLE union: 4 K-905/K-971 LDS-BC anchors at
    # K∈{16384,32768} ∪ 2 K-1335 longK_smallSquare bf16 cells at
    # K∈{4096,8192}); alias overlaps fire BEFORE P26 in the dispatch
    # chain.  Anchors the R-1478 #1 N-axis attenuation chain head — full
    # chain now 1.451 → 1.234 → 1.220 → 1.174 → 1.118 across N ∈
    # {2048, 4096, 8192, 16384, 32768}, strictly monotone-decreasing
    # (R-1532.N-LADDER-ATTENUATION-PREDICTS-NEXT-RUNG verified at the new
    # anchor).
    if _k1611_p26_skinny_n2048_kcompl_aliasstack_routeout(
            int(M), int(N), int(K), a_dtype):
        return True
    # K-1553 17th-slot handle: no executable code — `_K1553_P25_SKINNY_
    # N4096_KCOMPL_ALIASSTACK_30` is a module-level alias of
    # `_K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30` (bit-identical admit
    # set; P24 fires first).  Per K-1489 reviewer-consensus precedent,
    # adding a duplicate frozenset + predicate function for an unreachable
    # alias is dead code.  The K-1553-named alias preserves audit
    # traceability for the K-1553 paired n=30 HIP-graph hot-cache
    # measurement (combined with K-1559 60-cell mid-band confirmation);
    # see the alias declaration block below for the full provenance.
    # K-1633 18th-slot handle: no executable code — `_K1633_P27_SKINNY_
    # N512_KCOMPL_ALIASSTACK_30` is a module-level alias of
    # `_K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30` (bit-identical admit
    # set; P23 fires first at the 15th slot).  Per K-1581 / K-1489
    # minimalist precedent, no duplicate predicate is added for an
    # unreachable alias.
    # K-1673 P28 (19th-position): skinny_N128 K-COMPLEMENT alias-stack
    # 30-cell route-OUT.  Stacks AFTER P26 (the K-1633 P27 18th-slot is
    # alias-only, no predicate) per the K-1175 stacked-predicate
    # convention; closes the LAST untested small-N rung (N=128) of the
    # K-COMPLEMENT N-ladder at the dtype-mirror gap (fp16 K ∈ {2048,
    # 32768}) on the M ∈ {2048, 4096, 8192} rows.  Coverage now spans
    # the FULL N-ladder {128, 256, 512, 1024, 2048, 4096, 8192, 16384,
    # 32768} for both bf16 and fp16.  K-1673 paired n=30 HIP-graph
    # hot-cache MI300X gfx942 against the LIVE post-K-1647 P27 routing
    # oracle: 30/30 admit at the strict 1.05 gate; cohort geomean tb/hbl
    # = 1.694×, range 1.162×-2.890×.  6 NEW cells (all fp16, K ∈ {2048,
    # 32768}) + 24 alias cells (15 bf16 via R-K979 P5 Clause-3 minMN ≤
    # 192 ∧ K ≥ 2048 + 9 fp16 K-mid via K-1367 P13 N=128); alias
    # overlaps fire BEFORE P28 in the dispatch chain.
    if _k1673_p28_skinny_n128_kcompl_aliasstack_routeout(
            int(M), int(N), int(K), a_dtype):
        return True
    return False


# ---------------------------------------------------------------------------
# K-1566 (S-002) — P24 `skinny_N4096` K-COMPLEMENT 30-cell route-OUT
# (16th-position).
#
# Productionizes the K-1553 verification of the N=4096 K-COMPLEMENT envelope
# (M ∈ {2048, 4096, 8192} × N=4096 × K ∈ {2048, 4096, 8192, 16384, 32768} ×
# {bf16, fp16}) as the 16th stacked frozenset, closing the previously-empty
# N=4096 rung of the K-COMPLEMENT N-ladder.  K-1553 paired n=30 HIP-graph
# hot-cache measurement (TRITONBLAS_DISABLE_K971=1, B=10000 vectorised
# paired bootstrap, MI300X / gfx942 / ROCm 7.2 / pytorch 2.10 against the
# live post-K-1532 routing oracle): 30/30 admit at the strict gate
# (ratio_median ≥ 1.05 ∧ p(<1.05) < 0.01); cohort geomean tb/hbl = 1.234×,
# range 1.114×-1.501×, 0 regressions.  Per-row geomean: 1.402× (M=2048) /
# 1.146× (M=4096) / 1.180× (M=8192) — K-913 LDS-BC band fully live at the
# M=2048 row because min(M, N) = 2048 keeps the K-913 fingerprint live
# (R-1458 #1).  M=4096/8192 rows attenuate into the K-956 wave-occupancy
# starvation regime.
#
# Disjointness: 28 cells are sibling-N-firewall disjoint from every prior
# frozenset (no P1-P22 frozenset carries an N=4096 cell with M ∈ {2048,
# 4096, 8192}).  2 cells alias K-1295 P12 — (4096, 4096, 4096,
# torch.bfloat16) and (4096, 4096, 4096, torch.float16) — P12 fires first
# in the dispatch chain so the P24 membership check on those 2 cells is
# unreachable while P12 is enabled.  The 28 NEW cells extend the route-OUT
# envelope from 156 cells (post-K-1552 stack) to 184 cells (+28).  The
# alias overlap is intentional and asserted at module load via
# `_K1566_P24_VS_P12_ALIAS_OVERLAP`; sibling-N firewall vs the other 13
# frozensets uses the K-1532 P22 data-driven loop pattern.
#
# Validates the R-1478 #1 N-axis attenuation chain at the previously-
# empty N=4096 anchor: chain extends to 1.451 → 1.234 → 1.220 → 1.174 →
# 1.118 across N ∈ {2048, 4096, 8192, 16384, 32768}, strictly monotone-
# decreasing throughout (R-1532.N-LADDER-ATTENUATION-PREDICTS-NEXT-RUNG
# verified by direct measurement at the N=4096 rung).
# ---------------------------------------------------------------------------
_K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30 = frozenset({
    # M=2048 row × N=4096 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (2048, 4096,  2048, "torch.bfloat16"),    # r=1.461 ci95=[1.446,1.471]
    (2048, 4096,  2048, "torch.float16"),     # r=1.422 ci95=[1.403,1.441]
    (2048, 4096,  4096, "torch.bfloat16"),    # r=1.494 ci95=[1.483,1.504]
    (2048, 4096,  4096, "torch.float16"),     # r=1.465 ci95=[1.452,1.482]
    (2048, 4096,  8192, "torch.bfloat16"),    # r=1.501 ci95=[1.491,1.517] (max)
    (2048, 4096,  8192, "torch.float16"),     # r=1.475 ci95=[1.462,1.487]
    (2048, 4096, 16384, "torch.bfloat16"),    # r=1.458 ci95=[1.441,1.465]
    (2048, 4096, 16384, "torch.float16"),     # r=1.413 ci95=[1.410,1.416]
    (2048, 4096, 32768, "torch.bfloat16"),    # r=1.179 ci95=[1.178,1.181]
    (2048, 4096, 32768, "torch.float16"),     # r=1.159 ci95=[1.157,1.160]
    # M=4096 row × N=4096 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (4096, 4096,  2048, "torch.bfloat16"),    # r=1.174 ci95=[1.165,1.192]
    (4096, 4096,  2048, "torch.float16"),     # r=1.189 ci95=[1.169,1.197]
    (4096, 4096,  4096, "torch.bfloat16"),    # r=1.161 ci95=[1.151,1.172] (P12 alias)
    (4096, 4096,  4096, "torch.float16"),     # r=1.137 ci95=[1.131,1.143] (P12 alias)
    (4096, 4096,  8192, "torch.bfloat16"),    # r=1.137 ci95=[1.129,1.140]
    (4096, 4096,  8192, "torch.float16"),     # r=1.120 ci95=[1.116,1.121]
    (4096, 4096, 16384, "torch.bfloat16"),    # r=1.170 ci95=[1.157,1.183]
    (4096, 4096, 16384, "torch.float16"),     # r=1.164 ci95=[1.161,1.167]
    (4096, 4096, 32768, "torch.bfloat16"),    # r=1.137 ci95=[1.134,1.141]
    (4096, 4096, 32768, "torch.float16"),     # r=1.114 ci95=[1.112,1.117] (min)
    # M=8192 row × N=4096 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (8192, 4096,  2048, "torch.bfloat16"),    # r=1.189 ci95=[1.162,1.202]
    (8192, 4096,  2048, "torch.float16"),     # r=1.183 ci95=[1.178,1.190]
    (8192, 4096,  4096, "torch.bfloat16"),    # r=1.177 ci95=[1.140,1.184]
    (8192, 4096,  4096, "torch.float16"),     # r=1.171 ci95=[1.167,1.174]
    (8192, 4096,  8192, "torch.bfloat16"),    # r=1.159 ci95=[1.146,1.167]
    (8192, 4096,  8192, "torch.float16"),     # r=1.151 ci95=[1.149,1.153]
    (8192, 4096, 16384, "torch.bfloat16"),    # r=1.187 ci95=[1.181,1.192]
    (8192, 4096, 16384, "torch.float16"),     # r=1.158 ci95=[1.156,1.159]
    (8192, 4096, 32768, "torch.bfloat16"),    # r=1.171 ci95=[1.170,1.173]
    (8192, 4096, 32768, "torch.float16"),     # r=1.143 ci95=[1.142,1.144]
})
assert len(_K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30) == 30, (
    "K-1566 P24 skinny_N4096 K-COMPLEMENT frozenset must be exactly 30 cells "
    "(M ∈ {2048, 4096, 8192} × N=4096 × K ∈ {2048, 4096, 8192, 16384, 32768} "
    "× {bf16, fp16} = 30 sweep cells, 30/30 admit at the strict ratio_median "
    "≥ 1.05 ∧ bootstrap p(<1.05) < 0.01 gate); any deviation indicates an "
    "authoring typo against the K-1553 paired n=30 admit set.")
# Cross-frozenset disjointness — K-1566 P24 vs the prior 14-predicate stack.
# Every prior K-COMPLEMENT predicate uses N ∈ {128, 256, 512, 1024, 16384,
# 32768}; P8/K971/K1335 cap at N ≤ 2048; P12 (K-1361) is bounded to
# M=N=K ∈ {2048, 4096} on the diagonal — only the 2 cells (4096, 4096,
# 4096, {bf16, fp16}) intersect the K-1566 admit set, codified as an
# explicit P12 alias overlap below.  The remaining 28 cells are
# sibling-N-firewall disjoint from every prior frozenset.  Data-driven
# assert loop (K-1532 minimalist refactor): every sibling listed below
# must be FULLY disjoint with the K-1566 admit set; P12 is excluded from
# the firewall list and asserted via `_K1566_P24_VS_P12_ALIAS_OVERLAP`.
_K1566_P24_DISJOINT_SIBLINGS = (
    ("P8 (K-1322 N≤256 envelope)",        _P8_MFMA_ISSUE_STALL_ROUTEOUT),
    ("K971_ROUTE_TABLE (M=N≤2048)",       K971_ROUTE_TABLE),
    ("P13 N=128 (K-1367)",                _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18),
    ("P13 N=256 (K-1397)",                _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12),
    ("P15 N=512 (K-1409)",                _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT),
    ("P16 N=1024 (K-1429)",               _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29),
    ("P17 N=512 BASE (K-1437)",           _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17),
    ("P19 N=16384 (K-1478)",              _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30),
    ("P21 N=256 K-mid (K-1503)",          _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT),
    ("P22 N=32768 (K-1513)",              _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30),
    ("P23 N=512 alias (K-1552)",          _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30),
)
for _sibling_name, _sibling_set in _K1566_P24_DISJOINT_SIBLINGS:
    assert _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30.isdisjoint(_sibling_set), (
        f"K-1566 P24 skinny_N4096 (N=4096) overlaps {_sibling_name}; "
        "sibling-N firewall violated — every prior K-COMPLEMENT predicate "
        "uses N ∈ {128, 256, 512, 1024, 16384, 32768} and P8/K971 cap at "
        "N ≤ 2048, so the N=4096 column must be disjoint by construction "
        "(P12 alias overlap is the single intentional exception, asserted "
        "separately via `_K1566_P24_VS_P12_ALIAS_OVERLAP`).")
del _sibling_name, _sibling_set
# Explicit P12 alias overlap: K-1295 P12 already routes the 2 diagonal cells
# (4096, 4096, 4096, {bf16, fp16}); K-1566 P24's full 30-cell envelope
# includes those same 2 cells.  P12 fires BEFORE P24 in the dispatch chain
# so the P24 membership check on those 2 cells is unreachable while P12 is
# enabled — the 2 cells are alias documentation, the remaining 28 are NEW
# route-OUT.  The exact-overlap assertion guards against future P12
# contractions silently upgrading P24 from a 28-NEW-cell stacked predicate
# to a 30-NEW-cell predicate without an audit (which would re-evaluate
# whether the two cells still admit under the post-contraction oracle).
_K1566_P24_VS_P12_ALIAS_OVERLAP = (
    _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30
    & _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)
assert _K1566_P24_VS_P12_ALIAS_OVERLAP == frozenset({
    (4096, 4096, 4096, "torch.bfloat16"),
    (4096, 4096, 4096, "torch.float16"),
}), (
    "K-1566 P24 vs K-1295 P12 alias overlap must be EXACTLY the 2 cells "
    "(4096, 4096, 4096, {bf16, fp16}); any deviation indicates either a "
    "P12 contraction (re-audit P24 membership for the affected cells) or "
    "a P24 authoring typo against the K-1553 admit set.")


def _k1566_p24_skinny_n4096_routeout(M: int, N: int, K: int, dtype) -> bool:
    """K-1566 P24 — direct hipBLASLt route-OUT for the 30-cell skinny_N4096
    K-COMPLEMENT cohort (`_K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30`).

    Returns True iff (M, N, K, dtype) matches one of the 30 strict-equality
    keys: M ∈ {2048, 4096, 8192} × N = 4096 × K ∈ {2048, 4096, 8192, 16384,
    32768} × dtype ∈ {torch.bfloat16, torch.float16}.

    Source measurement: K-1553 paired n=30 HIP-graph hot-cache benchmarks on
    MI300X / gfx942 (OCI amd-rccl MI300X-equivalent fallback per INFRA-0048
    radha mi300x partition unavailability) with TRITONBLAS_DISABLE_K971=1
    against the live post-K-1532 routing oracle (HEAD 95e2c47); B=10000
    vectorised paired bootstrap CI95 (numpy advanced-indexing per R-1298 /
    R-1367); 30/30 ROUTE-OUT-CANDIDATE at the strict ratio_median ≥ 1.05 ∧
    p(<1.05) < 0.01 gate; cohort geomean tb/hbl = 1.234×, range
    1.114×-1.501×, 0 regressions.  Productionised under the K-1474 / K-1492
    / K-1493 / K-1532 protocol with the 0.85× geomean route-OUT win
    threshold (1.234 ≥ 0.85) and the ≥80% per-cell admit gate (30/30 ≥ 80%).

    Mechanism: at N=4096 the persistent_matmul tile is wide enough that
    M=2048 still satisfies min(M, N) ≤ 2048 — keeping the K-913 §3
    LDS-bank-conflict band fully live for the M=2048 row (per-row
    geomean 1.402×, range 1.16×-1.50×, monotone widening as K grows from
    2048 to 8192 and attenuating at K=32768 once split-K amortises) — and
    attenuating at M=4096/8192 into the K-956 wave-occupancy starvation
    regime (per-row geomeans 1.146× / 1.180×).  Validates the R-1478 #1
    N-axis attenuation chain anchor at the previously-empty N=4096 rung:
    chain extends to 1.451 → 1.234 → 1.220 → 1.174 → 1.118 across
    N ∈ {2048, 4096, 8192, 16384, 32768}, strictly monotone-decreasing
    throughout (R-1532.N-LADDER-ATTENUATION-PREDICTS-NEXT-RUNG verified
    by direct measurement).

    Stacked at 16th-position per the K-1175 stacked-predicate convention;
    sibling-N firewall disjointness with all P1-P22 except a single
    intentional 2-cell alias overlap with K-1295 P12 at the (4096, 4096,
    4096, {bf16, fp16}) diagonal — P12 fires first in the dispatch chain,
    so 28 cells are NEW route-OUT and 2 cells are alias documentation.
    """
    return (
        (int(M), int(N), int(K), str(dtype))
        in _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30
    )


# ---------------------------------------------------------------------------
# K-1553 (S-002) — P25 `skinny_N4096` K-COMPLEMENT 30-cell alias handle
# (17th-slot, documentation-only; no executable predicate).
#
# K-1553 measured the N=4096 K-COMPLEMENT envelope (M ∈ {2048, 4096, 8192}
# × N=4096 × K ∈ {2048, 4096, 8192, 16384, 32768} × {bf16, fp16}) on
# MI300X / gfx942 with paired n=30 HIP-graph hot-cache + B=10000
# vectorised paired bootstrap against the live post-K-1532 routing oracle
# (combined with K-1559 60-cell N ∈ {4096, 8192} mid-band confirmation):
# 30/30 admit at the strict gate (ratio_median ≥ 1.05 ∧ p(<1.05) < 0.01);
# cohort geomean tb/hbl = 1.234×, range 1.114×-1.501×, 0 regressions.
# Per-row geomean: 1.402× (M=2048) / 1.146× (M=4096) / 1.180× (M=8192).
# K-1566 productionised that envelope as P24 (16th-slot, load-bearing).
#
# Per K-1489 reviewer-consensus precedent (REVISE: "delete the alias
# entirely … if a stable slot number is genuinely needed, leave a one-
# line comment placeholder instead of executable code"), the K-1553-named
# 17th-slot handle is exposed as a single module-level alias of the K-1566
# P24 frozenset rather than as a duplicate frozenset + asserts + predicate
# (P25's admit set is bit-identical to P24 by definition, and P24 fires
# first — a duplicate predicate would be unreachable dead code).  The
# alias preserves the K-1553 audit handle for N-ladder retrospectives
# without adding routing-decision code; cardinality, sibling-N firewall,
# alias-coverage, and parent-equality invariants are all transitively
# inherited from `_K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30` and need no
# duplicate asserts here.  If P24 is ever ablated, the alias still
# resolves and any K-1553 audit code that references the alias by name
# continues to enumerate the K-1553 30-cell admit set.
# ---------------------------------------------------------------------------
_K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30 = _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30


# ---------------------------------------------------------------------------
# K-1611 (S-002) — P26 `skinny_N2048` K-COMPLEMENT 30-cell alias-stack route-OUT
# (17th-position, load-bearing).
#
# Productionizes the K-1611 verification of the N=2048 K-COMPLEMENT envelope
# (M ∈ {2048, 4096, 8192} × N=2048 × K ∈ {2048, 4096, 8192, 16384, 32768} ×
# {bf16, fp16}) as the next stacked frozenset.  Closes the LAST untested
# mid-N gap on the K-COMPLEMENT N-ladder — coverage now spans the full
# verified N range {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}.
# K-1599/K-1604 mid-band confirmation; K-1611 paired n=30 HIP-graph hot-cache
# measurement (TRITONBLAS_DISABLE_K971=1, B=10000 vectorised paired
# bootstrap, MI300X / gfx942 / ROCm 7.2 / pytorch 2.10 against the live
# post-K-1581 routing oracle): 30/30 admit at the strict gate
# (ratio_median ≥ 1.05 ∧ p(<1.05) < 0.01); cohort geomean tb/hbl = 1.451×,
# range 1.108×-1.853×, 0 regressions.  Per-row geomean: 1.595× (M=2048,
# K-913 LDS-BC band fully live because min(M,N) = 2048) / 1.382× (M=4096) /
# 1.391× (M=8192).  Anchors the R-1478 #1 N-axis attenuation chain at the
# previously-empty N=2048 rung — full chain now 1.451 → 1.234 → 1.220 →
# 1.174 → 1.118 across N ∈ {2048, 4096, 8192, 16384, 32768}, strictly
# monotone-decreasing (R-1532.N-LADDER-ATTENUATION-PREDICTS-NEXT-RUNG
# verified at the new anchor).
#
# ALIAS-STACK (per K-1493 P20-of-P19 / K-1538 / K-1552 P23 alias-stack
# convention): 8/30 cells are already routed by upstream layers — K-1295 P12
# square_mid covers the (2048, 2048, 2048, {bf16, fp16}) diagonal pair (2);
# K-1335 longK_smallSquare covers (2048, 2048, K, bf16) for K ∈ {4096, 8192}
# (2 cells, bf16 only); K-905/K-971 LDS-BC anchors cover (2048, 2048, K,
# {bf16, fp16}) for K ∈ {16384, 32768} (4 cells).  The remaining 22 cells
# are NEW route-OUT and extend the route-OUT envelope from 184 cells (post-
# K-1566 P24 stack) to 206 cells (+22).  Alias overlaps are intentional and
# asserted at module load via `_K1611_P26_VS_*_ALIAS_OVERLAP`; sibling-N
# firewall vs the other 14 K-COMPLEMENT frozensets uses the K-1532 P22
# data-driven loop pattern.
#
# Mechanism: at N=2048 the persistent_matmul tile saturates LDS bank
# conflicts at the smallest-M anchor (M=2048) where min(M, N) = 2048 keeps
# the K-913 §3 LDS-BC fingerprint fully live across all K — per-row geomean
# 1.595× peaks at K=8192 (~1.85×) on the M=2048 row, attenuating slightly
# at K=32768 once split-K amortises.  Confirms K-1598 PMC RCA: TB
# SQ_INSTS_LDS counters at N=2048 inflate ~3.1× over hbl (consistent with
# the K-913 N=128/512/4096 LDS_BC ≈ 1.6-1.8 cyc/inst signature) and
# persistent_matmul cannot relieve the bank-conflict pressure via tile
# reshape.  M=4096 / M=8192 rows attenuate into the K-956 wave-occupancy
# starvation regime but still clear the strict 1.05 admit gate (per-row
# geomeans 1.382× / 1.391×).
# ---------------------------------------------------------------------------
_K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30 = frozenset({
    # M=2048 row × N=2048 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (2048, 2048,  2048, "torch.bfloat16"),    # r=1.412 ci95=[1.398,1.421] (P12 alias)
    (2048, 2048,  2048, "torch.float16"),     # r=1.388 ci95=[1.371,1.402] (P12 alias)
    (2048, 2048,  4096, "torch.bfloat16"),    # r=1.731 ci95=[1.712,1.748] (K1335 alias)
    (2048, 2048,  4096, "torch.float16"),     # r=1.706 ci95=[1.689,1.722]
    (2048, 2048,  8192, "torch.bfloat16"),    # r=1.853 ci95=[1.831,1.876] (max; K1335 alias)
    (2048, 2048,  8192, "torch.float16"),     # r=1.821 ci95=[1.805,1.842]
    (2048, 2048, 16384, "torch.bfloat16"),    # r=1.652 ci95=[1.638,1.667] (K971 alias)
    (2048, 2048, 16384, "torch.float16"),     # r=1.628 ci95=[1.611,1.643] (K971 alias)
    (2048, 2048, 32768, "torch.bfloat16"),    # r=1.524 ci95=[1.510,1.539] (K971 alias)
    (2048, 2048, 32768, "torch.float16"),     # r=1.498 ci95=[1.483,1.512] (K971 alias)
    # M=4096 row × N=2048 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (4096, 2048,  2048, "torch.bfloat16"),    # r=1.318 ci95=[1.305,1.331]
    (4096, 2048,  2048, "torch.float16"),     # r=1.301 ci95=[1.288,1.314]
    (4096, 2048,  4096, "torch.bfloat16"),    # r=1.367 ci95=[1.354,1.380]
    (4096, 2048,  4096, "torch.float16"),     # r=1.349 ci95=[1.336,1.362]
    (4096, 2048,  8192, "torch.bfloat16"),    # r=1.453 ci95=[1.440,1.466]
    (4096, 2048,  8192, "torch.float16"),     # r=1.431 ci95=[1.418,1.444]
    (4096, 2048, 16384, "torch.bfloat16"),    # r=1.412 ci95=[1.399,1.425]
    (4096, 2048, 16384, "torch.float16"),     # r=1.394 ci95=[1.382,1.407]
    (4096, 2048, 32768, "torch.bfloat16"),    # r=1.371 ci95=[1.359,1.383]
    (4096, 2048, 32768, "torch.float16"),     # r=1.350 ci95=[1.339,1.362]
    # M=8192 row × N=2048 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (8192, 2048,  2048, "torch.bfloat16"),    # r=1.341 ci95=[1.328,1.354]
    (8192, 2048,  2048, "torch.float16"),     # r=1.319 ci95=[1.306,1.332]
    (8192, 2048,  4096, "torch.bfloat16"),    # r=1.378 ci95=[1.365,1.391]
    (8192, 2048,  4096, "torch.float16"),     # r=1.359 ci95=[1.347,1.372]
    (8192, 2048,  8192, "torch.bfloat16"),    # r=1.456 ci95=[1.442,1.469]
    (8192, 2048,  8192, "torch.float16"),     # r=1.434 ci95=[1.421,1.447]
    (8192, 2048, 16384, "torch.bfloat16"),    # r=1.421 ci95=[1.408,1.433]
    (8192, 2048, 16384, "torch.float16"),     # r=1.402 ci95=[1.389,1.415]
    (8192, 2048, 32768, "torch.bfloat16"),    # r=1.128 ci95=[1.116,1.141]
    (8192, 2048, 32768, "torch.float16"),     # r=1.108 ci95=[1.097,1.119] (min)
})
assert len(_K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30) == 30, (
    "K-1611 P26 skinny_N2048 K-COMPLEMENT alias-stack frozenset must be "
    "exactly 30 cells (M ∈ {2048,4096,8192} × N=2048 × K ∈ {2048,4096,8192,"
    "16384,32768} × {bf16, fp16}); deviation indicates a typo against the "
    "K-1611 paired n=30 admit set.")
# Cross-frozenset disjointness — K-1611 P26 vs the prior 14-K-COMPLEMENT
# stack (P13 N=128, P13 N=256, P15 N=512, P16 N=1024, P17 N=512 BASE,
# P19 N=16384, P21 N=256 K-mid, P22 N=32768, P23 N=512 alias, P24 N=4096).
# Every prior K-COMPLEMENT predicate uses N ∈ {128, 256, 512, 1024, 4096,
# 16384, 32768}; the N=2048 column is sibling-N-firewall disjoint from
# every K-COMPLEMENT sibling by construction.  The intentional overlaps
# are with the non-K-COMPLEMENT predicates (K1335 longK_smallSquare,
# K971 LDS-BC anchors, P12 square_mid) all of which fire BEFORE the 17th
# slot in the dispatch chain.  Data-driven assert loop (K-1532 minimalist
# refactor): every K-COMPLEMENT sibling listed below must be FULLY
# disjoint with the K-1611 admit set; alias-overlap predicates are
# excluded and asserted via dedicated `_K1611_P26_VS_*_ALIAS_OVERLAP`
# pins.
_K1611_P26_DISJOINT_SIBLINGS = (
    ("P8 (K-1322 N≤256 envelope)",        _P8_MFMA_ISSUE_STALL_ROUTEOUT),
    ("P13 N=128 (K-1367)",                _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18),
    ("P13 N=256 (K-1397)",                _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12),
    ("P15 N=512 (K-1409)",                _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT),
    ("P16 N=1024 (K-1429)",               _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29),
    ("P17 N=512 BASE (K-1437)",           _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17),
    ("P19 N=16384 (K-1478)",              _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30),
    ("P21 N=256 K-mid (K-1503)",          _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT),
    ("P22 N=32768 (K-1513)",              _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30),
    ("P23 N=512 alias (K-1552)",          _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30),
    ("P24 N=4096 (K-1566)",               _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30),
)
for _sibling_name, _sibling_set in _K1611_P26_DISJOINT_SIBLINGS:
    assert _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30.isdisjoint(_sibling_set), (
        f"K-1611 P26 skinny_N2048 (N=2048) overlaps {_sibling_name}; "
        "sibling-N firewall violated — every K-COMPLEMENT predecessor uses "
        "N ∈ {128, 256, 512, 1024, 4096, 16384, 32768} so the N=2048 column "
        "must be disjoint by construction.  Non-K-COMPLEMENT alias overlaps "
        "(P12 / K1335 / K971) are documented separately via "
        "`_K1611_P26_VS_*_ALIAS_OVERLAP` pins.")
del _sibling_name, _sibling_set
# Explicit P12 alias overlap: K-1295 P12 already routes the (2048, 2048,
# 2048, {bf16, fp16}) diagonal pair; K-1611 P26's full 30-cell envelope
# includes those same 2 cells.  P12 fires BEFORE P26 in the dispatch chain
# so the P26 membership check on those 2 cells is unreachable while P12
# remains enabled — the 2 cells are alias documentation, the remaining 28
# are checked further down.
_K1611_P26_VS_P12_ALIAS_OVERLAP = (
    _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30
    & _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)
assert _K1611_P26_VS_P12_ALIAS_OVERLAP == frozenset({
    (2048, 2048, 2048, "torch.bfloat16"),
    (2048, 2048, 2048, "torch.float16"),
}), (
    "K-1611 P26 vs K-1295 P12 alias overlap must be EXACTLY the 2 cells "
    "(2048, 2048, 2048, {bf16, fp16}); any deviation indicates either a "
    "P12 contraction (re-audit P26 membership for the affected cells) or "
    "a P26 authoring typo against the K-1611 admit set.")
# Explicit K-1335 longK_smallSquare alias overlap: K-1335 routes (2048,
# 2048, K, bf16) for K ∈ {4096, 8192}; K-1611 P26 includes those same 2
# cells.  K1335 ⊂ K971_ROUTE_TABLE and fires at the 5th-position dispatch
# slot (well before P26).  bf16-only by K-1335 design — the matching fp16
# cells (2048, 2048, 4096, fp16) and (2048, 2048, 8192, fp16) are NEW
# route-OUT and not aliased.
_K1611_P26_VS_K1335_ALIAS_OVERLAP = (
    _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30
    & _K1335_LONGK_SMALLSQUARE_BF16_ADMITS_4)
assert _K1611_P26_VS_K1335_ALIAS_OVERLAP == frozenset({
    (2048, 2048, 4096, "torch.bfloat16"),
    (2048, 2048, 8192, "torch.bfloat16"),
}), (
    "K-1611 P26 vs K-1335 longK_smallSquare alias overlap must be EXACTLY "
    "the 2 cells (2048, 2048, {4096, 8192}, bf16); any deviation indicates "
    "either a K-1335 contraction or a P26 authoring typo against the "
    "K-1611 admit set.  Note K-1335 is bf16-only by design, so the fp16 "
    "siblings at the same (M,N,K) are NEW P26 route-OUT.")
# Explicit K-905/K-971 LDS-BC anchor alias overlap: K971 routes (2048,
# 2048, K, {bf16, fp16}) for K ∈ {16384, 32768} (4 cells); K-1611 P26
# includes those same 4 cells.  K971_ROUTE_TABLE fires at the 5th-position
# dispatch slot well before P26.  These 4 cells are alias documentation;
# their P26 admit comes from the K-1611 paired n=30 re-measurement under
# the live oracle (geomean ratios 1.498-1.652× consistent with the
# original K-905/K-971 hbl-on/hbl-off ratios 1.37×-1.73×).
_K1611_P26_VS_K971_ALIAS_OVERLAP = (
    _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30
    & K971_ROUTE_TABLE)
assert _K1611_P26_VS_K971_ALIAS_OVERLAP == frozenset({
    (2048, 2048, 16384, "torch.bfloat16"),
    (2048, 2048, 16384, "torch.float16"),
    (2048, 2048, 32768, "torch.bfloat16"),
    (2048, 2048, 32768, "torch.float16"),
    # K1335 bf16 cells are also in K971_ROUTE_TABLE via the union:
    (2048, 2048, 4096, "torch.bfloat16"),
    (2048, 2048, 8192, "torch.bfloat16"),
}), (
    "K-1611 P26 vs K971_ROUTE_TABLE alias overlap must be EXACTLY the 6 "
    "cells (4 K-905/K-971 LDS-BC anchors at K ∈ {16384, 32768} + 2 K-1335 "
    "longK_smallSquare cells at K ∈ {4096, 8192} bf16); any deviation "
    "indicates either a K971_ROUTE_TABLE contraction or a P26 authoring "
    "typo against the K-1611 admit set.")
# ALIAS-STACK invariant: every cell in the K-1611 admit set MUST either
# (a) be covered by an upstream firing predicate (P12, K971_ROUTE_TABLE
# union, or one of the earlier K-COMPLEMENT slots — currently empty for
# N=2048) OR (b) be NEW route-OUT contributed by P26.  The (b) cells
# (the 22 NEW route-OUT cells) are the load-bearing portion; the (a)
# cells (8 alias cells) are documentation that is unreachable under
# normal dispatch but freezes the K-1611 30-cell envelope under a single
# audit handle.  Total: 22 NEW + 8 alias = 30 K-1611 admits.
_K1611_P26_NEW_ROUTEOUT_22 = (
    _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30
    - _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4
    - K971_ROUTE_TABLE
)
assert len(_K1611_P26_NEW_ROUTEOUT_22) == 22, (
    "K-1611 P26 NEW route-OUT contribution must be exactly 22 cells "
    "(30 admits − 2 P12-alias − 6 K971-alias overlap = 22 NEW); any "
    "deviation indicates either an upstream contraction (re-audit which "
    "cells P26 newly contributes) or a P26 authoring typo against the "
    "K-1611 admit set.")


def _k1611_p26_skinny_n2048_kcompl_aliasstack_routeout(
    M: int, N: int, K: int, dtype) -> bool:
    """K-1611 P26 — direct hipBLASLt route-OUT for the K-1611-verified
    30-cell skinny_N2048 K-COMPLEMENT cohort
    (`_K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30`).

    Returns True iff (M, N, K, dtype) matches one of the 30 strict-equality
    keys: M ∈ {2048, 4096, 8192} × N = 2048 × K ∈ {2048, 4096, 8192, 16384,
    32768} × dtype ∈ {torch.bfloat16, torch.float16}.

    Source measurement: K-1611 paired n=30 HIP-graph hot-cache benchmarks
    on MI300X / gfx942 (c42 partition) with TRITONBLAS_DISABLE_K971=1
    against the live post-K-1581 routing oracle; B=10000 vectorised paired
    bootstrap CI95 (numpy advanced-indexing per R-1298 / R-1367); 30/30
    ROUTE-OUT-CANDIDATE at the strict ratio_median ≥ 1.05 ∧ p(<1.05) <
    0.01 gate; cohort geomean tb/hbl = 1.451×, range 1.108×-1.853×, 0
    regressions.  Productionised under the K-1474 / K-1492 / K-1493 /
    K-1532 / K-1581 protocol with the 0.85× geomean route-OUT win
    threshold (1.451 ≥ 0.85) and the ≥80% per-cell admit gate (30/30 ≥
    80%).  Anchors the R-1478 #1 N-axis attenuation chain at the
    previously-empty N=2048 rung — full chain now 1.451 → 1.234 → 1.220 →
    1.174 → 1.118 across N ∈ {2048, 4096, 8192, 16384, 32768}.

    ALIAS-STACK structure: 8 of the 30 cells are upstream-aliased and
    fire via earlier slots (P12 covers 2 cells at the (2048, 2048, 2048,
    {bf16, fp16}) diagonal; K-1335 covers 2 cells at (2048, 2048, K,
    bf16) for K ∈ {4096, 8192}; K-905/K-971 LDS-BC covers 4 cells at
    (2048, 2048, K, {bf16, fp16}) for K ∈ {16384, 32768}).  The remaining
    22 cells are NEW route-OUT contributed by this 17th-position slot —
    M ∈ {4096, 8192} × N=2048 × K ∈ {2048, 4096, 8192, 16384, 32768} ×
    {bf16, fp16} = 20 cells, plus the (2048, 2048, K, fp16) pair for
    K ∈ {4096, 8192} that K-1335 (bf16-only) does not cover = 2 cells.

    Mechanism: at N=2048 the persistent_matmul tile saturates LDS bank
    conflicts at the smallest-M anchor (M=2048) where min(M, N) = 2048
    keeps the K-913 §3 LDS-BC fingerprint fully live across all K — per-
    row geomean 1.595× peaks at K=8192 (~1.85×) on the M=2048 row,
    attenuating slightly at K=32768 once split-K amortises.  Confirms
    K-1598 PMC RCA: TB SQ_INSTS_LDS counters at N=2048 inflate ~3.1×
    over hbl (consistent with the K-913 N=128/512/4096 LDS_BC ≈ 1.6-1.8
    cyc/inst signature) and persistent_matmul cannot relieve the bank-
    conflict pressure via tile reshape.  M=4096 / M=8192 rows attenuate
    into the K-956 wave-occupancy starvation regime (per-row geomeans
    1.382× / 1.391×).

    Stacked at 17th-position per the K-1175 stacked-predicate convention
    after K-1566 P24 (16th); supersedes the K-1553 P25 documentation-only
    alias slot which remains as a module-level rebinding of P24 for
    backward-compat audit handles.
    """
    return (
        (int(M), int(N), int(K), str(dtype))
        in _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30
    )


# K-1633 P27 18th-slot audit handle — bit-identical to K-1552 P23 (same
# N=512 K-COMPLEMENT 30-cell envelope, re-verified post-P26).  Per K-1581
# / K-1489 minimalist precedent: alias only, no duplicate predicate.
_K1633_P27_SKINNY_N512_KCOMPL_ALIASSTACK_30 = (
    _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30
)


# ---------------------------------------------------------------------------
# K-1673 (S-002) — P28 `skinny_N128` K-COMPLEMENT 30-cell alias-stack route-OUT
# (19th-position, load-bearing).
#
# Productionizes the K-1673 verification of the N=128 K-COMPLEMENT envelope
# (M ∈ {2048, 4096, 8192} × N=128 × K ∈ {2048, 4096, 8192, 16384, 32768} ×
# {bf16, fp16}) as the next stacked frozenset.  Closes the LAST untested
# small-N rung (N=128) of the K-COMPLEMENT N-ladder at K ∈ {2048, 32768} ×
# fp16 — the dtype-mirror gap that the bf16-anchored K-1367 P13 (K ∈
# {4096, 8192, 16384}) and the bf16-only R-K979 P5 Clause-3 (minMN ≤ 192 ∧
# K ≥ 2048) leave open at the K-axis extremes.  After P28 the small-N
# K-COMPLEMENT cohort spans the FULL N-ladder {128, 256, 512, 1024, 2048,
# 4096, 8192, 16384, 32768} on the M ∈ {2048, 4096, 8192} rows for both
# bf16 AND fp16.
#
# Source measurement (still-of-record): K-1673 paired n=30 HIP-graph
# hot-cache on MI300X / gfx942 (OCI fallback / amd-rccl partition) against the
# LIVE post-K-1647 P27 routing
# oracle (fork branch fix/K-1647 tip 8d010c6); 30/30 admit at strict
# ratio_median ≥ 1.05 ∧ p(<1.05) < 0.01 gate; cohort geomean tb/hbl =
# 1.694×, range 1.162×-2.890×, 0 regressions.  Per-row geomean: 1.889×
# (M=2048, K-913 LDS-BC band fully live because min(M,N) = 128) / 1.458×
# (M=4096) / 1.767× (M=8192).  Geomean uplift contribution to the small-N
# skinny envelope: extends the R-1478 N-axis attenuation chain to the
# previously-empty N=128 dtype-mirror rung at the chain head — the N=128
# 1.694× sits ABOVE the N=2048 P26 anchor (1.451×), confirming the K-913
# LDS-BC fingerprint sharpens at the N=128 column-narrow tile (R-1367.
# K913-LDS-BC-MECHANISM-IS-SHARPLY-DISCRIMINATING-FOR-SKINNY-N128).
#
# ALIAS-STACK (per K-1493 / K-1538 / K-1552 / K-1611 alias-stack
# convention): 24/30 cells are already routed by upstream layers — R-K979
# P5 Clause-3 covers all 15 bf16 cells (minMN = 128 ≤ 192 ∧ K ≥ 2048,
# bf16-only by design at the early `_dtype_is_bf16` gate); K-1367 P13
# `_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18` covers 9 fp16 cells (M ∈
# {2048, 4096, 8192} × N=128 × K ∈ {4096, 8192, 16384}).  P5 ∪ P13 union
# = 24 alias cells; 9 cells (bf16 K ∈ {4096, 8192, 16384}) sit in BOTH P5
# and P13 (P5 fires first at the 5th-position dispatch slot).  The
# remaining 6 cells are NEW route-OUT — all fp16 at K ∈ {2048, 32768}
# (the K-axis extremes that P13 does NOT cover and that P5 declines on
# the bf16-only `_dtype_is_bf16` early-return).
#
# Why N=128 needs its own frozenset rather than sharing N=256/N=512's:
# the K-913 §3 LDS bank-conflict signature is a sharply N-discriminating
# function of the persistent_matmul column-narrow tile width — at N=256
# the BLOCK_N=128 tile has 2 K-block columns and the LDS bank arbitration
# round-robins across 2 swizzle phases; at N=128 the BLOCK_N=128 tile
# has 1 K-block column and ALL accumulator lanes serialise on one bank
# group, yielding the K-913 LDS_BC ≈ 1.45-2.13 cyc/inst signature
# (vs N=256's ≈ 0.6-0.9 cyc/inst).  Sharing P21's N=256 frozenset would
# (a) admit cells that P21 was never measured against (sibling-N firewall
# violation), and (b) couple the N=128 and N=256 audit handles such that
# any future P21 contraction silently breaks the K-1673 N=128 alias
# coverage.  Per the K-1175 stacked-predicate convention, each N-bucket
# carries its own frozenset; the K-1633 P27 alias-of-P23 precedent only
# applies when the admit set is BIT-IDENTICAL (same N axis, same K-grid,
# same dtype-set) — not the case here (P21 N=256 K ∈ {4096, 8192, 16384}
# vs P28 N=128 K ∈ {2048, 4096, 8192, 16384, 32768}).
#
# Mechanism (K-1673 PMC RCA, source `output/k1673_rca_note.md`): TB
# SQ_LDS_BANK_CONFLICT/inst = 1.45-2.13 cyc (vs HBL = 0.000 exactly) —
# the K-913 §3 dtype-invariant LDS-bank-conflict signature already proved
# at K-1338 / K-1361 / K-1397 / K-1611, here applied to the N=128 column-
# narrow tile.  persistent_matmul cannot relieve the bank-conflict
# pressure via tile reshape; route-OUT to hipBLASLt is the only remaining
# mechanism (R-1329.CROSS-BRANCH-RATIO-INVARIANT-IMPLIES-ROUTE-OUT-ONLY).
# The per-row peak at M=2048 (1.889×) confirms K-913 LDS-BC is fully live
# across all K when min(M,N) = 128; M ∈ {4096, 8192} attenuate slightly
# but still clear the strict 1.05 admit gate.
# ---------------------------------------------------------------------------
_K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 = frozenset({
    # M=2048 row × N=128 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (2048, 128,  2048, "torch.bfloat16"),    # r=1.313 (P5 alias)
    (2048, 128,  2048, "torch.float16"),     # r=1.296 NEW
    (2048, 128,  4096, "torch.bfloat16"),    # r=1.420 (P5 + P13 alias)
    (2048, 128,  4096, "torch.float16"),     # r=1.430 (P13 alias)
    (2048, 128,  8192, "torch.bfloat16"),    # r=1.929 (P5 + P13 alias)
    (2048, 128,  8192, "torch.float16"),     # r=1.962 (P13 alias)
    (2048, 128, 16384, "torch.bfloat16"),    # r=2.340 (P5 + P13 alias)
    (2048, 128, 16384, "torch.float16"),     # r=2.286 (P13 alias)
    (2048, 128, 32768, "torch.bfloat16"),    # r=2.890 (P5 alias; max-ratio bf16)
    (2048, 128, 32768, "torch.float16"),     # r=2.861 NEW (max-ratio overall)
    # M=4096 row × N=128 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (4096, 128,  2048, "torch.bfloat16"),    # r=1.179 (P5 alias)
    (4096, 128,  2048, "torch.float16"),     # r=1.162 NEW (min-ratio overall)
    (4096, 128,  4096, "torch.bfloat16"),    # r=1.171 (P5 + P13 alias)
    (4096, 128,  4096, "torch.float16"),     # r=1.167 (P13 alias)
    (4096, 128,  8192, "torch.bfloat16"),    # r=1.522 (P5 + P13 alias)
    (4096, 128,  8192, "torch.float16"),     # r=1.427 (P13 alias)
    (4096, 128, 16384, "torch.bfloat16"),    # r=1.782 (P5 + P13 alias)
    (4096, 128, 16384, "torch.float16"),     # r=1.782 (P13 alias)
    (4096, 128, 32768, "torch.bfloat16"),    # r=1.841 (P5 alias)
    (4096, 128, 32768, "torch.float16"),     # r=1.821 NEW
    # M=8192 row × N=128 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
    (8192, 128,  2048, "torch.bfloat16"),    # r=1.473 (P5 alias)
    (8192, 128,  2048, "torch.float16"),     # r=1.413 NEW
    (8192, 128,  4096, "torch.bfloat16"),    # r=1.752 (P5 + P13 alias)
    (8192, 128,  4096, "torch.float16"),     # r=1.728 (P13 alias)
    (8192, 128,  8192, "torch.bfloat16"),    # r=2.010 (P5 + P13 alias)
    (8192, 128,  8192, "torch.float16"),     # r=1.805 (P13 alias)
    (8192, 128, 16384, "torch.bfloat16"),    # r=1.917 (P5 + P13 alias)
    (8192, 128, 16384, "torch.float16"),     # r=1.924 (P13 alias)
    (8192, 128, 32768, "torch.bfloat16"),    # r=1.889 (P5 alias)
    (8192, 128, 32768, "torch.float16"),     # r=1.859 NEW
})
assert len(_K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30) == 30, (
    "K-1673 P28 skinny_N128 K-COMPLEMENT alias-stack frozenset must be "
    "exactly 30 cells (M ∈ {2048,4096,8192} × N=128 × K ∈ {2048,4096,8192,"
    "16384,32768} × {bf16, fp16}); deviation indicates a typo against the "
    "K-1673 paired n=30 admit set.")
# Cross-frozenset disjointness — K-1673 P28 vs the prior 15-K-COMPLEMENT
# stack (P13 N=128 18-cell K-extremes-complement, P13 N=256, P15 N=512,
# P16 N=1024, P17 N=512 BASE, P19 N=16384, P21 N=256 K-mid, P22 N=32768,
# P23 N=512 alias, P24 N=4096, P26 N=2048 alias).  The K-1367 P13 N=128
# 18-cell envelope intentionally OVERLAPS P28 (alias of P13 by design, the
# 9 fp16 K ∈ {4096, 8192, 16384} cells are the alias subset documented
# below); every OTHER predecessor uses N ∈ {256, 512, 1024, 4096, 16384,
# 32768, 2048} and is sibling-N-firewall disjoint from N=128 by
# construction.  Data-driven assert loop (K-1532 minimalist refactor):
# every K-COMPLEMENT sibling listed below must be FULLY disjoint with the
# K-1673 admit set; the P13_N128 alias-overlap predicate is excluded and
# asserted via the dedicated `_K1673_P28_VS_P13_N128_ALIAS_OVERLAP` pin.
_K1673_P28_DISJOINT_SIBLINGS = (
    ("P8 (K-1322 N≤256 envelope)",        _P8_MFMA_ISSUE_STALL_ROUTEOUT),
    ("P12 (K-1361 M=N=K∈{2048,4096})",    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4),
    ("P13 N=256 (K-1397)",                _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12),
    ("P15 N=512 (K-1409)",                _K1409_P15_SKINNY_N512_KCOMPL_ROUTEOUT),
    ("P16 N=1024 (K-1429)",               _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29),
    ("P17 N=512 BASE (K-1437)",           _K1437_P17_SKINNY_N512_KCOMPL_BASE_ROUTEOUT_17),
    ("P19 N=16384 (K-1478)",              _K1478_P19_SKINNY_N16384_KCOMPL_ROUTEOUT_30),
    ("P21 N=256 K-mid (K-1503)",          _K1503_P21_SKINNY_N256_KCOMPL_KMID_ROUTEOUT),
    ("P22 N=32768 (K-1513)",              _K1513_P22_SKINNY_N32768_KCOMPL_ROUTEOUT_30),
    ("P23 N=512 alias (K-1552)",          _K1552_P23_SKINNY_N512_KCOMPL_ALIASSTACK_30),
    ("P24 N=4096 (K-1566)",               _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30),
    ("P26 N=2048 alias (K-1611)",         _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30),
)
for _sibling_name, _sibling_set in _K1673_P28_DISJOINT_SIBLINGS:
    assert _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30.isdisjoint(_sibling_set), (
        f"K-1673 P28 skinny_N128 (N=128) overlaps {_sibling_name}; "
        "sibling-N firewall violated — every K-COMPLEMENT predecessor "
        "EXCEPT the K-1367 P13 N=128 18-cell envelope (intentionally "
        "aliased here) uses N ∈ {256, 512, 1024, 2048, 4096, 16384, "
        "32768}, so the N=128 column must be disjoint by construction.  "
        "The intentional P13 N=128 alias is asserted separately via "
        "`_K1673_P28_VS_P13_N128_ALIAS_OVERLAP`.")
del _sibling_name, _sibling_set
# Explicit K-1367 P13 N=128 alias overlap: K-1367 P13 routes M ∈ {2048,
# 4096, 8192} × N=128 × K ∈ {4096, 8192, 16384} × {bf16, fp16} (18 cells);
# K-1673 P28's full 30-cell envelope INCLUDES those same 18 cells.  P13
# fires at the 7th-position dispatch slot well before P28's 19th slot, so
# the P28 membership check on those 18 cells is unreachable while P13
# remains enabled — the 18 cells are alias documentation, the remaining
# 12 cells (K ∈ {2048, 32768} × {bf16, fp16}) are addressed by the P5
# alias (bf16) plus the 6 NEW route-OUT cells (fp16).
_K1673_P28_VS_P13_N128_ALIAS_OVERLAP = (
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30
    & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)
assert _K1673_P28_VS_P13_N128_ALIAS_OVERLAP == frozenset({
    (M, 128, K, dtype)
    for M in (2048, 4096, 8192)
    for K in (4096, 8192, 16384)
    for dtype in ("torch.bfloat16", "torch.float16")
}), (
    "K-1673 P28 vs K-1367 P13 N=128 alias overlap must be EXACTLY the 18 "
    "cells (M ∈ {2048,4096,8192} × N=128 × K ∈ {4096,8192,16384} × {bf16, "
    "fp16}); any deviation indicates either a P13 contraction (re-audit "
    "P28 membership for the affected cells) or a P28 authoring typo "
    "against the K-1673 admit set.")
# Explicit R-K979 P5 Clause-3 alias overlap: P5 Clause-3 routes ALL bf16
# cells with min(M,N) ≤ 192 ∧ K ≥ 2048 (a structural envelope, not a
# strict-equality table); for the K-1673 P28 30-cell envelope this catches
# all 15 bf16 cells (M ∈ {2048,4096,8192} × N=128 × K ∈ {2048,4096,8192,
# 16384,32768}).  P5 fires at the 5th-position dispatch slot well before
# P28's 19th slot.  bf16-only by P5's `_dtype_is_bf16` early gate; the 15
# fp16 mirror cells are NOT covered by P5 (9 of them are caught by P13 at
# the 7th slot, the remaining 6 are the NEW route-OUT contribution of
# P28).  Asserted dynamically because P5 is a closed-form predicate, not
# a frozenset.
_K1673_P28_VS_P5_ALIAS_OVERLAP = frozenset({
    cell for cell in _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30
    if R_K979_P5_route_to_hbl(cell[0], cell[1], cell[2], cell[3])
})
assert _K1673_P28_VS_P5_ALIAS_OVERLAP == frozenset({
    (M, 128, K, "torch.bfloat16")
    for M in (2048, 4096, 8192)
    for K in (2048, 4096, 8192, 16384, 32768)
}), (
    "K-1673 P28 vs R-K979 P5 Clause-3 alias overlap must be EXACTLY the "
    "15 bf16 cells (M ∈ {2048,4096,8192} × N=128 × K ∈ {2048,4096,8192,"
    "16384,32768} × bf16); P5 Clause-3 (minMN ≤ 192 ∧ K ≥ 2048, bf16-only) "
    "structurally catches every bf16 cell in the P28 envelope.  Any "
    "deviation indicates either a P5 K-floor / minMN-ceiling contraction "
    "or a P28 authoring typo against the K-1673 admit set.")
# ALIAS-STACK invariant: every cell in the K-1673 admit set MUST either
# (a) be covered by an upstream firing predicate (R-K979 P5 Clause-3 for
# bf16, K-1367 P13 N=128 for fp16 K ∈ {4096, 8192, 16384}) OR (b) be NEW
# route-OUT contributed by P28.  The (b) cells (the 6 NEW fp16 cells at
# K ∈ {2048, 32768}) are the load-bearing portion; the (a) cells (24
# alias cells) are documentation that is unreachable under normal
# dispatch but freezes the K-1673 30-cell envelope under a single audit
# handle.  Total: 6 NEW + 24 alias = 30 K-1673 admits.
_K1673_P28_NEW_ROUTEOUT_6 = (
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30
    - _K1673_P28_VS_P5_ALIAS_OVERLAP
    - _K1673_P28_VS_P13_N128_ALIAS_OVERLAP
)
assert _K1673_P28_NEW_ROUTEOUT_6 == frozenset({
    (M, 128, K, "torch.float16")
    for M in (2048, 4096, 8192)
    for K in (2048, 32768)
}), (
    "K-1673 P28 NEW route-OUT contribution must be exactly 6 cells "
    "(30 admits − 15 P5-bf16-alias − 18 P13-N128-alias + 9 P5∩P13 "
    "double-counted overlap = 6 NEW; equivalently the 6 fp16 cells at "
    "M ∈ {2048,4096,8192} × N=128 × K ∈ {2048, 32768} that fall outside "
    "both P5 (bf16-only) and P13 (K ∈ {4096,8192,16384})); any deviation "
    "indicates either an upstream contraction (re-audit which cells P28 "
    "newly contributes) or a P28 authoring typo against the K-1673 "
    "admit set.")
assert len(_K1673_P28_NEW_ROUTEOUT_6) == 6


def _k1673_p28_skinny_n128_kcompl_aliasstack_routeout(
    M: int, N: int, K: int, dtype) -> bool:
    """K-1673 P28 — direct hipBLASLt route-OUT for the K-1673-verified
    30-cell skinny_N128 K-COMPLEMENT cohort
    (`_K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30`).

    Returns True iff (M, N, K, dtype) matches one of the 30 strict-equality
    keys: M ∈ {2048, 4096, 8192} × N = 128 × K ∈ {2048, 4096, 8192, 16384,
    32768} × dtype ∈ {torch.bfloat16, torch.float16}.

    Source measurement: K-1673 paired n=30 HIP-graph hot-cache benchmarks
    on MI300X / gfx942 (OCI fallback amd-rccl partition) against
    the LIVE post-K-1647 P27 routing oracle (fork branch fix/K-1647 tip
    8d010c6); 30/30 ROUTE-OUT-CANDIDATE at the strict ratio_median ≥ 1.05
    ∧ p(<1.05) < 0.01 gate; cohort geomean tb/hbl = 1.694×, range
    1.162×-2.890×, 0 regressions.  Productionised under the K-1474 /
    K-1492 / K-1493 / K-1532 / K-1581 protocol with the 0.85× geomean
    route-OUT win threshold (1.694 ≥ 0.85) and the ≥80% per-cell admit
    gate (30/30 ≥ 80%).  Closes the LAST untested small-N rung (N=128) of
    the K-COMPLEMENT N-ladder at the dtype-mirror gap (fp16 K ∈ {2048,
    32768}) — coverage now spans the FULL N-ladder {128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768} on the M ∈ {2048, 4096, 8192} rows
    for both bf16 and fp16.

    ALIAS-STACK structure: 24 of the 30 cells are upstream-aliased and
    fire via earlier slots (R-K979 P5 Clause-3 catches all 15 bf16 cells
    via the structural minMN ≤ 192 ∧ K ≥ 2048 envelope; K-1367 P13 N=128
    catches 9 fp16 cells at M ∈ {2048,4096,8192} × N=128 × K ∈ {4096,
    8192, 16384}).  The remaining 6 cells are NEW route-OUT contributed
    by this 19th-position slot — the fp16 mirror at K ∈ {2048, 32768} ×
    M ∈ {2048, 4096, 8192} that P5 (bf16-only) and P13 (K-mid only)
    structurally exclude.  Per-cell admit ratios for the 6 NEW cells:
    1.162× (M=4096, K=2048, fp16) – 2.861× (M=2048, K=32768, fp16); NEW
    sub-cohort geomean = 1.656×.

    Mechanism (K-1673 PMC RCA, source `output/k1673_rca_note.md`): TB
    SQ_LDS_BANK_CONFLICT/inst = 1.45-2.13 cyc (vs HBL = 0.000 exactly) —
    the K-913 §3 dtype-invariant LDS-bank-conflict signature already
    proved at K-1338 / K-1361 / K-1397 / K-1611, here applied to the
    N=128 column-narrow tile.  The bank-arbitration topology lives at
    the LDS swizzle layer (BELOW the dtype lane mux) so the fp16 cells
    inherit the same BC tax as their bf16 cousins; the existing oracle
    predicates were authored on bf16-anchored evidence campaigns
    (K-984 / K-989 / K-1335) and never extended to fp16 mirrors.
    K-984 / K-1020 / K-1028 already established the dtype-invariant fp16-
    mirror pattern as a productionization shape on three prior LDS-bound
    campaigns (K-905 / K-971 / K-1335); P28 is the same shape applied to
    the N=128 K-COMPLEMENT extreme-K dtype-mirror gap.

    Why N=128 needs its own frozenset rather than sharing N=256/N=512's:
    the K-913 §3 LDS bank-conflict signature is sharply N-discriminating
    — at N=128 the BLOCK_N=128 tile has 1 K-block column and ALL
    accumulator lanes serialise on one bank group (LDS_BC ≈ 1.45-2.13
    cyc/inst); at N=256 the same tile has 2 K-block columns and the LDS
    bank arbitration round-robins across 2 swizzle phases (LDS_BC ≈
    0.6-0.9 cyc/inst).  Sharing P21's N=256 frozenset would (a) admit
    cells that P21 was never measured against (sibling-N firewall
    violation) and (b) couple the N=128 and N=256 audit handles such
    that any future P21 contraction silently breaks the K-1673 N=128
    alias coverage.  The K-1633 P27 alias-of-P23 precedent only applies
    when the admit set is BIT-IDENTICAL — not the case for P28 vs any
    existing 30-cell K-COMPLEMENT frozenset.

    Stacked at 19th-position per the K-1175 stacked-predicate convention
    after K-1633 P27 (18th-slot module-level alias of P23).
    """
    return (
        (int(M), int(N), int(K), str(dtype))
        in _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30
    )


# K-1700 P29 (20th-slot): N=64 K-COMPLEMENT alias-stack — 29 K-1709 admit cells
# (M ∈ {2048,4096,8192} × N=64 × K ∈ {2048,4096,8192,16384,32768} × {bf16, fp16}
# minus the bubble cell (2048, 64, 4096, fp16) which fails the strict 1.05 gate
# at CI95-lo=1.045).  15 bf16 cells alias R-K979 P5 Clause-3; 14 fp16 cells are
# NEW route-OUT.  Sibling-N disjoint by construction (no prior K-COMPL slot N=64).
_K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 = frozenset(
    (M, 64, K, dt) for M in (2048, 4096, 8192)
    for K in (2048, 4096, 8192, 16384, 32768) for dt in ("torch.bfloat16", "torch.float16")
) - frozenset({(2048, 64, 4096, "torch.float16")})

# K-1748 P30 (21st-slot): K-1711 N-mid (N ∈ {384, 768, 1536}) K-COMPLEMENT
# alias-stack, 34 cells (10 / 14 / 10 per N).  Audit (K-1748): post-K-1709
# oracle had 0/34 active (K-1720 was a parallel K-1685 branch never merged).
# Sibling-N disjoint vs P29 (N=64) and vs P16 (N=1024 covered control).
# K-1711 paired n=30 HIP-graph hot-cache MI300X (geomean 1.456×, 1.18-1.83×).
_K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 = frozenset(
    (M, N, K, dt) for (M, N, K) in (
        (2048,384,8192),(2048,384,32768),(4096,384,8192),(4096,384,32768),(8192,384,32768),
        (2048,768,8192),(2048,768,32768),(4096,768,8192),(4096,768,32768),
        (8192,768,2048),(8192,768,8192),(8192,768,32768),
        (2048,1536,32768),(4096,1536,2048),(4096,1536,8192),(4096,1536,32768),(8192,1536,32768),
    ) for dt in ("torch.bfloat16", "torch.float16")
)
# Cardinality (==34) gated by tests/test_k1748_p30_skinny_nmid_alias_stack.py per
# the minimalist split: src holds data, tests hold invariants (R-1532 / R-1720).

# P31 (22nd-slot): N=256 K-COMPLEMENT verified-winner subset — 28 cells (the
# 30-cell M ∈ {2048,4096,8192} × N=256 × K ∈ {2048,4096,8192,16384,32768} ×
# {bf16,fp16} cohort minus 2 paired-n30 LOSER cells at (M=2048, K=2048): TB-native
# beats hipBLASLt by 2% in bfloat16 (HBL/TB=0.98) and ties in float16 (HBL/TB=1.02,
# p=0.79 — fails the strict ≥1.05 ∧ p<0.05 gate).
# 28-cell winner subset paired n=30 hot-cache HIP-graph TB-native-vs-HBL on MI300X
# (route-OUT ablated): geomean HBL/TB = 1.392×, range 1.07×–2.12×, 28/28 pass the
# ≥1.05 ∧ p<0.05 gate; full-cohort geomean (all 30) = 1.359×, 28/30 pass the gate.
_P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28 = frozenset(
    (M, 256, K, dt) for M in (2048, 4096, 8192)
    for K in (2048, 4096, 8192, 16384, 32768) for dt in ("torch.bfloat16", "torch.float16")
) - frozenset({
    (2048, 256, 2048, "torch.bfloat16"),
    (2048, 256, 2048, "torch.float16"),
})
# Cardinality (==28) gated by tests/test_p31_skinny_n256_alias_stack.py per the
# minimalist split: src holds data, tests hold invariants.

# K-1881 (S-002): consolidated P32–P38 K-COMPLEMENT verified-winner alias-stack.
# Extends K-1864's 89-cell P32–P36 consolidation in-place to absorb P37
# (N=352, K-1868, 17 cells) and P38 (N=384, K-1880, 14 cells) → ONE
# 120-cell frozenset replacing what would otherwise be a 7-probe chain.
# All seven slots share the BLOCK_N=128 wave-MISalignment / K-913 §3
# LDS-bank-conflict root cause (R-1811) and the hipBLASLt routing target.
# Per-slot mechanism + PMC RCA + alias-overlap discipline lives in the
# linked per-slot tickets (P32 K-1794 / P33 K-1794 / P34 K-1818 / P35
# K-1832 / P36 K-1843 / P37 K-1843 / P38 K-1857).  Cells are inlined as
# literal tuples (single source of truth, sorted (N,M,K,dtype) for diff-
# ability); the 7 per-slot views below are N-axis projections consumed by
# `tests/test_p3{2..6}_*_alias_stack.py` only — the dispatcher probes the
# canonical set directly.  Pairwise disjointness is by construction
# (distinct N integers).
#
# Compact-predicate audit (per K-1858 RCA): `N % 64 != 0 and N <= 384 and
# K >= 4096` MISSES N=320,384 (both ≡ 0 mod 64); `N % 128 != 0 and ...`
# over-includes N ∈ {32,64,192} cells outside the productionised set; 6
# cell-level exclusions ((2048,320,4096,bf16), (2048,352,4096,bf16),
# (2048,384,8192,*), (4096,384,8192,*)) preclude any clean structural
# predicate.  Conclusion: literal frozenset is minimum-information; NO
# predicate substitution.  Boundary cases are pinned in
# `tests/test_k1881_p32_p38_consolidation.py::test_upstream_aliased_
# exclusions_are_preserved` and `_predicate_audit_boundary_*`.
_K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120 = frozenset({
    # ---- P34 (N=96) — 18 cells (K-1818 sub-cohort, full grid) ----
    (2048,  96,  4096, "torch.bfloat16"),
    (2048,  96,  4096, "torch.float16"),
    (2048,  96,  8192, "torch.bfloat16"),
    (2048,  96,  8192, "torch.float16"),
    (2048,  96, 16384, "torch.bfloat16"),
    (2048,  96, 16384, "torch.float16"),
    (4096,  96,  4096, "torch.bfloat16"),
    (4096,  96,  4096, "torch.float16"),
    (4096,  96,  8192, "torch.bfloat16"),
    (4096,  96,  8192, "torch.float16"),
    (4096,  96, 16384, "torch.bfloat16"),
    (4096,  96, 16384, "torch.float16"),
    (8192,  96,  4096, "torch.bfloat16"),
    (8192,  96,  4096, "torch.float16"),
    (8192,  96,  8192, "torch.bfloat16"),
    (8192,  96,  8192, "torch.float16"),
    (8192,  96, 16384, "torch.bfloat16"),
    (8192,  96, 16384, "torch.float16"),
    # ---- P32 (N=160) — 18 cells (K-1794 sub-cohort, full grid;
    #      9 bf16 cells alias-overlap with R-K979 P5 by design) ----
    (2048, 160,  4096, "torch.bfloat16"),
    (2048, 160,  4096, "torch.float16"),
    (2048, 160,  8192, "torch.bfloat16"),
    (2048, 160,  8192, "torch.float16"),
    (2048, 160, 16384, "torch.bfloat16"),
    (2048, 160, 16384, "torch.float16"),
    (4096, 160,  4096, "torch.bfloat16"),
    (4096, 160,  4096, "torch.float16"),
    (4096, 160,  8192, "torch.bfloat16"),
    (4096, 160,  8192, "torch.float16"),
    (4096, 160, 16384, "torch.bfloat16"),
    (4096, 160, 16384, "torch.float16"),
    (8192, 160,  4096, "torch.bfloat16"),
    (8192, 160,  4096, "torch.float16"),
    (8192, 160,  8192, "torch.bfloat16"),
    (8192, 160,  8192, "torch.float16"),
    (8192, 160, 16384, "torch.bfloat16"),
    (8192, 160, 16384, "torch.float16"),
    # ---- P33 (N=224) — 18 cells (K-1794 sub-cohort, full grid) ----
    (2048, 224,  4096, "torch.bfloat16"),
    (2048, 224,  4096, "torch.float16"),
    (2048, 224,  8192, "torch.bfloat16"),
    (2048, 224,  8192, "torch.float16"),
    (2048, 224, 16384, "torch.bfloat16"),
    (2048, 224, 16384, "torch.float16"),
    (4096, 224,  4096, "torch.bfloat16"),
    (4096, 224,  4096, "torch.float16"),
    (4096, 224,  8192, "torch.bfloat16"),
    (4096, 224,  8192, "torch.float16"),
    (4096, 224, 16384, "torch.bfloat16"),
    (4096, 224, 16384, "torch.float16"),
    (8192, 224,  4096, "torch.bfloat16"),
    (8192, 224,  4096, "torch.float16"),
    (8192, 224,  8192, "torch.bfloat16"),
    (8192, 224,  8192, "torch.float16"),
    (8192, 224, 16384, "torch.bfloat16"),
    (8192, 224, 16384, "torch.float16"),
    # ---- P35 (N=288) — 18 cells (K-1832 sub-cohort, full grid) ----
    (2048, 288,  4096, "torch.bfloat16"),
    (2048, 288,  4096, "torch.float16"),
    (2048, 288,  8192, "torch.bfloat16"),
    (2048, 288,  8192, "torch.float16"),
    (2048, 288, 16384, "torch.bfloat16"),
    (2048, 288, 16384, "torch.float16"),
    (4096, 288,  4096, "torch.bfloat16"),
    (4096, 288,  4096, "torch.float16"),
    (4096, 288,  8192, "torch.bfloat16"),
    (4096, 288,  8192, "torch.float16"),
    (4096, 288, 16384, "torch.bfloat16"),
    (4096, 288, 16384, "torch.float16"),
    (8192, 288,  4096, "torch.bfloat16"),
    (8192, 288,  4096, "torch.float16"),
    (8192, 288,  8192, "torch.bfloat16"),
    (8192, 288,  8192, "torch.float16"),
    (8192, 288, 16384, "torch.bfloat16"),
    (8192, 288, 16384, "torch.float16"),
    # ---- P36 (N=320) — 17 cells (K-1843 full grid MINUS the
    #      (2048, 320, 4096, bf16) upstream-aliased exclusion) ----
    (2048, 320,  4096, "torch.float16"),
    (2048, 320,  8192, "torch.bfloat16"),
    (2048, 320,  8192, "torch.float16"),
    (2048, 320, 16384, "torch.bfloat16"),
    (2048, 320, 16384, "torch.float16"),
    (4096, 320,  4096, "torch.bfloat16"),
    (4096, 320,  4096, "torch.float16"),
    (4096, 320,  8192, "torch.bfloat16"),
    (4096, 320,  8192, "torch.float16"),
    (4096, 320, 16384, "torch.bfloat16"),
    (4096, 320, 16384, "torch.float16"),
    (8192, 320,  4096, "torch.bfloat16"),
    (8192, 320,  4096, "torch.float16"),
    (8192, 320,  8192, "torch.bfloat16"),
    (8192, 320,  8192, "torch.float16"),
    (8192, 320, 16384, "torch.bfloat16"),
    (8192, 320, 16384, "torch.float16"),
    # ---- P37 (N=352) — 17 cells (K-1843 N=352 sub-cohort, full grid
    #      MINUS (2048, 352, 4096, bf16) upstream-aliased exclusion;
    #      paired-n30 ratio_TB/HBL=1.0031, p=0.167 — parity band) ----
    (2048, 352,  4096, "torch.float16"),
    (2048, 352,  8192, "torch.bfloat16"),
    (2048, 352,  8192, "torch.float16"),
    (2048, 352, 16384, "torch.bfloat16"),
    (2048, 352, 16384, "torch.float16"),
    (4096, 352,  4096, "torch.bfloat16"),
    (4096, 352,  4096, "torch.float16"),
    (4096, 352,  8192, "torch.bfloat16"),
    (4096, 352,  8192, "torch.float16"),
    (4096, 352, 16384, "torch.bfloat16"),
    (4096, 352, 16384, "torch.float16"),
    (8192, 352,  4096, "torch.bfloat16"),
    (8192, 352,  4096, "torch.float16"),
    (8192, 352,  8192, "torch.bfloat16"),
    (8192, 352,  8192, "torch.float16"),
    (8192, 352, 16384, "torch.bfloat16"),
    (8192, 352, 16384, "torch.float16"),
    # ---- P38 (N=384) — 14 cells (K-1857 N=384 sub-cohort, full grid
    #      MINUS the 4 (M∈{2048,4096})×K=8192×{bf16,fp16} cells already
    #      routed by upstream P30 _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK
    #      _34 — R-K1825 alias-coverage-map exclusion) ----
    (2048, 384,  4096, "torch.bfloat16"),
    (2048, 384,  4096, "torch.float16"),
    (2048, 384, 16384, "torch.bfloat16"),
    (2048, 384, 16384, "torch.float16"),
    (4096, 384,  4096, "torch.bfloat16"),
    (4096, 384,  4096, "torch.float16"),
    (4096, 384, 16384, "torch.bfloat16"),
    (4096, 384, 16384, "torch.float16"),
    (8192, 384,  4096, "torch.bfloat16"),
    (8192, 384,  4096, "torch.float16"),
    (8192, 384,  8192, "torch.bfloat16"),
    (8192, 384,  8192, "torch.float16"),
    (8192, 384, 16384, "torch.bfloat16"),
    (8192, 384, 16384, "torch.float16"),
})

# Per-slot N-axis projection views — for tests/test_p3{2..6}_*_alias_stack.py
# only; the dispatcher never references these.
def _k1881_proj(_N): return frozenset(c for c in _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120 if c[1] == _N)
_P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18 = _k1881_proj(160)
_P33_SKINNY_N224_KCOMPL_VERIFIED_WIN_18 = _k1881_proj(224)
_P34_SKINNY_N96_KCOMPL_VERIFIED_WIN_18  = _k1881_proj(96)
_P35_SKINNY_N288_KCOMPL_VERIFIED_WIN_18 = _k1881_proj(288)
_P36_SKINNY_N320_KCOMPL_VERIFIED_WIN_17 = _k1881_proj(320)
_P37_SKINNY_N352_KCOMPL_VERIFIED_WIN_17 = _k1881_proj(352)
_P38_SKINNY_N384_KCOMPL_VERIFIED_WIN_14 = _k1881_proj(384)
del _k1881_proj

# K-1922 (S-002): P40 (30th-slot) skinny_N544 K-COMPLEMENT alias-stack —
# 18-cell verified-winner subset for the off-by-32 wave-misaligned N=544 rung
# (544 mod 128 = 32 — same off-by-32 band as P32 N=160, P35 N=288, P37 N=352).
# K-1912 paired n=30 HIP-graph hot-cache MI300X / gfx942 measurement vs the
# live post-K-1881 oracle confirmed 18/18 admit at the strict ≥1.05 ∧ p<0.05
# gate over the M ∈ {2048,4096,8192} × N=544 × K ∈ {4096,8192,16384} ×
# {bf16,fp16} grid; cohort geomean tb/hbl ≈ 1.51× (range ≈ 1.18×–2.05×, no
# upstream alias overlap at this off-band rung).  Sibling-N firewall disjoint
# by construction with every prior K-COMPLEMENT alias-stack slot (P28 N=128,
# P29 N=64, P30 N ∈ {384, 768, 1536}, P31 N=256, consolidated P32–P38
# N ∈ {96,160,224,288,320,352,384}, plus K-1367/K-1397 P13 N ∈ {128, 256}).
#
# Mechanism (R-1811 wave-misalignment + K-913 §3 LDS-bank-conflict):
# BLOCK_N=128 packs N=544 into 4.25 BLOCK_N tiles per N-row — same SCHEDULER_
# LDS A4 failure mode as the K-1681 / K-1710 / K-1781 / K-1812 / K-1824 /
# K-1832 / K-1843 wave-misaligned skinny-N class.  hipBLASLt's split-K kernel
# selection clears the band by ~50% on average.  Per K-1908 compact-predicate
# analysis, N=544 is NOT covered by the existing S1-form
# `(N % 64 != 0) ∧ (N <= 384) ∧ (K >= 4096)` predicate (N=544 > 384 and N % 64
# == 32 — the predicate's N-ceiling cuts off below this rung), so explicit
# alias-stack promotion is required until S1 is extended in a separate task.
# K-1900's compact-predicate replacement angle failed at depth-2 closure and
# is NOT retried here.
_K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18 = frozenset(
    (M, 544, K, dt) for M in (2048, 4096, 8192)
    for K in (4096, 8192, 16384) for dt in ("torch.bfloat16", "torch.float16")
)
# Cardinality (==18) gated by tests/test_k1922_p40_skinny_n544_alias_stack.py
# per the minimalist split: src holds data, tests hold invariants
# (R-1532 / R-1720 / R-1775).

# K-1963 (S-002): P41–P44 (31st–34th-slot) wave-misaligned skinny-N
# K-COMPLEMENT alias-stack quad — N ∈ {608, 672, 704, 736}.  Extends the
# K-1922 P40 N=544 promotion one rung at a time across the next four
# successive K-COMPLEMENT N-buckets validated by the live-oracle
# paired-n=30 HIP-graph hot-cache MI300X / gfx942 cohorts of
# K-1938 (N=608), K-1923 / K-1927 / K-1931 (N=672), K-1945 (N=704), and
# K-1941 / K-1943 (N=736).  Each slot is a full M ∈ {2048,4096,8192} × N ×
# K ∈ {4096,8192,16384} × {bf16,fp16} 18-cell verified-winner subset (no
# upstream-aliased exclusions at these off-band rungs — confirmed via
# `_k971_route_to_hbl(...)` direct call on each candidate cell against the
# fix/K-1922 HEAD oracle).  Union cohort = 4 × 18 = 72 cells, geomean
# tb/hbl ≥ ~1.4× per rung, 0/72 regressions on the K-1925 231-cell drift
# baseline.
#
# Wave-/tile-alignment fingerprint per rung (BLOCK_N=128 packing):
#   N=608: 608 mod 64 = 32 (wave-misaligned), 608 mod 128 = 96 — packs into
#          4 full BN=128 tiles + 1 BN=96 tail (3/4-tile tail).
#   N=672: 672 mod 64 = 32 (wave-misaligned), 672 mod 128 = 32 — packs into
#          5 full BN=128 tiles + 1 BN=32 tail (1/4-tile tail; same off-by-32
#          tail-fragment band as P40 N=544 = 4.25 BN-tiles).
#   N=704: 704 mod 64 = 0 (wave-ALIGNED at the 64-lane SIMD; arithmetic
#          correction vs the K-1963 PRD's blanket "N mod 64 == 32" claim —
#          704 = 11×64 exactly), 704 mod 128 = 64 — packs into 5 full
#          BN=128 tiles + 1 BN=64 *half-tile* tail.  Same K-913 §3
#          SCHEDULER_LDS A4 mechanism attacked from a different geometric
#          angle (LDS-double-buffer prologue collision rather than
#          MFMA-lane half-utilisation).
#   N=736: 736 mod 64 = 32 (wave-misaligned), 736 mod 128 = 96 — packs into
#          5 full BN=128 tiles + 1 BN=96 tail (3/4-tile tail; same
#          tail-fragment fingerprint as N=608 one rung down).
#
# Mechanism (R-1811 wave-misalignment + K-913 §3 LDS-bank-conflict +
# K-1945 BN-half-tile LDS-double-buffer collision): the persistent_matmul
# kernel cannot trade tile reshape for atomic-reduction across these tail
# fragments; hipBLASLt's split-K kernel selection clears the band by ~50%
# on average per the K-1938 / K-1923 / K-1945 / K-1941 paired-n=30 cohorts.
# Per K-1908 compact-predicate analysis, none of the four rungs are covered
# by the existing S1-form `(N % 64 != 0) ∧ (N <= 384) ∧ (K >= 4096)`
# predicate (all four N > 384 — predicate's N-ceiling cuts off below this
# rung), so explicit alias-stack promotion is required until S1 is extended
# in a separate task.  Per K-1946 a depth-4 closed-form predicate over the
# {544, 608, 672, 704, 736} union is *available* but is NOT yet substituted
# here pending K-1942-style bit-equiv revalidation on the expanded
# P41–P44 set; this is left to a follow-up task.  Sibling-N firewall
# disjoint by construction with every prior K-COMPLEMENT alias-stack slot
# (P28 N=128, P29 N=64, P30 N ∈ {384,768,1536}, P31 N=256, the
# consolidated P32–P38 N ∈ {96,160,224,288,320,352,384}, P40 N=544, plus
# K-1367/K-1397 P13 N ∈ {128,256}) and pairwise disjoint across P41–P44
# themselves (each rung is a distinct N-axis bucket).
_K1963_P41_SKINNY_N608_KCOMPL_ALIASSTACK_18 = frozenset(
    (M, 608, K, dt) for M in (2048, 4096, 8192)
    for K in (4096, 8192, 16384) for dt in ("torch.bfloat16", "torch.float16")
)
_K1963_P42_SKINNY_N672_KCOMPL_ALIASSTACK_18 = frozenset(
    (M, 672, K, dt) for M in (2048, 4096, 8192)
    for K in (4096, 8192, 16384) for dt in ("torch.bfloat16", "torch.float16")
)
_K1963_P43_SKINNY_N704_KCOMPL_ALIASSTACK_18 = frozenset(
    (M, 704, K, dt) for M in (2048, 4096, 8192)
    for K in (4096, 8192, 16384) for dt in ("torch.bfloat16", "torch.float16")
)
_K1963_P44_SKINNY_N736_KCOMPL_ALIASSTACK_18 = frozenset(
    (M, 736, K, dt) for M in (2048, 4096, 8192)
    for K in (4096, 8192, 16384) for dt in ("torch.bfloat16", "torch.float16")
)
# Per-slot cardinality (==18 each) and union cardinality (==72) gated by
# tests/test_k1963_p41_p44_skinny_n608_672_704_736_alias_stack.py per the
# minimalist split: src holds data, tests hold structural invariants
# (R-1532 / R-1720 / R-1775).
assert len(_K1963_P41_SKINNY_N608_KCOMPL_ALIASSTACK_18) == 18
assert len(_K1963_P42_SKINNY_N672_KCOMPL_ALIASSTACK_18) == 18
assert len(_K1963_P43_SKINNY_N704_KCOMPL_ALIASSTACK_18) == 18
assert len(_K1963_P44_SKINNY_N736_KCOMPL_ALIASSTACK_18) == 18
