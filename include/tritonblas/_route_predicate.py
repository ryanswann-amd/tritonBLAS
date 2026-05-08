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
    """
    if not _dtype_is_bf16(dtype):
        return False
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


def k971_route_decision(M, N, K, a_dtype, b_dtype, enable_streamk,
                        work_stealing, disable_env_set: bool = False) -> bool:
    """Pure routing decision — same logic as ``matmul._k971_route_to_hbl``
    but without reading the environment (caller passes ``disable_env_set``).

    Useful for tests that want to exercise the *full* dispatch decision
    (predicate + strict-equality + streamk/work-stealing/dtype carve-outs)
    without monkey-patching ``os.environ``.

    K-1089 P6 admit gate: structural surrogate of K-1037 P6 takes priority
    over P5 route-OUT for cells in the MFMA-issue-stall fingerprint. When
    P6 admit fires, the caller dispatches in-kernel with waves_per_eu=1.
    """
    if disable_env_set:
        return False
    if enable_streamk or work_stealing or str(a_dtype) != str(b_dtype):
        return False
    # K-1089: P6 admit overrides P5 route-OUT for MFMA-stall structural
    # signature; cells fall through to in-kernel dispatch.
    if R_K1037_P6_admit_wpeu1(int(M), int(N), int(K), a_dtype):
        return False
    if R_K979_P5_route_to_hbl(int(M), int(N), int(K), a_dtype):
        return True
    return (int(M), int(N), int(K), str(a_dtype)) in K971_ROUTE_TABLE
