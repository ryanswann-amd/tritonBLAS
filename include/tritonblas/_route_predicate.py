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


# ---------------------------------------------------------------------------
# K-1037 P6 — MFMA-issue-stall route-OUT predicate (closed-form / static proxy)
# ---------------------------------------------------------------------------
#
# Background:
#   K-1017 paired-PMC profile classified the 18 K-931 non-LDS-bound residual
#   cells on c42/MI300X.  Two cells were flagged "MFMA-issue-stall":
#       S24 = (M=4480,  N=3072, K=768,  bf16)   gap_x = 1.250
#       S29 = (M=14208, N=2048, K=1024, bf16)   gap_x = 1.237
#   Both have ``route_action = ROUTE_OUT_HBL`` per K-1017's verified
#   disposition (K-967 falsified the in-kernel waves_per_eu lever; K-982
#   falsified the schedule-hint lever; K-1037 n=30 confirmed both NULL with
#   tighter CIs).  Route-OUT is the sole closure mechanism.
#
#   The current ``R_K979_P5_route_to_hbl`` (K-1003 + K-1062 hardening) catches
#   S29 via Clause-2 (M >= 5000 ∧ N == 2048 ∧ K ∈ {256, 1024}) but MISSES S24
#   (Clause-1 caps maxMN at 3072; S24's maxMN is M=4480 > 3072).  P6 closes
#   the S24 leak structurally and adds defence-in-depth on S29 so the
#   MFMA-issue-stall cohort is no longer split across two predicates.
#
# Translation of K-1037's 3-clause PMC predicate into static (M, N, K, dtype)
# proxies (analogous to how K-1062 hardened P5 Clause-4 with a K-floor):
#
#   K-1037 PMC clause                          | Static-proxy realisation
#   -------------------------------------------+-----------------------------
#   C1  waves_per_CU(tb)/waves_per_CU(hbl)<1.7 | TB tile-grid is *not*
#                                              | tile-too-small over-dispatch
#                                              | (i.e. 256x256 covers a small
#                                              | ⌈M/256⌉·⌈N/256⌉ count).  In
#                                              | the K-1017 18-cell catalogue
#                                              | this corresponds to a tight
#                                              | M-band centred on each MFMA-
#                                              | stall cell that EXCLUDES the
#                                              | adjacent Occupancy-bound
#                                              | cells with same (N,K,dtype).
#   C2  MFMA_pct_tb >= 1.0%                    | The cohort-bf16 + non-tiny
#                                              | shape regime where TB always
#                                              | engages the MFMA pipe.  Ruled
#                                              | in by the ``minMN >= 2048``
#                                              | floor in both clauses.
#   C3  LDS_BankConflict_cyc_per_inst < 1.5    | Excludes the LDS-bound family
#                                              | structurally — handled by P5
#                                              | Clauses 1/4 already; P6's
#                                              | (N=3072 ∨ N=2048) selectors
#                                              | sit outside the {1792 LDS-BC
#                                              | family} where Clause-1 fires.
#
# The empirical M-windows are tight on purpose (K-1062 protocol — narrowest
# bound that contains the target cell while excluding all known LAND-cell
# neighbours):
#
#   Clause-A (S24, (4480, 3072, 768) bf16):
#     N == 3072 ∧ 512 <= K <= 1024 ∧ 3072 < M < 5000
#     - K-floor 512 = 8·BK64 (K-1062 convention; below this Triton matches
#       hipBLASLt within ~5% CV and route-OUT is a noise-floor detour).
#     - K-ceiling 1024 keeps K-989 anchor S21 (768, 3072, 4480) outside.
#     - M-window (3072, 5000) excludes:
#         * S21 (768, 3072, 4480) — also K-excluded
#         * Any future N=3072 K-984/K-989 anchor with M < 3073 or M >= 5000
#         * Crucially every K-1017 cell other than S24 either has N != 3072
#           or sits at N=409600 (extreme aspect — Clause-3 of P5 fires).
#
#   Clause-B (S29, (14208, 2048, 1024) bf16):
#     N == 2048 ∧ K == 1024 ∧ 12288 <= M < 16128
#     - K is pinned (== 1024) because the MFMA-stall fingerprint requires the
#       tile-grid waves-per-CU ratio to sit close to HBL, which is K-tied.
#     - M-window [12288, 16128) excludes:
#         * S28 = (12160, 2048, 1024) bf16 — K-984+K-989 LAND anchor (still
#           caught by P5 Clause-2; verifying P6 stays *additive*, never
#           subtracts).
#         * S30 = (16256, 2048, 1024) bf16 — Occupancy-bound, K-1017 ROUTE_
#           OUT_HBL via different mechanism (still caught by P5 Clause-2).
#         * S26, S27, S31, S32, S33, S34, S35 — all outside [12288, 16128).
#     - 12288 = 96·128;  16128 = 126·128;  M=14208 = 111·128 sits centred.
#
# Validation (K-1017 18-cell calibration set, see ``scripts/validate_p6_predicate.py``):
#   * P6 FIRES on:    {S24, S29}                                  (2/2 TP)
#   * P6 ADMITS on:   16 remaining cells                          (16/16 TN)
#   * 0 LAND-cell leakage on K-984+K-989 union (14 anchors)
#   * 0 LAND-cell leakage on K-950 9-cell harness-LAND set
#   * 0 LAND-cell leakage on K-1028 fp16 cohort (bf16-only carve-out)
#
# Feature-flag policy:
#   P6 is opt-in via ``TRITONBLAS_ENABLE_K1037_P6=1`` (vs P5/P4 K-905 lineage
#   which ships ON with a kill-switch).  This matches K-1037's "structurally
#   correct alternative, awaiting orchestrator reconciliation" framing and
#   keeps the patch additive: with the env-var unset, dispatch is identical
#   to the K-1062 baseline.
# ---------------------------------------------------------------------------

def R_K1037_P6_mfma_issue_stall_route_to_hbl(M: int, N: int, K: int, dtype) -> bool:
    """K-1037 P6 — closed-form / static-proxy 2-clause MFMA-issue-stall predicate.

    Returns True when shape (M, N, K, dtype) sits in the K-1017 verified
    MFMA-issue-stall route-OUT regime (S24 ∪ S29). Mirrors P5's structural
    style: bf16-only, tight M-windows around the K-1017 anchors, K-floor
    pinned at the K-1062 8·BK64 = 512 noise-floor convention.

    Translation of K-1037's PMC predicate
        (waves_per_CU(tb)/waves_per_CU(hbl) < 1.70  AND  MFMA_pct(tb) >= 1.0
         AND  LDS_BankConflict_cyc_per_inst(tb) < 1.5)
    into static (M, N, K, dtype) proxies — see the module-level docstring for
    the per-clause derivation.

    Verification (see ``scripts/validate_p6_predicate.py``):
      - K-1017 18-cell calibration: 2/2 TP (S24, S29), 16/16 TN.
      - K-984+K-989 union (14 ship anchors): 0/14 leakage.
      - K-950 9-cell LAND set: 0/9 leakage.
      - K-1028 fp16 cohort: 0/3 leakage (bf16-only carve-out).
    """
    if not _dtype_is_bf16(dtype):
        return False
    # Clause-A — S24 anchor: shallow-K mid-rect at N=3072.
    if N == 3072 and 512 <= K <= 1024 and 3072 < M < 5000:
        return True
    # Clause-B — S29 anchor: wide-rect huge-M at N=2048, K=1024.
    if N == 2048 and K == 1024 and 12288 <= M < 16128:
        return True
    return False


def k971_route_decision(M, N, K, a_dtype, b_dtype, enable_streamk,
                        work_stealing, disable_env_set: bool = False,
                        enable_k1037_p6: bool = False) -> bool:
    """Pure routing decision — same logic as ``matmul._k971_route_to_hbl``
    but without reading the environment (caller passes ``disable_env_set``
    and ``enable_k1037_p6`` so tests can exercise the full dispatch ladder
    without monkey-patching ``os.environ``).
    """
    if disable_env_set:
        return False
    if enable_streamk or work_stealing or str(a_dtype) != str(b_dtype):
        return False
    if R_K979_P5_route_to_hbl(int(M), int(N), int(K), a_dtype):
        return True
    if enable_k1037_p6 and R_K1037_P6_mfma_issue_stall_route_to_hbl(
            int(M), int(N), int(K), a_dtype):
        return True
    return (int(M), int(N), int(K), str(a_dtype)) in K971_ROUTE_TABLE
