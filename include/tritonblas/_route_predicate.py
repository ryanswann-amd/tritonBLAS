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
    on the remaining 15 occupancy + 1 HBM-L2-stall cells).

    Surrogate-C2 (MFMA_pct_tb >= 1.0%): K >= 512.
    Surrogate-C3 (LDS_BC_cpi_tb < 1.5): not in P5 LDS-bound regime
                                       (caller composes with P5 first).
    Surrogate-C1 (waves_per_CU_ratio < 1.70x — load-bearing):
        Envelope A (S29 wide-rect family):
            N == 2048 AND K == 1024 AND 13000 <= M <= 14999.
        Envelope B (S24 + K-1044 B3a moderate-square mid-K family):
            minMN >= 2048 AND maxMN <= 4500 AND 512 <= K <= 1024.

    bf16-only by design (K-1037 / K-1017 / K-1044 ground truth scope).

    Calibration verification on K-1017 18-cell set:
        S24 -> True (P6 positive, ratio=1.532)        ENVELOPE B
        S29 -> True (P6 positive, ratio=1.514)        ENVELOPE A
        all 16 negatives -> False                     neither envelope
    """
    if not _dtype_is_bf16(dtype):
        return False
    if K < 512:
        return False  # Surrogate-C2: insufficient MFMA work density
    minMN, maxMN = min(M, N), max(M, N)
    # Envelope A: K-931 wide-rect (N=2048, K=1024) family at moderate M.
    if N == 2048 and K == 1024 and 13000 <= M <= 14999:
        return True
    # Envelope B: moderate-square mid-K family (S24 fingerprint).
    if 2048 <= minMN and maxMN <= 4500 and 512 <= K <= 1024:
        return True
    return False


def R_K1106_P7_route_to_hbl(M: int, N: int, K: int, dtype) -> bool:
    """K-1106 — P7 sibling: OR-gate of R-K979 P5 (K-1066/K-1062 K-floor
    hardened) with K-1037 P6 MFMA-issue-stall structural classifier.

    Both inputs are bf16-only; the OR-gate inherits that scope. P7 fires
    when EITHER P5 says route-to-hbl OR P6's structural envelope admits
    the shape into the MFMA-issue-stall fingerprint (K-1089 surrogate).

    The composite preserves K-1031 leakage immunity (P5 Clause-4 K-floor
    blocks S07; P6 K>=512 floor also blocks S07 -> both False -> P7 False)
    and adds S24 (P5=False, P6 Envelope-B=True -> P7=True) as the only
    NEW route-OUT cell vs P5 alone on the K-1074+K-1079+K-1085+K-1080
    bench union.
    """
    return (R_K979_P5_route_to_hbl(M, N, K, dtype)
            or R_K1037_P6_admit_wpeu1(M, N, K, dtype))


def should_admit_p6_streamk(M, N, K, a_dtype, b_dtype, enable_streamk,
                            work_stealing, disable_env_set: bool = False) -> bool:
    """K-1089 (S-002) — should the dispatcher flip ``enable_streamk=True`` for
    this cell because R-K1037 P6 admits it?

    Pure decision, callable from tests without ``os.environ`` monkey-patching.
    """
    if disable_env_set:
        return False
    if enable_streamk or work_stealing or str(a_dtype) != str(b_dtype):
        return False
    return R_K1037_P6_admit_wpeu1(int(M), int(N), int(K), a_dtype)


def k971_route_decision(M, N, K, a_dtype, b_dtype, enable_streamk,
                        work_stealing, disable_env_set: bool = False) -> bool:
    """Pure routing decision — same logic as ``matmul._k971_route_to_hbl``
    but without reading the environment (caller passes ``disable_env_set``).

    K-1106 P7 OR-gate replaces the prior K-1089 P6-admit-overrides-P5
    composition: route-OUT now fires on P5 OR P6, with strict-tuple
    K971_ROUTE_TABLE retained as a final fall-through for the fp16
    K-905/K-971 anchors.
    """
    if disable_env_set:
        return False
    if enable_streamk or work_stealing or str(a_dtype) != str(b_dtype):
        return False
    if R_K1106_P7_route_to_hbl(int(M), int(N), int(K), a_dtype):
        return True
    return (int(M), int(N), int(K), str(a_dtype)) in K971_ROUTE_TABLE
