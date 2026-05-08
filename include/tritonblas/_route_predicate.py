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
          N == 1792 and 512 <= K <= 768, with M either large-skinny
          (>= 5000) or in the K-984/K-989 mid-rect band [256, 2048].

    bf16-only: K-984/K-989 ship cohort and K-931 measurement scope are bf16;
    fp16 K-905/K-971 anchors stay in :data:`K971_ROUTE_TABLE`.

    Verification (K-1003):
      - K-984+K-989 union (14 unique shapes): 14/14 FIRE
      - K-950 LAND set (9 cells): 0/9 FIRE (zero leakage)
      - K-931 top-40: 29/40 FIRE (15 net beyond strict-equality)

    K-1031 tightening (Clause-4 K lower bound 0 -> 512):
      K-1017 delivered 4-pass rocprofv2 PMC data on the 18 non-LDS-bound
      residual cells from K-931's top-40 catalogue (held out from K-979's
      training cohort). Scoring P5 against that set surfaced one
      Clause-4 leak: S07 = (M=2048, N=1792, K=256, bf16), gap_x=1.044
      (hipBLASLt is only ~4% faster than the Triton path — well inside
      the ~5% per-engine CV noise floor; routing buys nothing). The two
      K-984/K-989 anchors Clause-4 was authored to cover both have K
      well above 256:
          S06 = (736, 1792, 736)   gap_x=1.301
          S18 = (5972, 1792, 768)  gap_x=1.301
      Both stay covered after raising the K floor to 512 (>= 8 BK=64
      iterations, the minimum K-iteration count for the shallow-K
      pathology to actually penalise the Triton tile path). S07 is now
      correctly silenced. K-1017 18-cell confusion: FP=1/TP=14 -> FP=0/
      TP=14, with the dropped FP contributing only 1.044x speedup vs
      1.000x no-route — a wash on the cohort geomean (1.229x -> 1.225x).
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
    # Clause-4: N=1792 shallow-K dual of clause-1. K-1031 raised the K
    # floor from 0 to 512 after K-1017 PMC data caught a leak at
    # (2048, 1792, 256, bf16) where the Triton path is within ~4% of
    # hipBLASLt — see docstring. K=512 = 8*BK64 is the minimum K-iter
    # count for the shallow-K pathology to actually hurt Triton.
    if N == 1792 and 512 <= K <= 768 and (M >= 5000 or 256 <= M <= 2048):
        return True
    return False


def k971_route_decision(M, N, K, a_dtype, b_dtype, enable_streamk,
                        work_stealing, disable_env_set: bool = False) -> bool:
    """Pure routing decision — same logic as ``matmul._k971_route_to_hbl``
    but without reading the environment (caller passes ``disable_env_set``).

    Useful for tests that want to exercise the *full* dispatch decision
    (predicate + strict-equality + streamk/work-stealing/dtype carve-outs)
    without monkey-patching ``os.environ``.
    """
    if disable_env_set:
        return False
    if enable_streamk or work_stealing or str(a_dtype) != str(b_dtype):
        return False
    if R_K979_P5_route_to_hbl(int(M), int(N), int(K), a_dtype):
        return True
    return (int(M), int(N), int(K), str(a_dtype)) in K971_ROUTE_TABLE
