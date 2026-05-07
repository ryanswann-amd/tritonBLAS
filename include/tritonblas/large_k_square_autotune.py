"""K-273 (S-002) — large-K square cohort autotune cache for tritonblas.

Cohort definition (PRD K-273):
    M, N in {1024, 2048, 4096, 8192} AND M == N (square)
    K   in {8192, 16384, 32768}
    dtype in {fp16, bf16}

Gate predicate enforced at apply time (matmul.py):
    M >= 1024 AND N >= 1024 AND K >= 8192 AND dtype in {fp16, bf16}

═══════════════════════════════════════════════════════════════════════════
HEADLINE — cohort geomean speedup 1.0379x over Origami baseline
═══════════════════════════════════════════════════════════════════════════

PRD bar:        cohort geomean tritonblas/hipBLASLt improvement >= 1.05x
Result:         **1.0379x** (12/24 cohort shapes win via streamk routing)
Status:         BAR NOT MET (1.2% short).  Kernel-level Origami persistent
                path cannot reach the bar; **streamk routing**
                (selector.sk_grid K-slicing) closes most of the gap.
                Remaining 1.2% deficit vs the 1.05x bar is in the
                1024² and 2048² shapes where streamk hurts (too few
                output tiles to amortize the streamk reduction overhead).

Single-session warm-cache A/B on MI300X (node j07u37, container
rocm/pytorch (current ROCm release container),
30 timed iters median per measurement after 10 warmups, lru_cache
reset between baseline and tuned runs).  See output/verify_streamk_ab_final.csv:

    cohort_geomean_baseline (empty AUTOTUNE_TABLE) = 1.4241
    cohort_geomean_tuned    (12-entry table)       = 1.3720
    cohort_speedup_ratio    (base / tuned)         = 1.0379x
    n_win=12 / n_marginal=7 / n_regress=5 / n_corr_fail=0

(The 5 "regress" entries are 1024²/2048² shapes that have NO autotune
override applied — their delta is run-to-run noise on the unmodified
persistent path.  In_table=False for all of them; see CSV.)

Per-shape decision (cross-process probe, warm-cache MI300X, see
output/streamk_quick_all.csv):

    shape           dtype  persist/hip  streamk/hip  decision       Δ%
    ─────────────────────────────────────────────────────────────────────
    1024×1024×8192  fp16    1.95         2.38       persistent  -21.91
    1024×1024×8192  bf16    1.91         2.77       persistent  -44.62
    1024×1024×16384 fp16    1.89         5.29       persistent -180.38
    1024×1024×16384 bf16    1.83         5.20       persistent -184.71
    1024×1024×32768 fp16    2.12         6.61       persistent -211.71
    1024×1024×32768 bf16    2.42         7.23       persistent -198.88
    2048×2048×8192  fp16    1.52         1.54       persistent   -0.68
    2048×2048×8192  bf16    1.41         1.39       persistent   +0.86 (under guard)
    2048×2048×16384 fp16    1.52         2.42       persistent  -59.41
    2048×2048×16384 bf16    1.57         2.52       persistent  -61.11
    2048×2048×32768 fp16    1.37         2.18       persistent  -58.70
    2048×2048×32768 bf16    1.43         2.33       persistent  -62.57
    4096×4096×8192  fp16    1.10         1.05       STREAMK      +4.37 ← WIN
    4096×4096×8192  bf16    1.09         1.05       STREAMK      +4.02 ← WIN
    4096×4096×16384 fp16    1.18         1.15       STREAMK      +3.12 ← WIN
    4096×4096×16384 bf16    1.22         1.19       STREAMK      +2.66 ← WIN
    4096×4096×32768 fp16    1.15         1.09       STREAMK      +5.03 ← WIN
    4096×4096×32768 bf16    1.18         1.13       STREAMK      +4.18 ← WIN
    8192×8192×8192  fp16    1.13         1.08       STREAMK      +5.11 ← WIN
    8192×8192×8192  bf16    1.17         1.08       STREAMK      +7.33 ← WIN
    8192×8192×16384 fp16    1.24         1.12       STREAMK      +9.62 ← WIN
    8192×8192×16384 bf16    1.21         1.13       STREAMK      +6.74 ← WIN
    8192×8192×32768 fp16    1.29         1.13       STREAMK     +12.40 ← WIN
    8192×8192×32768 bf16    1.32         1.15       STREAMK     +13.08 ← WIN

cohort geomean tb/hip (cross-process probe — slightly more pessimistic
than the single-session A/B headline above due to per-process noise)
    baseline (Origami persistent)         1.4291
    tuned    (streamk on 12 shapes)       1.3816
    cohort speedup ratio (base / tuned)   1.0344x

Same configuration verified end-to-end by single-session A/B at 1.0379x
(see verify_streamk_ab_final.csv).

═══════════════════════════════════════════════════════════════════════════
WHY STREAMK WINS — mechanistic interpretation
═══════════════════════════════════════════════════════════════════════════

The persistent path schedules one program per output tile.  At 4096²
(16 tiles for BM=BN=256) and 8192² (64 tiles), tile count is comparable
to the MI300X CU count, leaving CUs idle on residual tiles.

The streamk path (kernels/streamk_gemm.py) decomposes the K dimension
into stream-K segments and uses atomic-free LDS-buffered cross-segment
reduction (selector.sk_grid programs).  This:

  * Keeps all CUs busy throughout the long mainloop.
  * Hides the wave-mismatch tail latency that the persistent path pays.
  * Trades a small fixed reduction cost for steady-state SM occupancy.

For the 1024² and 2048² shapes the trade reverses: tile count is small
enough that the streamk reduction overhead exceeds the occupancy gain.

The PRD called out "SPLIT_K in {1, 2, 4, 8}" — the persistent kernel
has no SPLIT_K parameter; the streamk path is the architectural
equivalent (implicit K-slicing via sk_grid).  We adopt the streamk
*default* sk_grid in this commit; a separate sk_grid sweep
(output/skgrid_all.csv) confirmed the selector default beats
overrides for 10/12 winning shapes (the other 2 gain <0.6% from
sk_grid override — under the 2.5% guard, dropped).

═══════════════════════════════════════════════════════════════════════════
COMPANION CHANGE — selector LRU cache (cold-start latency, separate axis)
═══════════════════════════════════════════════════════════════════════════

A separate change in matmul.py re-enables ``@functools.lru_cache(maxsize=
1024)`` on ``_make_matmul_selector``.  This is a **cold-start-only fix**:
the first call for a given (M, N, K, dtype) pays ~150-200µs in selector
construction; subsequent calls hit the cache and pay ~0µs.

In steady-state (e.g., training/inference loops calling matmul on the
same shape repeatedly), the lru_cache hit rate is ~100% and the
re-enable has no measurable effect on warm-cache geomean.  It is shipped
as a bugfix for one-shot / many-distinct-shape workloads where cold
dispatch dominates.

═══════════════════════════════════════════════════════════════════════════
FUTURE WORK (recommended next-ticket scope)
═══════════════════════════════════════════════════════════════════════════

1.  **Closing the 1.05x bar.**  The remaining 1.6% deficit is in the
    1024² and 2048² shapes.  These need a non-streamk path improvement
    (e.g., `triton_op` dispatch bypass, or a microkernel for
    BM=BN=64 small-tile cases).
2.  **Tile/block tuning on the streamk path.**  This commit gates the
    *path* but uses Origami's selector blocks unchanged.  Joint
    (block_m, block_n, block_k, sk_grid) tuning on the 12 winning
    shapes may add another ~0.5-1.5% per shape.
3.  **Selector cache key reduction.**  Currently keyed on (M, N, K,
    a_dtype, b_dtype, c_dtype, device, mx_block_size, streamk,
    num_stages); for the matmul_lt fp16/bf16 common case a coarser
    key would raise hit rate.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Gate predicate (cohort membership)
# ---------------------------------------------------------------------------

# PRD: large-K square cohort applies when (M, N) >= 1024 and K >= 8192.
GATE_M_MIN = 1024
GATE_N_MIN = 1024
GATE_K_MIN = 8192

# K-654 mainloop_iters floor — preserved for safety even though every
# in-cohort shape already satisfies it (K_min / BLOCK_K_max = 8192/256 = 32).
MAINLOOP_ITERS_FLOOR = 16


def is_large_k_square(M: int, N: int, K: int, dtype: str) -> bool:
    """Cohort gate: M, N >= 1024, K >= 8192, dtype in {fp16, bf16}."""
    return (
        M >= GATE_M_MIN
        and N >= GATE_N_MIN
        and K >= GATE_K_MIN
        and dtype.lower() in ("fp16", "float16", "half", "bf16", "bfloat16")
    )


def _safe_to_apply(cfg: Dict[str, Any], M: int, N: int, K: int) -> bool:
    """Defense-in-depth: reject any cached entry whose BLOCK_K would violate
    the K-654 mainloop_iters >= 16 floor or whose tile would exceed the shape.
    Streamk-only entries (no kernel kwargs) are always safe.
    """
    if "BLOCK_K" not in cfg:
        # streamk-only entry — uses selector's blocks, which are pre-validated
        return True
    bm = cfg["BLOCK_M"]
    bn = cfg["BLOCK_N"]
    bk = cfg["BLOCK_K"]
    if K // bk < MAINLOOP_ITERS_FLOOR:
        return False
    if bm > M or bn > N:
        return False
    return True


# ---------------------------------------------------------------------------
# Per-shape autotune table
# ---------------------------------------------------------------------------
# Keys: (M, N, K, dtype-string).  Values: dict with optional fields
#   enable_streamk : bool  - if True, dispatch through streamk_matmul_lt
#   sk_grid        : int   - override sk_grid (None = use selector default)
#   BLOCK_M/N/K, num_stages, num_warps, matrix_instr_nonkdim, kpack,
#       waves_per_eu, GROUP_SIZE_M  - persistent-path kernel kwargs override
#
# All entries here pass an apples-to-apples >= 2.5% gain guard at the
# user-facing tritonblas.matmul() API level, measured warm-cache on MI300X
# against the unmodified Origami persistent baseline (see
# output/streamk_quick_all.csv and output/skgrid_all.csv).

AUTOTUNE_TABLE: Dict = {
    # ─── streamk path winners (12 shapes, +2.66% to +13.08% gain) ───────
    # 4096² shapes — moderate streamk gain (3-5%)
    (4096, 4096, 8192,  "fp16"): dict(enable_streamk=True),
    (4096, 4096, 8192,  "bf16"): dict(enable_streamk=True),
    (4096, 4096, 16384, "fp16"): dict(enable_streamk=True),
    (4096, 4096, 16384, "bf16"): dict(enable_streamk=True),
    (4096, 4096, 32768, "fp16"): dict(enable_streamk=True),
    (4096, 4096, 32768, "bf16"): dict(enable_streamk=True),
    # 8192² shapes — large streamk gain (5-13%)
    (8192, 8192, 8192,  "fp16"): dict(enable_streamk=True),
    (8192, 8192, 8192,  "bf16"): dict(enable_streamk=True),
    (8192, 8192, 16384, "fp16"): dict(enable_streamk=True),
    (8192, 8192, 16384, "bf16"): dict(enable_streamk=True),
    (8192, 8192, 32768, "fp16"): dict(enable_streamk=True),
    (8192, 8192, 32768, "bf16"): dict(enable_streamk=True),
}


# ---------------------------------------------------------------------------
# Public lookup
# ---------------------------------------------------------------------------

def _normalize_dtype(dtype) -> str:
    """Coerce torch dtype / string into the canonical 'fp16' / 'bf16' tag."""
    s = str(dtype).lower()
    if "bf16" in s or "bfloat16" in s:
        return "bf16"
    if "fp16" in s or "float16" in s or s.endswith("half"):
        return "fp16"
    return s


def lookup(M: int, N: int, K: int, dtype) -> Optional[Dict]:
    """Return cached config for (M, N, K, dtype), or None if no override should
    be applied.  Out-of-cohort and out-of-table shapes return None.

    Returned dict may contain any subset of:
      enable_streamk : bool   - dispatch through streamk_matmul_lt
      sk_grid        : int    - override sk_grid for streamk path
      BLOCK_M/N/K, num_stages, num_warps, matrix_instr_nonkdim, kpack,
          waves_per_eu, GROUP_SIZE_M  - persistent-path kernel kwargs override
    """
    d = _normalize_dtype(dtype)
    if not is_large_k_square(M, N, K, d):
        return None
    cfg = AUTOTUNE_TABLE.get((M, N, K, d))
    if cfg is None:
        return None
    if not _safe_to_apply(cfg, M, N, K):
        return None
    return cfg


def should_use_streamk(M: int, N: int, K: int, dtype) -> bool:
    """Convenience predicate for the matmul() entry point: True if the cached
    entry asks for streamk dispatch.  False otherwise (including out-of-cohort
    and out-of-table shapes — those use whatever the caller passed)."""
    cfg = lookup(M, N, K, dtype)
    if cfg is None:
        return False
    return bool(cfg.get("enable_streamk", False))


def streamk_sk_grid_override(M: int, N: int, K: int, dtype) -> Optional[int]:
    """Convenience: return sk_grid override if the cached entry specifies one,
    else None (caller passes sk_grid=None → selector picks default)."""
    cfg = lookup(M, N, K, dtype)
    if cfg is None:
        return None
    return cfg.get("sk_grid", None)
