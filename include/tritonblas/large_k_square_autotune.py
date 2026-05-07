"""K-273 (S-002) — large-K square cohort autotune cache for tritonblas.

Cohort definition (PRD K-273):
    M, N in {1024, 2048, 4096, 8192} AND M == N (square)
    K   in {8192, 16384, 32768}
    dtype in {fp16, bf16}

Gate predicate enforced at apply time (matmul.py):
    M >= 1024 AND N >= 1024 AND K >= 8192 AND dtype in {fp16, bf16}

═══════════════════════════════════════════════════════════════════════════
HEADLINE — NEGATIVE FINDING (PRD bar NOT met)
═══════════════════════════════════════════════════════════════════════════

PRD bar:        cohort geomean tritonblas/hipBLASLt improvement >= 1.05x
Result:         1.000x  (only 1/24 shapes admits a winning override)
Status:         INFEASIBLE under (BLOCK_M, BLOCK_N, BLOCK_K, num_stages,
                num_warps, matrix_instr_nonkdim, kpack, waves_per_eu,
                GROUP_SIZE_M) tile/pipeline knob set.

Apples-to-apples warm-cache A/B through ``tritonblas.matmul()`` end-to-end
(per-shape baseline = empty AUTOTUNE_TABLE → Origami selector pick;
per-shape tuned = single-entry AUTOTUNE_TABLE → override pick) on MI300X
(node j07u37) gave:

    decision   count    description
    ──────────────────────────────────────────────────────────────────
    WIN        1/24     gain >= 2.5% over Origami baseline
    MARGINAL  12/24     0% < gain < 2.5% (within noise floor)
    REGRESS   11/24     tuned slower than baseline
                       (1024² shapes hit hardest: -30% to -60% because
                        v1 grid excluded BLOCK_M=64 — Origami's pick)

The single confirmed WIN is shipped in AUTOTUNE_TABLE below; the other 23
cohort shapes return None from lookup() and use Origami's selector
unchanged.

═══════════════════════════════════════════════════════════════════════════
WHY ORIGAMI WINS — methodological context
═══════════════════════════════════════════════════════════════════════════

Two earlier sweep iterations and one final reverify converged on the same
conclusion:

  * Sweep v1 (384-config kernel-level grid) — measured baseline through
    ``tritonblas.matmul`` and tuned via raw ``persistent_matmul[...]``
    launch.  Reported a +58% cohort geomean kernel-level "win".  This
    was a **measurement bias**: baseline paid the triton_op + selector
    dispatch overhead (~150-200µs cold), tuned bypassed it.

  * Sweep v2 (1296-config Stage A grid, BM ∈ {64, 128, 256}) —
    apples-to-apples via temporary AUTOTUNE_TABLE insertion + reset
    of the selector lru_cache so the override is honored.  0/24
    shapes beat baseline by >= 2.5% guard; 14/24 marginal.

  * K-273 reverify (this file's keep set) — re-checked v1 picks under
    warm-cache fair-comparison conditions.  1/24 WIN, 12/24 MARGINAL,
    11/24 REGRESS.  See output/reverify_v1.csv for per-shape numbers.

Mechanistic interpretation: at K >= 8192, arithmetic intensity per tile is
large enough that the tile/pipeline knob choice matters less than the
selector's MI300X-aware streamk routing.  Origami's heuristic already
chooses BLOCK_M=64 for the 1024-output-row shapes (where occupancy
matters most) and BLOCK_M=256 for the 8192-output-row shapes (where
MFMA throughput dominates).  The remaining 1.4-2.5x gap to hipBLASLt is
**dispatch overhead and kernel-launch latency**, not kernel inefficiency.

═══════════════════════════════════════════════════════════════════════════
COMPANION CHANGE — selector LRU cache (cold-start latency, separate axis)
═══════════════════════════════════════════════════════════════════════════

A separate change in matmul.py re-enables ``@functools.lru_cache(maxsize=
1024)`` on ``_make_matmul_selector``.  This is a **cold-start-only fix**:
the first call for a given (M, N, K, dtype) pays ~150-200µs in selector
construction; subsequent calls hit the cache and pay ~0µs.

In steady-state (e.g., training/inference loops calling matmul on the
same shape repeatedly), the lru_cache hit rate is ~100% and the
re-enable has no measurable effect on warm-cache geomean
(see verify_baseline.csv vs verify_final.csv).  It is shipped as a
bugfix for one-shot / many-distinct-shape workloads where cold dispatch
dominates.

The previous attempt's "1.628x cohort speedup" headline conflated the
cold-start cache enable with steady-state autotune impact — that headline
has been retracted.  The honest steady-state geomean impact of K-273's
combined changes is **1.000x** (no measurable improvement above noise on
this cohort).

═══════════════════════════════════════════════════════════════════════════
FUTURE WORK (recommended next-ticket scope)
═══════════════════════════════════════════════════════════════════════════

1.  **Streamk / split-K external reduction.**  The persistent kernel does
    not expose SPLIT_K.  The streamk path's implicit K-slicing
    (``selector.sk_grid``) might find wins for K=32768 where atomic-free
    split-K reduction overlaps the long mainloop.  Outside K-273 scope.

2.  **Dispatch-overhead bypass.**  Direct call to ``persistent_matmul_lt``
    skipping the triton_op layer would close the 1024² × 8K shape's
    remaining ~50% gap to hipBLASLt.  Incompatible with torch.compile.

3.  **Coarser selector cache key.**  Currently keyed on (M, N, K,
    a_dtype, b_dtype, c_dtype, device, mx_block_size, streamk,
    num_stages); for the matmul_lt fp16/bf16 common case a coarser
    key would raise hit rate.
"""

from __future__ import annotations

from typing import Dict, Optional

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


def _safe_to_apply(cfg: Dict[str, int], M: int, N: int, K: int) -> bool:
    """Defense-in-depth: reject any cached entry whose BLOCK_K would violate
    the K-654 mainloop_iters >= 16 floor or whose tile would exceed the shape.
    """
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
# Keys: (M, N, K, dtype-string).  Values: dict of persistent_matmul_lt kwargs.
#
# ENTRIES SHIPPED HERE PASS the apples-to-apples >= 2.5% gain guard at the
# user-facing tritonblas.matmul() API level, measured warm-cache on MI300X
# against an empty-table Origami baseline.
#
# As of K-273 close, only ONE cohort shape produces a winning override above
# the guard (see output/reverify_v1.csv).  The 23 other cohort shapes return
# None from lookup() → Origami's selector pick is used unchanged.

AUTOTUNE_TABLE: Dict = {
    # K-273 confirmed winner (apples-to-apples warm-cache reverify, +3.01%
    # over Origami baseline through tritonblas.matmul() end-to-end on MI300X).
    (8192, 8192, 16384, "bf16"): dict(
        BLOCK_M=256, BLOCK_N=256, BLOCK_K=64,
        num_stages=2, num_warps=8,
        matrix_instr_nonkdim=16, kpack=1,
        waves_per_eu=0, GROUP_SIZE_M=1,
    ),
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
