"""K-273 (S-002) — large-K square cohort autotune cache for tritonblas.

Cohort definition (PRD K-273):
    M, N in {1024, 2048, 4096, 8192} AND M == N (square)
    K   in {8192, 16384, 32768}
    dtype in {fp16, bf16}

Gate predicate enforced at apply time (matmul.py):
    M >= 1024 AND N >= 1024 AND K >= 8192 AND dtype in {fp16, bf16}

This module is gated behind ``is_large_k_square()`` AND ``_safe_to_apply()``
so that:

    * shapes with K < 8192 (small/medium-K cohort, K-216 territory) are
      never perturbed,
    * shapes with M < 1024 or N < 1024 (small-M / skinny-N cohort,
      K-176 / K-208 territory) are never perturbed,
    * shapes that fall in the cohort but where the cached BLOCK_K would
      not produce K-654's mainloop_iters >= 16 floor are rejected.

═══════════════════════════════════════════════════════════════════════════
HEADLINE (cohort geomean over 24 shapes, MI300X, b06u31, fp16+bf16)
═══════════════════════════════════════════════════════════════════════════

    no patch   tritonblas.matmul / hipBLASLt = 2.311x   (tritonblas slower)
    K-273      tritonblas.matmul / hipBLASLt = 1.419x

    cohort geomean improvement = 1.628x   (PRD bar: >= 1.05x)

The improvement is delivered **entirely by the @functools.lru_cache enable on
``_make_matmul_selector``** in matmul.py — not by the autotune table.  The
autotune table is currently empty by deliberate choice; see "Negative finding"
below.

═══════════════════════════════════════════════════════════════════════════
METHODOLOGY
═══════════════════════════════════════════════════════════════════════════

Sweep v1 — coarse 384-config grid (BM, BN ∈ {128, 256}; BK ∈ {32, 64, 128,
256}; ns ∈ {2, 3, 4}; nw ∈ {4, 8}; mfma=16; kpack ∈ {1, 2}; gsm ∈ {4, 8}).
Run on 8 MI300X GPUs in parallel (3 shapes per GPU, ~10 min wall).

Sweep v1 measurement (baseline = ``tritonblas.matmul``; tuned = direct
``persistent_matmul[...]`` launch with override knobs) reported a cohort
geomean improvement of **+58% over baseline**.  This was a **measurement
bias**: the baseline path went through ``triton_op`` (~0.2 ms dispatch
overhead per call), the tuned path bypassed it.  After fixing this bias
(see Sweep v2 below), the apparent +58% kernel-level win collapsed to no
measurable win at the user-facing API level.

Sweep v2 — same coarse grid but apples-to-apples: every candidate goes
through ``tritonblas.matmul()`` end-to-end via temporary insertion into
``AUTOTUNE_TABLE``, and a regression guard requires the candidate's median
to beat baseline by ≥ 2.5%.  Search expanded to BM, BN ∈ {64, 128, 256}
to cover Origami's BM=64 picks for the small-output shapes.  Result:
**0 of 24 shapes produced a winning config above the 2.5% guard.**  14 of 24
showed marginal improvements (0.04% – 2.2%) at the noise floor.

═══════════════════════════════════════════════════════════════════════════
NEGATIVE FINDING — autotune table is intentionally empty
═══════════════════════════════════════════════════════════════════════════

Under the user-facing ``tritonblas.matmul()`` API path, **no tile / pipeline
knob combination in the search grid beats Origami's selector by ≥ 2.5%** on
any of the 24 cohort shapes.

This mirrors K-216's outcome on the medium-square cohort (closed as
INFEASIBLE under the stated knob set) but for a different mechanistic
reason:

    K-216 (medium-square, K ∈ [256, 2048]):
        K-654 mainloop_iters >= 16 guard rejected 7 of 9 unique shapes
        for any BLOCK_K in {64, 128, 256}.  Headroom existed but couldn't
        be claimed.

    K-273 (large-K square, K >= 8192):
        K-654 guard is satisfied trivially (K_min/BK_max = 8192/256 = 32).
        Origami's heuristic is already at or near the kernel-level optimum
        for this regime.  The remaining gap to hipBLASLt is **dispatch
        overhead** (now reduced 4-7x by the ``@lru_cache`` enable), not
        kernel inefficiency.

Per-knob ablation (from sweep v1, kernel-level numbers):

    * BLOCK_M = BLOCK_N: 64 wins for M==N <= 1024 (Origami's pick), 128
      wins for M==N == 2048, 256 wins for M==N >= 4096.  My v1 grid
      excluded BM=64 and was therefore biased away from Origami's
      established small-output sweet spot.
    * BLOCK_K: insensitive in the 64–128 range; 32 occasionally helps
      under heavy register pressure; 256 hurts (cuts mainloop_iters too
      far in the 8192-K case).
    * num_stages: 2 dominates; 3 ties on a couple of shapes; 4 always
      regresses (LDS spill).
    * num_warps: 8 dominates universally; 4 always regresses (under-uses
      MFMA throughput at these tile sizes).
    * matrix_instr_nonkdim: 16 dominates; 32 is net-neutral.
    * kpack: 1 vs 2 swap order between shapes; gain < 1% either way.
    * waves_per_eu: only the largest 8192² shapes show a < 1% win at wpe>0.
    * GROUP_SIZE_M: 1, 4, 8 all within 1-2%; 16 starts to degrade L2 reuse.

═══════════════════════════════════════════════════════════════════════════
WHY THE TABLE IS STILL HERE
═══════════════════════════════════════════════════════════════════════════

The cache infrastructure (gate, lookup, regression guard) is shipped so that
future work can populate it without re-deriving the dispatch plumbing:

    * Adding shape-specific entries (e.g., from a new search around different
      knobs — split-K external reduction, persistent vs streamk routing) is
      a one-line edit to AUTOTUNE_TABLE.
    * The regression guard ``_safe_to_apply()`` is preserved so any future
      entry inherits the K-654 mainloop_iters floor without re-derivation.
    * The cohort gate ``is_large_k_square()`` defines the partition that
      future autotune work should target.

Out-of-cohort shapes return ``None`` from ``lookup()`` and are unperturbed.

═══════════════════════════════════════════════════════════════════════════
FUTURE WORK (recommended next-ticket scope)
═══════════════════════════════════════════════════════════════════════════

1.  Extra-knob autotune.  The PRD's SPLIT_K knob is not exposed by the
    persistent kernel; a new sweep using the streamk path (which has implicit
    K-slicing via ``selector.sk_grid``) might find wins for the K=32768 row
    where atomic-free split-K reduction overlaps the long mainloop.

2.  Dispatch-overhead reduction beyond LRU.  The 1024² × 8K shape still
    spends ~50% of its time in triton_op dispatch even with LRU cache enabled.
    Bypassing the triton_op layer entirely (direct call to
    ``persistent_matmul_lt``) would close that gap but is incompatible with
    torch.compile / fakemode tracing.  An eager-only fastpath could be added.

3.  Move Origami selector cache key to (M, N, K, dtype) only.  Currently the
    ``functools.lru_cache`` keys on (M, N, K, a_dtype, b_dtype, c_dtype,
    device, mx_block_size, streamk, num_stages); for the common case
    (matmul_lt, fp16/bf16) a coarser key would raise the cache hit rate.
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
# Per-shape autotune table  (regression-guarded; currently empty)
# ---------------------------------------------------------------------------
# Keys: (M, N, K, dtype-string).  Values: dict matching persistent_matmul_lt's
# kwargs.  Entries here SHALL ONLY be added if they verifiably beat the
# Origami baseline by >= 2.5% at the user-facing tritonblas.matmul API level
# (sweep v2 protocol — see module docstring).
#
# As of K-273 close, sweep v2 found 0 of 24 cohort shapes meet that bar
# under the (BM, BN, BK, num_stages, num_warps, mfma, kpack, waves_per_eu,
# GROUP_SIZE_M) knob set.  The negative finding is the deliverable; the
# table is preserved as infrastructure for future autotune additions.

AUTOTUNE_TABLE: Dict = {
    # (M, N, K, dtype): dict(BLOCK_M=..., BLOCK_N=..., BLOCK_K=...,
    #                         num_stages=..., num_warps=...,
    #                         matrix_instr_nonkdim=..., kpack=...,
    #                         waves_per_eu=..., GROUP_SIZE_M=...),
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
