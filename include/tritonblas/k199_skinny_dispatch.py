"""K-199 — medium-K skinny dispatch overrides for MI300X (gfx942).

Cohort: ``K in [128, 512] AND (M <= 128 OR N <= 128)``.

Per-shape winners came from a focused autotune sweep (K-199) over

    BLOCK_K   in {32, 64, 128, 256}
    num_warps in {4, 8}
    num_stages in {2, 3}
    split_k   in {1, 2, 4, 8, 16}

run on c42 / mi300x in
``rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0``,
torch 2.10.0+rocm7.2.0, triton 3.6.0+rocm7.2.0, gfx942.

Three empirical findings shaped what landed here:

1. **split_k=1 wins 60/60 sweeps.** The K-direction loop is already short
   (``K/BLK_K in {1..16}``) and the streamk overhead (atomic locks,
   partial-tile reductions) is strictly larger than the K-iteration savings
   on this cohort.  Do not propose ``split_k > 1`` for medium-K skinny
   shapes again without first showing the streamk lock cost is amortized.
   This is the BLOCK_K x split_k anti-correlation called out in the K-199
   PRD.

2. **The cohort is launch/dispatch-bound, not compute-bound.** Direct
   kernel-time A/B (mode=on vs mode=off via the env var below) on MI300X
   shows the per-shape sweep "winners" sit within ±5% of the analytical
   Origami selection at the kernel level. Many configs that looked like
   wins against the wrapper-level baseline (~0.27 ms ``tritonblas.matmul``
   call) were actually just measuring the K-180 Python-wrapper tax; once
   that wrapper overhead is excluded, the headroom is at the noise floor.
   The remaining cohort gap to hipBLASLt is the Triton kernel-launch
   floor (~50 us) plus K-180-style dispatch overhead — neither addressable
   from per-shape tile selection alone.

3. **fp16 produced no kernel-level winners.** Every swept fp16 shape
   regressed by 1-5% under the override (Origami's fp16 picks are already
   small-tile and at the launch floor). No fp16 entries are shipped.

Given those findings, the table below contains **only entries with a
strict kernel-level win above the measurement noise floor** in the
verification A/B (``spd_def >= 1.02x``, the per-shape standard deviation
of the median-of-30-reps timing). 2 of 30 swept bf16 shapes met that bar
(geomean +2.4% on the kept entries, vs noise-level +0.5% if the previous
``>=1.00x`` bar were retained). All other swept configs are intentionally
NOT here — see ``output/verify_kernel_bf16.json`` and
``output/verify_kernel_fp16.json`` in the K-199 workspace for the full
per-shape numbers and re-derivation if the production wrapper (K-180) or
launch floor improves enough that more entries clear the noise floor.

**Post-filter cohort A/B** (the SHIPPED 2-entry table A/B'd against the
analytical default on the full 30-shape cohort):
``output/verify_kernel_filtered_bf16.json`` — 2/30 cohort hits, geomean
kernel speedup vs default = **1.006x** (≥ 1.00x — confirms the SHIPPED
table does not regress the cohort, addressing reviewer concerns that
selecting from a single noisy run could overfit).
``output/verify_kernel_filtered_fp16.json`` — 0/30 hits, geomean 1.009x
(noise on identical default kernels on both arms — confirms the dtype
filter rejects all fp16 callers).

Gating layers (per K-654: never apply a knob unconditionally):

* **Exact (M, N, K, dtype) match** — the table only contains entries that
  satisfy the cohort predicate ``_is_in_cohort``; non-cohort callers can
  never hit it, and even within the cohort only the verified subset is
  served.
* **gfx942 only** — the table was tuned on MI300X (gfx942). Other archs
  bypass the lookup.
* **LDS-capacity check at apply time** — see ``origami.py`` where the
  override tile is checked against the Triton LDS budget at the requested
  ``num_stages`` before being applied. If it overflows, we silently fall
  back to the analytical pick rather than crashing.
* **Env-var kill-switch** — set ``TRITONBLAS_DISABLE_K199_OVERRIDES=1``
  to bypass the lookup entirely (rollback / A/B).
* **streamk path untouched** — selector built with ``streamk=True`` skips
  the override entirely, because the swept winners are all
  ``split_k=1`` (persistent path).
"""
from __future__ import annotations

import os
from typing import Optional


# Per-shape overrides for the K-199 cohort. Only entries with a kernel-level
# win above the measurement noise floor (spd_def >= 1.02x on median of 30
# CUDA-event reps, fresh selector + fresh kernel cache) made it in.
#
# Source data: output/verify_kernel_bf16.json in the K-199 workspace
# (n=30 swept shapes, MI300X g08u07 / c42, persistent_matmul_lt path).
# 28 of 30 swept bf16 shapes were dropped because the kernel-level delta
# was within noise (|spd-1| < 0.02); shipping them would have been a wash
# at best and a regression risk on cooler runs (per K-199 reviewer feedback
# and the K-654 lesson against shipping noise-level "wins").
#
# All swept fp16 shapes regressed and are not represented.
_K199_OVERRIDES = {
    # Both shipped entries are bf16 on MI300X.
    (32, 1024, 512, 'bf16'): dict(BLK_M=32, BLK_N=128, BLK_K=128, num_warps=4, num_stages=2, split_k=1),    # verified spd_def=1.022x
    (1024, 32, 128, 'bf16'): dict(BLK_M=128, BLK_N=32, BLK_K=128, num_warps=4, num_stages=2, split_k=1),    # verified spd_def=1.026x
}


def _is_in_cohort(M: int, N: int, K: int) -> bool:
    """Return True iff (M, N, K) lies in the K-199 medium-K skinny cohort.

    The dispatch table only contains entries that satisfy this predicate, so
    the lookup itself is sufficient guarding. This helper is exposed for
    callers that want to short-circuit unrelated work.
    """
    return 128 <= K <= 512 and (M <= 128 or N <= 128)


def lookup_k199_override(
    M: int,
    N: int,
    K: int,
    a_dtype_str: str,
    arch_name: Optional[str] = None,
) -> Optional[dict]:
    """Look up a K-199 dispatch override for this exact (M, N, K, dtype).

    Returns:
        dict with keys ``BLK_M, BLK_N, BLK_K, num_warps, num_stages, split_k``
        if a tuned entry exists for this exact shape on gfx942, else ``None``.

    Notes:
      * The table was tuned on gfx942 (MI300X) only. On other architectures,
        the function returns None to avoid leaking the override across archs.
      * Honors ``TRITONBLAS_DISABLE_K199_OVERRIDES=1`` for fast rollback.
      * The (M, N, K) tuple is matched exactly — no fuzzy matching. Adding new
        shapes requires running the K-199 sweep and re-emitting the table.
    """
    if os.environ.get("TRITONBLAS_DISABLE_K199_OVERRIDES", "0") == "1":
        return None
    if arch_name is not None and arch_name != "gfx942":
        return None
    if not _is_in_cohort(M, N, K):
        return None
    return _K199_OVERRIDES.get((M, N, K, a_dtype_str))


__all__ = ["lookup_k199_override", "_is_in_cohort"]
