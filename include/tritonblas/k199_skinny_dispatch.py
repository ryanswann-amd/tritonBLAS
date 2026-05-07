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

Two empirical findings shaped what landed here:

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

Given those findings, the table below contains **only entries that did NOT
regress kernel time** in the verification A/B (``spd_def >= 1.00x``).
14 of 30 swept bf16 shapes met that bar (geomean +1.3% within ~5% noise);
all swept fp16 shapes regressed by ~1-5% (because Origami's fp16 picks are
already small-tile and the override forced larger asymmetric tiles), so no
fp16 entries land. This makes the change a kernel-level safe-or-better
override rather than a "knob-on-by-default" of the kind K-654 warned
against. The unverified 16 bf16 + 30 fp16 candidate configs are in the
K-199 workspace artefacts (``output/k199_dispatch_table.py``,
``output/winners_aggregated.json``, ``output/verify_kernel_*.json``) for
re-derivation should the production wrapper get faster (K-180) or the
launch path shorter.

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
from typing import Optional, Tuple


# Per-shape overrides for the K-199 cohort. Only entries that did NOT regress
# kernel time vs Origami's analytical pick in the verification A/B made it in;
# see verify_kernel_bf16.json + verify_kernel_fp16.json (in the K-199 workspace
# output/) for the full per-shape numbers. The 16 bf16 + 30 fp16 swept candidates
# that regressed by 0.5-5% (kernel-level noise floor) are intentionally NOT here.
#
# spd_def values are the verified (median over 30 reps, CUDA events) ratio of
# Origami-default kernel time over K-199 kernel time on a fresh selector + fresh
# kernel cache, MI300X g08u07 / c42, on the production persistent_matmul_lt path.
_K199_OVERRIDES = {
    # All entries are bf16 on MI300X. The fp16 sweep produced no kernel-level
    # non-regressing winners on this cohort (Origami's small-tile fp16 picks
    # are already at the launch floor).
    (16, 8192, 256, 'bf16'): dict(BLK_M=16, BLK_N=256, BLK_K=32, num_warps=4, num_stages=2, split_k=1),  # verified spd_def=1.01x
    (32, 1024, 512, 'bf16'): dict(BLK_M=32, BLK_N=128, BLK_K=128, num_warps=4, num_stages=2, split_k=1),  # verified spd_def=1.02x
    (32, 4096, 256, 'bf16'): dict(BLK_M=32, BLK_N=256, BLK_K=32, num_warps=8, num_stages=2, split_k=1),  # verified spd_def=1.01x
    (128, 1024, 128, 'bf16'): dict(BLK_M=128, BLK_N=128, BLK_K=128, num_warps=4, num_stages=2, split_k=1),  # verified spd_def=1.02x
    (128, 4096, 256, 'bf16'): dict(BLK_M=128, BLK_N=128, BLK_K=64, num_warps=8, num_stages=3, split_k=1),  # verified spd_def=1.00x
    (1024, 16, 512, 'bf16'): dict(BLK_M=256, BLK_N=16, BLK_K=64, num_warps=4, num_stages=2, split_k=1),  # verified spd_def=1.02x
    (1024, 32, 128, 'bf16'): dict(BLK_M=128, BLK_N=32, BLK_K=128, num_warps=4, num_stages=2, split_k=1),  # verified spd_def=1.03x
    (1024, 128, 128, 'bf16'): dict(BLK_M=128, BLK_N=128, BLK_K=64, num_warps=8, num_stages=3, split_k=1),  # verified spd_def=1.00x
    (2048, 32, 256, 'bf16'): dict(BLK_M=128, BLK_N=32, BLK_K=128, num_warps=4, num_stages=2, split_k=1),  # verified spd_def=1.01x
    (4096, 16, 256, 'bf16'): dict(BLK_M=256, BLK_N=16, BLK_K=32, num_warps=4, num_stages=3, split_k=1),  # verified spd_def=1.00x
    (4096, 32, 256, 'bf16'): dict(BLK_M=128, BLK_N=32, BLK_K=64, num_warps=4, num_stages=3, split_k=1),  # verified spd_def=1.00x
    (4096, 64, 256, 'bf16'): dict(BLK_M=128, BLK_N=64, BLK_K=128, num_warps=4, num_stages=2, split_k=1),  # verified spd_def=1.01x
    (4096, 128, 256, 'bf16'): dict(BLK_M=128, BLK_N=128, BLK_K=64, num_warps=8, num_stages=2, split_k=1),  # verified spd_def=1.01x
    (8192, 16, 256, 'bf16'): dict(BLK_M=256, BLK_N=16, BLK_K=32, num_warps=4, num_stages=3, split_k=1),  # verified spd_def=1.02x
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
