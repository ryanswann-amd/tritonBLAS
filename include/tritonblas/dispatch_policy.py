"""Single owner of the public-API auto-dispatch policy for matmul.

Consolidates the three pieces that were previously scattered across
``matmul.py`` (the ``enable_streamk`` resolver) and ``origami.py`` (the
``is_large_skinny_shape`` classifier and the ``recommend_streamk``
recommendation).  Centralising them here lets future tile / codegen
variants extend the dispatch policy without re-threading sentinel
handling through the public API surface again.

The module exports:

  * Threshold constants — :data:`LARGE_SKINNY_LONG_DIM_MIN`,
    :data:`LARGE_SKINNY_ASPECT_MIN`, :data:`LARGE_SKINNY_K_MIN`
    (large-skinny classifier) and :data:`STREAMK_MIN_K`,
    :data:`STREAMK_MAX_WAVES`, :data:`STREAMK_LAST_WAVE_FRAC_MAX`,
    :data:`STREAMK_REGRESSION_GUARD_RATIO_FLOOR`,
    :data:`STREAMK_DEFAULT_TILE` (stream-K recommendation).
  * Pure-Python predicates — :func:`is_large_skinny_shape`,
    :func:`recommend_streamk`.
  * The ``EnableStreamKArg`` type alias and the public-API resolver
    :func:`resolve_enable_streamk`.

``origami.py`` re-exports the classifier and the recommendation so that
existing imports continue to work; ``matmul.py`` imports the resolver
and the type alias from here directly.

Empirical calibration was done on MI300X (gfx942) — see the module-level
docstrings for the regimes each threshold was measured against and the
per-shape rationale.
"""

from __future__ import annotations

from typing import Union

# --------------------------------------------------------------------------
# Large-skinny rectangle classifier.
#
# Flags shapes in the difficult regime where last-wave imbalance dominates
# performance loss vs hipBLASLt.  The classifier is *informational* — used
# for logging / KB queries and as a *necessary* (not sufficient) condition
# inside ``recommend_streamk``.
#
# Calibrated against the K-654 PRD residual rectangles on gfx942:
#   1024x8192x8192 bf16, 6144x4096x4096 fp16, 8192x2048x4096 fp16.
# --------------------------------------------------------------------------
LARGE_SKINNY_LONG_DIM_MIN = 4096
LARGE_SKINNY_ASPECT_MIN = 2.0
LARGE_SKINNY_K_MIN = 2048


def is_large_skinny_shape(m: int, n: int, k: int) -> bool:
    """Classify ``(m, n, k)`` as a large-skinny rectangle GEMM.

    Returns ``True`` iff *all* of the following hold:

      1. ``max(M, N) >= LARGE_SKINNY_LONG_DIM_MIN`` (currently 4096).
      2. ``K >= LARGE_SKINNY_K_MIN`` (currently 2048).
      3. ``max(M, N) / min(M, N) >= LARGE_SKINNY_ASPECT_MIN`` (currently 2.0).

    Reject degenerate / non-positive shapes (``m <= 0`` etc.) outright so
    we don't silently divide-by-zero on the aspect ratio.
    """
    if m <= 0 or n <= 0 or k <= 0:
        return False
    long_dim = max(m, n)
    short_dim = min(m, n)
    if long_dim < LARGE_SKINNY_LONG_DIM_MIN:
        return False
    if k < LARGE_SKINNY_K_MIN:
        return False
    return (long_dim / short_dim) >= LARGE_SKINNY_ASPECT_MIN


# --------------------------------------------------------------------------
# Stream-K recommendation.
#
# Stream-K wins over the data-parallel persistent kernel exactly when the
# tile grid has a meaningful *last-wave imbalance*: ``total_tiles mod n_cu``
# is a small fraction of ``n_cu`` so a chunk of CUs would otherwise idle
# while the partial final wave drains.
#
# The recommendation is conservative:
#
#   * It fires only when the K-loop is long enough to amortise the
#     inter-WG reduction overhead (``K >= STREAMK_MIN_K``).
#   * It fires only when the grid is big enough to fill at least one full
#     wave (``total_tiles >= n_cu``).
#   * It bails out for many-wave shapes
#     (``waves > STREAMK_MAX_WAVES``) where the imbalance is amortised
#     away by sheer volume.
#   * It requires the partial last wave to be small enough that
#     K-splitting actually fills idle CUs
#     (``last_wave_frac < STREAMK_LAST_WAVE_FRAC_MAX``).
#
# These bounds were calibrated empirically on MI300X (gfx942) with the
# 4-mode kernel sweep in
# ``state/mc2/workspaces/<task>/output/bench_large_skinny.csv``.  The
# regression-guard ratio floor pins the public default so smaller squares
# already in the 0.80-0.95 ratio band are *not* silently flipped under
# auto mode.
# --------------------------------------------------------------------------
STREAMK_MIN_K = 1024
STREAMK_MAX_WAVES = 4.5
STREAMK_LAST_WAVE_FRAC_MAX = 0.6
STREAMK_DEFAULT_TILE = 256

# Minimum total waves before stream-K is considered.  Calibrated on
# MI300X: at ~1-2 waves the partial last wave is the entire kernel,
# and the inter-WG reduction added by stream-K outweighs the imbalance
# fix on a single sub-wave (measured -4% on 6144x4096x4096 fp16,
# 1.26 waves on 304 CUs, where the data-parallel persistent kernel
# wins by 18 TF — see the large-skinny benchmark CSV in the PR).
# Stream-K only delivers a robust win once the partial last wave is
# *amortised* against >=2 full waves of compute.
STREAMK_MIN_WAVES = 2.0

# Stream-K MUST NOT silently flip shapes whose ``ratio_persistent`` is
# already in the 0.80-0.95 PRD band — those are stable wins for the
# data-parallel kernel.  The regression guard is encoded as a "perfectly-
# tiled small-square" carve-out below; this constant is exposed so tests
# can pin the documented behaviour.
STREAMK_REGRESSION_GUARD_RATIO_FLOOR = 0.80


def recommend_streamk(
    m: int,
    n: int,
    k: int,
    n_cu: int,
    default_tile: int = STREAMK_DEFAULT_TILE,
) -> bool:
    """Recommend the persistent stream-K kernel over the data-parallel one.

    Returns ``True`` when *all* the bounds above hold; ``False`` (the
    safe default) otherwise.  See module docstring for the rationale.

    Empirically validated wins on MI300X (gfx942):

      * ``8192 x 8192 x 8192`` bf16 — +5% over persistent
      * ``8192 x 8192 x 8192`` fp16 — +5% over persistent
      * ``6144 x 4096 x 4096`` fp16 — +9% over persistent (rectangle case)

    Empirically validated *non*-wins (heuristic correctly returns False):

      * ``1024 x 8192 x 8192`` bf16 — sub-wave; K-split overhead dominates
      * ``8192 x 2048 x 4096`` fp16 — sub-wave; ditto
      * ``4096 x 4096 x 4096`` bf16 — perfectly aligned; no last-wave imbalance
    """
    if m <= 0 or n <= 0 or k <= 0 or n_cu <= 0:
        return False
    if k < STREAMK_MIN_K:
        return False
    tiles_m = (m + default_tile - 1) // default_tile
    tiles_n = (n + default_tile - 1) // default_tile
    total_tiles = tiles_m * tiles_n
    if total_tiles < n_cu:
        # K-split helps in theory but per-WG overhead dominates in practice.
        return False
    waves = total_tiles / n_cu
    if waves < STREAMK_MIN_WAVES:
        # Sub-2-wave grids: the partial last wave IS the whole kernel and
        # stream-K's inter-WG reduction overhead outweighs the imbalance
        # fix.  Empirically -4% on 6144x4096x4096 fp16 (1.26 waves) when
        # auto-streamk fired in the prior calibration.
        return False
    if waves > STREAMK_MAX_WAVES:
        # Many waves -> last-wave imbalance is a tiny fraction of runtime.
        return False
    last_wave = total_tiles % n_cu
    if last_wave == 0:
        return False  # perfectly aligned grid -> no imbalance to fix
    return last_wave < int(n_cu * STREAMK_LAST_WAVE_FRAC_MAX)


# --------------------------------------------------------------------------
# Public-API resolver for the ``enable_streamk`` parameter.
# --------------------------------------------------------------------------

EnableStreamKArg = Union[bool, str, None]
"""Type of the public ``enable_streamk`` argument: explicit boolean,
the literal string sentinel ``"auto"``, or ``None`` (back-compat alias
for ``"auto"``).  Any other value raises :class:`TypeError` at dispatch
time."""


def resolve_enable_streamk(
    enable_streamk: EnableStreamKArg,
    M: int,
    N: int,
    K: int,
    n_cu: int,
) -> bool:
    """Resolve the ``enable_streamk`` parameter on the public API.

    The public functions accept:

      * ``True`` / ``False`` — explicit caller choice; bypasses the
        heuristic.  Use this when a caller has already done its own
        autotuning.
      * ``"auto"`` (string sentinel, the default) — consult
        :func:`recommend_streamk` and auto-enable stream-K on shapes
        whose data-parallel tile grid would leave a meaningful chunk of
        CUs idle in the last wave.
      * ``None`` — alias for ``"auto"``, retained so old callers that
        passed a positional ``Optional[bool]`` from a wrapper continue
        to opt in to the heuristic.

    Identity-checks ``True`` / ``False`` so Python's ``bool`` /
    ``int`` subclass relationship cannot accept ``0`` / ``1`` silently.

    Raises ``TypeError`` for anything else (e.g. ``"on"``, ``"true"``)
    so a caller mistake surfaces immediately at dispatch time.
    """
    if enable_streamk is True or enable_streamk is False:
        return enable_streamk
    if enable_streamk is None or enable_streamk == "auto":
        return recommend_streamk(M, N, K, n_cu)
    raise TypeError(
        f"enable_streamk must be True, False, 'auto', or None; "
        f"got {enable_streamk!r}"
    )


__all__ = [
    "EnableStreamKArg",
    "LARGE_SKINNY_ASPECT_MIN",
    "LARGE_SKINNY_K_MIN",
    "LARGE_SKINNY_LONG_DIM_MIN",
    "STREAMK_DEFAULT_TILE",
    "STREAMK_LAST_WAVE_FRAC_MAX",
    "STREAMK_MAX_WAVES",
    "STREAMK_MIN_K",
    "STREAMK_REGRESSION_GUARD_RATIO_FLOOR",
    "is_large_skinny_shape",
    "recommend_streamk",
    "resolve_enable_streamk",
]
