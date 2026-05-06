"""Warm-cache pre-population for the tritonBLAS autotuner.

K-486 audited cold-start latency on shapes outside the analytical model's
training distribution and found two compounding costs on the first call:

1.  Origami's ``select_config`` runs a full search (~20us per call after
    LDS pruning, larger before).
2.  Triton JIT-compiles a fresh kernel for the chosen tile/constexpr
    signature (1-3 seconds on MI300X).

For long-running services (vLLM, text-generation servers) the JIT cost
matters most because each unseen shape stalls the first request to that
shape by seconds.  This module addresses both costs:

* :func:`prepopulate` walks a curated list of "training shapes" and forces
  Origami selection + (optionally) Triton compilation for each.  After the
  pass, Triton's on-disk cache holds compiled binaries for every distinct
  tile/constexpr signature that Origami picks for the curated set.
* :class:`KNNSelectorCache` indexes the resulting selectors by
  ``(log2 M, log2 N, log2 K, dtype-key)`` so any new shape lookup finds
  the K nearest training shapes in O(N) time and votes on the tile config.
* :class:`_WarmSelector` exposes the same surface as
  ``OrigamiMatmulSelector`` but is built from a K-NN-voted config plus the
  *new* problem's M/N/K/even_k.  Because the constexpr signature
  (BLOCK_M/N/K, GROUP_M, NUM_XCDS, EVEN_K, num_warps, num_stages, MFMA dim,
  kpack) matches one already present in Triton's cache, the kernel launch
  hits the cache and skips JIT.

The cache is opt-in -- callers run ``tritonblas.warmup(...)`` once at
process start.  When the cache is empty (or no neighbour is close enough),
``_make_matmul_selector`` falls back to the original Origami path so
behaviour is unchanged.

All public state is process-global and protected by a re-entrant lock so
multiple threads can safely warm or query the cache concurrently.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


# ----------------------------------------------------------------------
# Default training distribution
# ----------------------------------------------------------------------

# Curated set of "production shape" archetypes used at warmup.  These come
# from common LLM serving (Llama 7B/13B/70B FFN + projection), decode-time
# small-M shapes, and benchmark squares.  A real deployment is expected to
# extend this list with its own measured workload; the defaults exist so
# that ``tritonblas.warmup()`` with no arguments still produces a useful
# pre-population.
DEFAULT_TRAINING_SHAPES: Sequence[Tuple[int, int, int]] = (
    # Square benchmark shapes (compute-bound)
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    # Llama-7B prefill (FFN + projection)
    (4096, 11008, 4096),
    (4096, 4096, 11008),
    (4096, 4096, 4096),
    # Llama-13B prefill
    (5120, 13824, 5120),
    (5120, 5120, 13824),
    # Llama-70B prefill (with TP=8 sharding)
    (8192, 3584, 8192),
    (8192, 8192, 3584),
    # Decode (small-M skewed)
    (1, 4096, 4096),
    (1, 8192, 8192),
    (32, 4096, 4096),
    (128, 4096, 4096),
    (512, 4096, 4096),
)

# Default (a_dtype, b_dtype, c_dtype) tuples to warm.  bf16/bf16 dominates
# modern LLM serving; fp16 is still common for older stacks; fp32 is
# included so the scientific-compute path is not stranded.
DEFAULT_TRAINING_DTYPES: Sequence[Tuple[torch.dtype, torch.dtype, torch.dtype]] = (
    (torch.bfloat16, torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float16, torch.float16),
)


# ----------------------------------------------------------------------
# K-NN registry data structures
# ----------------------------------------------------------------------


# Key used to bucket cache entries.  Two queries can only match if every
# field is identical; only M/N/K may interpolate via K-NN.
_DtypeKey = Tuple[torch.dtype, torch.dtype, torch.dtype, int, bool, int]


def _dtype_key(
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_dtype: torch.dtype,
    mx_block_size: int,
    streamk: bool,
    num_stages: int,
) -> _DtypeKey:
    return (a_dtype, b_dtype, c_dtype, mx_block_size, streamk, num_stages)


@dataclass(frozen=True)
class _ConfigSnapshot:
    """Tile + launch parameters extracted from an OrigamiMatmulSelector.

    Captures everything we need to re-launch the kernel without re-running
    the analytical model, while leaving shape-dependent values
    (``COUNTERS_PER_XCD``, ``group_m``, ``even_k``, ``sk_grid``) to be
    recomputed for the new problem inside :class:`_WarmSelector`.
    """

    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_xcds: int  # i.e. xcc_workgroup_mapping
    n_cu: int
    active_cu: int
    waves_per_eu: int
    num_stages: int


@dataclass
class _Entry:
    """A single registry record."""

    log_shape: Tuple[float, float, float]
    raw_shape: Tuple[int, int, int]
    config: _ConfigSnapshot


class KNNSelectorCache:
    """Process-global registry of pre-warmed selectors.

    Linear-scan K-NN is more than fast enough at the sizes we expect
    (hundreds of training shapes) and avoids pulling in any dependency
    beyond the Python standard library.
    """

    # Below this log-distance we treat a hit as exact.  ``log2(1.05) ~=
    # 0.07``: anything within 5% on every dimension is effectively the
    # same shape from a tile-selection standpoint.
    EXACT_MATCH_THRESHOLD = 0.07

    # Maximum log-distance at which we are still willing to vote a config
    # for the new shape.  log2(2) == 1.0; anything farther is considered
    # outside the training neighbourhood and we fall back to Origami.
    MAX_NEIGHBOUR_DISTANCE = 1.0

    def __init__(self, k: int = 3) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._k = k
        self._lock = threading.RLock()
        self._buckets: dict[_DtypeKey, List[_Entry]] = {}
        self._enabled = True
        self._stats = {"hits_exact": 0, "hits_knn": 0, "misses": 0}

    # ------------------------------------------------------------------
    # Public read-only surface
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def k(self) -> int:
        return self._k

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def size(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._buckets.values())

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()
            self._stats = {"hits_exact": 0, "hits_knn": 0, "misses": 0}

    def set_enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    # ------------------------------------------------------------------
    # Insertion + lookup
    # ------------------------------------------------------------------
    @staticmethod
    def _log_shape(m: int, n: int, k: int) -> Tuple[float, float, float]:
        # ``max(1, ...)`` guards against decode-time M=1 (log2 0 is undefined).
        return (
            math.log2(max(1, m)),
            math.log2(max(1, n)),
            math.log2(max(1, k)),
        )

    @staticmethod
    def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        # L2 distance in log2 space -- proportional to log of geometric
        # ratio between shapes, which matches how tile selection scales.
        return math.sqrt(
            (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
        )

    def insert(
        self,
        m: int,
        n: int,
        k: int,
        dtype_key: _DtypeKey,
        config: _ConfigSnapshot,
    ) -> None:
        """Add (or overwrite an exact-match for) a training shape."""
        log_pt = self._log_shape(m, n, k)
        new = _Entry(log_pt, (m, n, k), config)
        with self._lock:
            bucket = self._buckets.setdefault(dtype_key, [])
            for i, existing in enumerate(bucket):
                if existing.raw_shape == new.raw_shape:
                    bucket[i] = new
                    return
            bucket.append(new)

    def lookup(
        self,
        m: int,
        n: int,
        k: int,
        dtype_key: _DtypeKey,
    ) -> Optional[Tuple[_ConfigSnapshot, float, str]]:
        """Return ``(config, distance, kind)`` if a usable neighbour exists.

        ``kind`` is ``"exact"`` (closest neighbour within threshold) or
        ``"knn"`` (K-NN vote among neighbours within ``MAX_NEIGHBOUR_DISTANCE``).
        Returns ``None`` if the bucket is empty or the closest neighbour is
        farther than ``MAX_NEIGHBOUR_DISTANCE`` -- in that case the caller
        should fall back to Origami.
        """
        if not self._enabled:
            return None
        log_pt = self._log_shape(m, n, k)
        with self._lock:
            bucket = self._buckets.get(dtype_key)
            if not bucket:
                self._stats["misses"] += 1
                return None
            # Compute distances and sort.  O(N) per query is fine for the
            # registry sizes we expect (<= a few hundred entries).
            ranked = sorted(
                ((self._distance(log_pt, e.log_shape), e) for e in bucket),
                key=lambda t: t[0],
            )
            best_d, best_e = ranked[0]
            if best_d <= self.EXACT_MATCH_THRESHOLD:
                self._stats["hits_exact"] += 1
                return best_e.config, best_d, "exact"
            if best_d > self.MAX_NEIGHBOUR_DISTANCE:
                self._stats["misses"] += 1
                return None
            # K-NN vote.  Pick the most common (block_m, block_n, block_k)
            # tuple among the top-k, then the most common num_stages /
            # num_xcds for that winner.  Tiebreak by closest distance.
            top_k = ranked[: self._k]
            tile_votes: dict[Tuple[int, int, int], List[Tuple[float, _ConfigSnapshot]]] = {}
            for d, e in top_k:
                key = (e.config.block_m, e.config.block_n, e.config.block_k)
                tile_votes.setdefault(key, []).append((d, e.config))
            # Winner: most votes; tiebreak by lowest min-distance.
            winner_key = max(
                tile_votes,
                key=lambda key: (
                    len(tile_votes[key]),
                    -min(d for d, _ in tile_votes[key]),
                ),
            )
            # Within the winning tile bucket, the closest entry's other
            # fields (group_m, num_xcds, etc.) are the safest pick.
            chosen = min(tile_votes[winner_key], key=lambda t: t[0])[1]
            self._stats["hits_knn"] += 1
            return chosen, best_d, "knn"


# ----------------------------------------------------------------------
# Synthetic selector
# ----------------------------------------------------------------------


class _WarmSelector:
    """OrigamiMatmulSelector look-alike built from a cached config.

    Mirrors every attribute that ``persistent_matmul_lt`` /
    ``streamk_matmul_lt`` / ``matmul_preamble`` actually read.  Shape-
    dependent fields (``even_k``, ``COUNTERS_PER_XCD``, workgroup mapping,
    ``sk_grid``) are recomputed for the new problem; tile parameters are
    inherited from the cached snapshot.
    """

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        config: _ConfigSnapshot,
        hardware,
        problem,
        streamk: bool,
    ) -> None:
        self._m = m
        self._n = n
        self._k = k
        self._config = config
        self._hardware = hardware
        self._problem = problem
        self.streamk = streamk
        self._num_stages = config.num_stages
        self._N_CU = config.n_cu
        self._ACTIVE_CU = config.active_cu
        self._active_cus = None  # matches OrigamiMatmulSelector default
        # Recompute shape-sensitive scheduling parameters using the same
        # heuristics OrigamiMatmulSelector applies in ``_select_ws_params``.
        self._select_ws_params()
        self._grid = self._compute_grid()

    # ------------------------------------------------------------------
    # OrigamiMatmulSelector property surface
    # ------------------------------------------------------------------
    @property
    def block_m(self) -> int:
        return self._config.block_m

    @property
    def block_n(self) -> int:
        return self._config.block_n

    @property
    def block_k(self) -> int:
        return self._config.block_k

    @property
    def group_m(self) -> int:
        return self._workgroup_mapping

    @property
    def num_sms(self) -> int:
        return self._config.num_xcds

    @property
    def num_stages(self) -> int:
        return self._num_stages

    @property
    def waves_per_eu(self) -> int:
        return self._config.waves_per_eu

    @property
    def even_k(self) -> bool:
        return self._k % self._config.block_k == 0

    @property
    def sk_grid(self) -> int:
        return self._grid

    # ------------------------------------------------------------------
    # Internal helpers (mirror OrigamiMatmulSelector)
    # ------------------------------------------------------------------
    def _select_ws_params(self) -> None:
        bm = self._config.block_m
        bn = self._config.block_n
        total_tiles = ((self._m + bm - 1) // bm) * ((self._n + bn - 1) // bn)
        tiles_m = (self._m + bm - 1) // bm
        if total_tiles <= 512:
            self.COUNTERS_PER_XCD = 8
        elif total_tiles <= 1536:
            self.COUNTERS_PER_XCD = 4
        elif total_tiles <= 2048:
            self.COUNTERS_PER_XCD = 2
        else:
            self.COUNTERS_PER_XCD = 1
        self._workgroup_mapping = min(8, tiles_m)

    def hierarchical_split(self, num_xcds: int) -> Tuple[int, int]:
        # Same logic as OrigamiMatmulSelector.hierarchical_split.
        bm = self._config.block_m
        bn = self._config.block_n
        total_tiles = ((self._m + bm - 1) // bm) * ((self._n + bn - 1) // bn)
        hw_cus = self._hardware.NUM_XCD * self._hardware.CU_per_L2
        tiles_per_cu = total_tiles / max(hw_cus, 1)
        local_frac = max(0.5, 1.0 - max(0.0, tiles_per_cu - 4.0) * 0.05)
        local_per_xcd = int(total_tiles * local_frac) // num_xcds
        local_per_xcd = max(local_per_xcd, 1)
        global_tiles = total_tiles - local_per_xcd * num_xcds
        return local_per_xcd, global_tiles

    def _compute_grid(self) -> int:
        if not self.streamk:
            return self._hardware.N_CU
        # Use the same logic as OrigamiMatmulSelector._compute_sk_grid but
        # without the tile-fraction search heuristic -- the cached config
        # plus the new shape is enough to compute a sane stream-K grid.
        from math import ceil

        cu_count = self._hardware.N_CU
        bm, bn, bk = self._config.block_m, self._config.block_n, self._config.block_k
        tiles = ceil(self._m / bm) * ceil(self._n / bn)
        sk_grid = tiles
        if tiles > cu_count:
            # Fall back to "full grid" -- mirrors the conservative branch
            # OrigamiMatmulSelector takes when no fractional split fits.
            sk_grid = tiles
        elif tiles < cu_count:
            iters_per_tile = max(1, ceil(self._k / bk))
            for factor in (8, 6, 4, 3, 2, 1):
                split_grid = tiles * factor
                iters_per_cu = iters_per_tile // factor
                if split_grid <= cu_count and iters_per_cu >= 8:
                    sk_grid = split_grid
                    break
        if tiles % sk_grid != 0:
            sk_grid = tiles
        return sk_grid


# ----------------------------------------------------------------------
# Snapshot extraction
# ----------------------------------------------------------------------


def _snapshot(selector) -> _ConfigSnapshot:
    """Capture the launch-relevant fields of a real Origami selector."""
    return _ConfigSnapshot(
        block_m=selector.block_m,
        block_n=selector.block_n,
        block_k=selector.block_k,
        group_m=selector.group_m,
        num_xcds=selector.num_sms,
        n_cu=selector._N_CU,
        active_cu=selector._ACTIVE_CU,
        waves_per_eu=selector.waves_per_eu,
        num_stages=selector.num_stages,
    )


# ----------------------------------------------------------------------
# Singleton accessor
# ----------------------------------------------------------------------

_GLOBAL_CACHE: Optional[KNNSelectorCache] = None
_GLOBAL_CACHE_LOCK = threading.Lock()


def get_cache() -> KNNSelectorCache:
    """Return the process-global :class:`KNNSelectorCache`."""
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        with _GLOBAL_CACHE_LOCK:
            if _GLOBAL_CACHE is None:
                # K is configurable for experimentation but defaults to 3
                # following the standard K-NN heuristic for tabular voting.
                k = int(os.environ.get("TBLAS_WARMUP_K", "3"))
                _GLOBAL_CACHE = KNNSelectorCache(k=k)
    return _GLOBAL_CACHE


# ----------------------------------------------------------------------
# Public warmup / lookup APIs
# ----------------------------------------------------------------------


def lookup_warm_selector(
    m: int,
    n: int,
    k: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_dtype: torch.dtype,
    device: torch.device,
    mx_block_size: int = 0,
    streamk: bool = False,
    num_stages: int = 2,
) -> Optional[_WarmSelector]:
    """Return a warm selector for the requested problem, or ``None``.

    On miss the caller is expected to fall back to ``OrigamiMatmulSelector``.
    Hits include both exact matches and K-NN-voted neighbours within
    :attr:`KNNSelectorCache.MAX_NEIGHBOUR_DISTANCE`.
    """
    cache = get_cache()
    if not cache.enabled:
        return None
    # FP4 ('f4') and other non-torch dtypes don't currently flow through
    # this path; restrict to torch-dtype problems for safety.
    if not (
        isinstance(a_dtype, torch.dtype)
        and isinstance(b_dtype, torch.dtype)
        and isinstance(c_dtype, torch.dtype)
    ):
        return None
    key = _dtype_key(a_dtype, b_dtype, c_dtype, mx_block_size, streamk, num_stages)
    result = cache.lookup(m, n, k, key)
    if result is None:
        return None
    config, _distance, _kind = result
    # Build the lightweight selector.  The hardware/problem objects are
    # cheap to construct via the real selector path -- but to keep warm
    # lookups fast we lazily build them on first request and cache them
    # per-device.  Since constructing problem_t / hardware_t is microsecond
    # work, we just call into origami here.
    import origami  # imported lazily to avoid module-import-time cost
    hardware = origami.get_hardware_for_device(device.index)
    # Build a minimal problem_t -- only needed for downstream callers that
    # introspect the selector.  Most kernel paths only touch the tile
    # config + hardware, so this is defensive.
    problem = _build_problem(m, n, k, a_dtype, b_dtype, c_dtype, mx_block_size)
    return _WarmSelector(
        m=m,
        n=n,
        k=k,
        config=config,
        hardware=hardware,
        problem=problem,
        streamk=streamk,
    )


def _build_problem(
    m: int,
    n: int,
    k: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_dtype: torch.dtype,
    mx_block_size: int,
):
    import origami

    # Same dtype string mapping the real selector uses.
    from .origami import OrigamiMatmulSelector

    a_str = OrigamiMatmulSelector.dtype_to_str.get(a_dtype, a_dtype)
    b_str = OrigamiMatmulSelector.dtype_to_str.get(b_dtype, b_dtype)
    c_str = OrigamiMatmulSelector.dtype_to_str.get(c_dtype, c_dtype)
    a_orig = origami.string_to_datatype(a_str)
    b_orig = origami.string_to_datatype(b_str)
    c_orig = origami.string_to_datatype(c_str)
    problem = origami.problem_t()
    problem.size = origami.dim3_t(m, n, k)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = a_orig
    problem.b_dtype = b_orig
    problem.c_dtype = c_orig
    problem.d_dtype = c_orig
    problem.mi_dtype = c_orig
    problem.a_mx_block_size = mx_block_size
    problem.b_mx_block_size = mx_block_size
    return problem


def warmup(
    shapes: Optional[Iterable[Tuple[int, int, int]]] = None,
    dtypes: Optional[Iterable[Tuple[torch.dtype, torch.dtype, torch.dtype]]] = None,
    device: Optional[torch.device] = None,
    streamk: bool = False,
    num_stages: int = 2,
    compile_kernels: bool = True,
    log: bool = False,
) -> int:
    """Pre-populate the selector cache (and optionally the Triton JIT cache).

    Parameters
    ----------
    shapes
        Iterable of ``(M, N, K)`` triples.  Defaults to
        :data:`DEFAULT_TRAINING_SHAPES`.
    dtypes
        Iterable of ``(a_dtype, b_dtype, c_dtype)`` triples.  Defaults to
        :data:`DEFAULT_TRAINING_DTYPES`.
    device
        Device to warm.  Defaults to the current CUDA device.
    streamk
        Warm the persistent path (default) or the stream-K path.
    num_stages
        Pipeline stages to warm; matches the kernel-launch parameter.
    compile_kernels
        When ``True``, also launch a tiny dummy matmul for each warmed
        shape to force Triton to JIT-compile and cache the kernel binary.
        Set ``False`` to only warm the analytical selector cache (much
        faster; useful for unit tests).
    log
        Print a one-line progress summary per warmed entry.

    Returns
    -------
    int
        Number of cache entries successfully inserted.
    """
    from .origami import OrigamiMatmulSelector

    if shapes is None:
        shapes = DEFAULT_TRAINING_SHAPES
    if dtypes is None:
        dtypes = DEFAULT_TRAINING_DTYPES
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    cache = get_cache()
    inserted = 0

    for (a_dtype, b_dtype, c_dtype) in dtypes:
        for (m, n, k) in shapes:
            try:
                selector = OrigamiMatmulSelector(
                    m,
                    n,
                    k,
                    a_dtype,
                    b_dtype,
                    c_dtype,
                    device,
                    streamk=streamk,
                    num_stages=num_stages,
                )
            except Exception as exc:  # pragma: no cover - defensive
                if log:
                    print(f"[warmup] skip ({m},{n},{k}) {a_dtype}: {exc}")
                continue
            snap = _snapshot(selector)
            cache.insert(
                m,
                n,
                k,
                _dtype_key(a_dtype, b_dtype, c_dtype, 0, streamk, num_stages),
                snap,
            )
            inserted += 1
            if compile_kernels:
                try:
                    _force_compile(m, n, k, a_dtype, b_dtype, c_dtype, device, streamk)
                except Exception as exc:  # pragma: no cover - defensive
                    # Surface compile failures unconditionally -- a silent
                    # skip here would silently break the whole warmup
                    # pipeline (the cache would still hold the snapshot,
                    # but the corresponding Triton binary would not be
                    # cached, and first-call would still pay the JIT cost).
                    print(f"[warmup] compile-skip ({m},{n},{k}) {a_dtype}: {exc}")
            if log:
                print(
                    f"[warmup] {m}x{n}x{k} {a_dtype} -> "
                    f"BLOCK={snap.block_m}x{snap.block_n}x{snap.block_k} "
                    f"GM={snap.group_m} XCD={snap.num_xcds}"
                )
    return inserted


def _force_compile(
    m: int,
    n: int,
    k: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_dtype: torch.dtype,
    device: torch.device,
    streamk: bool,
) -> None:
    """Force Triton to JIT-compile the kernel for this shape.

    We launch a real matmul on small tensors that share the warmed
    constexpr signature.  Because Triton's cache is keyed on the constexpr
    values (BLOCK_*/GROUP_M/NUM_XCDS/EVEN_K/num_warps/num_stages/MFMA dim
    /kpack), any subsequent launch with the same signature -- regardless
    of run-time M/N/K -- reuses the cached binary.

    Allocating the *real* problem-sized tensors here would consume tens of
    GB during warmup; instead we run with the actual M/N/K so the launch
    matches the warmed selector exactly.  Memory pressure is bounded by
    the largest shape in the training set (~256MB for 8192^3 bf16) and is
    freed before the next iteration.
    """
    # NOTE: ``from . import matmul`` would resolve to the ``matmul``
    # function re-exported by ``tritonblas/__init__.py``, not the
    # ``matmul.py`` submodule -- import the function explicitly to avoid
    # that ambiguity.
    from .matmul import matmul as _matmul_fn

    # Allocate the real shape so the EVEN_K constexpr matches what the
    # selector recorded.  Cheap to free immediately afterwards.
    a = torch.empty((m, k), device=device, dtype=a_dtype)
    b = torch.empty((k, n), device=device, dtype=b_dtype)
    out = torch.empty((m, n), device=device, dtype=c_dtype)
    try:
        if streamk:
            _matmul_fn(a, b, out=out, enable_streamk=True)
        else:
            _matmul_fn(a, b, out=out)
        torch.cuda.synchronize(device)
    finally:
        del a, b, out


def is_warm() -> bool:
    """Return True if the cache contains at least one entry."""
    return get_cache().size() > 0
