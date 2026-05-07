import os
import torch

# 256-byte separation between atomic counters to avoid false sharing
# across L2 cache lines.  Each int32 is 4 bytes -> stride = 256 / 4 = 64 elements.
COUNTER_STRIDE = 64

MAX_SK_TILES = 512


# ---------------------------------------------------------------------------
# K-312 large-square fp16/bf16 cohort overrides.
#
# Background: K-268 measured a 0.838x geomean on M,N in {4096,8192},
# K in {4096,8192,16384} for fp16/bf16 vs hipBLASLt; K-250 ATT traces
# attributed the residual to (a) epilogue store coalescing, (b) L2 hit-rate,
# and (c) occupancy lost to register spills - NOT mainloop MFMA throughput.
#
# This helper centralises the cohort gate.  Per the K-654 lesson, the
# gating predicate MUST live in select_*_config() itself (not be merely
# consulted by the autotuner) so that:
#   (a) mode=on cannot regress shapes outside the envelope,
#   (b) cache hits on out-of-envelope shapes fall back to baseline,
#   (c) sister cohorts (e.g. K-539 small-K) are not poisoned.
#
# The gate is conservative:
#   - dtype must be fp16 or bf16 (fp8 / int8 / fp4 take dispatch paths
#     with different epilogues; out of scope here),
#   - M*N >= 16M (covers 4096*4096 and larger; rejects 2048-square),
#   - K     >= 4096  (rejects K-539 small-K cohort that K-654 protected).
#
# IMPORTANT — the K-312 BUG_FIX initial measurement (24-shape MI300X
# sweep, 2026-05-07, container rocm/pytorch:rocm7.2_ubuntu24.04, c42
# node g09u19) showed that the proposed (waves_per_eu=2, kpack=2,
# EPILOGUE_VECTOR_WIDTH=8) lever REGRESSES this cohort:
#
#     geomean tb_on / tb_off = 0.9352  over 24 shapes
#     22 / 24 shapes regress 3.2% – 10.9%
#     ratio vs torch falls 0.6310 -> 0.5922
#
# This corroborates K-895's NEGATIVE_RESULT_VERIFIED finding for the
# sister kpack=2 + num_warps=4 lever on the same anchor shape
# (1024×8192×8192 bf16, –16.3 pp). The rerouted plan in
# project_context/tritonblas/persistent_residuals.md is K-633 upstream
# codegen OR per-shape Triton autotune — neither lands in this PR.
#
# So this gate ships **OFF by default** (opt-in via
# TRITONBLAS_ENABLE_K312_OVERRIDES=1) so the cohort detection,
# kernel-side EPILOGUE_VECTOR_WIDTH constexpr, and override plumbing
# are all in place for the future autotune-driven landing without
# regressing production today.  The K-654 anti-pattern is "knob
# applied unconditionally" — opt-in OFF-by-default is the safe form.
# ---------------------------------------------------------------------------

_K312_FP_DTYPES = (torch.float16, torch.bfloat16)
_K312_MN_MIN = 16 * 1024 * 1024  # 16M elements (4096*4096)
_K312_K_MIN = 4096
_K312_DEFAULT_OVERRIDES = {
    "waves_per_eu": 2,           # K-250 ATT: recover occupancy from register spills
    "kpack": 2,                  # K-580/K-654: pair LDS reads, akin to lds_swizzle ON
    "epilogue_vector_width": 8,  # K-250 epilogue store coalescing (currently 4)
}


def large_square_cohort_overrides(M, N, K, a_dtype, b_dtype):
    """Return kernel knob overrides for the K-312 cohort, else None.

    Returns a dict with keys ``waves_per_eu``, ``kpack``,
    ``epilogue_vector_width`` ONLY if (M,N,K,dtype) is in the K-312
    envelope AND the opt-in env var ``TRITONBLAS_ENABLE_K312_OVERRIDES``
    is set.  Returns None otherwise (default: baseline behavior).

    Default is OFF because the initial measurement (see module docstring
    above) showed a ~6.5% geomean regression on the cohort.  Land the
    plumbing now; opt-in stays OFF until a per-shape autotune table
    (K-895 reroute) supplies a measured-positive override set.
    """
    if os.environ.get("TRITONBLAS_ENABLE_K312_OVERRIDES", "").lower() not in ("1", "true", "yes"):
        return None
    if a_dtype not in _K312_FP_DTYPES or b_dtype not in _K312_FP_DTYPES:
        return None
    try:
        mn = int(M) * int(N)
    except (TypeError, ValueError):
        return None
    if mn < _K312_MN_MIN or int(K) < _K312_K_MIN:
        return None
    return dict(_K312_DEFAULT_OVERRIDES)


class MatmulConfig:
    """
    Pre-allocated GPU buffers for GEMM kernel launches.

    Create via :func:`matmul_preamble` with an ``OrigamiMatmulSelector``.
    Buffer sizes are derived from the selector's tile configuration.

    Attributes:
        device:           ``torch.device`` the buffers live on.
        tile_counter:     ``int32[num_counters * COUNTER_STRIDE]`` work-stealing
                          counters, padded to 256B per slot to avoid false sharing.
        global_counter:   ``int32[COUNTER_STRIDE]`` single global counter for
                          hierarchical mode's Level 2 fallback pool.
        mask:             ``int32[N_CU]`` per-CU enable mask (1=active, 0=skip).
        locks:            ``uint8[sk_grid]`` stream-K lock array.
        P:                ``float32[sk_grid, block_size]`` stream-K partial buffer.
        sk_iter_counter:  ``int32[COUNTER_STRIDE]`` global atomic for dynamic SK.
        sk_locks:         ``int32[MAX_SK_TILES]`` per-tile spin-lock.
        sk_done:          ``int32[MAX_SK_TILES]`` per-tile K-iteration completion.
        sk_P:             ``float32[MAX_SK_TILES, block_size]`` per-tile partial acc.
    """

    def __init__(self, device: torch.device, tile_counter: torch.Tensor,
                 streamk_tile_counter: torch.Tensor, locks: torch.Tensor,
                 P: torch.Tensor, global_atomic: bool = False,
                 global_counter: torch.Tensor = None,
                 mask: torch.Tensor = None,
                 sk_iter_counter: torch.Tensor = None,
                 sk_locks: torch.Tensor = None,
                 sk_done: torch.Tensor = None,
                 sk_P: torch.Tensor = None):
        self.device = device
        self.tile_counter = tile_counter
        self.streamk_tile_counter = streamk_tile_counter
        self.locks = locks
        self.P = P
        self.global_atomic = global_atomic
        self.global_counter = global_counter
        self.mask = mask
        self.sk_iter_counter = sk_iter_counter
        self.sk_locks = sk_locks
        self.sk_done = sk_done
        self.sk_P = sk_P

    def reset(self, streamk: bool = False, work_stealing: bool = False):
        """Reset mutable state based on the active kernel mode.

        Args:
            streamk:        Zero the stream-K lock array.
            work_stealing:  Zero the work-stealing tile counter(s).
        """
        if work_stealing:
            self.tile_counter.zero_()
            if self.global_counter is not None:
                self.global_counter.zero_()
        if streamk:
            self.locks.zero_()
            self.P.zero_()
            if self.sk_iter_counter is not None:
                self.sk_iter_counter.zero_()
            if self.sk_locks is not None:
                self.sk_locks.zero_()
            if self.sk_done is not None:
                self.sk_done.zero_()
            if self.sk_P is not None:
                self.sk_P.zero_()

    def __repr__(self):
        return (
            f"MatmulConfig(device={self.device!r}, "
            f"tile_counter={list(self.tile_counter.shape)}, "
            f"locks={list(self.locks.shape)}, "
            f"P={list(self.P.shape)})"
        )


def matmul_preamble(selector, device: torch.device = None) -> MatmulConfig:
    """
    Allocate all GPU-side buffers needed by the tritonBLAS GEMM kernels.

    Call this once per problem shape (or once with the largest expected shape)
    and pass the returned config into ``matmul_lt``, ``matmul_a8w8_lt``, etc.

    Args:
        selector:  An ``OrigamiMatmulSelector`` providing tile sizes, XCD count,
                   stream-K grid, and ``COUNTERS_PER_XCD``.
        device:    ``torch.device`` for buffer allocation (default: current CUDA device).

    Returns:
        A :class:`MatmulConfig` ready for kernel launches.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    num_xcds = selector._hardware.NUM_XCD
    counters_per_xcd = selector.COUNTERS_PER_XCD
    block_size = selector.block_m * selector.block_n
    sk_grid = selector.sk_grid

    num_counters = num_xcds * counters_per_xcd
    tile_counter = torch.zeros(num_counters * COUNTER_STRIDE, device=device, dtype=torch.int32)
    streamk_tile_counter = torch.zeros(num_counters * COUNTER_STRIDE, device=device, dtype=torch.int32)
    locks = torch.zeros(sk_grid, device=device, dtype=torch.uint8)
    P = torch.empty(sk_grid, block_size, device=device, dtype=torch.float32)

    global_counter = torch.zeros(COUNTER_STRIDE, device=device, dtype=torch.int32)

    n_cu = selector._N_CU
    active_cu = selector._ACTIVE_CU
    mask = torch.ones(n_cu, dtype=torch.int32, device=device)
    if active_cu < n_cu:
        mask[active_cu:] = 0

    max_sk = MAX_SK_TILES
    sk_iter_counter = torch.zeros(COUNTER_STRIDE, device=device, dtype=torch.int32)
    sk_locks = torch.zeros(max_sk, device=device, dtype=torch.int32)
    sk_done = torch.zeros(max_sk, device=device, dtype=torch.int32)
    sk_P = torch.zeros(max_sk, block_size, device=device, dtype=torch.float32)

    return MatmulConfig(device=device, tile_counter=tile_counter,
                        streamk_tile_counter=streamk_tile_counter,
                        locks=locks, P=P, mask=mask,
                        global_counter=global_counter,
                        sk_iter_counter=sk_iter_counter,
                        sk_locks=sk_locks, sk_done=sk_done, sk_P=sk_P)
