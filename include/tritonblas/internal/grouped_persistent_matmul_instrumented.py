"""Instrumented grouped persistent GEMM kernel with s_memrealtime timing probes.

This is a separate copy of grouped_persistent_matmul.py with cycle-accurate
timing instrumentation.  The original kernel is NOT modified.

Timing probes use s_memrealtime (CDNA3 / gfx942) via tl.inline_asm_elementwise.
Each SM writes 7 int64 deltas for its first tile into a timing buffer:
  [group_search, metadata_load, tile_setup, k_loop, store_epilogue, total, total_with_loop]
"""

import triton
import triton.language as tl

from .pid_transforms import chiplet_transform_chunked


# ---------------------------------------------------------------------------
# Helper: read the GPU 64-bit memory-realtime counter
# ---------------------------------------------------------------------------
@triton.jit
def _read_clock() -> tl.int64:
    """Read s_memrealtime after draining all pending memory/compute ops.

    s_waitcnt 0 ensures every prior load/store/MFMA has retired so the
    timestamp is not speculative.  The result is a scalar 64-bit cycle count.
    """
    return tl.inline_asm_elementwise(
        "s_waitcnt 0\n"
        "s_memrealtime $0",
        "=s",
        [],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit()
def grouped_persistent_matmul_instrumented(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_gemm_sizes,
    gemm_offsets,
    g_lds,
    timing_buf,  # int64 pointer, shape [NUM_SMS, 7]
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GROUP_COUNT: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    MATMUL_DTYPE: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Persistent grouped GEMM kernel — instrumented with s_memrealtime probes."""
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    # ── t0: kernel entry ──────────────────────────────────────────────
    t0 = _read_clock()

    total_tiles = tl.load(gemm_offsets + GROUP_COUNT)

    # We only record timing for the *first* tile this SM processes
    first_tile = True

    for tile_id in range(pid, total_tiles, NUM_SMS):
        # ── Group search ──────────────────────────────────────────────
        g = 0
        for g_idx in range(GROUP_COUNT):
            if tile_id >= tl.load(gemm_offsets + g_idx + 1):
                g = g_idx + 1

        # ── t1: after group search ────────────────────────────────────
        t1 = _read_clock()

        g_start = tl.load(gemm_offsets + g)
        tile_in_group = tile_id - g_start

        M = tl.load(group_gemm_sizes + g * 3)
        N = tl.load(group_gemm_sizes + g * 3 + 1)
        K = tl.load(group_gemm_sizes + g * 3 + 2)

        A = tl.load(group_a_ptrs + g).to(tl.pointer_type(MATMUL_DTYPE))
        B = tl.load(group_b_ptrs + g).to(tl.pointer_type(MATMUL_DTYPE))
        C = tl.load(group_c_ptrs + g).to(tl.pointer_type(MATMUL_DTYPE))

        stride_am = tl.load(g_lds + g * 6)
        stride_bk = tl.load(g_lds + g * 6 + 2)
        stride_cm = tl.load(g_lds + g * 6 + 4)

        tl.assume(stride_am > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_cm > 0)

        # ── t2: after metadata loads ──────────────────────────────────
        t2 = _read_clock()

        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

        # L2-friendly tile ordering
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_in_group // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_in_group % num_pid_in_group) % group_size_m)
        pid_n = (tile_in_group % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # Modulo wrapping (no masking on loads)
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)

        A_BASE = A + rm[:, None] * stride_am + rk[None, :]
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :]

        # ── t3: after tile index computation & pointer setup ──────────
        t3 = _read_clock()

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # K-loop with vectorized loads
        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        for k in range(0, loop_k):
            a = tl.load(tl.max_contiguous(tl.multiple_of(A_BASE, (1, 16)), (1, BLOCK_SIZE_K)))
            b = tl.load(tl.max_contiguous(tl.multiple_of(B_BASE, (1, 16)), (1, BLOCK_SIZE_N)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K
            B_BASE += BLOCK_SIZE_K * stride_bk

        # Remainder K-block
        if not EVEN_K:
            rk_last = loop_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_LAST = A + rm[:, None] * stride_am + rk_last[None, :]
            B_LAST = B + rk_last[:, None] * stride_bk + rn[None, :]
            k_mask = rk_last < K
            a = tl.load(A_LAST, mask=k_mask[None, :], other=0.0)
            b = tl.load(B_LAST, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)

        # ── t4: after K-loop ─────────────────────────────────────────
        t4 = _read_clock()

        c = acc.to(C.type.element_ty)

        # Store with boundary masking
        rm_store = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn_store = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_mask = (rm_store[:, None] < M) & (rn_store[None, :] < N)
        rm_store = tl.max_contiguous(tl.multiple_of(rm_store % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn_store = tl.max_contiguous(tl.multiple_of(rn_store % N, BLOCK_SIZE_N), BLOCK_SIZE_N)
        C_ = C + rm_store[:, None] * stride_cm + rn_store[None, :]
        tl.store(C_, c, mask=c_mask)

        # ── t5: after output store ────────────────────────────────────
        t5 = _read_clock()

        # ── Write timing deltas for the first tile processed by this SM ──
        if first_tile:
            # ── t6: capture loop overhead while still in the first-tile branch
            t6 = _read_clock()
            base = pid * 7
            tl.store(timing_buf + base + 0, t1 - t0)   # group search
            tl.store(timing_buf + base + 1, t2 - t1)   # metadata load
            tl.store(timing_buf + base + 2, t3 - t2)   # tile setup
            tl.store(timing_buf + base + 3, t4 - t3)   # K-loop compute
            tl.store(timing_buf + base + 4, t5 - t4)   # store epilogue
            tl.store(timing_buf + base + 5, t5 - t0)   # total (entry → store done)
            tl.store(timing_buf + base + 6, t6 - t0)   # total including loop overhead
            first_tile = False
