"""Instrumented copy of tritonBLAS `streamk_matmul` (source commit 95e2c47).

Adds `s_memrealtime`-based timers (via triton.language.extra.hip.memrealtime,
100 MHz constant clock) around each phase so we can separate:
  - comp_full : data-parallel full-tile compute (top loop)
  - comp_sk   : Stream-K partial-tile MFMA compute
  - red       : Stream-K reduction/fixup (consumer: spin + load partials + add)
  - spin      : subset of `red` spent busy-waiting on producer locks
  - prod_store: producer-side partial store + lock set

Per-program (per-CU) results are written to `trace_ptr` [NUM_SMS x TRACE_SLOTS]
(int64). Only the timers + trace stores are added; the numerics are unchanged.
"""
import triton
import triton.language as tl

from tritonblas.kernels.stages.indexing.pid_transforms import chiplet_transform_chunked
from trace_helpers import read_realtime, get_cu_id

# trace slot layout (int64 per program id)
T_START, T_END = 0, 1
COMP_FULL, COMP_SK, RED, SPIN, PROD = 2, 3, 4, 5, 6
N_FULL, N_SK, N_CONS, N_PROD = 7, 8, 9, 10
CU_ID, PID_SLOT = 11, 12
TRACE_SLOTS = 16


@triton.jit()
def streamk_matmul_traced(
    A, B, C,
    A_scale_ptr, B_scale_ptr, bias_ptr,
    P, locks,
    trace_ptr,
    M, N, K,
    stride_am, stride_bn, stride_cm, stride_cn, stride_bias,
    stride_ak: tl.constexpr, stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr, STREAMK_TILES: tl.constexpr,
    BIAS: tl.constexpr, EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr, CACHE_MODIFIER_B: tl.constexpr,
    TRACE_SLOTS: tl.constexpr,
    QUANTIZED: tl.constexpr = False, ALLOW_TF32: tl.constexpr = True,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    total_full_tiles = total_tiles - STREAMK_TILES

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    # ---- timers (loop-carried) ----
    t_start = read_realtime()
    comp_full = tl.cast(0, tl.int64)
    comp_sk = tl.cast(0, tl.int64)
    red = tl.cast(0, tl.int64)
    spin = tl.cast(0, tl.int64)
    prod = tl.cast(0, tl.int64)
    n_full = tl.cast(0, tl.int64)
    n_sk = tl.cast(0, tl.int64)
    n_cons = tl.cast(0, tl.int64)
    n_prod = tl.cast(0, tl.int64)

    # Full tiles loop
    for tile_id in range(pid, total_full_tiles, NUM_SMS):
        tf0 = read_realtime()
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rn * stride_bias
            bias = tl.load(bias_, mask=rn < N, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1
        tl.assume(loop_k > 1)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_A)
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_A)
            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_B)
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_B)
            if QUANTIZED:
                acc += tl.dot(a, b, input_precision="ieee")
            else:
                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            if stride_ak == 1:
                A_BASE = tl.multiple_of(A_BASE, (1, 16))
            else:
                A_BASE = tl.multiple_of(A_BASE, (16, 1))
            if stride_bk == 1:
                B_BASE = tl.multiple_of(B_BASE, (16, 1))
            else:
                B_BASE = tl.multiple_of(B_BASE, (1, 16))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)
            if QUANTIZED:
                acc += tl.dot(a, b, input_precision="ieee")
            else:
                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

        if QUANTIZED:
            rm_A_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
            rn_B_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
            A_scale = tl.load(A_scale_ptr + rm_A_scale)
            B_scale = tl.load(B_scale_ptr + rn_B_scale)
            acc *= A_scale[:, None] * B_scale[None, :]

        if BIAS:
            if QUANTIZED:
                bias_float = bias.to(tl.float32)
                c = acc + bias_float[None, :]
                c = c.to(C.type.element_ty)
            else:
                c = acc.to(C.type.element_ty)
                c += bias[None, :]
        else:
            c = acc.to(C.type.element_ty)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, mask=mask)
        comp_full += read_realtime() - tf0
        n_full += 1

    if STREAMK_TILES == 0:
        t_end = read_realtime()
        cu = get_cu_id()
        base = trace_ptr + pid * TRACE_SLOTS
        # slot layout: 0 t_start,1 t_end,2 comp_full,3 comp_sk,4 red,5 spin,
        #   6 prod,7 n_full,8 n_sk,9 n_cons,10 n_prod,11 cu_id,12 pid
        tl.store(base + 0, t_start)
        tl.store(base + 1, t_end)
        tl.store(base + 2, comp_full)
        tl.store(base + 3, comp_sk)
        tl.store(base + 4, red)
        tl.store(base + 5, spin)
        tl.store(base + 6, prod)
        tl.store(base + 7, n_full)
        tl.store(base + 8, n_sk)
        tl.store(base + 9, n_cons)
        tl.store(base + 10, n_prod)
        tl.store(base + 11, cu.to(tl.int64))
        tl.store(base + 12, pid.to(tl.int64))
        return

    # Initialize shared buffers for Stream-K
    rm1 = tl.arange(0, BLOCK_SIZE_M)
    rn1 = tl.arange(0, BLOCK_SIZE_N)
    rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
    P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
    tl.store(P_, 0.0, cache_modifier=".wt")
    tl.debug_barrier()
    tl.store(locks + pid, 0, cache_modifier=".wt")

    tl.assume(pid >= 0)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_streamk_iters = STREAMK_TILES * iters_per_tile
    streamk_iters_pcu = total_streamk_iters // NUM_SMS
    streamk_remainder_iters = total_streamk_iters % NUM_SMS
    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)

    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder
        if stride_ak == 1:
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
        else:
            A_BASE = tl.multiple_of(A_BASE, (16, 1))
        if stride_bk == 1:
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
        else:
            B_BASE = tl.multiple_of(B_BASE, (1, 16))

        if BIAS:
            bias_ = bias_ptr + rn * stride_bias
            bias = tl.load(bias_, mask=rn < N, other=0.0)

        tc0 = read_realtime()
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(A_BASE, cache_modifier=CACHE_MODIFIER_A)
                b = tl.load(B_BASE, cache_modifier=CACHE_MODIFIER_B)
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0, cache_modifier=CACHE_MODIFIER_A)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0, cache_modifier=CACHE_MODIFIER_B)
            if QUANTIZED:
                acc += tl.dot(a, b, input_precision="ieee")
            else:
                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk
        comp_sk += read_realtime() - tc0
        n_sk += 1

        if QUANTIZED:
            rm_A_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
            rn_B_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
            A_scale = tl.load(A_scale_ptr + rm_A_scale)
            B_scale = tl.load(B_scale_ptr + rn_B_scale)
            acc *= A_scale[:, None] * B_scale[None, :]

        tile_iter = tile_id * iters_per_tile

        if start_iter != tile_iter:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc, cache_modifier=".wt")
            tl.debug_barrier()
            tl.store(locks + pid, 1, cache_modifier=".wt")
            n_prod += 1
        else:
            tr0 = read_realtime()
            next_pid = pid + 1
            tile_iter_end = tile_iter + iters_per_tile
            end = end_iter

            acc_m_reshaped = tl.reshape(acc, (2, BLOCK_SIZE_M // 2, BLOCK_SIZE_N))
            acc_m_permuted = tl.permute(acc_m_reshaped, (1, 2, 0))
            acc_top, acc_bottom = tl.split(acc_m_permuted)
            acc_top = tl.reshape(acc_top, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N))
            acc_bottom = tl.reshape(acc_bottom, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N))
            acc_top_reshaped = tl.reshape(acc_top, (BLOCK_SIZE_M // 2, 2, BLOCK_SIZE_N // 2))
            acc_top_permuted = tl.permute(acc_top_reshaped, (0, 2, 1))
            acc00, acc01 = tl.split(acc_top_permuted)
            acc_bottom_reshaped = tl.reshape(acc_bottom, (BLOCK_SIZE_M // 2, 2, BLOCK_SIZE_N // 2))
            acc_bottom_permuted = tl.permute(acc_bottom_reshaped, (0, 2, 1))
            acc10, acc11 = tl.split(acc_bottom_permuted)
            acc00 = tl.reshape(acc00, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))
            acc01 = tl.reshape(acc01, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))
            acc10 = tl.reshape(acc10, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))
            acc11 = tl.reshape(acc11, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))

            while (end < tile_iter_end and next_pid < NUM_SMS):
                while tl.load(locks + next_pid, cache_modifier=".cv", volatile=True) != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_base = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N
                P_00 = P_base + tl.arange(0, BLOCK_SIZE_M // 2)[:, None] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 2)[None, :]
                acc00 += tl.load(P_00, cache_modifier=".cv")
                P_01 = P_base + tl.arange(0, BLOCK_SIZE_M // 2)[:, None] * BLOCK_SIZE_N + (tl.arange(0, BLOCK_SIZE_N // 2)[None, :] + BLOCK_SIZE_N // 2)
                acc01 += tl.load(P_01, cache_modifier=".cv")
                P_10 = P_base + (tl.arange(0, BLOCK_SIZE_M // 2)[:, None] + BLOCK_SIZE_M // 2) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 2)[None, :]
                acc10 += tl.load(P_10, cache_modifier=".cv")
                P_11 = P_base + (tl.arange(0, BLOCK_SIZE_M // 2)[:, None] + BLOCK_SIZE_M // 2) * BLOCK_SIZE_N + (tl.arange(0, BLOCK_SIZE_N // 2)[None, :] + BLOCK_SIZE_N // 2)
                acc11 += tl.load(P_11, cache_modifier=".cv")
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            if BIAS:
                rn_bias_left = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 2)) % N
                rn_bias_right = (pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N // 2 + tl.arange(0, BLOCK_SIZE_N // 2)) % N
                bias_left = tl.load(bias_ptr + rn_bias_left * stride_bias, mask=rn_bias_left < N, other=0.0)
                bias_right = tl.load(bias_ptr + rn_bias_right * stride_bias, mask=rn_bias_right < N, other=0.0)
                bias_left_reshaped = tl.reshape(bias_left, (1, BLOCK_SIZE_N // 2))
                bias_right_reshaped = tl.reshape(bias_right, (1, BLOCK_SIZE_N // 2))
                if QUANTIZED:
                    bias_left_float = bias_left_reshaped.to(tl.float32)
                    bias_right_float = bias_right_reshaped.to(tl.float32)
                    acc00 += bias_left_float
                    acc01 += bias_right_float
                    acc10 += bias_left_float
                    acc11 += bias_right_float
                else:
                    acc00 += bias_left_reshaped
                    acc01 += bias_right_reshaped
                    acc10 += bias_left_reshaped
                    acc11 += bias_right_reshaped

            c00 = acc00.to(C.type.element_ty)
            c01 = acc01.to(C.type.element_ty)
            c10 = acc10.to(C.type.element_ty)
            c11 = acc11.to(C.type.element_ty)

            rm_top = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M // 2)) % M
            rm_bottom = (pid_m * BLOCK_SIZE_M + tl.arange(BLOCK_SIZE_M // 2, BLOCK_SIZE_M)) % M
            rn_left = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 2)) % N
            rn_right = (pid_n * BLOCK_SIZE_N + tl.arange(BLOCK_SIZE_N // 2, BLOCK_SIZE_N)) % N
            mask00 = (rm_top < M)[:, None] & (rn_left < N)[None, :]
            tl.store(C + rm_top[:, None] * stride_cm + rn_left[None, :] * stride_cn, c00, mask=mask00)
            mask01 = (rm_top < M)[:, None] & (rn_right < N)[None, :]
            tl.store(C + rm_top[:, None] * stride_cm + rn_right[None, :] * stride_cn, c01, mask=mask01)
            mask10 = (rm_bottom < M)[:, None] & (rn_left < N)[None, :]
            tl.store(C + rm_bottom[:, None] * stride_cm + rn_left[None, :] * stride_cn, c10, mask=mask10)
            mask11 = (rm_bottom < M)[:, None] & (rn_right < N)[None, :]
            tl.store(C + rm_bottom[:, None] * stride_cm + rn_right[None, :] * stride_cn, c11, mask=mask11)

            red += read_realtime() - tr0
            n_cons += 1

        start_iter = end_iter

    t_end = read_realtime()
    cu = get_cu_id()
    base = trace_ptr + pid * TRACE_SLOTS
    tl.store(base + 0, t_start)
    tl.store(base + 1, t_end)
    tl.store(base + 2, comp_full)
    tl.store(base + 3, comp_sk)
    tl.store(base + 4, red)
    tl.store(base + 5, spin)
    tl.store(base + 6, prod)
    tl.store(base + 7, n_full)
    tl.store(base + 8, n_sk)
    tl.store(base + 9, n_cons)
    tl.store(base + 10, n_prod)
    tl.store(base + 11, cu.to(tl.int64))
    tl.store(base + 12, pid.to(tl.int64))
