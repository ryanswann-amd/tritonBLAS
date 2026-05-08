"""K-811 split-K=N GEMM kernel with FP32 atomic-add reduction.

Origin: K-795 prototype (split_k_kernel.py, 117 LOC) verbatim, integrated
as a routed override behind a narrow shape+dtype gate in matmul.py.

Targets the M=N=2048 × K∈{4096,8192,16384} × {fp16,bf16} sub-band where
rocprofv2 evidence (K-795 §1) showed hipBLASLt's GSUAMBSK kernel
dispatches GSU=2 with atomic-multi-buffer-single-kernel reduction.

Triton constraint: tl.arange must be power-of-2; we use BLOCK_N=128 instead
of hipBLASLt's BLOCK_N=112. With BLOCK_M=128 BLOCK_K=64 SPLIT_K=2 the grid
is 16x16x2=512 work-groups for M=N=2048.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _split_k_gemm_kernel(
    A, B, C_fp32,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    rn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)

    K_per_shard = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * K_per_shard
    k_end = tl.minimum(k_start + K_per_shard, K)

    rk = tl.arange(0, BLOCK_K)
    A_ptr = A + (rm[:, None] * stride_am + (k_start + rk)[None, :] * stride_ak)
    B_ptr = B + ((k_start + rk)[:, None] * stride_bk + rn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_iters = tl.cdiv(k_end - k_start, BLOCK_K)
    for i in range(0, n_iters):
        k_off = i * BLOCK_K
        a = tl.load(A_ptr + k_off * stride_ak,
                    mask=(k_start + k_off + rk[None, :]) < k_end, other=0.0)
        b = tl.load(B_ptr + k_off * stride_bk,
                    mask=(k_start + k_off + rk[:, None]) < k_end, other=0.0)
        acc += tl.dot(a, b)

    c_off = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(C_fp32 + c_off, acc, mask=mask)
    else:
        tl.atomic_add(C_fp32 + c_off, acc, mask=mask, sem='relaxed')


def split_k_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                   split_k: int = 2,
                   block_m: int = 128,
                   block_n: int = 128,
                   block_k: int = 64,
                   group_m: int = 8,
                   num_warps: int = 8,
                   num_stages: int = 2,
                   waves_per_eu: int = 0,
                   matrix_instr_nonkdim: int = 16,
                   kpack: int = 2):
    """Split-K GEMM: c = a @ b, FP32 accumulator, atomic-add reduction.

    The output is written into ``c`` (which must already be allocated with
    ``a.dtype`` and shape (M, N)). Internally allocates an FP32 partial-sum
    buffer; final downcast is performed with a copy. This matches K-795's
    correctness-checked prototype.
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"K mismatch: {K} vs {K2}"
    assert a.is_contiguous() and b.is_contiguous()
    assert c.shape == (M, N), f"c shape {c.shape} != (M, N) = ({M}, {N})"

    C_fp32 = torch.zeros((M, N), dtype=torch.float32, device=a.device)

    grid_m = triton.cdiv(M, block_m)
    grid_n = triton.cdiv(N, block_n)
    grid = (grid_m * grid_n, split_k)

    _split_k_gemm_kernel[grid](
        a, b, C_fp32,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        C_fp32.stride(0), C_fp32.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        SPLIT_K=split_k, GROUP_M=group_m,
        num_warps=num_warps, num_stages=num_stages,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=matrix_instr_nonkdim,
        kpack=kpack,
    )
    c.copy_(C_fp32.to(c.dtype))
    return c
