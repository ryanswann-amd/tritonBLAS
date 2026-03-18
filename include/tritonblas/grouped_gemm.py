import torch
import triton
import triton.language as tl
import math
import functools
from .internal.grouped_persistent_matmul import grouped_persistent_matmul
from .origami import GroupedGemmSelector, MatmulHeuristicResult
from .matmul import persistent_matmul_lt

current_device_index = torch.cuda.current_device()
current_device = torch.cuda.get_device_properties(current_device_index)
MAX_SMS = current_device.multi_processor_count

_torch_to_triton_dtype = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


@functools.lru_cache(maxsize=32)
def _cached_homogeneous_selector(m, n, k, a_dtype, b_dtype, c_dtype):
    return MatmulHeuristicResult(m, n, k, a_dtype, b_dtype, c_dtype, streamk=False)


@functools.lru_cache(maxsize=32)
def _cached_grouped_config(group_shapes, in_dtype, out_dtype, device_index):
    selector = GroupedGemmSelector(
        list(group_shapes), in_dtype, in_dtype, out_dtype,
        device_index=device_index,
    )
    return selector.get_config()


def _is_homogeneous(group_shapes):
    return all(s == group_shapes[0] for s in group_shapes)


def _homogeneous_per_group_dispatch(group_a, group_b, group_c, m, n, k, group_size):
    """Per-group cached dispatch for small group counts (G<4)."""
    selector = _cached_homogeneous_selector(
        m, n, k, group_a[0].dtype, group_b[0].dtype, group_c[0].dtype,
    )
    for i in range(group_size):
        persistent_matmul_lt(group_a[i], group_b[i], group_c[i], selector)


def _homogeneous_bmm_dispatch(group_a, group_b):
    """Fast path for homogeneous groups (G>=4): single batched GEMM launch."""
    A_batch = torch.stack(group_a)  # [G, M, K]
    B_batch = torch.stack(group_b)  # [G, K, N]
    C_batch = torch.bmm(A_batch, B_batch)  # [G, M, N]
    return list(C_batch.unbind(0))


_NUM_XCDS = 8
_GROUP_SIZE_M = int(math.ceil(math.sqrt(MAX_SMS / _NUM_XCDS)))


def _heterogeneous_dispatch(group_a, group_b, group_c, group_shapes, group_size,
                             BLK_M, BLK_N, BLK_K, triton_dtype):
    """Single-kernel dispatch for heterogeneous groups."""
    even_k = all(k % BLK_K == 0 for _, _, k in group_shapes)

    # Check if all tensors are row-major contiguous (inner stride = 1)
    contig = all(
        group_a[i].stride(1) == 1 and group_b[i].stride(1) == 1 and group_c[i].stride(1) == 1
        for i in range(group_size)
    )

    G = group_size
    all_ptrs = [0] * (3 * G)
    # Int32 metadata: [g_sizes (3*G) | g_lds (6*G if not contig) | gemm_offsets (G+1)]
    ofs_lds = 3 * G
    if contig:
        ofs_gemm = ofs_lds  # skip g_lds entirely
    else:
        ofs_gemm = 9 * G
    all_i32 = [0] * (ofs_gemm + G + 1)

    cumulative = 0
    ceil_m = math.ceil
    for i in range(G):
        m, n, k = group_shapes[i]
        a, b, c = group_a[i], group_b[i], group_c[i]
        all_ptrs[i] = a.data_ptr()
        all_ptrs[G + i] = b.data_ptr()
        all_ptrs[2 * G + i] = c.data_ptr()
        j = 3 * i
        all_i32[j] = m
        all_i32[j + 1] = n
        all_i32[j + 2] = k
        if not contig:
            j = ofs_lds + 6 * i
            all_i32[j] = a.stride(0)
            all_i32[j + 1] = a.stride(1)
            all_i32[j + 2] = b.stride(0)
            all_i32[j + 3] = b.stride(1)
            all_i32[j + 4] = c.stride(0)
            all_i32[j + 5] = c.stride(1)
        cumulative += ceil_m(m / BLK_M) * ceil_m(n / BLK_N)
        all_i32[ofs_gemm + i + 1] = cumulative

    d_ptrs = torch.tensor(all_ptrs, device="cuda", dtype=torch.int64)
    d_i32 = torch.tensor(all_i32, device="cuda", dtype=torch.int32)

    d_a_ptrs = d_ptrs[:G]
    d_b_ptrs = d_ptrs[G:2 * G]
    d_c_ptrs = d_ptrs[2 * G:]
    d_g_sizes = d_i32[:ofs_lds]
    if contig:
        d_g_lds = d_g_sizes  # unused when CONTIG=True, pass dummy
    else:
        d_g_lds = d_i32[ofs_lds:ofs_gemm]
    d_gemm_offsets = d_i32[ofs_gemm:]

    chunk_size = max(1, min(_GROUP_SIZE_M * _GROUP_SIZE_M, cumulative // _NUM_XCDS))

    grouped_persistent_matmul[(MAX_SMS,)](
        d_a_ptrs, d_b_ptrs, d_c_ptrs,
        d_g_sizes, d_gemm_offsets, d_g_lds,
        BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=_GROUP_SIZE_M,
        GROUP_COUNT=group_size,
        NUM_SMS=MAX_SMS, NUM_XCDS=_NUM_XCDS, CHUNK_SIZE=chunk_size,
        MATMUL_DTYPE=triton_dtype,
        EVEN_K=even_k,
        CONTIG=contig,
        num_stages=2, num_warps=8,
        waves_per_eu=0, matrix_instr_nonkdim=16, kpack=1,
    )


def grouped_gemm(
    group_a: list[torch.Tensor],
    group_b: list[torch.Tensor],
    group_c: list[torch.Tensor] = None,
    BLK_M: int = None,
    BLK_N: int = None,
    BLK_K: int = None,
):
    """Grouped GEMM with automatic dispatch optimization.

    Homogeneous groups: persistent_matmul per group (near-peak constexpr strides).
    Heterogeneous groups: single-kernel persistent dispatch.
    """
    group_size = len(group_a)
    assert group_size == len(group_b)

    in_dtype = group_a[0].dtype
    out_dtype = group_c[0].dtype if group_c is not None else in_dtype

    if group_c is None:
        group_c = []
        for i in range(group_size):
            group_c.append(torch.empty((group_a[i].shape[0], group_b[i].shape[1]),
                                       device="cuda", dtype=out_dtype))
    else:
        assert group_size == len(group_c)

    group_shapes = []
    for i in range(group_size):
        A, B = group_a[i], group_b[i]
        assert A.shape[1] == B.shape[0], f"Group {i}: incompatible A={A.shape}, B={B.shape}"
        group_shapes.append((A.shape[0], B.shape[1], A.shape[1]))

    if _is_homogeneous(group_shapes):
        m, n, k = group_shapes[0]
        if group_size >= 4:
            results = _homogeneous_bmm_dispatch(group_a, group_b)
            for i in range(group_size):
                group_c[i].copy_(results[i])
        else:
            _homogeneous_per_group_dispatch(group_a, group_b, group_c, m, n, k, group_size)
    else:
        if BLK_M is None or BLK_N is None or BLK_K is None:
            BLK_M, BLK_N, BLK_K = _cached_grouped_config(
                tuple(group_shapes), in_dtype, out_dtype, current_device_index,
            )

        triton_dtype = _torch_to_triton_dtype.get(in_dtype)
        if triton_dtype is None:
            raise ValueError(f"Unsupported dtype: {in_dtype}")

        _heterogeneous_dispatch(group_a, group_b, group_c, group_shapes,
                                group_size, BLK_M, BLK_N, BLK_K, triton_dtype)

    return group_c
