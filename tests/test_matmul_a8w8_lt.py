import pytest
import torch  # type: ignore
import tritonblas  # type: ignore
from tritonblas.utils import generate_matmul_inputs, get_fp8_dtypes  # type: ignore


def run_torch(a, b, a_scale, b_scale, bias=None, dtype=torch.bfloat16):
    """Reference computation matching kernel behavior: float32 accumulation with scale application."""
    # 1. Matrix multiplication in float32 (like kernel's tl.dot accumulation)
    acc = torch.matmul(a.to(torch.float32), b.to(torch.float32))

    if a_scale is not None and b_scale is not None:
        # Scales from generate_matmul_inputs are 1D: (M,) and (N,)
        # Apply scales: (M, 1) * (1, N) -> (M, N)
        scale = a_scale[:, None] * b_scale[None, :]
        acc = acc * scale  # Keep in float32

    if bias is not None:
        # 4. Add bias in float32 (like kernel: acc + bias_float[:, None])
        acc = acc + bias.to(torch.float32)

    # 5. Convert to output dtype at the very end (like kernel: c = acc.to(C.type.element_ty))
    if "float8" in str(dtype):
        dtype_max = torch.finfo(dtype).max
        acc = torch.clamp(acc, -dtype_max, dtype_max)
    elif dtype == torch.int8:
        # INT8 has range [-128, 127], but we use symmetric range [-127, 127] like the kernel
        dtype_max = 127.0
        acc = torch.clamp(acc, -dtype_max, dtype_max)

    return acc.to(dtype)


def run_triton(a, b, a_scale, b_scale, bias=None, dtype=torch.bfloat16, c=None):
    # Helper function that matches the actual API signature
    # Note: matmul_a8w8 creates the selector internally
    if c is None:
        c = torch.zeros((a.shape[0], b.shape[1]), device="cuda", dtype=dtype)
    return tritonblas.matmul_a8w8(a, b, a_scale, b_scale, c, enable_streamk=False)


def _get_fp8_e4m3_dtype():
    """Get the architecture-correct e4m3 FP8 dtype."""
    _, e4m3 = get_fp8_dtypes()
    return e4m3


def _get_fp8_e5m2_dtype():
    """Get the architecture-correct e5m2 FP8 dtype."""
    e5m2, _ = get_fp8_dtypes()
    return e5m2


@pytest.mark.parametrize(
    "m, n, k",
    [
        (8192, 8192, 8192),  # Large
        (4096, 4096, 4096),  # Medium-large
        (1024, 1024, 1024),  # Medium
        (512, 512, 512),     # Small
        (256, 256, 256),     # Very small
    ],
)
@pytest.mark.parametrize(
    "transA, transB",
    [
        ("T", "T"),  # A^T @ B^T
    ],
)
@pytest.mark.parametrize(
    "enable_streamk",
    [
        False,
    ],
)
def test_matmul_a8w8_fp8_e4m3(m, n, k, transA, transB, enable_streamk):
    """Test quantized FP8 e4m3 matmul with architecture-correct dtypes."""
    in_dtype = _get_fp8_e4m3_dtype()
    out_dtype = torch.bfloat16  # Use bf16 output for easier correctness checking
    init_type = "randn"

    inputs = generate_matmul_inputs(m, n, k, in_dtype, out_dtype, transA, transB, init_type)

    selector = tritonblas.OrigamiMatmulSelector(
        m, n, k, inputs.A.dtype, inputs.B.dtype, inputs.C.dtype, inputs.A.device,
        streamk=enable_streamk,
    )
    config = tritonblas.matmul_preamble(selector)
    tritonblas.matmul_a8w8_lt(
        inputs.A, inputs.B, inputs.scaleA, inputs.scaleB, inputs.C, selector, config,
        enable_streamk,
    )

    # Check correctness using reference computation
    torch_c = run_torch(
        inputs.A, inputs.B, inputs.scaleA, inputs.scaleB, bias=None, dtype=out_dtype
    )

    # Relaxed tolerance: FP8 quantization introduces error from limited precision
    torch.testing.assert_close(
        inputs.C.to(torch.float32), torch_c.to(torch.float32), atol=1.0, rtol=0.15
    )


@pytest.mark.parametrize(
    "m, n, k",
    [
        (8192, 8192, 8192),
        (4096, 4096, 4096),
        (1024, 1024, 1024),
        (512, 512, 512),
        (256, 256, 256),
    ],
)
@pytest.mark.parametrize(
    "transA, transB",
    [
        ("T", "T"),
    ],
)
@pytest.mark.parametrize(
    "enable_streamk",
    [
        False,
    ],
)
def test_matmul_a8w8_fp8_e5m2(m, n, k, transA, transB, enable_streamk):
    """Test quantized FP8 e5m2 matmul with architecture-correct dtypes."""
    in_dtype = _get_fp8_e5m2_dtype()
    out_dtype = torch.bfloat16
    init_type = "randn"

    inputs = generate_matmul_inputs(m, n, k, in_dtype, out_dtype, transA, transB, init_type)

    selector = tritonblas.OrigamiMatmulSelector(
        m, n, k, inputs.A.dtype, inputs.B.dtype, inputs.C.dtype, inputs.A.device,
        streamk=enable_streamk,
    )
    config = tritonblas.matmul_preamble(selector)
    tritonblas.matmul_a8w8_lt(
        inputs.A, inputs.B, inputs.scaleA, inputs.scaleB, inputs.C, selector, config,
        enable_streamk,
    )

    torch_c = run_torch(
        inputs.A, inputs.B, inputs.scaleA, inputs.scaleB, bias=None, dtype=out_dtype
    )

    torch.testing.assert_close(
        inputs.C.to(torch.float32), torch_c.to(torch.float32), atol=1.0, rtol=0.15
    )
