import math
import pytest
import torch
import tritonblas


@pytest.mark.parametrize(
    "z, m, n, k",
    [
        (2, 256, 256, 256),
        (4, 512, 512, 512),
        (8, 256, 256, 512),
        (16, 128, 128, 512),
        (4, 1024, 1024, 1024),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
)
def test_bmm_correctness(z, m, n, k, dtype):
    """Single-launch batched matmul matches torch.bmm within numerical bounds."""
    torch.manual_seed(0)
    a = torch.randn(z, m, k, device="cuda", dtype=dtype)
    b = torch.randn(z, k, n, device="cuda", dtype=dtype)

    out = tritonblas.bmm(a, b)
    ref = torch.bmm(a.float(), b.float()).to(dtype)

    eps = torch.finfo(dtype).eps
    bound = 8 * math.sqrt(k) * eps
    rel = (out.float() - ref.float()).norm() / (ref.float().norm() + 1e-12)
    assert rel.item() <= bound, (
        f"bmm({z},{m},{n},{k},{dtype}) rel_err {rel.item():.3e} "
        f"exceeds bound {bound:.3e}"
    )


@pytest.mark.parametrize("z, m, n, k", [(4, 256, 256, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_rank3_dispatches_to_bmm(z, m, n, k, dtype):
    """tritonblas.matmul(rank-3, rank-3) routes through the batched entrypoint."""
    torch.manual_seed(0)
    a = torch.randn(z, m, k, device="cuda", dtype=dtype)
    b = torch.randn(z, k, n, device="cuda", dtype=dtype)

    out = tritonblas.matmul(a, b)
    ref = torch.bmm(a.float(), b.float()).to(dtype)

    eps = torch.finfo(dtype).eps
    bound = 8 * math.sqrt(k) * eps
    rel = (out.float() - ref.float()).norm() / (ref.float().norm() + 1e-12)
    assert rel.item() <= bound


def test_bmm_out_argument():
    """The out= argument is honoured."""
    z, m, n, k = 4, 256, 256, 256
    a = torch.randn(z, m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(z, k, n, device="cuda", dtype=torch.float16)
    out = torch.empty(z, m, n, device="cuda", dtype=torch.float16)
    rv = tritonblas.bmm(a, b, out=out)
    assert rv is None  # out= path returns None
    ref = torch.bmm(a.float(), b.float()).to(torch.float16)
    rel = (out.float() - ref.float()).norm() / (ref.float().norm() + 1e-12)
    assert rel.item() <= 8 * math.sqrt(k) * torch.finfo(torch.float16).eps
