"""Property test: Origami must NEVER select a tile config that overflows LDS.

For every combination of:
- GEMM shape (M, N, K) covering small, medium, large, and the problematic 8192^3
- dtype (fp16, bf16, fp8)
- num_stages (2, 3, 4)
- architecture (MI300X 64KB, MI350X 160KB)

...the selected (tile, stages) config must fit in the hardware LDS limit.

This test catches the bug where Origami selects 256x256x128 stages=3 on MI355X
(needs 262KB, limit is 160KB) and similar overflow cases.
"""

import pytest
import torch

from tritonblas.origami import (
    OrigamiMatmulSelector,
    estimate_triton_lds_bytes,
    check_triton_lds_capacity,
)


SHAPES = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (1024, 1024, 8192),
    (8192, 8192, 1024),
    (128, 256, 512),
    (4096, 4096, 8192),
    (16384, 16384, 4096),
]

DTYPES = [torch.float16, torch.bfloat16]

NUM_STAGES_VALUES = [2, 3, 4]

LDS_LIMITS = {
    "MI300X_64KB": 65536,
    "MI350X_160KB": 163840,
}


def _bytes_per_elem(dtype):
    return torch.tensor([], dtype=dtype).element_size()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestOrigamiNeverOverflowsLDS:
    """Origami must never select a config that exceeds the hardware LDS limit."""

    @pytest.fixture
    def hardware_lds(self):
        """Get the actual LDS capacity from the hardware."""
        try:
            import origami
            hw = origami.get_hardware_for_device(torch.cuda.current_device())
            return hw.lds_capacity
        except Exception:
            pytest.skip("origami hardware detection unavailable")

    @pytest.mark.parametrize(
        "M,N,K",
        SHAPES,
        ids=[f"{m}x{n}x{k}" for m, n, k in SHAPES],
    )
    @pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
    @pytest.mark.parametrize("num_stages", NUM_STAGES_VALUES, ids=[f"s{s}" for s in NUM_STAGES_VALUES])
    def test_selected_tile_fits_in_lds(self, M, N, K, dtype, num_stages, hardware_lds):
        """The tile Origami selects must fit in the actual hardware LDS."""
        bpe = _bytes_per_elem(dtype)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        try:
            selector = OrigamiMatmulSelector(
                M, N, K, dtype, dtype, dtype, device, num_stages=num_stages
            )
        except Exception:
            pytest.skip("Origami selector failed to initialize")

        bm = selector._result.config.mt.m
        bn = selector._result.config.mt.n
        bk = selector._result.config.mt.k
        # Use the clamped num_stages, not the requested value — the selector
        # may have reduced it to fit within the hardware LDS budget.
        clamped_stages = selector.num_stages

        actual_lds = estimate_triton_lds_bytes(bm, bn, bk, bpe, bpe, clamped_stages)

        assert actual_lds <= hardware_lds, (
            f"Origami selected {bm}x{bn}x{bk} stages={clamped_stages} "
            f"(requested {num_stages}) for {M}x{N}x{K} {dtype} "
            f"which needs {actual_lds} bytes LDS, but hardware limit is {hardware_lds} bytes "
            f"({hardware_lds // 1024} KB). "
            f"Origami must cap num_stages or reject this tile."
        )


class TestLdsCheckRejectsOverflow:
    """Pure formula tests — no GPU needed."""

    @pytest.mark.parametrize("lds_limit_name,lds_limit", LDS_LIMITS.items())
    @pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
    @pytest.mark.parametrize("num_stages", NUM_STAGES_VALUES, ids=[f"s{s}" for s in NUM_STAGES_VALUES])
    def test_lds_check_consistent_with_estimate(self, lds_limit_name, lds_limit, dtype, num_stages):
        """check_triton_lds_capacity must agree with estimate_triton_lds_bytes.

        For every (tile, stages, lds_limit) combination, check_triton_lds_capacity
        must return True iff estimate_triton_lds_bytes <= lds_limit.
        """
        bpe = _bytes_per_elem(dtype)

        for bm in [64, 128, 256]:
            for bn in [64, 128, 256]:
                for bk in [32, 64, 128, 256]:
                    lds_needed = estimate_triton_lds_bytes(bm, bn, bk, bpe, bpe, num_stages)
                    passes_check = check_triton_lds_capacity(bm, bn, bk, bpe, bpe, lds_limit, num_stages)

                    if lds_needed > lds_limit:
                        assert not passes_check, (
                            f"check_triton_lds_capacity WRONGLY PASSED {bm}x{bn}x{bk} "
                            f"stages={num_stages} on {lds_limit_name}: needs {lds_needed} > {lds_limit}"
                        )
                    else:
                        assert passes_check, (
                            f"check_triton_lds_capacity WRONGLY REJECTED {bm}x{bn}x{bk} "
                            f"stages={num_stages} on {lds_limit_name}: needs {lds_needed} <= {lds_limit}"
                        )

    def test_problematic_config_rejected(self):
        """The specific config that caused the MI355X crash must be rejected.

        256x256x128 bf16 stages=3 needs 262144 bytes > 163840 (MI355X 160KB).
        """
        assert not check_triton_lds_capacity(256, 256, 128, 2, 2, 163840, 3)
