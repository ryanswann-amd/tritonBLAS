"""Unit tests for LDS (shared memory) estimation functions.

These functions are critical for preventing "out of resource: shared memory"
errors (issue #62). They estimate Triton's AMD LDS usage with
swizzled_shared / amd_rotating_shared encodings (no padding overhead).

The AMD LDS model (validated 35/35 against Triton 3.6.0+rocm7.2.0 on gfx942):
  ns == 1:  max(A_bytes, B_bytes)            — sequential inline alloc
  ns >= 2:  (ns - 1) * (A_bytes + B_bytes)   — software-pipelined buffers

Tests cover:
- estimate_triton_lds_bytes: AMD LDS model
- check_triton_lds_capacity: capacity check
- Architecture-dependent tests: use origami.get_hardware_for_device() to verify
  LDS capacity checks against real hardware (gfx942, gfx950, etc.)
- Compiled kernel validation: compare estimates against metadata.shared from
  actual Triton compiled kernels
"""

import pytest
import torch

from tritonblas.origami import (
    estimate_triton_lds_bytes,
    check_triton_lds_capacity,
)


def _get_hardware():
    """Get hardware from origami for current device. Returns hardware or None if unavailable."""
    if not torch.cuda.is_available():
        return None
    try:
        import origami
        device_id = torch.cuda.current_device()
        return origami.get_hardware_for_device(device_id)
    except Exception:
        return None


class TestEstimateTritonLdsBytes:
    """Test estimate_triton_lds_bytes: AMD LDS model (no padding)."""

    def test_num_stages_1_uses_max(self):
        """ns=1: LDS = max(A_bytes, B_bytes), no pipelining."""
        # Square tile: A == B, so max = either
        assert estimate_triton_lds_bytes(128, 128, 64, 2, 2, 1) == 128 * 64 * 2
        # Rectangular: A > B
        assert estimate_triton_lds_bytes(256, 64, 64, 2, 2, 1) == 256 * 64 * 2
        # Rectangular: B > A
        assert estimate_triton_lds_bytes(64, 256, 64, 2, 2, 1) == 64 * 256 * 2

    def test_num_stages_2_one_buffer_set(self):
        """ns=2: LDS = (A_bytes + B_bytes), one pipeline buffer set."""
        # 128x128x64 bf16: (128*64 + 64*128) * 2 = 32768
        assert estimate_triton_lds_bytes(128, 128, 64, 2, 2, 2) == 32768
        # 256x256x64 bf16: (256*64 + 64*256) * 2 = 65536
        assert estimate_triton_lds_bytes(256, 256, 64, 2, 2, 2) == 65536

    def test_num_stages_3_two_buffer_sets(self):
        """ns=3: LDS = 2 * (A_bytes + B_bytes), two pipeline buffer sets."""
        # 128x128x64 bf16: 2 * (128*64 + 64*128) * 2 = 65536
        assert estimate_triton_lds_bytes(128, 128, 64, 2, 2, 3) == 65536

    def test_num_stages_scaling_linear(self):
        """LDS scales as (ns-1) for ns >= 2."""
        bm, bn, bk = 128, 128, 64
        lds_2 = estimate_triton_lds_bytes(bm, bn, bk, 2, 2, 2)
        lds_3 = estimate_triton_lds_bytes(bm, bn, bk, 2, 2, 3)
        lds_4 = estimate_triton_lds_bytes(bm, bn, bk, 2, 2, 4)
        assert lds_3 == lds_2 * 2  # (3-1)/(2-1) = 2x
        assert lds_4 == lds_2 * 3  # (4-1)/(2-1) = 3x

    def test_rectangular_tiles(self):
        """Non-square tiles compute A and B sizes independently."""
        # 128x64x64 bf16 ns=2: (128*64 + 64*64) * 2 = 24576
        assert estimate_triton_lds_bytes(128, 64, 64, 2, 2, 2) == 24576
        # 256x128x64 bf16 ns=2: (256*64 + 64*128) * 2 = 49152
        assert estimate_triton_lds_bytes(256, 128, 64, 2, 2, 2) == 49152

    def test_known_configs_bf16(self):
        """Known configs validated against Triton metadata.shared on gfx942."""
        assert estimate_triton_lds_bytes(64, 64, 32, 2, 2, 2) == 8192
        assert estimate_triton_lds_bytes(128, 128, 64, 2, 2, 2) == 32768
        assert estimate_triton_lds_bytes(256, 256, 64, 2, 2, 2) == 65536
        assert estimate_triton_lds_bytes(128, 128, 64, 2, 2, 3) == 65536
        assert estimate_triton_lds_bytes(128, 256, 64, 2, 2, 2) == 49152

    def test_known_configs_f32(self):
        """Known configs validated against Triton metadata.shared on gfx942."""
        assert estimate_triton_lds_bytes(64, 64, 32, 4, 4, 2) == 16384
        assert estimate_triton_lds_bytes(128, 128, 32, 4, 4, 2) == 32768
        assert estimate_triton_lds_bytes(128, 128, 64, 4, 4, 2) == 65536

    def test_fp32_doubles_bf16(self):
        """FP32 (4 bytes) uses 2x LDS compared to bf16 (2 bytes) for same tile."""
        lds_bf16 = estimate_triton_lds_bytes(64, 64, 32, 2, 2, 2)
        lds_fp32 = estimate_triton_lds_bytes(64, 64, 32, 4, 4, 2)
        assert lds_fp32 == lds_bf16 * 2

    def test_mixed_ab_dtypes(self):
        """Mixed A/B byte sizes (e.g. f8 A with bf16 B)."""
        # A: 128*64*1 = 8192, B: 64*128*2 = 16384, total = 24576
        lds_mixed = estimate_triton_lds_bytes(128, 128, 64, 1, 2, 2)
        assert lds_mixed == 24576
        lds_uniform = estimate_triton_lds_bytes(128, 128, 64, 2, 2, 2)
        assert lds_mixed < lds_uniform

    def test_sub_byte_dtype(self):
        """Sub-byte types (e.g. f4 = 0.5 bytes) must produce non-zero LDS estimates."""
        lds = estimate_triton_lds_bytes(128, 128, 64, 0.5, 0.5, 2)
        assert lds > 0
        lds_bf16 = estimate_triton_lds_bytes(128, 128, 64, 2, 2, 2)
        assert lds == lds_bf16 / 4


class TestNS1FalseAttractorInvariant:
    """Regression guard for K-745: codifies the LDS-estimator branch behaviour
    that motivates the documented "NS=1 + larger BK" false attractor on the
    K-570 large-K square FP16/BF16 cohort (MI300X / gfx942).

    The NOTE in `estimate_triton_lds_bytes` documents that the ns==1 branch
    admits BK in {128,256} at the 256x256 fp16/bf16 tile that ns==2 statically
    rejects. A 22-tile x 18-shape paired CUDA-graph sweep on MI300X falsified
    the resulting hypothesis (best NS=1 tile per cell regresses -13.7%..-27.6%
    vs HEAD; the (256,256,128,NS=1) candidate is ~5x slower than HEAD).

    These assertions guard the branch arithmetic the NOTE relies on so the
    docstring cannot silently rot if the estimator math changes — if any of
    these flip, either the NOTE must be revised or the underlying false
    attractor has changed and the empirical sweep must be re-run.

    Pure arithmetic — no GPU required.
    """

    # gfx942 LDS capacity (bytes), matches hardware.lds_capacity on MI300X.
    GFX942_LDS = 65536
    # FP16 / BF16 element size in bytes.
    BYTES_FP16 = 2

    def test_brief_tile_fits_at_ns1(self):
        """The (BM=BN=256, BK=128) fp16/bf16 tile fits at NS=1 (max-branch).

        max(256*128*2, 128*256*2) = 65536 — exactly the gfx942 LDS budget.
        This is the LDS-budget arithmetic that makes the K-745 brief tile
        look attractive on paper.
        """
        lds = estimate_triton_lds_bytes(
            256, 256, 128, self.BYTES_FP16, self.BYTES_FP16, 1
        )
        assert lds == 65536
        assert lds <= self.GFX942_LDS

    def test_brief_tile_infeasible_at_ns2(self):
        """The same (256,256,128) tile is INFEASIBLE_LDS at NS=2.

        (2-1) * (256*128*2 + 128*256*2) = 131072 > 65536. This is why
        Triton's NS=2 default rejects the tile and why dropping NS to 1
        appears to "unlock" larger BK at the 256x256 geometry.
        """
        lds = estimate_triton_lds_bytes(
            256, 256, 128, self.BYTES_FP16, self.BYTES_FP16, 2
        )
        assert lds == 131072
        assert lds > self.GFX942_LDS

    def test_origami_pick_fits_at_ns2(self):
        """Origami's currently-shipped (256,256,64,NS=2) pick fits exactly.

        (2-1) * (256*64*2 + 64*256*2) = 65536 — exactly the gfx942 LDS budget.
        This is the HEAD pick on the M=N>=4096 sub-cohort that the NS=1 +
        larger-BK hypothesis was attempting to displace.
        """
        lds = estimate_triton_lds_bytes(
            256, 256, 64, self.BYTES_FP16, self.BYTES_FP16, 2
        )
        assert lds == 65536
        assert lds <= self.GFX942_LDS

    def test_ns1_halves_lds_vs_ns2_at_square_tile(self):
        """At square BM==BN tiles: NS=1 LDS is half of NS=2 LDS.

        max(A,B) = A (when A==B), and (ns-1)*(A+B) = A+B at ns=2. So the
        NS=1 branch halves the LDS budget vs NS=2 for square tiles — the
        "free" budget that motivated the K-745 hypothesis. Asserted across
        the square-tile geometries that bracket the K-570 cohort.
        """
        for bm_bn in (128, 256):
            for bk in (32, 64, 128, 256):
                ns1 = estimate_triton_lds_bytes(
                    bm_bn, bm_bn, bk, self.BYTES_FP16, self.BYTES_FP16, 1
                )
                ns2 = estimate_triton_lds_bytes(
                    bm_bn, bm_bn, bk, self.BYTES_FP16, self.BYTES_FP16, 2
                )
                # NS=1 = max(A,B) = A; NS=2 = (ns-1)*(A+B) = A+B = 2A
                assert ns2 == 2 * ns1, (
                    f"({bm_bn},{bm_bn},{bk},fp16): "
                    f"NS=2={ns2} should be 2*NS=1={ns1}"
                )

    def test_ns1_admits_bk128_at_256x256_that_ns2_rejects(self):
        """The branch-asymmetry the K-745 NOTE codifies: at (BM=BN=256,BK=128)
        on gfx942, the NS=1 branch admits the tile where the NS=2 branch
        rejects it. This is the static LDS-budget invariant; the run-time
        consequence (~5x slowdown) is empirically falsified per the NOTE and
        not re-asserted here.
        """
        bk = 128
        ns1 = estimate_triton_lds_bytes(
            256, 256, bk, self.BYTES_FP16, self.BYTES_FP16, 1
        )
        ns2 = estimate_triton_lds_bytes(
            256, 256, bk, self.BYTES_FP16, self.BYTES_FP16, 2
        )
        assert ns1 <= self.GFX942_LDS, (
            f"NS=1 (256,256,{bk},fp16) should fit gfx942 LDS budget"
        )
        assert ns2 > self.GFX942_LDS, (
            f"NS=2 (256,256,{bk},fp16) should exceed gfx942 LDS budget"
        )
        assert check_triton_lds_capacity(
            256, 256, bk, self.BYTES_FP16, self.BYTES_FP16,
            self.GFX942_LDS, 1,
        ) is True
        assert check_triton_lds_capacity(
            256, 256, bk, self.BYTES_FP16, self.BYTES_FP16,
            self.GFX942_LDS, 2,
        ) is False

    def test_bk256_at_256x256_infeasible_at_both_ns1_and_ns2(self):
        """Sanity bound on the NOTE's "BK in {128,256} admitted at NS=1"
        wording: for the 256x256 fp16/bf16 tile, BK=256 is infeasible at
        NS=1 too (max(256*256*2, 256*256*2) = 131072 > 65536). So the NOTE's
        admitted-tile set at NS=1 is {BK=128} only on the 256x256 tile, and
        the K-745 brief's "BK=128 or 256" wording is bounded by this LDS
        constraint, not by branch arithmetic.
        """
        ns1 = estimate_triton_lds_bytes(
            256, 256, 256, self.BYTES_FP16, self.BYTES_FP16, 1
        )
        assert ns1 == 131072
        assert ns1 > self.GFX942_LDS


class TestCheckTritonLdsCapacity:
    """Test check_triton_lds_capacity: capacity check."""

    def test_fits_within_capacity(self):
        """Config that fits should return True."""
        # 64x64x32 bf16 ns=2 uses 8192 bytes
        assert check_triton_lds_capacity(64, 64, 32, 2, 2, 65536, 2) is True

    def test_exceeds_capacity(self):
        """Config that exceeds capacity should return False."""
        # 256x256x64 f32 ns=2: (256*64 + 64*256) * 4 = 131072 > 65536
        assert check_triton_lds_capacity(256, 256, 64, 4, 4, 65536, 2) is False

    def test_256x256x64_bf16_fits_64kb(self):
        """256x256x64 bf16 ns=2 uses exactly 65536 bytes — fits 64KB."""
        assert check_triton_lds_capacity(256, 256, 64, 2, 2, 65536, 2) is True

    def test_128x128x64_bf16_fits_64kb(self):
        """128x128x64 bf16 ns=2 uses 32768 bytes — fits 64KB easily."""
        assert check_triton_lds_capacity(128, 128, 64, 2, 2, 65536, 2) is True

    def test_num_stages_affects_capacity(self):
        """Higher num_stages can push config over capacity."""
        # 128x128x64 bf16: ns=2 uses 32768, ns=3 uses 65536, ns=4 uses 98304
        fits_2 = check_triton_lds_capacity(128, 128, 64, 2, 2, 65536, 2)
        fits_3 = check_triton_lds_capacity(128, 128, 64, 2, 2, 65536, 3)
        fits_4 = check_triton_lds_capacity(128, 128, 64, 2, 2, 65536, 4)
        assert fits_2 is True
        assert fits_3 is True   # exactly 65536
        assert fits_4 is False  # 98304 > 65536

    def test_sub_byte_dtype_not_free(self):
        """Sub-byte types (0.5 bytes/elem) must not pass capacity=0; LDS > 0."""
        assert check_triton_lds_capacity(128, 128, 64, 0.5, 0.5, 0, 2) is False


# N_CU -> expected LDS capacity (from origami common.hpp / hardware)
# gfx950: 256 CUs, 160KB; gfx942/gfx90a: 64KB
N_CU_TO_LDS = {256: 163840, 304: 65536, 80: 65536, 64: 65536, 228: 65536, 104: 65536}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLdsArchitectureDependent:
    """Architecture-dependent LDS tests using origami.get_hardware_for_device().

    These tests verify that check_triton_lds_capacity correctly filters configs
    based on the actual GPU's LDS capacity. Uses hardware.N_CU for arch detection.
    """

    def test_hardware_lds_capacity_matches_expected(self):
        """Detected hardware lds_capacity should match known N_CU values."""
        hardware = _get_hardware()
        if hardware is None:
            pytest.skip("Could not get hardware")
        n_cu = hardware.N_CU
        if n_cu not in N_CU_TO_LDS:
            pytest.skip(f"N_CU={n_cu} not in known LDS capacity map")
        expected = N_CU_TO_LDS[n_cu]
        assert hardware.lds_capacity == expected, (
            f"N_CU={n_cu}: expected lds_capacity={expected}, got {hardware.lds_capacity}"
        )

    def test_small_config_fits_on_all_archs(self):
        """64x64x32 bf16 should fit within LDS on all supported archs."""
        hardware = _get_hardware()
        if hardware is None:
            pytest.skip("Could not get hardware")
        lds_cap = hardware.lds_capacity
        fits = check_triton_lds_capacity(64, 64, 32, 2, 2, lds_cap, 2)
        assert fits, f"64x64x32 should fit in {lds_cap} bytes (N_CU={hardware.N_CU})"

    def test_256x256x64_bf16_fits_on_all_archs(self):
        """256x256x64 bf16 ns=2 uses 65536 bytes — fits on all archs (64KB+)."""
        hardware = _get_hardware()
        if hardware is None:
            pytest.skip("Could not get hardware")
        lds_cap = hardware.lds_capacity
        usage = estimate_triton_lds_bytes(256, 256, 64, 2, 2, 2)
        fits = check_triton_lds_capacity(256, 256, 64, 2, 2, lds_cap, 2)
        assert fits, (
            f"256x256x64 bf16 ns=2 uses {usage} bytes, capacity={lds_cap} (N_CU={hardware.N_CU})"
        )

    def test_256x256x64_f32_exceeds_on_64kb_archs(self):
        """256x256x64 f32 ns=2 uses 131072 bytes — exceeds 64KB archs."""
        hardware = _get_hardware()
        if hardware is None:
            pytest.skip("Could not get hardware")
        lds_cap = hardware.lds_capacity
        if lds_cap != 65536:
            pytest.skip(f"Requires 64KB LDS, got {lds_cap} (N_CU={hardware.N_CU})")
        fits = check_triton_lds_capacity(256, 256, 64, 4, 4, lds_cap, 2)
        assert not fits, "256x256x64 f32 ns=2 should NOT fit in 64KB"

    def test_selector_filters_by_lds_on_current_arch(self):
        """OrigamiMatmulSelector should not select configs that exceed LDS."""
        hardware = _get_hardware()
        if hardware is None:
            pytest.skip("Could not get hardware")
        from tritonblas import OrigamiMatmulSelector
        m, n, k = 4096, 4096, 4096
        dtype = torch.bfloat16
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        selector = OrigamiMatmulSelector(m, n, k, dtype, dtype, dtype, device)
        usage = estimate_triton_lds_bytes(
            selector.block_m, selector.block_n, selector.block_k,
            2, 2, selector.num_stages
        )
        assert usage <= hardware.lds_capacity, (
            f"Selector chose {selector.block_m}x{selector.block_n}x{selector.block_k} "
            f"(LDS={usage}) which exceeds {hardware.lds_capacity} (N_CU={hardware.N_CU})"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLdsVsCompiledKernel:
    """Validate LDS estimates against metadata.shared from compiled Triton kernels.

    This is the ground truth: Triton's compiler reports exact LDS allocation.
    Our model must match exactly.
    """

    @staticmethod
    def _compile_and_get_lds(block_m, block_n, block_k, num_stages, dtype):
        """Compile a Triton matmul kernel and return metadata.shared."""
        import triton
        import triton.language as tl

        # Import the kernel from the validation script
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "lds_validation",
            os.path.join(os.path.dirname(__file__), "..", "tools", "lds_model_validation.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        kernel = mod.matmul_bf16 if dtype == torch.bfloat16 else mod.matmul_f32
        return mod.get_lds(kernel, block_m, block_n, block_k, num_stages, dtype)

    @pytest.mark.parametrize("block_m,block_n,block_k,num_stages", [
        (64, 64, 32, 1),
        (64, 64, 32, 2),
        (128, 128, 64, 1),
        (128, 128, 64, 2),
        (128, 128, 64, 3),
        (256, 256, 64, 1),
        (256, 256, 64, 2),
        (128, 64, 64, 2),
        (64, 128, 64, 2),
        (256, 128, 64, 2),
        (128, 256, 64, 2),
    ])
    def test_bf16_estimate_matches_compiled(self, block_m, block_n, block_k, num_stages):
        """estimate_triton_lds_bytes must match Triton metadata.shared for bf16."""
        estimated = estimate_triton_lds_bytes(block_m, block_n, block_k, 2, 2, num_stages)
        actual = self._compile_and_get_lds(block_m, block_n, block_k, num_stages, torch.bfloat16)
        assert estimated == actual, (
            f"{block_m}x{block_n}x{block_k} ns={num_stages} bf16: "
            f"estimated={estimated}, compiled={actual}"
        )

    @pytest.mark.parametrize("block_m,block_n,block_k,num_stages", [
        (64, 64, 32, 1),
        (64, 64, 32, 2),
        (128, 128, 32, 2),
        (128, 128, 64, 1),
        (128, 128, 64, 2),
    ])
    def test_f32_estimate_matches_compiled(self, block_m, block_n, block_k, num_stages):
        """estimate_triton_lds_bytes must match Triton metadata.shared for f32."""
        estimated = estimate_triton_lds_bytes(block_m, block_n, block_k, 4, 4, num_stages)
        actual = self._compile_and_get_lds(block_m, block_n, block_k, num_stages, torch.float32)
        assert estimated == actual, (
            f"{block_m}x{block_n}x{block_k} ns={num_stages} f32: "
            f"estimated={estimated}, compiled={actual}"
        )
