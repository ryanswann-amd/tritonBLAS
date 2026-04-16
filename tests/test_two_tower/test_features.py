"""Unit tests for two_tower_triton.features module.

Tests the GEMM (query) and kernel (document) feature engineering
pipelines used by the Two Tower embedding model.
"""

import numpy as np
import pandas as pd
import pytest

from two_tower_triton.features import (
    gemm_preprocess,
    kernel_preprocess,
    get_continuous_gemm_cols,
    get_continuous_kernel_cols,
    get_categorical_kernel_cols,
    GPU_SPECS,
    DTYPE_BYTES,
)


# ---------------------------------------------------------------------------
# gemm_preprocess tests (query tower)
# ---------------------------------------------------------------------------


class TestGemmPreprocess:
    """Tests for GEMM problem feature engineering."""

    def test_basic_output_shape(self):
        """Output has same number of rows as input."""
        df = pd.DataFrame({"M": [4096, 2048], "N": [4096, 1024], "K": [512, 2048]})
        result = gemm_preprocess(df)
        assert len(result) == 2

    def test_no_nans(self):
        """Features should never contain NaN values."""
        df = pd.DataFrame({"M": [4096, 2048, 1], "N": [4096, 1024, 1], "K": [512, 2048, 1]})
        result = gemm_preprocess(df)
        assert not result.isnull().any().any(), f"NaN found in columns: {result.columns[result.isnull().any()].tolist()}"

    def test_log_dims_present(self):
        """Core log dimension features must exist."""
        df = pd.DataFrame({"M": [1024], "N": [512], "K": [256]})
        result = gemm_preprocess(df)
        for col in ["log_m", "log_n", "log_k"]:
            assert col in result.columns, f"Missing expected column: {col}"

    def test_log_dims_values(self):
        """Log2 dimension values should be correct."""
        df = pd.DataFrame({"M": [1024], "N": [512], "K": [256]})
        result = gemm_preprocess(df)
        assert np.isclose(result["log_m"].iloc[0], 10.0)
        assert np.isclose(result["log_n"].iloc[0], 9.0)
        assert np.isclose(result["log_k"].iloc[0], 8.0)

    def test_arithmetic_intensity_present(self):
        """Arithmetic intensity feature must be computed."""
        df = pd.DataFrame({"M": [4096], "N": [4096], "K": [4096]})
        result = gemm_preprocess(df)
        assert "arithmetic_intensity" in result.columns
        assert result["arithmetic_intensity"].iloc[0] > 0

    def test_arithmetic_intensity_formula(self):
        """AI = 2*M*N*K / ((M*K + K*N + M*N) * elem_size)."""
        M, N, K = 1024, 1024, 1024
        df = pd.DataFrame({"M": [M], "N": [N], "K": [K]})
        result = gemm_preprocess(df, dtype="bf16")
        elem = 2  # bf16
        expected_ai = (2.0 * M * N * K) / ((M * K + K * N + M * N) * elem)
        assert np.isclose(result["arithmetic_intensity"].iloc[0], expected_ai, rtol=1e-6)

    def test_roofline_flags(self):
        """Compute/memory bound flags should sum to 1."""
        df = pd.DataFrame({"M": [4096], "N": [4096], "K": [4096]})
        result = gemm_preprocess(df)
        assert "is_compute_bound" in result.columns
        assert "is_memory_bound" in result.columns
        cb = result["is_compute_bound"].iloc[0]
        mb = result["is_memory_bound"].iloc[0]
        assert np.isclose(cb + mb, 1.0), f"Bound flags don't sum to 1: {cb} + {mb}"

    def test_shape_flags_square(self):
        """Square matrix should have is_square=1, is_tall=0, is_wide=0."""
        df = pd.DataFrame({"M": [256], "N": [256], "K": [128]})
        result = gemm_preprocess(df)
        assert result["is_square"].iloc[0] == 1.0
        assert result["is_tall"].iloc[0] == 0.0
        assert result["is_wide"].iloc[0] == 0.0

    def test_shape_flags_tall(self):
        """Tall matrix (M > N) should have is_tall=1."""
        df = pd.DataFrame({"M": [4096], "N": [128], "K": [256]})
        result = gemm_preprocess(df)
        assert result["is_tall"].iloc[0] == 1.0
        assert result["is_wide"].iloc[0] == 0.0

    def test_shape_flags_wide(self):
        """Wide matrix (N > M) should have is_wide=1."""
        df = pd.DataFrame({"M": [128], "N": [4096], "K": [256]})
        result = gemm_preprocess(df)
        assert result["is_wide"].iloc[0] == 1.0
        assert result["is_tall"].iloc[0] == 0.0

    def test_power_of_2_detection(self):
        """Power-of-2 flags should be correct."""
        df = pd.DataFrame({"M": [1024], "N": [768], "K": [256]})
        result = gemm_preprocess(df)
        assert result["m_is_po2"].iloc[0] == 1.0
        assert result["n_is_po2"].iloc[0] == 0.0  # 768 is not power of 2
        assert result["k_is_po2"].iloc[0] == 1.0

    def test_alignment_flags(self):
        """Alignment features should detect 128-alignment correctly."""
        df = pd.DataFrame({"M": [256], "N": [192], "K": [64]})
        result = gemm_preprocess(df)
        assert result["m_align_128"].iloc[0] == 1.0  # 256 % 128 == 0
        assert result["n_align_128"].iloc[0] == 0.0  # 192 % 128 != 0
        assert result["k_align_8"].iloc[0] == 1.0     # 64 % 8 == 0

    def test_batch_count_default(self):
        """Without batch_count column, should default to 1."""
        df = pd.DataFrame({"M": [1024], "N": [1024], "K": [1024]})
        result = gemm_preprocess(df)
        assert result["log_batch"].iloc[0] == 0.0  # log2(1) = 0

    def test_batch_count_explicit(self):
        """Explicit batch_count should be reflected in features."""
        df = pd.DataFrame({"M": [1024], "N": [1024], "K": [1024], "batch_count": [8]})
        result = gemm_preprocess(df)
        assert np.isclose(result["log_batch"].iloc[0], 3.0)  # log2(8) = 3

    def test_different_gpu_archs(self):
        """Should work with both mi300x and mi350x."""
        df = pd.DataFrame({"M": [1024], "N": [1024], "K": [1024]})
        r1 = gemm_preprocess(df, gpu_arch="mi300x")
        r2 = gemm_preprocess(df, gpu_arch="mi350x")
        # Same dims but different hw → different roofline features
        assert r1.shape == r2.shape
        # Memory bandwidth differs, so cache pressure should differ
        assert not np.isclose(
            r1["log_bandwidth_pressure"].iloc[0],
            r2["log_bandwidth_pressure"].iloc[0],
        )

    def test_gemv_detection(self):
        """M=1 should trigger is_gemv flag."""
        df = pd.DataFrame({"M": [1], "N": [4096], "K": [1024]})
        result = gemm_preprocess(df)
        assert result["is_gemv"].iloc[0] == 1.0

    def test_wastage_features(self):
        """Wastage features should exist for standard tile sizes."""
        df = pd.DataFrame({"M": [1024], "N": [1024], "K": [1024]})
        result = gemm_preprocess(df)
        for tile in [32, 64, 128, 192, 256]:
            col = f"wastage_{tile}"
            assert col in result.columns, f"Missing wastage column: {col}"

    def test_wastage_zero_for_aligned(self):
        """Wastage should be 0 when dims are multiples of tile size."""
        df = pd.DataFrame({"M": [256], "N": [256], "K": [256]})
        result = gemm_preprocess(df)
        assert np.isclose(result["wastage_128"].iloc[0], 0.0)
        assert np.isclose(result["wastage_256"].iloc[0], 0.0)
        assert np.isclose(result["wastage_64"].iloc[0], 0.0)

    def test_many_rows(self):
        """Should handle larger DataFrames efficiently."""
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "M": np.random.choice([64, 128, 256, 512, 1024, 2048, 4096], n),
            "N": np.random.choice([64, 128, 256, 512, 1024, 2048, 4096], n),
            "K": np.random.choice([64, 128, 256, 512, 1024, 2048], n),
        })
        result = gemm_preprocess(df)
        assert len(result) == n
        assert not result.isnull().any().any()


# ---------------------------------------------------------------------------
# kernel_preprocess tests (document tower)
# ---------------------------------------------------------------------------


class TestKernelPreprocess:
    """Tests for kernel config feature engineering."""

    @staticmethod
    def _make_kernel_df(**overrides):
        """Create a kernel config DataFrame with sensible defaults."""
        defaults = {
            "BLOCK_SIZE_M": [128],
            "BLOCK_SIZE_N": [128],
            "BLOCK_SIZE_K": [64],
            "GROUP_SIZE_M": [8],
            "num_warps": [4],
            "num_stages": [2],
            "waves_per_eu": [0],
            "matrix_instr_nonkdim": [16],
        }
        defaults.update(overrides)
        return pd.DataFrame(defaults)

    def test_basic_output_shape(self):
        """Output has same number of rows as input."""
        df = self._make_kernel_df(
            BLOCK_SIZE_M=[128, 64],
            BLOCK_SIZE_N=[128, 128],
            BLOCK_SIZE_K=[64, 32],
            GROUP_SIZE_M=[8, 4],
            num_warps=[8, 4],
            num_stages=[2, 3],
            waves_per_eu=[0, 2],
            matrix_instr_nonkdim=[16, 32],
        )
        result = kernel_preprocess(df)
        assert len(result) == 2

    def test_no_nans(self):
        """Features should never contain NaN values."""
        df = self._make_kernel_df()
        result = kernel_preprocess(df)
        assert not result.isnull().any().any()

    def test_tile_area_computed(self):
        """tile_area = BLOCK_SIZE_M * BLOCK_SIZE_N."""
        df = self._make_kernel_df(BLOCK_SIZE_M=[128], BLOCK_SIZE_N=[256])
        result = kernel_preprocess(df)
        assert "tile_area" in result.columns
        assert result["tile_area"].iloc[0] == 128 * 256

    def test_tile_aspect_square(self):
        """Square tiles should have aspect ratio ~0 (log2(1)=0)."""
        df = self._make_kernel_df(BLOCK_SIZE_M=[128], BLOCK_SIZE_N=[128])
        result = kernel_preprocess(df)
        assert np.isclose(result["tile_aspect"].iloc[0], 0.0)

    def test_tile_shape_flags(self):
        """Tile shape flags should be mutually exclusive."""
        # Square tile
        df = self._make_kernel_df(BLOCK_SIZE_M=[128], BLOCK_SIZE_N=[128])
        r = kernel_preprocess(df)
        assert r["tile_is_square"].iloc[0] == 1.0
        assert r["tile_is_tall"].iloc[0] == 0.0
        assert r["tile_is_wide"].iloc[0] == 0.0

        # Tall tile
        df = self._make_kernel_df(BLOCK_SIZE_M=[256], BLOCK_SIZE_N=[64])
        r = kernel_preprocess(df)
        assert r["tile_is_tall"].iloc[0] == 1.0
        assert r["tile_is_wide"].iloc[0] == 0.0

    def test_lds_utilization(self):
        """LDS utilization should be between 0 and a reasonable upper bound."""
        df = self._make_kernel_df()
        result = kernel_preprocess(df)
        util = result["lds_utilization"].iloc[0]
        assert util >= 0, f"LDS utilization negative: {util}"

    def test_lds_bytes_formula(self):
        """lds_bytes = (BM + BN) * BK * elem_size * num_stages."""
        BM, BN, BK, NS = 128, 128, 64, 2
        df = self._make_kernel_df(
            BLOCK_SIZE_M=[BM], BLOCK_SIZE_N=[BN],
            BLOCK_SIZE_K=[BK], num_stages=[NS],
        )
        result = kernel_preprocess(df, dtype="bf16")
        expected = (BM + BN) * BK * 2 * NS  # elem_size=2 for bf16
        assert np.isclose(result["lds_bytes"].iloc[0], expected)

    def test_threads_per_block(self):
        """threads_per_block = num_warps * 64."""
        df = self._make_kernel_df(num_warps=[8])
        result = kernel_preprocess(df)
        assert result["threads_per_block"].iloc[0] == 8 * 64

    def test_group_size_features(self):
        """Group size features should reflect input values."""
        df = self._make_kernel_df(GROUP_SIZE_M=[1])
        r = kernel_preprocess(df)
        assert r["group_is_1"].iloc[0] == 1.0
        assert r["group_is_large"].iloc[0] == 0.0

        df = self._make_kernel_df(GROUP_SIZE_M=[32])
        r = kernel_preprocess(df)
        assert r["group_is_1"].iloc[0] == 0.0
        assert r["group_is_large"].iloc[0] == 1.0

    def test_alignment_flags(self):
        """Block alignment flags should be correct."""
        df = self._make_kernel_df(BLOCK_SIZE_M=[64], BLOCK_SIZE_N=[96])
        result = kernel_preprocess(df)
        assert result["block_m_align_16"].iloc[0] == 1.0
        assert result["block_m_align_64"].iloc[0] == 1.0
        assert result["block_n_align_16"].iloc[0] == 1.0
        assert result["block_n_align_64"].iloc[0] == 0.0  # 96 % 64 != 0

    def test_tile_ai(self):
        """Tile AI = 2*BM*BN*BK / ((BM*BK + BN*BK) * elem_size)."""
        BM, BN, BK = 128, 128, 64
        df = self._make_kernel_df(
            BLOCK_SIZE_M=[BM], BLOCK_SIZE_N=[BN], BLOCK_SIZE_K=[BK],
        )
        result = kernel_preprocess(df, dtype="bf16")
        expected = (2 * BM * BN * BK) / ((BM * BK + BN * BK) * 2)
        assert np.isclose(result["tile_ai"].iloc[0], expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Column name helpers
# ---------------------------------------------------------------------------


class TestColumnHelpers:
    """Tests for get_continuous_*_cols and get_categorical_kernel_cols."""

    def test_gemm_cols_nonempty(self):
        """Should return a non-empty list of feature column names."""
        cols = get_continuous_gemm_cols()
        assert isinstance(cols, list)
        assert len(cols) >= 10, f"Expected >=10 gemm cols, got {len(cols)}"

    def test_kernel_cols_nonempty(self):
        """Should return a non-empty list of feature column names."""
        cols = get_continuous_kernel_cols()
        assert isinstance(cols, list)
        assert len(cols) >= 5, f"Expected >=5 kernel cols, got {len(cols)}"

    def test_gemm_cols_match_preprocess(self):
        """Column list should match what gemm_preprocess actually produces."""
        cols = get_continuous_gemm_cols()
        df = pd.DataFrame({"M": [1024], "N": [512], "K": [256]})
        result = gemm_preprocess(df)
        assert set(cols) == set(result.columns)

    def test_kernel_cols_match_preprocess(self):
        """Column list should match what kernel_preprocess actually produces."""
        cols = get_continuous_kernel_cols()
        df = pd.DataFrame({
            "BLOCK_SIZE_M": [128], "BLOCK_SIZE_N": [128], "BLOCK_SIZE_K": [32],
            "GROUP_SIZE_M": [8], "num_warps": [4], "num_stages": [2],
            "waves_per_eu": [0], "matrix_instr_nonkdim": [16],
        })
        result = kernel_preprocess(df)
        assert set(cols) == set(result.columns)

    def test_categorical_kernel_cols(self):
        """Should return dict with matrix_instr_nonkdim."""
        cats = get_categorical_kernel_cols()
        assert isinstance(cats, dict)
        assert "matrix_instr_nonkdim" in cats
        assert cats["matrix_instr_nonkdim"] == 2


# ---------------------------------------------------------------------------
# GPU_SPECS and DTYPE_BYTES
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_gpu_specs_has_required_archs(self):
        assert "mi300x" in GPU_SPECS
        assert "mi350x" in GPU_SPECS

    def test_gpu_specs_fields(self):
        for arch in ["mi300x", "mi350x"]:
            spec = GPU_SPECS[arch]
            for key in ["n_cu", "wave_size", "mem_bandwidth", "L1_size", "L2_size"]:
                assert key in spec, f"Missing {key} in {arch} specs"
                assert spec[key] > 0

    def test_dtype_bytes(self):
        assert DTYPE_BYTES["fp16"] == 2
        assert DTYPE_BYTES["bf16"] == 2
        assert DTYPE_BYTES["fp32"] == 4
        assert DTYPE_BYTES["fp64"] == 8
