"""Feature engineering for Two Tower kernel selection model.

Adapted from GemmKernelSelection/Embedding/two_tower/data.py for
tritonBLAS's Triton config space (11 params, ~28,800 configs).

Two main functions:
  gemm_preprocess(df, gpu_arch)   - Query tower features (GEMM problem → embedding)
  kernel_preprocess(df)           - Document tower features (kernel config → embedding)
"""

import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# GPU hardware specs (ported from GemmKernelSelection data.py)
# ---------------------------------------------------------------------------
GPU_SPECS = {
    "mi300x": {
        "n_cu": 304,
        "wave_size": 64,
        "max_occupancy": 8,
        "peak_flops": {2: 2.3e15, 4: 163.4e12},
        "mem_bandwidth": 5.3e12,
        "L1_size": 32 * 1024,
        "L2_size": 4 * 1024 * 1024,
        "L3_size": 256 * 1024 * 1024,
        "LDS_size": 64 * 1024,
        "VGPR_per_CU": 512 * 1024,
        "HBM_size": 192 * 1024**3,
        "num_xcd": 8,
        "matrix_inst_latency": 8,
        "cvt_latency": 1,
    },
    "mi350x": {
        "n_cu": 256,
        "wave_size": 64,
        "max_occupancy": 8,
        "peak_flops": {2: 2.3e15, 4: 144.2e12},
        "mem_bandwidth": 8e12,
        "L1_size": 32 * 1024,
        "L2_size": 4 * 1024 * 1024,
        "L3_size": 256 * 1024 * 1024,
        "LDS_size": 160 * 1024,
        "VGPR_per_CU": 512 * 1024,
        "HBM_size": 288 * 1024**3,
        "num_xcd": 8,
        "matrix_inst_latency": 8,
        "cvt_latency": 1,
    },
}

# Dtype byte sizes
DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "tf32": 4,
    "fp64": 8,
}

# ---------------------------------------------------------------------------
# Query tower: GEMM problem features
# ---------------------------------------------------------------------------

def gemm_preprocess(df, gpu_arch="mi300x", dtype="bf16"):
    """Compute query-tower features from GEMM problem dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'M', 'N', 'K'. Optionally 'batch_count'.
    gpu_arch : str
        Key into GPU_SPECS ('mi300x' or 'mi350x').
    dtype : str
        Data type string for element size lookup.

    Returns
    -------
    pd.DataFrame
        Feature columns only (original columns are dropped).
    """
    hw = GPU_SPECS[gpu_arch]
    elem_size = DTYPE_BYTES.get(dtype, 2)

    # Hardware constants
    N_CU = hw["n_cu"]
    PEAK_FLOPS = hw["peak_flops"][elem_size]
    MEM_BW = hw["mem_bandwidth"]
    L1_SIZE = hw["L1_size"]
    L2_SIZE = hw["L2_size"]
    L3_SIZE = hw["L3_size"]
    WAVE_SIZE = hw["wave_size"]

    gdf = pd.DataFrame(index=df.index)

    M = df["M"].values.astype(np.float64)
    N = df["N"].values.astype(np.float64)
    K = df["K"].values.astype(np.float64)
    batch = df["batch_count"].values.astype(np.float64) if "batch_count" in df.columns else np.ones(len(df))

    # --- 1. Raw log dims ---
    gdf["log_m"] = np.log2(np.maximum(M, 1))
    gdf["log_n"] = np.log2(np.maximum(N, 1))
    gdf["log_k"] = np.log2(np.maximum(K, 1))
    gdf["log_batch"] = np.log2(np.maximum(batch, 1))

    # --- 2. Log ratios ---
    gdf["log_ratio_m_n"] = np.log2(np.maximum(M / np.maximum(N, 1), 1e-6))
    gdf["log_ratio_m_k"] = np.log2(np.maximum(M / np.maximum(K, 1), 1e-6))
    gdf["log_ratio_n_k"] = np.log2(np.maximum(N / np.maximum(K, 1), 1e-6))

    # --- 3. Products ---
    gdf["log_mn"] = np.log2(np.maximum(M * N, 1))
    gdf["log_mnk"] = np.log2(np.maximum(M * N * K, 1))

    # --- 4. Arithmetic intensity ---
    flops = 2.0 * M * N * K * batch
    bytes_moved = (M * K + K * N + M * N) * elem_size
    gdf["arithmetic_intensity"] = flops / np.maximum(bytes_moved, 1)
    gdf["log_ai"] = np.log1p(gdf["arithmetic_intensity"].values)

    gdf["log_flops"] = np.log1p(flops)
    gdf["log_bytes"] = np.log1p(bytes_moved)

    # --- 5. Roofline ---
    balance_ai = PEAK_FLOPS / MEM_BW
    memory_peak = MEM_BW * gdf["arithmetic_intensity"].values
    gdf["is_compute_bound"] = (memory_peak > PEAK_FLOPS).astype(np.float32)
    gdf["is_memory_bound"] = 1.0 - gdf["is_compute_bound"].values
    gdf["ai_vs_balance"] = gdf["arithmetic_intensity"].values / balance_ai
    gdf["log_ai_vs_balance"] = np.log1p(gdf["ai_vs_balance"].values)
    gdf["memory_headroom"] = np.clip(memory_peak / PEAK_FLOPS, 0, 2)

    # --- 6. Cache hierarchy pressure ---
    gdf["log_ws_l1_ratio"] = np.log1p(bytes_moved / L1_SIZE)
    gdf["fits_in_l1"] = (bytes_moved <= L1_SIZE).astype(np.float32)
    gdf["log_ws_l2_ratio"] = np.log1p(bytes_moved / L2_SIZE)
    gdf["fits_in_l2"] = (bytes_moved <= L2_SIZE).astype(np.float32)
    gdf["log_ws_l3_ratio"] = np.log1p(bytes_moved / L3_SIZE)
    gdf["fits_in_l3"] = (bytes_moved <= L3_SIZE).astype(np.float32)
    gdf["exceeds_l1"] = (bytes_moved > L1_SIZE).astype(np.float32)
    gdf["exceeds_l2"] = (bytes_moved > L2_SIZE).astype(np.float32)
    gdf["exceeds_l3"] = (bytes_moved > L3_SIZE).astype(np.float32)

    SWEET = 0.5
    gdf["in_l1_sweet_spot"] = ((bytes_moved > SWEET * L1_SIZE) & (bytes_moved <= L1_SIZE)).astype(np.float32)
    gdf["in_l2_sweet_spot"] = ((bytes_moved > SWEET * L2_SIZE) & (bytes_moved <= L2_SIZE)).astype(np.float32)
    gdf["in_l3_sweet_spot"] = ((bytes_moved > SWEET * L3_SIZE) & (bytes_moved <= L3_SIZE)).astype(np.float32)
    gdf["fits_in_l3_not_l2"] = ((bytes_moved <= L3_SIZE) & (bytes_moved > L2_SIZE)).astype(np.float32)

    # --- 7. K-dimension pressure ---
    gdf["log_k_l1_pressure"] = np.log1p((K * elem_size) / L1_SIZE)
    gdf["log_k_parallelism"] = np.log1p(K / WAVE_SIZE)
    gdf["k_underutilizes_wave"] = (K < WAVE_SIZE).astype(np.float32)
    gdf["k_saturates_waves"] = (K >= 4 * WAVE_SIZE).astype(np.float32)

    # --- 8. Bandwidth pressure ---
    gdf["log_bandwidth_pressure"] = np.log1p(bytes_moved / MEM_BW)

    # --- 9. Accumulator pressure ---
    acc_size = 4  # FP32 accumulator
    acc_bytes = acc_size * M * N
    gdf["log_acc_bytes"] = np.log1p(acc_bytes)
    gdf["log_acc_pressure"] = np.log1p(acc_bytes / L2_SIZE)
    gdf["log_acc_pressure_l3"] = np.log1p(acc_bytes / L3_SIZE)

    # --- 10. Wave alignment ---
    gdf["m_wave_misalignment"] = (M % WAVE_SIZE) / WAVE_SIZE
    gdf["n_wave_misalignment"] = (N % WAVE_SIZE) / WAVE_SIZE
    gdf["wave_misalignment_total"] = gdf["m_wave_misalignment"].values + gdf["n_wave_misalignment"].values
    gdf["m_wave_aligned"] = (M % WAVE_SIZE == 0).astype(np.float32)
    gdf["n_wave_aligned"] = (N % WAVE_SIZE == 0).astype(np.float32)
    gdf["both_wave_aligned"] = (gdf["m_wave_aligned"].values * gdf["n_wave_aligned"].values)

    # --- 11. Stream-K hints ---
    gdf["log_k_vs_mn"] = np.log1p(K / np.maximum(M * N, 1))
    gdf["log_streamk_imbalance"] = np.log1p(K / np.maximum(M + N, 1))
    gdf["streamk_favorable"] = ((K > 1024) & (M * N < 4096)).astype(np.float32)

    # --- 12. Aspect ratios ---
    gdf["aspect_mn"] = M / np.maximum(M + N, 1)
    gdf["squareness"] = np.minimum(M, N) / np.maximum(np.maximum(M, N), 1)
    gdf["log_aspect_m_n"] = np.log1p(M / np.maximum(N, 1))
    gdf["log_aspect_m_k"] = np.log1p(M / np.maximum(K, 1))
    gdf["log_aspect_n_k"] = np.log1p(N / np.maximum(K, 1))

    # --- 13. Power-of-2 flags ---
    gdf["m_is_po2"] = _is_power_of_2(M).astype(np.float32)
    gdf["n_is_po2"] = _is_power_of_2(N).astype(np.float32)
    gdf["k_is_po2"] = _is_power_of_2(K).astype(np.float32)

    # --- 14. Alignment ---
    gdf["m_align_128"] = (M % 128 == 0).astype(np.float32)
    gdf["n_align_128"] = (N % 128 == 0).astype(np.float32)
    gdf["k_align_8"] = (K % 8 == 0).astype(np.float32)
    gdf["m_align_256"] = (M % 256 == 0).astype(np.float32)
    gdf["n_align_256"] = (N % 256 == 0).astype(np.float32)
    gdf["k_align_32"] = (K % 32 == 0).astype(np.float32)
    gdf["k_align_64"] = (K % 64 == 0).astype(np.float32)
    gdf["k_align_128"] = (K % 128 == 0).astype(np.float32)

    # --- 15. Shape flags ---
    gdf["is_tall"] = (M > N).astype(np.float32)
    gdf["is_wide"] = (N > M).astype(np.float32)
    gdf["is_square"] = (M == N).astype(np.float32)
    gdf["is_tall_skinny"] = ((M > 4 * N) & (M > 4 * K)).astype(np.float32)
    gdf["is_short_wide"] = ((N > 4 * M) & (N > 4 * K)).astype(np.float32)
    gdf["is_deep_k"] = ((K > 4 * M) & (K > 4 * N)).astype(np.float32)

    # --- 16. K dimension features ---
    gdf["is_tiny_k"] = (K <= 16).astype(np.float32)
    gdf["is_very_small_k"] = (K <= 64).astype(np.float32)
    gdf["is_small_k"] = (K < 128).astype(np.float32)
    gdf["is_large_k"] = (K > 4096).astype(np.float32)
    gdf["is_huge_k"] = (K > 100000).astype(np.float32)
    gdf["k_div_32"] = np.log1p(K / 32)
    gdf["k_div_64"] = np.log1p(K / 64)
    gdf["k_exceeds_l2"] = ((K * elem_size) > L2_SIZE).astype(np.float32)
    gdf["k_exceeds_l3"] = ((K * elem_size) > L3_SIZE).astype(np.float32)

    # --- 17. Occupancy proxy ---
    est_tiles = np.ceil(M / 256) * np.ceil(N / 256)
    gdf["log_est_tiles"] = np.log1p(est_tiles)
    gdf["is_saturating"] = (est_tiles >= N_CU).astype(np.float32)
    gdf["log_est_waves"] = np.log1p(est_tiles / N_CU)

    # --- 18. Reuse factors ---
    gdf["low_reuse"] = ((N < 64) | (M < 64)).astype(np.float32)
    gdf["high_reuse"] = ((N >= 256) & (M >= 256)).astype(np.float32)

    # --- 19. Tile preference hints ---
    ws_l1_ratio = bytes_moved / L1_SIZE
    ws_l2_ratio = bytes_moved / L2_SIZE
    gdf["prefer_small_tile"] = (ws_l1_ratio > 2).astype(np.float32)
    gdf["prefer_large_tile"] = ((ws_l2_ratio < 0.5) & (M >= 512) & (N >= 512)).astype(np.float32)

    # --- 20. Wastage features ---
    for tile in [32, 64, 128, 192, 256]:
        tiles_m = np.ceil(M / tile)
        tiles_n = np.ceil(N / tile)
        work = tiles_m * tile * tiles_n * tile
        gdf[f"wastage_{tile}"] = (work - M * N) / np.maximum(work, 1)

    # --- 21. Tile alignment for common tile sizes ---
    for tile in [128, 192, 256]:
        gdf[f"m_tile_align_{tile}"] = (M % tile == 0).astype(np.float32)
        gdf[f"n_tile_align_{tile}"] = (N % tile == 0).astype(np.float32)

    # --- 22. Modulo features ---
    for col_name, col_val in [("m", M), ("n", N), ("k", K)]:
        gdf[f"{col_name}_mod_64"] = (col_val % 64 == 0).astype(np.float32)

    # --- 23. Size flags ---
    gdf["is_tiny_m"] = (M <= 32).astype(np.float32)
    gdf["is_tiny_n"] = (N <= 32).astype(np.float32)
    gdf["is_small_m"] = ((M > 32) & (M <= 128)).astype(np.float32)
    gdf["is_small_n"] = ((N > 32) & (N <= 128)).astype(np.float32)
    gdf["is_gemv"] = ((M == 1) | (N == 1)).astype(np.float32)
    gdf["is_large"] = ((M >= 8192) | (N >= 8192) | (K >= 8192)).astype(np.float32)
    gdf["extreme_aspect_ratio"] = ((N > 3 * M) | (M > 3 * N)).astype(np.float32)

    # --- 24. Output size features ---
    output_size = M * N
    gdf["log_output_size"] = np.log1p(output_size)
    gdf["log_k_vs_output"] = np.log1p(K / np.maximum(output_size, 1))
    gdf["output_vs_cu"] = output_size / N_CU
    gdf["log_output_vs_cu"] = np.log1p(gdf["output_vs_cu"].values)
    gdf["insufficient_parallelism"] = (output_size < N_CU).astype(np.float32)

    # --- 25. K memory features ---
    gdf["k_memory_bytes"] = K * elem_size * (M + N)
    gdf["log_k_memory"] = np.log1p(gdf["k_memory_bytes"].values)
    gdf["log_k_memory_vs_l3"] = np.log1p(gdf["k_memory_bytes"].values / L3_SIZE)

    # --- 26. Dimension dominance ---
    max_dim = np.maximum(np.maximum(M, N), K)
    gdf["k_is_max_dim"] = (K == max_dim).astype(np.float32)

    return gdf


def _is_power_of_2(x):
    """Check if values are powers of 2 (vectorized)."""
    xi = np.array(x, dtype=np.int64)
    return (xi > 0) & ((xi & (xi - 1)) == 0)


# ---------------------------------------------------------------------------
# Document tower: kernel config features
# ---------------------------------------------------------------------------

def kernel_preprocess(df, dtype="bf16"):
    """Compute document-tower features from Triton kernel configs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Triton config columns: BLOCK_SIZE_M, BLOCK_SIZE_N,
        BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, num_stages, waves_per_eu,
        matrix_instr_nonkdim.
    dtype : str
        Data type string for element size lookup.

    Returns
    -------
    pd.DataFrame
        Continuous feature columns (categorical handled separately).
    """
    elem_size = DTYPE_BYTES.get(dtype, 2)
    LDS_PER_WG = 65536  # 64 KB LDS per workgroup

    kdf = pd.DataFrame(index=df.index)

    BM = df["BLOCK_SIZE_M"].values.astype(np.float64)
    BN = df["BLOCK_SIZE_N"].values.astype(np.float64)
    BK = df["BLOCK_SIZE_K"].values.astype(np.float64)
    GM = df["GROUP_SIZE_M"].values.astype(np.float64)
    NW = df["num_warps"].values.astype(np.float64)
    NS = df["num_stages"].values.astype(np.float64)
    WPE = df["waves_per_eu"].values.astype(np.float64)

    # --- 1. Log block dims ---
    kdf["log_block_m"] = np.log2(np.maximum(BM, 1))
    kdf["log_block_n"] = np.log2(np.maximum(BN, 1))
    kdf["log_block_k"] = np.log2(np.maximum(BK, 1))

    # --- 2. Scheduling / execution params ---
    kdf["num_warps"] = NW
    kdf["num_stages"] = NS
    kdf["waves_per_eu"] = WPE
    kdf["group_size_m"] = GM

    # --- 3. Tile geometry ---
    tile_area = BM * BN
    kdf["tile_area"] = tile_area
    kdf["log_tile_area"] = np.log2(np.maximum(tile_area, 1))
    kdf["tile_aspect"] = np.log2(np.maximum(BM / np.maximum(BN, 1), 1e-6))
    kdf["tile_volume"] = BM * BN * BK
    kdf["log_tile_volume"] = np.log2(np.maximum(kdf["tile_volume"].values, 1))

    # --- 4. Tile shape flags ---
    kdf["tile_is_square"] = (BM == BN).astype(np.float32)
    kdf["tile_is_tall"] = (BM > BN).astype(np.float32)
    kdf["tile_is_wide"] = (BM < BN).astype(np.float32)

    # --- 5. LDS utilization ---
    lds_bytes = (BM + BN) * BK * elem_size * NS
    kdf["lds_bytes"] = lds_bytes
    kdf["lds_utilization"] = lds_bytes / LDS_PER_WG
    kdf["fits_lds"] = (lds_bytes <= LDS_PER_WG).astype(np.float32)
    kdf["lds_low"] = (kdf["lds_utilization"].values < 0.5).astype(np.float32)
    kdf["lds_high"] = (kdf["lds_utilization"].values >= 0.8).astype(np.float32)

    # --- 6. Thread-level features ---
    threads_per_block = NW * 64
    kdf["threads_per_block"] = threads_per_block
    kdf["elements_per_thread"] = tile_area / np.maximum(threads_per_block, 1)
    kdf["log_elements_per_thread"] = np.log2(np.maximum(kdf["elements_per_thread"].values, 1))

    # --- 7. Alignment ---
    kdf["block_m_align_16"] = (BM % 16 == 0).astype(np.float32)
    kdf["block_n_align_16"] = (BN % 16 == 0).astype(np.float32)
    kdf["block_m_align_64"] = (BM % 64 == 0).astype(np.float32)
    kdf["block_n_align_64"] = (BN % 64 == 0).astype(np.float32)

    # --- 8. Depth features ---
    kdf["depth_efficiency"] = BK / np.sqrt(np.maximum(tile_area, 1))
    kdf["depth_to_tileM"] = BK / np.maximum(BM, 1)
    kdf["depth_to_tileN"] = BK / np.maximum(BN, 1)

    # --- 9. Tile AI (arithmetic intensity of one tile) ---
    tile_flops = 2 * BM * BN * BK
    tile_bytes = (BM * BK + BN * BK) * elem_size
    kdf["tile_ai"] = tile_flops / np.maximum(tile_bytes, 1)
    kdf["log_tile_ai"] = np.log1p(kdf["tile_ai"].values)

    # --- 10. Tile size categories ---
    kdf["tile_too_small"] = ((BM <= 32) & (BN <= 32)).astype(np.float32)
    kdf["is_viable_tile"] = (tile_area >= 4096).astype(np.float32)

    # --- 11. Suitability scores ---
    kdf["suitability_small_n"] = ((BN <= 64).astype(np.float32) *
                                   (1 - BN / 256))
    kdf["suitability_large_problem"] = ((tile_area >= 32768).astype(np.float32) *
                                         (tile_area / 65536))

    # --- 12. Tile-depth balance ---
    kdf["tile_depth_balance"] = tile_area / np.maximum(BK * BK, 1)
    kdf["is_compute_heavy_tile"] = ((BK >= 128) & (tile_area <= 16384)).astype(np.float32)
    kdf["is_memory_heavy_tile"] = ((tile_area >= 32768) & (BK <= 64)).astype(np.float32)

    # --- 13. Occupancy hints ---
    kdf["may_have_low_occupancy"] = ((tile_area > 40000) |
                                      (kdf["lds_utilization"].values > 0.85)).astype(np.float32)
    kdf["may_have_high_occupancy"] = ((tile_area < 8192) &
                                       (kdf["lds_utilization"].values < 0.4)).astype(np.float32)

    # --- 14. Group size features ---
    kdf["log_group_size_m"] = np.log2(np.maximum(GM, 1))
    kdf["group_is_1"] = (GM == 1).astype(np.float32)
    kdf["group_is_large"] = (GM >= 16).astype(np.float32)

    return kdf


# ---------------------------------------------------------------------------
# Column name helpers
# ---------------------------------------------------------------------------

# Cache the column lists so they're computed once
_GEMM_COLS = None
_KERNEL_CONT_COLS = None


def get_continuous_gemm_cols():
    """Return list of continuous query feature column names."""
    global _GEMM_COLS
    if _GEMM_COLS is None:
        # Generate a tiny sample to discover column names
        sample = pd.DataFrame({
            "M": [1024], "N": [1024], "K": [1024],
        })
        _GEMM_COLS = list(gemm_preprocess(sample).columns)
    return _GEMM_COLS


def get_continuous_kernel_cols():
    """Return list of continuous document feature column names."""
    global _KERNEL_CONT_COLS
    if _KERNEL_CONT_COLS is None:
        sample = pd.DataFrame({
            "BLOCK_SIZE_M": [128], "BLOCK_SIZE_N": [128], "BLOCK_SIZE_K": [32],
            "GROUP_SIZE_M": [8], "num_warps": [4], "num_stages": [2],
            "waves_per_eu": [0], "matrix_instr_nonkdim": [16],
        })
        _KERNEL_CONT_COLS = list(kernel_preprocess(sample).columns)
    return _KERNEL_CONT_COLS


def get_categorical_kernel_cols():
    """Return dict of {col_name: num_categories} for categorical kernel features.

    These are handled by nn.Embedding layers, not continuous normalization.
    """
    return {
        "matrix_instr_nonkdim": 2,  # values: 16, 32 -> indices: 0, 1
    }
