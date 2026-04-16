"""
Feature engineering for Two Tower GEMM kernel selection model.

Adapted from GemmKernelSelection/Embedding/two_tower/data.py for tritonBLAS's
Triton config space. Provides query (GEMM problem) and document (kernel config)
feature extraction for the Two Tower embedding model.
"""

import numpy as np
import pandas as pd
import math

# ============================================================================
# GPU hardware specifications
# ============================================================================

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

# Data type configurations
DTYPE_CONFIG = {
    "fp16": {"elem_bytes": 2},
    "bf16": {"elem_bytes": 2},
    "fp32": {"elem_bytes": 4},
}


def _get_acc_size(dtype_str):
    """Get accumulator size in bytes for a given dtype."""
    cfg = DTYPE_CONFIG.get(dtype_str, DTYPE_CONFIG["bf16"])
    if cfg["elem_bytes"] == 8:
        return 8
    return 4  # BF16/FP16/FP32 -> FP32 accumulator


# ============================================================================
# Continuous GEMM (query) feature column names
# ============================================================================

_GEMM_CONTINUOUS_COLS = [
    # Raw dims (log-transformed at end)
    "m", "n", "k",
    # Log products
    "log_flops", "log_bytes",
    # Arithmetic intensity
    "arithmetic_intensity", "log_ai",
    # Log ratios
    "log_ratio_m_n", "log_ratio_n_k", "log_ratio_m_k",
    # Roofline
    "is_compute_bound", "is_memory_bound",
    "ai_vs_balance", "log_ai_vs_balance",
    "memory_headroom", "memory_headroom_clipped",
    # Cache pressure
    "log_ws_l1_ratio", "fits_in_l1",
    "log_ws_l2_ratio", "fits_in_l2",
    "log_ws_l3_ratio", "fits_in_l3",
    "exceeds_l1", "exceeds_l2",
    "in_l1_sweet_spot", "in_l2_sweet_spot",
    "exceeds_l3", "in_l3_sweet_spot",
    "fits_in_l3_not_l2", "exceeds_both_caches",
    # K-dimension pressure
    "log_k_l1_pressure",
    "log_k_parallelism", "k_underutilizes_wave", "k_saturates_waves",
    # Bandwidth pressure
    "log_bandwidth_pressure",
    # Accumulator pressure
    "log_acc_bytes", "log_acc_pressure", "log_acc_pressure_l3",
    # Wave alignment
    "m_wave_misalignment", "n_wave_misalignment", "wave_misalignment_total",
    "m_wave_aligned", "n_wave_aligned", "both_wave_aligned",
    # Stream-K hints
    "log_k_vs_mn", "log_streamk_imbalance", "streamk_favorable",
    # Reuse factors
    "low_reuse", "high_reuse",
    # Tile preference
    "prefer_small_tile", "prefer_large_tile",
    # Aspect ratios
    "sqrt_aspect_nm", "log_n_to_m_ratio",
    "log_aspect_m_n", "log_aspect_m_k", "log_aspect_n_k",
    # Tile alignment
    "m_tile_align_128", "m_tile_align_256",
    "n_tile_align_128", "n_tile_align_256",
    "k_tile_align_32", "k_tile_align_64", "k_tile_align_128",
    # Size ratios
    "n_div_tile128", "n_div_tile256",
    # Problem scale
    "is_large",
    # Shape flags
    "is_tall", "is_wide", "is_square",
    "is_tall_skinny", "is_short_wide", "is_deep_k",
    # K-dimension features
    "k_ultra_tiny", "is_tiny_k", "is_very_small_k", "is_small_k",
    "k_small_problem",
    "is_large_k", "is_huge_k",
    "k_div_32", "k_div_64",
    # Occupancy proxy
    "log_est_tiles", "is_saturating", "log_est_waves",
    # Modulo features
    "m_mod_64", "n_mod_64", "k_mod_64",
    # Tile count log features
    "log_tiles_64x64", "log_tiles_128x128", "log_tiles_256x256",
    # Wastage features
    "wastage_32", "wastage_64", "wastage_128", "wastage_256",
    # Underfill flags
    "m_underfills_256", "n_underfills_256",
    "m_underfills_128", "n_underfills_128",
    # Partial tiles
    "m_partial_128", "n_partial_128",
    "m_partial_256", "n_partial_256",
    # Wastage comparisons
    "wastage_256_vs_128",
    # Raw remainders
    "m_mod_256", "n_mod_256",
    # Tile count differences
    "tile_count_diff_256_128",
    # Edge cases
    "is_tiny_m", "is_tiny_n",
    "is_small_m", "is_small_n",
    "is_gemv", "is_all_tiny",
    # Small tile wastage
    "m_partial_32", "n_partial_32",
    "m_partial_64", "n_partial_64",
    # Critical interactions
    "tiny_m_tiny_n", "tiny_n_tiny_k", "gemv_tiny_k",
    # General features
    "n_small_misaligned", "k_small_misaligned",
    "n_small_wastage_ratio",
    "extreme_aspect_ratio", "very_extreme_aspect",
    "k_dominates_output",
    "multi_edge_case",
    # K extremes
    "is_ultra_huge_k",
    "k_exceeds_l3", "k_exceeds_l2",
    # Output size
    "log_output_size", "is_tiny_output",
    # K vs output
    "log_k_vs_output", "k_dominates_output_extreme",
    # K vs individual dims
    "log_k_vs_m", "log_k_vs_n",
    # Parallelization
    "log_output_vs_cu", "insufficient_parallelism",
    # Combined pathological
    "huge_k_tiny_output", "ultra_skinny_k",
    # K reuse
    "log_k_reuse",
    # K memory
    "log_k_memory", "log_k_memory_vs_l3",
    # Work distribution
    "log_work_per_output", "imbalanced_workload",
    # Dimension dominance
    "k_is_max_dim", "k_dominates_both", "k_ultra_dominates",
]


# ============================================================================
# Continuous kernel (document) feature column names
# ============================================================================

_KERNEL_CONTINUOUS_COLS = [
    # Log block dims
    "log_block_m", "log_block_n", "log_block_k",
    # Direct params
    "num_warps", "num_stages", "waves_per_eu", "GROUP_SIZE_M",
    # Derived tile features
    "tile_area", "log_tile_area", "tile_aspect",
    "tile_volume", "log_tile_volume",
    # Tile shape flags
    "tile_is_square", "tile_is_tall", "tile_is_wide",
    # LDS features
    "lds_bytes", "lds_utilization", "fits_lds",
    "lds_low", "lds_medium", "lds_high",
    # Thread features
    "threads_per_block", "elements_per_thread", "log_elements_per_thread",
    # Alignment
    "block_m_align_16", "block_n_align_16",
    "block_m_align_64", "block_n_align_64",
    "block_k_align_32",
    # Tile AI
    "tile_ai", "log_tile_ai",
    # Depth features
    "depth_efficiency", "depth_to_block_m", "depth_to_block_n",
    # Depth one-hot
    "depth_16", "depth_32", "depth_64", "depth_128", "depth_256", "depth_512",
    # Tile pattern flags (common Triton tile sizes)
    "is_64x64", "is_128x128", "is_128x64", "is_64x128",
    "is_128x256", "is_256x128", "is_256x256",
    "is_32x32", "is_64x32", "is_32x64",
    # Stage features
    "is_2stage", "is_3stage",
    # Waves per EU features
    "wpe_zero", "wpe_one", "wpe_high",
]

_KERNEL_CATEGORICAL_COLS = {
    "matrix_instr_nonkdim": 2,  # values: 16, 32 -> indices: 0, 1
}


# ============================================================================
# Public accessors
# ============================================================================

def get_continuous_gemm_cols():
    """Return list of continuous GEMM (query tower) feature column names."""
    return list(_GEMM_CONTINUOUS_COLS)


def get_continuous_kernel_cols():
    """Return list of continuous kernel (document tower) feature column names."""
    return list(_KERNEL_CONTINUOUS_COLS)


def get_categorical_kernel_cols():
    """Return dict of {col_name: num_categories} for kernel categorical features."""
    return dict(_KERNEL_CATEGORICAL_COLS)


# ============================================================================
# GEMM (query tower) feature engineering
# ============================================================================

def gemm_preprocess(df, gpu_arch="mi300x", dtype_str="bf16"):
    """
    Compute query tower features from GEMM problem dimensions.

    Adapted from GemmKernelSelection _gemm_preprocess(). Expects a DataFrame
    with columns: m, n, k (and optionally batch_count).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'm', 'n', 'k'. Optionally 'batch_count'.
    gpu_arch : str
        GPU architecture key into GPU_SPECS (default 'mi300x').
    dtype_str : str
        Data type string key into DTYPE_CONFIG (default 'bf16').

    Returns
    -------
    pd.DataFrame
        Feature DataFrame with one row per input row, columns matching
        get_continuous_gemm_cols().
    """
    hw = GPU_SPECS[gpu_arch]
    dtype_config = DTYPE_CONFIG.get(dtype_str, DTYPE_CONFIG["bf16"])
    dtype_size = dtype_config["elem_bytes"]

    gdf = df[["m", "n", "k"]].copy()

    if "batch_count" in df.columns:
        gdf["batch_count"] = df["batch_count"]
    else:
        gdf["batch_count"] = 1

    if (gdf[["m", "n", "k"]] == 0).any().any():
        raise ValueError("Zeros found in 'm, n, k'")

    N_CU = hw["n_cu"]
    PEAK_FLOPS = hw["peak_flops"][dtype_size]
    MEM_BW = hw["mem_bandwidth"]
    L1_SIZE = hw["L1_size"]
    L2_SIZE = hw["L2_size"]
    CACHE_SIZE = hw["L3_size"]
    WAVE_SIZE = hw["wave_size"]

    M = gdf["m"].values.astype(float)
    N = gdf["n"].values.astype(float)
    K = gdf["k"].values.astype(float)
    batch = gdf["batch_count"].values.astype(float)

    flops = 2 * M * N * K * batch
    bytes_moved = (M * K + K * N + M * N) * dtype_size

    gdf["log_flops"] = np.log1p(flops)
    gdf["log_bytes"] = np.log1p(bytes_moved)

    gdf["arithmetic_intensity"] = flops / bytes_moved
    gdf["log_ai"] = np.log1p(gdf["arithmetic_intensity"])

    # Log ratios
    gdf["log_ratio_m_n"] = np.log1p(M / N)
    gdf["log_ratio_n_k"] = np.log1p(N / K)
    gdf["log_ratio_m_k"] = np.log1p(M / K)

    # Roofline
    memory_peak = MEM_BW * gdf["arithmetic_intensity"].values
    compute_peak = PEAK_FLOPS
    gdf["is_compute_bound"] = (memory_peak > compute_peak).astype(float)
    gdf["is_memory_bound"] = 1.0 - gdf["is_compute_bound"]

    balance_ai = PEAK_FLOPS / MEM_BW
    gdf["ai_vs_balance"] = gdf["arithmetic_intensity"] / balance_ai
    gdf["log_ai_vs_balance"] = np.log1p(gdf["ai_vs_balance"])

    gdf["memory_headroom"] = memory_peak / compute_peak
    gdf["memory_headroom_clipped"] = np.clip(gdf["memory_headroom"], 0, 2)

    # Cache pressure
    ws_l1_ratio = bytes_moved / L1_SIZE
    gdf["log_ws_l1_ratio"] = np.log1p(ws_l1_ratio)
    gdf["fits_in_l1"] = (bytes_moved <= L1_SIZE).astype(float)

    ws_l2_ratio = bytes_moved / L2_SIZE
    gdf["log_ws_l2_ratio"] = np.log1p(ws_l2_ratio)
    gdf["fits_in_l2"] = (bytes_moved <= L2_SIZE).astype(float)

    gdf["log_ws_l3_ratio"] = np.log1p(bytes_moved / CACHE_SIZE)
    gdf["fits_in_l3"] = (bytes_moved <= CACHE_SIZE).astype(float)

    gdf["exceeds_l1"] = (bytes_moved > L1_SIZE).astype(float)
    gdf["exceeds_l2"] = (bytes_moved > L2_SIZE).astype(float)

    SWEET_SPOT_LOWER = 0.5
    gdf["in_l1_sweet_spot"] = (
        (bytes_moved > SWEET_SPOT_LOWER * L1_SIZE) & (bytes_moved <= L1_SIZE)
    ).astype(float)
    gdf["in_l2_sweet_spot"] = (
        (bytes_moved > SWEET_SPOT_LOWER * L2_SIZE) & (bytes_moved <= L2_SIZE)
    ).astype(float)

    gdf["exceeds_l3"] = (bytes_moved > CACHE_SIZE).astype(float)
    gdf["in_l3_sweet_spot"] = (
        (bytes_moved > SWEET_SPOT_LOWER * CACHE_SIZE) & (bytes_moved <= CACHE_SIZE)
    ).astype(float)

    gdf["fits_in_l3_not_l2"] = (
        (bytes_moved <= CACHE_SIZE) & (bytes_moved > L2_SIZE)
    ).astype(float)
    gdf["exceeds_both_caches"] = (bytes_moved > CACHE_SIZE).astype(float)

    # K-dimension pressure
    gdf["log_k_l1_pressure"] = np.log1p((K * dtype_size) / L1_SIZE)
    gdf["log_k_parallelism"] = np.log1p(K / WAVE_SIZE)
    gdf["k_underutilizes_wave"] = (K < WAVE_SIZE).astype(float)
    gdf["k_saturates_waves"] = (K >= 4 * WAVE_SIZE).astype(float)

    # Bandwidth pressure
    gdf["log_bandwidth_pressure"] = np.log1p(bytes_moved / MEM_BW)

    # Accumulator pressure
    acc_size = _get_acc_size(dtype_str)
    acc_bytes = acc_size * (M * N)
    gdf["log_acc_bytes"] = np.log1p(acc_bytes)
    gdf["log_acc_pressure"] = np.log1p(acc_bytes / L2_SIZE)
    gdf["log_acc_pressure_l3"] = np.log1p(acc_bytes / CACHE_SIZE)

    # Wave alignment
    gdf["m_wave_misalignment"] = (M % WAVE_SIZE) / WAVE_SIZE
    gdf["n_wave_misalignment"] = (N % WAVE_SIZE) / WAVE_SIZE
    gdf["wave_misalignment_total"] = (
        gdf["m_wave_misalignment"] + gdf["n_wave_misalignment"]
    )
    gdf["m_wave_aligned"] = (M % WAVE_SIZE == 0).astype(float)
    gdf["n_wave_aligned"] = (N % WAVE_SIZE == 0).astype(float)
    gdf["both_wave_aligned"] = (gdf["m_wave_aligned"] * gdf["n_wave_aligned"]).astype(float)

    # Stream-K hints
    gdf["log_k_vs_mn"] = np.log1p(K / (M * N))
    gdf["log_streamk_imbalance"] = np.log1p(K / (M + N))
    gdf["streamk_favorable"] = ((K > 1024) & (M * N < 4096)).astype(float)

    # Reuse factors
    gdf["low_reuse"] = ((N < 64) | (M < 64)).astype(float)
    gdf["high_reuse"] = ((N >= 256) & (M >= 256)).astype(float)

    # Tile preference
    gdf["prefer_small_tile"] = (ws_l1_ratio > 2).astype(float)
    gdf["prefer_large_tile"] = (
        (ws_l2_ratio < 0.5) & (M >= 512) & (N >= 512)
    ).astype(float)

    # Aspect ratios
    gdf["sqrt_aspect_nm"] = np.sqrt(N / M)
    gdf["log_n_to_m_ratio"] = np.log1p(N / M)
    gdf["log_aspect_m_n"] = np.log1p(M / N)
    gdf["log_aspect_m_k"] = np.log1p(M / K)
    gdf["log_aspect_n_k"] = np.log1p(N / K)

    # Tile alignment
    gdf["m_tile_align_128"] = (M % 128 == 0).astype(float)
    gdf["m_tile_align_256"] = (M % 256 == 0).astype(float)
    gdf["n_tile_align_128"] = (N % 128 == 0).astype(float)
    gdf["n_tile_align_256"] = (N % 256 == 0).astype(float)
    gdf["k_tile_align_32"] = (K % 32 == 0).astype(float)
    gdf["k_tile_align_64"] = (K % 64 == 0).astype(float)
    gdf["k_tile_align_128"] = (K % 128 == 0).astype(float)

    # Size ratios
    gdf["n_div_tile128"] = np.log1p(N / 128.0)
    gdf["n_div_tile256"] = np.log1p(N / 256.0)

    # Problem scale
    gdf["is_large"] = ((M >= 8192) | (N >= 8192) | (K >= 8192)).astype(float)

    # Shape flags
    gdf["is_tall"] = (M > N).astype(float)
    gdf["is_wide"] = (N > M).astype(float)
    gdf["is_square"] = (M == N).astype(float)
    gdf["is_tall_skinny"] = ((M > 4 * N) & (M > 4 * K)).astype(float)
    gdf["is_short_wide"] = ((N > 4 * M) & (N > 4 * K)).astype(float)
    gdf["is_deep_k"] = ((K > 4 * M) & (K > 4 * N)).astype(float)

    # K-dimension features
    gdf["k_ultra_tiny"] = (K <= 8).astype(float)
    gdf["is_tiny_k"] = (K <= 16).astype(float)
    gdf["is_very_small_k"] = (K <= 64).astype(float)
    gdf["is_small_k"] = (K < 128).astype(float)
    gdf["k_small_problem"] = ((K <= 128) & (M < 4096) & (N < 4096)).astype(float)
    gdf["is_large_k"] = (K > 4096).astype(float)
    gdf["is_huge_k"] = (K > 100000).astype(float)
    gdf["k_div_32"] = np.log1p(K / 32)
    gdf["k_div_64"] = np.log1p(K / 64)

    # Occupancy proxy
    est_tiles = np.ceil(M / 256) * np.ceil(N / 256)
    gdf["log_est_tiles"] = np.log1p(est_tiles)
    gdf["is_saturating"] = (est_tiles >= N_CU).astype(float)
    gdf["log_est_waves"] = np.log1p(est_tiles / N_CU)

    # Modulo features
    gdf["m_mod_64"] = (M % 64 == 0).astype(float)
    gdf["n_mod_64"] = (N % 64 == 0).astype(float)
    gdf["k_mod_64"] = (K % 64 == 0).astype(float)

    # Tile count features (for Triton-relevant tile sizes)
    for tm, tn in [(64, 64), (128, 128), (256, 256)]:
        gdf[f"log_tiles_{tm}x{tn}"] = np.log1p(np.ceil(M / tm) * np.ceil(N / tn))

    # Wastage features
    for tile in [32, 64, 128, 256]:
        tiles_m = np.ceil(M / tile)
        tiles_n = np.ceil(N / tile)
        work = tiles_m * tile * tiles_n * tile
        gdf[f"wastage_{tile}"] = (work - M * N) / work

    # Underfill flags
    gdf["m_underfills_256"] = (M < 256).astype(float)
    gdf["n_underfills_256"] = (N < 256).astype(float)
    gdf["m_underfills_128"] = (M < 128).astype(float)
    gdf["n_underfills_128"] = (N < 128).astype(float)

    # Partial tiles
    for tile in [128, 256]:
        gdf[f"m_partial_{tile}"] = (M % tile) / tile
        gdf[f"n_partial_{tile}"] = (N % tile) / tile

    # Wastage comparisons
    gdf["wastage_256_vs_128"] = gdf["wastage_256"] - gdf["wastage_128"]

    # Raw remainders
    gdf["m_mod_256"] = M % 256
    gdf["n_mod_256"] = N % 256

    # Tile count differences
    tiles_256 = np.ceil(M / 256) * np.ceil(N / 256)
    tiles_128 = np.ceil(M / 128) * np.ceil(N / 128)
    gdf["tile_count_diff_256_128"] = tiles_256 - tiles_128

    # Edge cases
    gdf["is_tiny_m"] = (M <= 32).astype(float)
    gdf["is_tiny_n"] = (N <= 32).astype(float)
    gdf["is_small_m"] = ((M > 32) & (M <= 128)).astype(float)
    gdf["is_small_n"] = ((N > 32) & (N <= 128)).astype(float)
    gdf["is_gemv"] = ((M == 1) | (N == 1)).astype(float)
    gdf["is_all_tiny"] = ((M <= 64) & (N <= 64) & (K <= 64)).astype(float)

    # Small tile wastage
    for tile in [32, 64]:
        gdf[f"m_partial_{tile}"] = (M % tile) / tile
        gdf[f"n_partial_{tile}"] = (N % tile) / tile

    # Critical interactions
    gdf["tiny_m_tiny_n"] = gdf["is_tiny_m"] * gdf["is_tiny_n"]
    gdf["tiny_n_tiny_k"] = gdf["is_tiny_n"] * gdf["is_tiny_k"]
    gdf["gemv_tiny_k"] = gdf["is_gemv"] * gdf["is_tiny_k"]

    # General features
    gdf["n_small_misaligned"] = ((N < 300) & (N % 16 != 0)).astype(float)
    gdf["k_small_misaligned"] = ((K < 300) & (K % 16 != 0)).astype(float)
    gdf["n_small_wastage_ratio"] = np.where(N < 300, (N % 16) / 16.0, 0)
    gdf["extreme_aspect_ratio"] = ((N > 3 * M) | (M > 3 * N)).astype(float)
    gdf["very_extreme_aspect"] = ((M > 10 * N) | (N > 10 * M)).astype(float)
    gdf["k_dominates_output"] = (K / (M * N + 1) > 10).astype(float)
    gdf["multi_edge_case"] = (
        (gdf["is_gemv"] + gdf["is_tiny_k"] + gdf["n_small_misaligned"] + gdf["k_ultra_tiny"]) > 1
    ).astype(float)

    # K extremes
    gdf["is_ultra_huge_k"] = (K > 10000000).astype(float)
    gdf["k_exceeds_l3"] = ((K * dtype_size) > CACHE_SIZE).astype(float)
    gdf["k_exceeds_l2"] = ((K * dtype_size) > L2_SIZE).astype(float)

    # Output size features
    output_size = M * N
    gdf["log_output_size"] = np.log1p(output_size)
    gdf["is_tiny_output"] = (output_size < 1000).astype(float)

    # K vs output
    k_vs_output = K / (output_size + 1)
    gdf["log_k_vs_output"] = np.log1p(k_vs_output)
    gdf["k_dominates_output_extreme"] = (k_vs_output > 1000).astype(float)

    # K vs individual dims
    gdf["log_k_vs_m"] = np.log1p(K / (M + 1))
    gdf["log_k_vs_n"] = np.log1p(K / (N + 1))

    # Parallelization
    gdf["log_output_vs_cu"] = np.log1p(output_size / N_CU)
    gdf["insufficient_parallelism"] = (output_size < N_CU).astype(float)

    # Combined pathological
    gdf["huge_k_tiny_output"] = ((K > 1000000) & (output_size < 1000)).astype(float)
    gdf["ultra_skinny_k"] = ((K > 10 * M * N) & (output_size < 10000)).astype(float)

    # K reuse
    gdf["log_k_reuse"] = np.log1p(K / (output_size + 1))

    # K memory
    k_memory_bytes = K * dtype_size * (M + N)
    gdf["log_k_memory"] = np.log1p(k_memory_bytes)
    gdf["log_k_memory_vs_l3"] = np.log1p(k_memory_bytes / CACHE_SIZE)

    # Work distribution
    gdf["log_work_per_output"] = np.log1p(K * 2)
    gdf["imbalanced_workload"] = ((K > 10000) & (output_size < 1000)).astype(float)

    # Dimension dominance
    max_dim = np.maximum(np.maximum(M, N), K)
    gdf["k_is_max_dim"] = (K == max_dim).astype(float)
    gdf["k_dominates_both"] = ((K > 10 * M) & (K > 10 * N)).astype(float)
    gdf["k_ultra_dominates"] = ((K > 100 * M) & (K > 100 * N)).astype(float)

    # Log-transform raw dims (done last, matching reference)
    for col in ["m", "n", "k"]:
        gdf[col] = np.log1p(gdf[col].astype(float))

    # Drop batch_count (not in feature list)
    if "batch_count" in gdf.columns:
        gdf.drop(columns=["batch_count"], inplace=True)

    # Coerce to float32
    gdf = gdf.apply(pd.to_numeric, errors="coerce").astype(np.float32)

    # Validate columns match spec
    missing = set(_GEMM_CONTINUOUS_COLS) - set(gdf.columns)
    if missing:
        raise RuntimeError(f"Missing GEMM feature columns: {missing}")

    return gdf[_GEMM_CONTINUOUS_COLS]


# ============================================================================
# Kernel (document tower) feature engineering
# ============================================================================

def kernel_preprocess(df, dtype_str="bf16", gpu_arch="mi300x"):
    """
    Compute document tower features from Triton kernel configurations.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        num_warps, num_stages, waves_per_eu, GROUP_SIZE_M,
        matrix_instr_nonkdim.
    dtype_str : str
        Data type string for element size (default 'bf16').
    gpu_arch : str
        GPU architecture key (default 'mi300x').

    Returns
    -------
    continuous_df : pd.DataFrame
        Continuous feature columns matching get_continuous_kernel_cols().
    categorical_df : pd.DataFrame
        Categorical feature columns (integer indices for nn.Embedding).
    """
    hw = GPU_SPECS[gpu_arch]
    dtype_config = DTYPE_CONFIG.get(dtype_str, DTYPE_CONFIG["bf16"])
    elem_bytes = dtype_config["elem_bytes"]
    LDS_CAP = hw["LDS_size"]

    kdf = pd.DataFrame(index=df.index)

    BM = df["BLOCK_SIZE_M"].values.astype(float)
    BN = df["BLOCK_SIZE_N"].values.astype(float)
    BK = df["BLOCK_SIZE_K"].values.astype(float)
    NW = df["num_warps"].values.astype(float)
    NS = df["num_stages"].values.astype(float)
    WPE = df["waves_per_eu"].values.astype(float)
    GSM = df["GROUP_SIZE_M"].values.astype(float)

    # Log block dims
    kdf["log_block_m"] = np.log2(np.maximum(BM, 1))
    kdf["log_block_n"] = np.log2(np.maximum(BN, 1))
    kdf["log_block_k"] = np.log2(np.maximum(BK, 1))

    # Direct params
    kdf["num_warps"] = NW
    kdf["num_stages"] = NS
    kdf["waves_per_eu"] = WPE
    kdf["GROUP_SIZE_M"] = GSM

    # Tile features
    tile_area = BM * BN
    kdf["tile_area"] = tile_area
    kdf["log_tile_area"] = np.log1p(tile_area)
    kdf["tile_aspect"] = np.log2(np.maximum(BM, 1) / np.maximum(BN, 1))

    tile_volume = BM * BN * BK
    kdf["tile_volume"] = tile_volume
    kdf["log_tile_volume"] = np.log1p(tile_volume)

    # Tile shape flags
    kdf["tile_is_square"] = (BM == BN).astype(float)
    kdf["tile_is_tall"] = (BM > BN).astype(float)
    kdf["tile_is_wide"] = (BM < BN).astype(float)

    # LDS features
    # In Triton, LDS holds A and B tiles for num_stages pipeline stages
    lds_bytes = (BM + BN) * BK * elem_bytes * NS
    kdf["lds_bytes"] = lds_bytes
    kdf["lds_utilization"] = lds_bytes / LDS_CAP
    kdf["fits_lds"] = (lds_bytes <= LDS_CAP).astype(float)
    kdf["lds_low"] = (kdf["lds_utilization"] < 0.5).astype(float)
    kdf["lds_medium"] = ((kdf["lds_utilization"] >= 0.5) & (kdf["lds_utilization"] < 0.8)).astype(float)
    kdf["lds_high"] = (kdf["lds_utilization"] >= 0.8).astype(float)

    # Thread features
    threads_per_block = NW * 64
    kdf["threads_per_block"] = threads_per_block
    elements_per_thread = tile_area / threads_per_block
    kdf["elements_per_thread"] = elements_per_thread
    kdf["log_elements_per_thread"] = np.log1p(elements_per_thread)

    # Alignment
    kdf["block_m_align_16"] = (BM % 16 == 0).astype(float)
    kdf["block_n_align_16"] = (BN % 16 == 0).astype(float)
    kdf["block_m_align_64"] = (BM % 64 == 0).astype(float)
    kdf["block_n_align_64"] = (BN % 64 == 0).astype(float)
    kdf["block_k_align_32"] = (BK % 32 == 0).astype(float)

    # Tile arithmetic intensity
    tile_flops = 2 * BM * BN * BK
    tile_bytes = (BM * BK + BN * BK) * elem_bytes
    kdf["tile_ai"] = tile_flops / np.maximum(tile_bytes, 1)
    kdf["log_tile_ai"] = np.log1p(kdf["tile_ai"])

    # Depth features
    kdf["depth_efficiency"] = BK / np.sqrt(tile_area + 1)
    kdf["depth_to_block_m"] = BK / np.maximum(BM, 1)
    kdf["depth_to_block_n"] = BK / np.maximum(BN, 1)

    # Depth one-hot
    kdf["depth_16"] = (BK == 16).astype(float)
    kdf["depth_32"] = (BK == 32).astype(float)
    kdf["depth_64"] = (BK == 64).astype(float)
    kdf["depth_128"] = (BK == 128).astype(float)
    kdf["depth_256"] = (BK == 256).astype(float)
    kdf["depth_512"] = (BK == 512).astype(float)

    # Tile pattern flags (common Triton tile sizes)
    tile_patterns = [
        (64, 64), (128, 128), (128, 64), (64, 128),
        (128, 256), (256, 128), (256, 256),
        (32, 32), (64, 32), (32, 64),
    ]
    for tm, tn in tile_patterns:
        kdf[f"is_{tm}x{tn}"] = ((BM == tm) & (BN == tn)).astype(float)

    # Stage features
    kdf["is_2stage"] = (NS == 2).astype(float)
    kdf["is_3stage"] = (NS == 3).astype(float)

    # Waves per EU features
    kdf["wpe_zero"] = (WPE == 0).astype(float)
    kdf["wpe_one"] = (WPE == 1).astype(float)
    kdf["wpe_high"] = (WPE >= 4).astype(float)

    # Coerce to float32
    kdf = kdf.apply(pd.to_numeric, errors="coerce").astype(np.float32)

    # Categorical features
    cat_df = pd.DataFrame(index=df.index)
    if "matrix_instr_nonkdim" in df.columns:
        # Map 16 -> 0, 32 -> 1
        cat_df["matrix_instr_nonkdim"] = (
            df["matrix_instr_nonkdim"].map({16: 0, 32: 1}).fillna(0).astype(int)
        )
    else:
        cat_df["matrix_instr_nonkdim"] = 0

    # Validate columns
    missing = set(_KERNEL_CONTINUOUS_COLS) - set(kdf.columns)
    if missing:
        raise RuntimeError(f"Missing kernel feature columns: {missing}")

    return kdf[_KERNEL_CONTINUOUS_COLS], cat_df
