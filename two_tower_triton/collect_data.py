"""Convert triton_gemm_bench YAML output into training CSVs for the Two Tower model.

Produces three CSV files:
  gemms.csv   - One row per unique GEMM shape (GEMMID, m, n, k, ...)
  kernels.csv - One row per unique kernel config (KernelID, BLOCK_SIZE_M, ...)
  perf.csv    - One row per (GEMM, kernel) measurement (GEMMID, KernelID, EFF, us)

Usage:
    python -m two_tower_triton.collect_data \
        --input /path/to/bench1.yaml /path/to/bench2.yaml \
        --output /path/to/training_data/ \
        --gpu_arch mi300x

Reference: GemmKernelSelection/Embedding/two_tower/data.py CSV format
"""

import argparse
import os
import random
import warnings
from collections import defaultdict

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None  # Handled at runtime if YAML loading is attempted

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Triton config fields we track (in order)
KERNEL_FIELDS = [
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
    "waves_per_eu",
    "matrix_instr_nonkdim",
    "kpack",
    "CHUNK_SIZE",
    "NUM_SMS",
]

# Defaults for optional fields
KERNEL_DEFAULTS = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 0,
    "waves_per_eu": 0,
    "matrix_instr_nonkdim": 16,
    "kpack": 1,
    "CHUNK_SIZE": 0,
    "NUM_SMS": 0,
}

# GEMM CSV columns (match GemmKernelSelection format)
GEMM_COLUMNS = [
    "GEMMID",
    "m",
    "n",
    "k",
    "lda",
    "stride_a",
    "ldb",
    "stride_b",
    "ldc",
    "stride_c",
    "batch_count",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shape_key(entry):
    """Return a hashable key for a GEMM shape from a bench entry."""
    sd = entry.get("sizeDict", entry)
    m = sd.get("M", sd.get("m", 0))
    n = sd.get("N", sd.get("n", 0))
    k = sd.get("K", sd.get("k", 0))
    batch = sd.get("batch_count", sd.get("B", 1))
    return (int(m), int(n), int(k), int(batch))


def _config_key(entry):
    """Return a hashable tuple of kernel config values from a bench entry."""
    cfg = entry.get("config", entry)
    vals = []
    for field in KERNEL_FIELDS:
        v = cfg.get(field, KERNEL_DEFAULTS.get(field, 0))
        vals.append(int(v))
    return tuple(vals)


def _parse_entries(yaml_data):
    """Normalise various YAML layouts into a flat list of entries.

    Each entry has keys: sizeDict (with M, N, K), config, tflops, time_us.
    """
    if isinstance(yaml_data, list):
        return yaml_data
    if isinstance(yaml_data, dict):
        # Some bench outputs use a top-level dict keyed by problem or shape
        entries = []
        for key, val in yaml_data.items():
            if isinstance(val, list):
                entries.extend(val)
            elif isinstance(val, dict):
                entries.append(val)
        return entries
    return []


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert_bench_to_training(input_paths, output_dir, gpu_arch="mi300x"):
    """Convert one or more triton_gemm_bench YAML files to training CSVs.

    Parameters
    ----------
    input_paths : list[str]
        Paths to YAML benchmark result files.
    output_dir : str
        Directory where gemms.csv, kernels.csv, perf.csv are written.
    gpu_arch : str
        GPU architecture tag (informational, stored for provenance).

    Returns
    -------
    dict
        Summary statistics: n_gemms, n_kernels, n_perf_rows.
    """
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml")

    # ------------------------------------------------------------------
    # 1. Load and merge all YAML files
    # ------------------------------------------------------------------
    all_entries = []
    for path in input_paths:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        entries = _parse_entries(data)
        all_entries.extend(entries)

    if not all_entries:
        raise ValueError(f"No benchmark entries found in {input_paths}")

    # ------------------------------------------------------------------
    # 2. Build shape -> config -> best measurement map (dedup keep best)
    # ------------------------------------------------------------------
    # Key: (shape_key, config_key) -> {tflops, time_us}
    best_measurements = {}
    for entry in all_entries:
        sk = _shape_key(entry)
        ck = _config_key(entry)
        tflops = float(entry.get("tflops", entry.get("TFLOPS", 0.0)))
        time_us = float(entry.get("time_us", entry.get("us", 0.0)))

        key = (sk, ck)
        if key in best_measurements:
            if tflops > best_measurements[key]["tflops"]:
                best_measurements[key] = {"tflops": tflops, "time_us": time_us}
        else:
            best_measurements[key] = {"tflops": tflops, "time_us": time_us}

    # ------------------------------------------------------------------
    # 3. Assign GEMM IDs and Kernel IDs
    # ------------------------------------------------------------------
    unique_shapes = {}  # shape_key -> GEMMID
    unique_configs = {}  # config_key -> KernelID

    shape_id_counter = 0
    config_id_counter = 0

    for (sk, ck) in best_measurements:
        if sk not in unique_shapes:
            unique_shapes[sk] = shape_id_counter
            shape_id_counter += 1
        if ck not in unique_configs:
            unique_configs[ck] = config_id_counter
            config_id_counter += 1

    # ------------------------------------------------------------------
    # 4. Build gemms.csv rows
    # ------------------------------------------------------------------
    gemm_rows = []
    for (m, n, k, batch), gid in sorted(unique_shapes.items(), key=lambda x: x[1]):
        # Default leading dimensions for TN layout
        lda = k
        ldb = k
        ldc = n
        stride_a = m * k
        stride_b = k * n
        stride_c = m * n
        gemm_rows.append(
            [gid, m, n, k, lda, stride_a, ldb, stride_b, ldc, stride_c, batch]
        )

    # ------------------------------------------------------------------
    # 5. Build kernels.csv rows
    # ------------------------------------------------------------------
    kernel_rows = []
    for cfg_tuple, kid in sorted(unique_configs.items(), key=lambda x: x[1]):
        row = [kid] + list(cfg_tuple)
        kernel_rows.append(row)

    # ------------------------------------------------------------------
    # 6. Build perf.csv rows, computing EFF
    # ------------------------------------------------------------------
    # First pass: find best TFLOPS per GEMM
    best_per_gemm = defaultdict(float)
    for (sk, ck), meas in best_measurements.items():
        gid = unique_shapes[sk]
        if meas["tflops"] > best_per_gemm[gid]:
            best_per_gemm[gid] = meas["tflops"]

    perf_rows = []
    for (sk, ck), meas in best_measurements.items():
        gid = unique_shapes[sk]
        kid = unique_configs[ck]
        best = best_per_gemm[gid]
        eff = meas["tflops"] / best if best > 0 else 1.0
        perf_rows.append([gid, kid, eff, meas["time_us"]])

    # Check for shapes with only one config
    configs_per_shape = defaultdict(int)
    for (sk, ck) in best_measurements:
        configs_per_shape[sk] += 1
    single_config_shapes = {sk for sk, cnt in configs_per_shape.items() if cnt == 1}
    if single_config_shapes:
        warnings.warn(
            f"{len(single_config_shapes)} shapes have only 1 config each "
            f"(EFF=1.0 for all). Consider collecting exhaustive benchmarks.",
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # 7. Write CSVs
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    _write_csv(
        os.path.join(output_dir, "gemms.csv"),
        GEMM_COLUMNS,
        gemm_rows,
    )

    _write_csv(
        os.path.join(output_dir, "kernels.csv"),
        ["KernelID"] + KERNEL_FIELDS,
        kernel_rows,
    )

    _write_csv(
        os.path.join(output_dir, "perf.csv"),
        ["GEMMID", "KernelID", "EFF", "us"],
        perf_rows,
    )

    stats = {
        "n_gemms": len(gemm_rows),
        "n_kernels": len(kernel_rows),
        "n_perf_rows": len(perf_rows),
        "n_single_config_shapes": len(single_config_shapes),
        "gpu_arch": gpu_arch,
    }

    print(f"[collect_data] Wrote training CSVs to {output_dir}")
    print(f"  gemms:   {stats['n_gemms']} shapes")
    print(f"  kernels: {stats['n_kernels']} configs")
    print(f"  perf:    {stats['n_perf_rows']} measurements")
    if single_config_shapes:
        print(
            f"  WARNING: {stats['n_single_config_shapes']} shapes with single config"
        )

    return stats


def _write_csv(path, columns, rows):
    """Write CSV without requiring pandas."""
    with open(path, "w") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


# ---------------------------------------------------------------------------
# Synthetic data generator (for testing)
# ---------------------------------------------------------------------------


def generate_synthetic_data(output_dir, n_gemms=100, n_kernels=50, seed=42):
    """Generate synthetic training data for smoke tests.

    Creates realistic-looking CSVs with random shapes and kernel configs.

    Parameters
    ----------
    output_dir : str
        Directory where gemms.csv, kernels.csv, perf.csv are written.
    n_gemms : int
        Number of unique GEMM shapes to generate.
    n_kernels : int
        Number of unique kernel configs to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Summary with n_gemms, n_kernels, n_perf_rows.
    """
    rng = random.Random(seed)

    os.makedirs(output_dir, exist_ok=True)

    # --- gemms.csv ---
    m_choices = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    n_choices = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    k_choices = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    gemm_rows = []
    used_shapes = set()
    for gid in range(n_gemms):
        while True:
            m = rng.choice(m_choices)
            n = rng.choice(n_choices)
            k = rng.choice(k_choices)
            if (m, n, k) not in used_shapes:
                used_shapes.add((m, n, k))
                break
        batch = 1
        lda = k
        stride_a = m * k
        ldb = k
        stride_b = k * n
        ldc = n
        stride_c = m * n
        gemm_rows.append(
            [gid, m, n, k, lda, stride_a, ldb, stride_b, ldc, stride_c, batch]
        )

    _write_csv(
        os.path.join(output_dir, "gemms.csv"),
        GEMM_COLUMNS,
        gemm_rows,
    )

    # --- kernels.csv ---
    block_m_choices = [16, 32, 64, 128, 256]
    block_n_choices = [16, 32, 64, 128, 256]
    block_k_choices = [16, 32, 64, 128, 256]
    group_m_choices = [1, 2, 4, 8, 16]
    warps_choices = [1, 2, 4, 8]
    stages_choices = [0, 1, 2, 3, 4]
    waves_choices = [0, 1, 2]
    mi_nonkdim_choices = [16, 32]
    kpack_choices = [1, 2]

    kernel_rows = []
    used_configs = set()
    for kid in range(n_kernels):
        while True:
            cfg = (
                rng.choice(block_m_choices),
                rng.choice(block_n_choices),
                rng.choice(block_k_choices),
                rng.choice(group_m_choices),
                rng.choice(warps_choices),
                rng.choice(stages_choices),
                rng.choice(waves_choices),
                rng.choice(mi_nonkdim_choices),
                rng.choice(kpack_choices),
                0,  # CHUNK_SIZE
                0,  # NUM_SMS
            )
            if cfg not in used_configs:
                used_configs.add(cfg)
                break
        kernel_rows.append([kid] + list(cfg))

    _write_csv(
        os.path.join(output_dir, "kernels.csv"),
        ["KernelID"] + KERNEL_FIELDS,
        kernel_rows,
    )

    # --- perf.csv ---
    # For each GEMM, pick a random subset of kernels, generate synthetic TFLOPS
    perf_rows = []
    for gid in range(n_gemms):
        n_configs = rng.randint(max(1, n_kernels // 5), n_kernels)
        chosen_kernels = rng.sample(range(n_kernels), n_configs)

        # Generate synthetic performance with realistic distribution
        base_tflops = rng.uniform(50, 500)
        tflops_values = {}
        for kid in chosen_kernels:
            relative = rng.betavariate(3, 2)  # Skewed towards higher values
            tflops_values[kid] = base_tflops * max(0.1, relative)

        best_tflops = max(tflops_values.values())

        for kid in chosen_kernels:
            eff = tflops_values[kid] / best_tflops
            # Synthetic time: inversely proportional to tflops
            flops = gemm_rows[gid][1] * gemm_rows[gid][2] * gemm_rows[gid][3] * 2
            time_us = (
                flops / (tflops_values[kid] * 1e12) * 1e6
                if tflops_values[kid] > 0
                else 9999.0
            )
            perf_rows.append([gid, kid, round(eff, 6), round(time_us, 2)])

    _write_csv(
        os.path.join(output_dir, "perf.csv"),
        ["GEMMID", "KernelID", "EFF", "us"],
        perf_rows,
    )

    stats = {
        "n_gemms": n_gemms,
        "n_kernels": n_kernels,
        "n_perf_rows": len(perf_rows),
    }

    print(f"[collect_data] Generated synthetic training data in {output_dir}")
    print(f"  gemms:   {stats['n_gemms']} shapes")
    print(f"  kernels: {stats['n_kernels']} configs")
    print(f"  perf:    {stats['n_perf_rows']} measurements")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert triton_gemm_bench YAML to Two Tower training CSVs"
    )
    sub = parser.add_subparsers(dest="command")

    # -- convert subcommand --
    conv = sub.add_parser("convert", help="Convert YAML bench files to training CSVs")
    conv.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Path(s) to triton_gemm_bench YAML file(s)",
    )
    conv.add_argument(
        "--output", required=True, help="Output directory for training CSVs"
    )
    conv.add_argument("--gpu_arch", default="mi300x", help="GPU architecture tag")

    # -- synthetic subcommand --
    syn = sub.add_parser("synthetic", help="Generate synthetic training data")
    syn.add_argument(
        "--output", required=True, help="Output directory for synthetic CSVs"
    )
    syn.add_argument(
        "--n_gemms", type=int, default=100, help="Number of GEMM shapes"
    )
    syn.add_argument(
        "--n_kernels", type=int, default=50, help="Number of kernel configs"
    )
    syn.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Support direct --input/--output (no subcommand) for the CLI interface
    if args.command is None:
        parser2 = argparse.ArgumentParser(
            description="Convert triton_gemm_bench YAML to Two Tower training CSVs"
        )
        parser2.add_argument(
            "--input",
            nargs="+",
            required=True,
            help="Path(s) to triton_gemm_bench YAML file(s)",
        )
        parser2.add_argument(
            "--output", required=True, help="Output directory for training CSVs"
        )
        parser2.add_argument(
            "--gpu_arch", default="mi300x", help="GPU architecture tag"
        )
        args = parser2.parse_args()
        convert_bench_to_training(args.input, args.output, args.gpu_arch)
    elif args.command == "convert":
        convert_bench_to_training(args.input, args.output, args.gpu_arch)
    elif args.command == "synthetic":
        generate_synthetic_data(args.output, args.n_gemms, args.n_kernels, args.seed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
