#!/usr/bin/env python3
"""
K-055 Overlap Sweep Analysis — Crossover Detection, Efficiency Metrics, Visualizations.

Reads CSV files produced by overlap.py standard mode and computes:
  - overlap_efficiency per config
  - crossover points where one backend overtakes another
  - CU sensitivity (dTFLOPS/dCU)
  - comm_compute_ratio

Produces matplotlib figures and a markdown summary.

Usage:
    python3 k055_analyze.py results1.csv results2.csv --output-dir ./analysis_out
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKENDS_ORDERED = ["persistent", "ws", "streamk", "ws-global", "ws-hierarchical", "torch"]
BACKEND_COLORS = {
    "persistent": "#1f77b4",
    "ws": "#ff7f0e",
    "streamk": "#2ca02c",
    "ws-global": "#d62728",
    "ws-hierarchical": "#9467bd",
    "torch": "#8c564b",
}
PHASE_STAT_PREFIXES = [
    "gemm_alone",
    "comm_alone",
    "rotating_gemm",
    "serial_gemm",
    "overlap_wall",
    "overlap_gemm",
    "overlap_comm",
    "overlap_rot_wall",
    "overlap_rot_gemm",
    "overlap_rot_comm",
]
STAT_SUFFIXES = ["min", "max", "mean", "median"]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_csvs(paths: List[str]) -> pd.DataFrame:
    """Load and concatenate one or more CSV files from overlap.py output."""
    frames: List[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        df["_source_file"] = os.path.basename(p)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # Coerce numeric columns
    numeric_cols = []
    for prefix in PHASE_STAT_PREFIXES:
        for suffix in STAT_SUFFIXES:
            col = f"{prefix}_{suffix}"
            if col in combined.columns:
                numeric_cols.append(col)

    for col in numeric_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    for col in [
        "m", "n", "k", "total_cus", "world_size", "warmup", "steps",
        "overlap_efficiency_pct", "overlap_efficiency_pct_mean",
        "gemm_slowdown_vs_warm", "gemm_slowdown_vs_rotating", "comm_slowdown",
    ]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    return combined


# ---------------------------------------------------------------------------
# Derived Metrics
# ---------------------------------------------------------------------------


def compute_shape_key(df: pd.DataFrame) -> pd.Series:
    """Create a string shape key (M,N,K) for grouping."""
    return df["m"].astype(str) + "," + df["n"].astype(str) + "," + df["k"].astype(str)


def compute_overlap_efficiency(df: pd.DataFrame) -> pd.Series:
    """
    overlap_efficiency = 1 - (overlap_time / (comm_time + compute_time))

    Uses median values. Overlap time is overlap_wall_median.
    Comm time is comm_alone_median. Compute time is gemm_alone_median.
    A value of 1.0 = perfect overlap; 0.0 = no overlap; negative = overhead.
    """
    comm_time = df["comm_alone_median"]
    compute_time = df["gemm_alone_median"]
    overlap_time = df["overlap_wall_median"]
    sequential_time = comm_time + compute_time
    # Guard against division by zero
    safe_seq = sequential_time.replace(0, np.nan)
    return 1.0 - (overlap_time / safe_seq)


def compute_comm_compute_ratio(df: pd.DataFrame) -> pd.Series:
    """comm_compute_ratio = comm_alone_median / gemm_alone_median."""
    gemm = df["gemm_alone_median"].replace(0, np.nan)
    return df["comm_alone_median"] / gemm


def compute_tflops(df: pd.DataFrame) -> pd.Series:
    """Compute TFLOPS from M, N, K and overlap_wall_median (seconds).
    FLOPS = 2 * M * N * K / time.  TFLOPS = FLOPS / 1e12.
    """
    flops = 2.0 * df["m"] * df["n"] * df["k"]
    time_s = df["overlap_wall_median"].replace(0, np.nan)
    return flops / time_s / 1e12


def compute_gemm_tflops(df: pd.DataFrame) -> pd.Series:
    """Compute TFLOPS from M, N, K and gemm_alone_median (seconds).
    Pure GEMM throughput without overlap.
    """
    flops = 2.0 * df["m"] * df["n"] * df["k"]
    time_s = df["gemm_alone_median"].replace(0, np.nan)
    return flops / time_s / 1e12


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived metric columns to the dataframe."""
    df = df.copy()
    df["shape_key"] = compute_shape_key(df)
    df["overlap_efficiency"] = compute_overlap_efficiency(df)
    df["comm_compute_ratio"] = compute_comm_compute_ratio(df)
    df["tflops_overlap"] = compute_tflops(df)
    df["gemm_tflops"] = compute_gemm_tflops(df)
    return df


def compute_cu_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (shape, backend), compute dTFLOPS/dCU as finite differences.
    Returns a dataframe with columns: shape_key, backend, total_cus, cu_sensitivity.
    """
    records = []
    for (shape, backend), grp in df.groupby(["shape_key", "backend"]):
        grp_sorted = grp.sort_values("total_cus")
        cus = grp_sorted["total_cus"].values
        tflops = grp_sorted["tflops_overlap"].values
        if len(cus) < 2:
            continue
        # Forward differences
        for i in range(len(cus) - 1):
            dcu = cus[i + 1] - cus[i]
            if dcu == 0:
                continue
            dtflops = tflops[i + 1] - tflops[i]
            records.append({
                "shape_key": shape,
                "backend": backend,
                "total_cus": (cus[i] + cus[i + 1]) / 2,  # midpoint
                "cu_sensitivity": dtflops / dcu,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Crossover Detection
# ---------------------------------------------------------------------------


def detect_crossovers(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each shape, find CU counts where one backend overtakes another.

    Returns a dataframe with columns:
        shape_key, cu_count, winner, loser, winner_tflops, loser_tflops
    """
    crossover_records = []
    backends = [b for b in BACKENDS_ORDERED if b in df["backend"].unique()]
    if len(backends) < 2:
        return pd.DataFrame()

    for shape, shape_grp in df.groupby("shape_key"):
        # Build a matrix: cu_count -> backend -> tflops
        pivot = shape_grp.pivot_table(
            index="total_cus", columns="backend", values="tflops_overlap", aggfunc="first"
        )
        pivot = pivot.sort_index()
        available = [b for b in backends if b in pivot.columns]
        if len(available) < 2:
            continue

        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                b1, b2 = available[i], available[j]
                vals1 = pivot[b1].values
                vals2 = pivot[b2].values
                cus = pivot.index.values
                # Check for sign changes in (b1 - b2)
                diff = vals1 - vals2
                for idx in range(len(diff) - 1):
                    if np.isnan(diff[idx]) or np.isnan(diff[idx + 1]):
                        continue
                    if diff[idx] * diff[idx + 1] < 0:
                        # Crossover between cus[idx] and cus[idx+1]
                        # Linearly interpolate
                        d0, d1 = diff[idx], diff[idx + 1]
                        frac = abs(d0) / (abs(d0) + abs(d1))
                        crossover_cu = cus[idx] + frac * (cus[idx + 1] - cus[idx])
                        # At crossover, who was winning before?
                        if diff[idx] > 0:
                            winner_before, winner_after = b1, b2
                        else:
                            winner_before, winner_after = b2, b1
                        crossover_records.append({
                            "shape_key": shape,
                            "cu_count": round(crossover_cu, 1),
                            "transition": f"{winner_before} -> {winner_after}",
                            "backend_a": b1,
                            "backend_b": b2,
                            "tflops_at_lower_cu": float(
                                np.nanmean([vals1[idx], vals2[idx]])
                            ),
                            "tflops_at_upper_cu": float(
                                np.nanmean([vals1[idx + 1], vals2[idx + 1]])
                            ),
                        })
    return pd.DataFrame(crossover_records)


def compute_winner_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (shape, CU count), determine the winning backend (highest TFLOPS).
    Returns a dataframe with shape_key, total_cus, winner, best_tflops.
    """
    records = []
    for (shape, cu), grp in df.groupby(["shape_key", "total_cus"]):
        best_row = grp.loc[grp["tflops_overlap"].idxmax()]
        records.append({
            "shape_key": shape,
            "total_cus": cu,
            "winner": best_row["backend"],
            "best_tflops": best_row["tflops_overlap"],
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def plot_crossover_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """
    Crossover heatmap: shape (M,N,K) vs CU count, colored by winning backend.
    """
    winner_df = compute_winner_map(df)
    if winner_df.empty:
        return ""

    backends_present = sorted(winner_df["winner"].unique())
    backend_to_idx = {b: i for i, b in enumerate(backends_present)}
    colors = [BACKEND_COLORS.get(b, "#333333") for b in backends_present]
    cmap = mcolors.ListedColormap(colors)
    bounds = list(range(len(backends_present) + 1))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    pivot = winner_df.pivot_table(
        index="shape_key", columns="total_cus", values="winner", aggfunc="first"
    )
    pivot_numeric = pivot.map(lambda x: backend_to_idx.get(x, -1) if isinstance(x, str) else -1)

    shapes = pivot_numeric.index.tolist()
    cus = pivot_numeric.columns.tolist()

    fig, ax = plt.subplots(figsize=(max(10, len(cus) * 0.6), max(4, len(shapes) * 0.5)))
    im = ax.imshow(pivot_numeric.values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(len(cus)))
    ax.set_xticklabels([str(int(c)) for c in cus], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(shapes)))
    ax.set_yticklabels(shapes, fontsize=7)
    ax.set_xlabel("Total CUs")
    ax.set_ylabel("Shape (M,N,K)")
    ax.set_title("Winning Backend by Shape and CU Count")

    # Legend
    patches = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i]) for i in range(len(backends_present))]
    ax.legend(patches, backends_present, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, "crossover_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cu_sensitivity(df: pd.DataFrame, output_dir: str) -> str:
    """
    TFLOPS vs CU count, one line per backend, for representative shapes.
    Pick up to 6 shapes by selecting the most data-rich ones.
    """
    # Select representative shapes (most data points)
    shape_counts = df.groupby("shape_key").size().sort_values(ascending=False)
    rep_shapes = shape_counts.head(6).index.tolist()
    if not rep_shapes:
        return ""

    n_shapes = len(rep_shapes)
    ncols = min(3, n_shapes)
    nrows = (n_shapes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, shape in enumerate(rep_shapes):
        ax = axes[idx // ncols][idx % ncols]
        shape_df = df[df["shape_key"] == shape]
        for backend in BACKENDS_ORDERED:
            bdf = shape_df[shape_df["backend"] == backend].sort_values("total_cus")
            if bdf.empty:
                continue
            ax.plot(
                bdf["total_cus"], bdf["tflops_overlap"],
                marker="o", markersize=3, linewidth=1.5,
                label=backend, color=BACKEND_COLORS.get(backend, "#333"),
            )
        ax.set_xlabel("Total CUs")
        ax.set_ylabel("TFLOPS (overlap)")
        ax.set_title(f"Shape {shape}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_shapes, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("CU Sensitivity: TFLOPS vs CU Count", fontsize=12)
    fig.tight_layout()
    path = os.path.join(output_dir, "cu_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_overlap_efficiency_histogram(df: pd.DataFrame, output_dir: str) -> str:
    """Distribution of overlap efficiency across all configs."""
    eff = df["overlap_efficiency"].dropna()
    if eff.empty:
        return ""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(eff, bins=50, edgecolor="black", alpha=0.75, color="#4C72B0")
    ax.axvline(eff.median(), color="red", linestyle="--", linewidth=1.5, label=f"Median: {eff.median():.3f}")
    ax.axvline(eff.mean(), color="orange", linestyle="--", linewidth=1.5, label=f"Mean: {eff.mean():.3f}")
    ax.set_xlabel("Overlap Efficiency")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Overlap Efficiency Across All Configs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "overlap_efficiency_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_backend_comparison(df: pd.DataFrame, output_dir: str) -> str:
    """
    Best TFLOPS per backend at key CU allocation points.
    Selects a few representative CU counts and shows a grouped bar chart.
    """
    all_cus = sorted(df["total_cus"].dropna().unique())
    if len(all_cus) == 0:
        return ""

    # Pick ~5 key CU points spread across the range
    n_points = min(5, len(all_cus))
    indices = np.linspace(0, len(all_cus) - 1, n_points, dtype=int)
    key_cus = [all_cus[i] for i in indices]

    backends_present = [b for b in BACKENDS_ORDERED if b in df["backend"].unique()]
    if not backends_present:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(key_cus))
    width = 0.8 / len(backends_present)

    for i, backend in enumerate(backends_present):
        best_tflops = []
        for cu in key_cus:
            subset = df[(df["backend"] == backend) & (df["total_cus"] == cu)]
            if subset.empty or subset["tflops_overlap"].isna().all():
                best_tflops.append(0)
            else:
                best_tflops.append(subset["tflops_overlap"].max())
        offset = (i - len(backends_present) / 2 + 0.5) * width
        ax.bar(
            x + offset, best_tflops, width,
            label=backend, color=BACKEND_COLORS.get(backend, "#333"),
        )

    ax.set_xlabel("Total CUs")
    ax.set_ylabel("Best TFLOPS (overlap)")
    ax.set_title("Best TFLOPS per Backend at Key CU Allocations")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in key_cus])
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "backend_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_comm_compute_tradeoff(df: pd.DataFrame, output_dir: str) -> str:
    """
    Scatter: overlap_efficiency vs comm_compute_ratio, colored by backend.
    """
    df_valid = df.dropna(subset=["overlap_efficiency", "comm_compute_ratio"])
    if df_valid.empty:
        return ""

    fig, ax = plt.subplots(figsize=(8, 6))
    for backend in BACKENDS_ORDERED:
        bdf = df_valid[df_valid["backend"] == backend]
        if bdf.empty:
            continue
        ax.scatter(
            bdf["comm_compute_ratio"], bdf["overlap_efficiency"],
            label=backend, color=BACKEND_COLORS.get(backend, "#333"),
            alpha=0.6, s=30, edgecolors="black", linewidths=0.3,
        )
    ax.set_xlabel("Comm/Compute Ratio")
    ax.set_ylabel("Overlap Efficiency")
    ax.set_title("Comm-Compute Tradeoff: Overlap Efficiency vs Comm/Compute Ratio")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "comm_compute_tradeoff.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------


def generate_summary(
    df: pd.DataFrame,
    crossovers: pd.DataFrame,
    output_dir: str,
    figure_paths: Dict[str, str],
) -> str:
    """Generate k055_summary.md with key findings."""
    lines: List[str] = []
    lines.append("# K-055 Overlap Sweep Analysis Summary\n")

    # -- Dataset overview --
    lines.append("## Dataset Overview\n")
    lines.append(f"- **Total configurations**: {len(df)}")
    backends = sorted(df["backend"].unique())
    lines.append(f"- **Backends**: {', '.join(backends)}")
    shapes = sorted(df["shape_key"].unique())
    lines.append(f"- **Shapes tested**: {len(shapes)}")
    cus = sorted(df["total_cus"].dropna().unique())
    if len(cus) > 0:
        lines.append(f"- **CU range**: {int(min(cus))} - {int(max(cus))} ({len(cus)} points)")
    lines.append("")

    # -- Overlap efficiency stats --
    lines.append("## Overlap Efficiency Statistics\n")
    eff = df["overlap_efficiency"].dropna()
    if not eff.empty:
        lines.append(f"- **Mean**: {eff.mean():.4f}")
        lines.append(f"- **Median**: {eff.median():.4f}")
        lines.append(f"- **Std**: {eff.std():.4f}")
        lines.append(f"- **Min**: {eff.min():.4f}")
        lines.append(f"- **Max**: {eff.max():.4f}")
        lines.append(f"- **Configs with >50% overlap**: {(eff > 0.5).sum()} / {len(eff)}")
    lines.append("")

    # -- Best backend per CU regime --
    lines.append("## Best Backend per CU Regime\n")
    winner_df = compute_winner_map(df)
    if not winner_df.empty:
        lines.append("| CU Count | Winning Backend | Best TFLOPS |")
        lines.append("|----------|----------------|-------------|")
        for cu in sorted(winner_df["total_cus"].unique()):
            cu_winners = winner_df[winner_df["total_cus"] == cu]
            # Aggregate: which backend wins most shapes?
            winner_counts = cu_winners["winner"].value_counts()
            dominant = winner_counts.index[0]
            avg_tflops = cu_winners[cu_winners["winner"] == dominant]["best_tflops"].mean()
            lines.append(
                f"| {int(cu)} | {dominant} ({winner_counts.iloc[0]}/{len(cu_winners)} shapes) "
                f"| {avg_tflops:.1f} |"
            )
    lines.append("")

    # -- Crossover points --
    lines.append("## Key Crossover Points\n")
    if crossovers.empty:
        lines.append("No crossover points detected (only one backend or insufficient CU sweep).\n")
    else:
        lines.append(f"**{len(crossovers)} crossover(s) detected**:\n")
        lines.append("| Shape | CU Count | Transition |")
        lines.append("|-------|----------|------------|")
        for _, row in crossovers.iterrows():
            lines.append(
                f"| {row['shape_key']} | {row['cu_count']:.0f} | {row['transition']} |"
            )
    lines.append("")

    # -- Per-backend summary --
    lines.append("## Per-Backend Performance\n")
    for backend in backends:
        bdf = df[df["backend"] == backend]
        tflops = bdf["tflops_overlap"].dropna()
        eff_b = bdf["overlap_efficiency"].dropna()
        lines.append(f"### {backend}\n")
        if not tflops.empty:
            lines.append(f"- **Peak TFLOPS**: {tflops.max():.1f}")
            lines.append(f"- **Median TFLOPS**: {tflops.median():.1f}")
        if not eff_b.empty:
            lines.append(f"- **Median overlap efficiency**: {eff_b.median():.4f}")
        lines.append("")

    # -- CU sensitivity --
    sens_df = compute_cu_sensitivity(df)
    if not sens_df.empty:
        lines.append("## CU Sensitivity (dTFLOPS/dCU)\n")
        for backend in backends:
            bsens = sens_df[sens_df["backend"] == backend]
            if bsens.empty:
                continue
            lines.append(
                f"- **{backend}**: mean={bsens['cu_sensitivity'].mean():.4f}, "
                f"max={bsens['cu_sensitivity'].max():.4f}"
            )
        lines.append("")

    # -- Figures --
    lines.append("## Generated Figures\n")
    for name, path in figure_paths.items():
        if path:
            rel = os.path.basename(path)
            lines.append(f"- **{name}**: ![{name}]({rel})")
    lines.append("")

    summary_text = "\n".join(lines)
    summary_path = os.path.join(output_dir, "k055_summary.md")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    return summary_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze overlap sweep CSV results: crossover detection, "
        "efficiency metrics, and visualizations."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="One or more CSV files from overlap.py standard mode",
    )
    parser.add_argument(
        "--output-dir",
        default="./k055_analysis",
        help="Directory for output figures and summary (default: ./k055_analysis)",
    )
    args = parser.parse_args()

    # Validate inputs
    for f in args.csv_files:
        if not os.path.isfile(f):
            print(f"ERROR: CSV file not found: {f}", file=sys.stderr)
            sys.exit(1)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {len(args.csv_files)} CSV file(s)...")
    df = load_csvs(args.csv_files)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Validate required columns
    required = ["m", "n", "k", "backend", "gemm_alone_median", "comm_alone_median", "overlap_wall_median"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        print(f"  Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Fill total_cus with 304 (MI300X default) if missing
    if "total_cus" not in df.columns:
        df["total_cus"] = 304
    else:
        df["total_cus"] = df["total_cus"].fillna(304)

    print("Computing derived metrics...")
    df = add_derived_columns(df)

    print("Detecting crossover points...")
    crossovers = detect_crossovers(df)
    if not crossovers.empty:
        print(f"  Found {len(crossovers)} crossover(s)")
        crossovers.to_csv(os.path.join(output_dir, "crossovers.csv"), index=False)
    else:
        print("  No crossovers detected")

    # Save enriched data
    enriched_path = os.path.join(output_dir, "enriched_data.csv")
    df.to_csv(enriched_path, index=False)
    print(f"  Enriched data saved to {enriched_path}")

    print("Generating visualizations...")
    figure_paths: Dict[str, str] = {}

    figure_paths["Crossover Heatmap"] = plot_crossover_heatmap(df, output_dir)
    figure_paths["CU Sensitivity"] = plot_cu_sensitivity(df, output_dir)
    figure_paths["Overlap Efficiency Histogram"] = plot_overlap_efficiency_histogram(df, output_dir)
    figure_paths["Backend Comparison"] = plot_backend_comparison(df, output_dir)
    figure_paths["Comm-Compute Tradeoff"] = plot_comm_compute_tradeoff(df, output_dir)

    generated = [k for k, v in figure_paths.items() if v]
    print(f"  Generated {len(generated)} figure(s): {', '.join(generated)}")

    print("Generating summary report...")
    summary_path = generate_summary(df, crossovers, output_dir, figure_paths)
    print(f"  Summary written to {summary_path}")

    print(f"\nDone. All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
