#!/bin/bash
#SBATCH --job-name=cu-alloc-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=cu_sweep_%j.log
#SBATCH --error=cu_sweep_%j.log

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/shared/ryaswann}"
RESULTS_DIR="${SCRATCH}/cu_sweep_results_$(date +%Y%m%d_%H%M)"

echo "=========================================="
echo "CU Allocation Sweep"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $(hostname)"
echo "GPUs:        $(rocm-smi --showid 2>/dev/null | grep -c GPU || echo 'unknown')"
echo "Results dir: $RESULTS_DIR"
echo "=========================================="

# Verify GPUs are idle
rocm-smi --showuse 2>/dev/null || true
echo ""

# Set up environment
export PYTHONPATH="${PWD}/include:${PYTHONPATH:-}"
export HSA_NO_SCRATCH_RECLAIM=1

# Tier selection: default to 'full', override with SWEEP_TIER env var
TIER="${SWEEP_TIER:-full}"
RESUME="${SWEEP_RESUME:-0}"

echo "Tier: $TIER"
echo "Resume from: $RESUME"
echo ""

python3 benchmarks/cu_allocation_sweep.py \
    --output-dir "$RESULTS_DIR" \
    --tier "$TIER" \
    --nproc 8 \
    --resume-from "$RESUME"

echo ""
echo "Results saved to: $RESULTS_DIR"
echo "CSV: $RESULTS_DIR/cu_allocation_sweep.csv"
ls -la "$RESULTS_DIR/"
