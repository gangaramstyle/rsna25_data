#!/bin/bash
#SBATCH --job-name=zarr_conversion
#SBATCH --output=slurm_logs/zarr_conv_%A_%a.out
#SBATCH --error=slurm_logs/zarr_conv_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# --- Configuration ---
# Set these paths to your actual directories
cd /cbica/home/gangarav/rsna_any/rsna_2025
export OUTPUT_ZARR_ROOT="/cbica/home/gangarav/nifti_brain"
export SERIES_CSV="OpenMind_series_to_process.csv"

# --- Setup ---
# Create log directory if it doesn't exist
mkdir -p slurm_logs

# --- Execution ---
echo "Starting job for SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

uv run python run_slurm_task.py

echo "Job finished."
