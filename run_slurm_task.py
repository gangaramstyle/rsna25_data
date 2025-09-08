import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import pandas as pd
    import subprocess
    import sys

    # Read environment variables passed from sbatch
    TASK_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    OUTPUT_ZARR_ROOT = os.environ['OUTPUT_ZARR_ROOT']
    SERIES_CSV = os.environ['SERIES_CSV']

    print(f"--- Starting SLURM Task ID: {TASK_ID} ---")

    # Read the CSV and select the row for this task
    # SLURM arrays are 1-based, pandas iloc is 0-based
    try:
        df = pd.read_csv(SERIES_CSV)
        series_path = df.iloc[TASK_ID - 1]['input_path']
        zarr_name = df.iloc[TASK_ID - 1]['zarr_name']
    except (FileNotFoundError, IndexError) as e:
        print(f"Error: Could not find or read row {TASK_ID-1} from {SERIES_CSV}. Error: {e}")
        sys.exit(1)

    print(f"Processing series at path: {series_path}")
    print(f"Series name: {zarr_name}")
    print(f"Zarr base folder: {OUTPUT_ZARR_ROOT}")
    return OUTPUT_ZARR_ROOT, TASK_ID, series_path, subprocess, sys, zarr_name


@app.cell
def _(OUTPUT_ZARR_ROOT, series_path, zarr_name):
    # Construct the command to run the main processing script
    command = [
        "python",
        "02_process_series.py",
        "--series-path", series_path,
        "--zarr-name", zarr_name,
        "--output-base-path", OUTPUT_ZARR_ROOT
    ]
    return (command,)


@app.cell
def _(TASK_ID, command, subprocess, sys):
    # Execute the command
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print("STDOUT:", process.stdout)
        print("STDERR:", process.stderr)
        print(f"--- SLURM Task ID: {TASK_ID} finished successfully. ---")
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR in SLURM Task ID: {TASK_ID} ---")
        print(f"Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        sys.exit(1)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
