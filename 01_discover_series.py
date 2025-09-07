import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pathlib
    import pandas as pd
    import sys
    return mo, pathlib, pd


@app.cell
def _(pathlib):
    def discover_series_generator(root_dir: pathlib.Path):
        """
        Yields all discovered DICOM series directories and NIfTI files under a root directory.
        Using a generator is memory-efficient for very large directories.
        """

        # Find NIfTI files
        nifti_files = list(root_dir.rglob('*.nii')) + list(root_dir.rglob('*.nii.gz'))
        for path in nifti_files:
            yield path.resolve()

        # Find DICOM directories by checking all subdirectories
        dicom_dirs = list(set(p.parent for p in root_dir.rglob('*.dcm')))
        for path in dicom_dirs:
            yield path.resolve() 
    return (discover_series_generator,)


@app.cell
def _(mo):
    input_dir = mo.ui.text(
        label="Root Directory to Scan:", 
        placeholder="/path/to/your/raw/data",
        value='../series',
        full_width=True
    )
    input_dir
    return (input_dir,)


@app.cell
def _(discover_series_generator, input_dir, pathlib, pd):
    found_paths = []
    if input_dir.value and pathlib.PosixPath(input_dir.value).is_dir():
        found_paths = list(discover_series_generator(pathlib.PosixPath(input_dir.value)))

        df = pd.DataFrame({
            "input_path": [str(p) for p in found_paths],
            "zarr_name": [str(p).split('/')[-1] for p in found_paths],
        })

    df
    return (df,)


@app.cell
def _(mo):
    output_filename = mo.ui.text(
        label="Output CSV:", 
        placeholder="/path/to/process.csv",
        value='series_to_process.csv',
        full_width=True
    )
    output_filename
    return (output_filename,)


@app.cell
def _(df, output_filename):
    if output_filename.value:
        df.to_csv(output_filename.value, index=False)
    return


if __name__ == "__main__":
    app.run()
