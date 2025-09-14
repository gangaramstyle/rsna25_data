import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pathlib
    import pydicom
    import nibabel as nib
    import numpy as np
    import zarr
    import pandas as pd
    from tqdm.auto import tqdm
    import traceback
    import hashlib
    import fastparquet
    return pathlib, pd, zarr


@app.cell
def _(pathlib, pd, zarr):
    def OpenMind_aggregator():
        base_zarr_path = '/cbica/home/gangarav/nifti_brain/'
        openmind_df = pd.read_csv('./OpenMind_series_to_process.csv')
        metadatas = []
        for _, row in openmind_df.iterrows():
            zarr_path = pathlib.Path(base_zarr_path) / row['zarr_name']
            if zarr_path.exists():
                zarr_group = zarr.open(zarr_path, mode='r')
                metadata = dict(zarr_group.attrs)
                metadata["patient_id"] = row['patient']
                metadata["series_uid"] = row['zarr_name']

                parts = row['zarr_name'].split(':')
                if len(parts) > 2 and 'ses' in parts[2]:
                    metadata["study_uid"] = parts[2]

                metadata["modality"] = "MR"
                metadatas.append(metadata)
            else:
                print(row)
        metadata_df = pd.DataFrame(metadatas)
        return metadata_df
    return (OpenMind_aggregator,)


@app.cell
def _(pathlib, pd, zarr):
    def FOMO_aggregator():
        base_zarr_path = '/cbica/home/gangarav/nifti_brain/'
        openmind_df = pd.read_csv('./FOMO_series_to_process.csv')
        metadatas = []
        for _, row in openmind_df.iterrows():
            zarr_path = pathlib.Path(base_zarr_path) / row['zarr_name']
            if zarr_path.exists():
                zarr_group = zarr.open(zarr_path, mode='r')
                metadata = dict(zarr_group.attrs)
                metadata["patient_id"] = row['patient']
                metadata["series_uid"] = row['zarr_name']

                parts = row['zarr_name'].split(':')
                if len(parts) > 1 and 'ses' in parts[1]:
                    metadata["study_uid"] = parts[1]

                metadata["modality"] = "MR"
                metadatas.append(metadata)
            else:
                print(row)
        metadata_df = pd.DataFrame(metadatas)
        return metadata_df
    return (FOMO_aggregator,)


@app.cell
def _(pathlib, pd, zarr):
    def RSNA_aggregator():
        base_zarr_path = '/cbica/home/gangarav/nifti_brain/'
        openmind_df = pd.read_csv('./series_to_process.csv')
        metadatas = []
        for _, row in openmind_df.iterrows():
            zarr_path = pathlib.Path(base_zarr_path) / row['zarr_name']
            if zarr_path.exists():
                zarr_group = zarr.open(zarr_path, mode='r')
                metadata = dict(zarr_group.attrs)
                metadata["series_uid"] = row['zarr_name']
                metadata["original_format"] = 'dicom'
                metadatas.append(metadata)
            else:
                print(row)
        metadata_df = pd.DataFrame(metadatas)
        return metadata_df
    return (RSNA_aggregator,)


@app.cell
def _(FOMO_aggregator, OpenMind_aggregator, RSNA_aggregator, pd):
    openmind_df = OpenMind_aggregator()
    fomo_df = FOMO_aggregator()
    rsna_df = RSNA_aggregator()

    combined_df = pd.concat([openmind_df, fomo_df, rsna_df], ignore_index=True)
    combined_df
    return (combined_df,)


@app.cell
def _(combined_df):
    combined_df.to_parquet('nifti_combined_metadata.parquet', index=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
