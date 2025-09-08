import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import argparse
    import pathlib
    import zarr
    import json
    import numpy as np
    import pydicom
    from pydicom.multival import MultiValue 
    import nibabel as nib
    import itertools
    import traceback
    return (
        MultiValue,
        argparse,
        itertools,
        mo,
        nib,
        np,
        pathlib,
        pydicom,
        traceback,
        zarr,
    )


@app.cell
def _(argparse, pathlib):
    parser = argparse.ArgumentParser(description="Phase 2: Process a single series into Zarr.")
    parser.add_argument("--series-path", type=pathlib.Path, required=False, help="Full path to the DICOM series dir or NIfTI file.")
    parser.add_argument("--zarr-name", type=pathlib.Path, required=False, help="Name for the outputted zarr file.")
    parser.add_argument("--output-base-path", type=pathlib.Path, required=False, help="Root directory to save Zarr files.")
    args = parser.parse_args()
    return (args,)


@app.cell
def _(mo):
    # /gpfs/fs001/cbica/home/gangarav/rsna_any/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647/

    if mo.running_in_notebook:
        series_path = mo.ui.text(
            label="Series Absolute Path:", 
            placeholder="/path/to/your/raw/data",
            full_width=True
        )
        zarr_name = mo.ui.text(
            label="Series Name (to use for zarr):", 
            placeholder="/path/to/your/raw/data",
            full_width=True
        )
        output_base_path = mo.ui.text(
            label="Output Base Path:", 
            placeholder="/path/to/your/raw/data",
            full_width=True
        )
        mo.output.replace(
            mo.vstack([
                series_path,
                zarr_name,
                output_base_path
            ])
        )
    return output_base_path, series_path, zarr_name


@app.cell
def _(args, pathlib, process_dicom_series, process_nifti_series, series_path):
    input_path = None
    if series_path.value:
        input_path = pathlib.PosixPath(series_path.value)
    elif args.series_path:
        input_path = pathlib.PosixPath(args.series_path)

    if input_path:
        if input_path.is_dir():
            processed_data = process_dicom_series(input_path)
        elif input_path.is_file() and input_path.name.endswith(('.nii', '.nii.gz')):
            processed_data = process_nifti_series(input_path)

    if processed_data:
        pixel_data, affines, dicom_headers, row = processed_data
    return affines, pixel_data, processed_data, row


@app.cell
def _(mo, pixel_data):
    if mo.running_in_notebook:
        POINT_SIZE = 4

        p1_d = mo.ui.slider(start=0, stop=pixel_data.shape[0]-1)
        p1_r = mo.ui.slider(start=POINT_SIZE, stop=pixel_data.shape[1]-1-POINT_SIZE)
        p1_c = mo.ui.slider(start=POINT_SIZE, stop=pixel_data.shape[2]-1-POINT_SIZE)

        mo.output.replace(
                mo.vstack([
                    p1_d,
                    p1_r,
                    p1_c
                ]),
        )
    return POINT_SIZE, p1_c, p1_d, p1_r


@app.cell
def _(POINT_SIZE, affines, mo, np, p1_c, p1_d, p1_r, pixel_data, to_pt_coords):
    if mo.running_in_notebook:
        drawn = np.repeat(pixel_data[:,:,:,np.newaxis], 3, axis=3)
        max_val = np.max(pixel_data)
        drawn[p1_d.value,p1_r.value-POINT_SIZE:p1_r.value+POINT_SIZE,p1_c.value-POINT_SIZE:p1_c.value+POINT_SIZE] = [max_val, 0, 0]
        mo.output.replace(
            mo.vstack([
                mo.image(src=drawn[p1_d.value], width=512),
                to_pt_coords(np.array([[p1_d.value, p1_r.value, p1_c.value]]), affines),
            ])
        )
    return


@app.cell
def _(row):
    row
    return


@app.cell
def _():
    # mo.md(
    #     f"""
    # ### DICOM Header Information for slice {p1_d.value}
    # ---
    # - **Slice Thickness:** {dicom_headers[p1_d.value].get('SliceThickness', 'N/A')}
    # - **Patient Position:** {dicom_headers[p1_d.value].get('PatientPosition', 'N/A')}
    # - **Image Position (Patient):** {dicom_headers[p1_d.value].get('ImagePositionPatient', 'N/A')}
    # - **Image Orientation (Patient):** {dicom_headers[p1_d.value].get('ImageOrientationPatient', 'N/A')}
    # - **Slice Location:** {dicom_headers[p1_d.value].get('SliceLocation', 'N/A')}
    # - **Pixel Spacing:** {dicom_headers[p1_d.value].get('PixelSpacing', 'N/A')}
    # - **Window Center:** {dicom_headers[p1_d.value].get('WindowCenter', 'N/A')}
    # - **Window Width:** {dicom_headers[p1_d.value].get('WindowWidth', 'N/A')}
    # - **Smallest Image Pixel Value:** {dicom_headers[p1_d.value].get('SmallestImagePixelValue', 'N/A')}
    # - **Largest Image Pixel Value:** {dicom_headers[p1_d.value].get('LargestImagePixelValue', 'N/A')}
    # """
    # )
    return


@app.cell
def _(
    affines,
    args,
    np,
    output_base_path,
    pathlib,
    pixel_data,
    processed_data,
    row,
    zarr,
    zarr_name,
):
    output_zarr_path = None

    if output_base_path.value and zarr_name.value:
        output_zarr_path = pathlib.PosixPath(output_base_path.value) / zarr_name.value
    elif args.output_base_path and args.zarr_name:
        output_zarr_path = pathlib.PosixPath(args.output_base_path) / args.zarr_name

    if processed_data and output_zarr_path:

        print(output_zarr_path)

        row['zarr_path'] = str(output_zarr_path)

        group = zarr.group(store=str(output_zarr_path), overwrite=True)

        _slices, height, width = pixel_data.shape

        pixel_data_arr = group.create_array(
            'pixel_data',
            data=pixel_data,
            chunks=(4, height//2, width//2)
        )

        # Create and write the 'slice_affines' array.
        slice_affines_arr = group.create_array(
            'slice_affines',
            data=affines,
        )

        # Convert numpy scalars to Python native types for JSON serialization
        json_compatible_row = {k: float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v for k, v in row.items()}

        group.attrs.put(json_compatible_row)
    return


@app.cell
def _(MultiValue, nib, np, pathlib, pydicom, traceback):
    def nifti_to_pt_coords(voxel_indices: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
        """
        Transforms voxel coordinates to patient coordinates (x, y, z) in RAS space.

        Args:
            voxel_indices (np.ndarray): An array of shape (N, 3) with voxel coordinates.
                The order must be (depth, row, column) to match the output of
                `process_nifti_series`.
            affine_matrix (np.ndarray): The 4x4 affine transformation matrix from
                `process_nifti_series`.

        Returns:
            np.ndarray: An array of shape (N, 3) with corresponding [x, y, z] patient
                coordinates (RAS+).
        """
        # nibabel's apply_affine expects the voxel coordinates as columns,
        # so we provide our (d, r, c) coordinates directly.
        patient_coords = nib.affines.apply_affine(affine_matrix, voxel_indices)
        return patient_coords

    def to_pt_coords(voxel_indices: np.ndarray, affine_matrices: np.ndarray) -> np.ndarray:
        """
        Transforms voxel coordinates (d, r, c) to patient coordinates (x, y, z)
        using a corresponding singular 4x4 of a stack of 4x4 affine matrices.
        """

        if len(affine_matrices.shape) == 2:
            # the source data is likely a nifti and just use the singular 4x4 matrix for the transform
            patient_coords = nib.affines.apply_affine(affine_matrices, voxel_indices)
            return patient_coords

        # 1. Get the slice index 'd' for each point.
        slice_indices = voxel_indices[:, 0].astype(int)

        # 2. Select the specific 4x4 affine matrix for each point's slice.
        # This results in a stack of shape (N, 4, 4).
        selected_affines = affine_matrices[slice_indices]

        # 3. Get the in-plane row and column indices (r, c).
        coords_rc = voxel_indices[:, 1:]  # Shape: (N, 2)

        # 4. Construct the 4-element homogeneous voxel coordinates [r, c, 0, 1].
        # The '0' indicates the position is on the 2D plane itself.
        # The '1' is the homogeneous coordinate.
        num_points = voxel_indices.shape[0]
        zeros_col = np.zeros((num_points, 1))
        ones_col = np.ones((num_points, 1))

        # We construct the vector [r, c, 0, 1] because our affine matrix was built
        # with column 0 for 'r' steps and column 1 for 'c' steps.
        homogeneous_coords = np.hstack([coords_rc, zeros_col, ones_col]) # Shape: (N, 4)

        # 5. Perform batched matrix multiplication.
        # We need to reshape homogeneous_coords for matmul: (N, 4) -> (N, 4, 1)
        # The operation is: (N, 4, 4) @ (N, 4, 1) -> (N, 4, 1)
        patient_coords_homogeneous = np.matmul(
            selected_affines,
            homogeneous_coords[:, :, np.newaxis]
        )

        # 6. Squeeze the result and extract the (x, y, z) components.
        # The result is (N, 4, 1), squeeze to (N, 4), then take the first 3 columns.
        patient_coords_xyz = patient_coords_homogeneous.squeeze(axis=2)[:, :3]

        return patient_coords_xyz

    def calculate_dicom_slice_affine(dcm: pydicom.FileDataset) -> np.ndarray:

        image_position = np.array(dcm.ImagePositionPatient, dtype=float)
        image_orientation = np.array(dcm.ImageOrientationPatient, dtype=float)
        pixel_spacing = np.array(dcm.PixelSpacing, dtype=float)
        # TODO: Possible improvement is to make the slice spacing be the delta between consecutive slices
        slice_spacing = float(getattr(dcm, 'SpacingBetweenSlices', getattr(dcm, 'SliceThickness', 0.0)))

        row_spacing, col_spacing = pixel_spacing[0], pixel_spacing[1]

        # The first is the direction cosine for rows (changes as column index increases).
        row_cosine = image_orientation[:3]
        # The second is the direction cosine for columns (changes as row index increases).
        col_cosine = image_orientation[3:]
        # This is realistically not used (theoretically could be used if using fractional slices)
        slice_cosine = np.cross(col_cosine, row_cosine)

        affine = np.identity(4)

        affine[:3, 0] = col_cosine * row_spacing
        affine[:3, 1] = row_cosine * col_spacing
        affine[:3, 2] = slice_cosine * slice_spacing
        affine[:3, 3] = image_position

        return affine

    def process_dicom_series(series_path: pathlib.Path):

        try:
            dicom_files = [pydicom.dcmread(p, force=True) for p in series_path.glob("*.dcm")]
            dicom_files = [dcm for dcm in dicom_files if hasattr(dcm, 'InstanceNumber')]

            if not dicom_files:
                print(series_path)
                return 1/0

            dicom_files.sort(key=lambda dcm: dcm.InstanceNumber)

            num_slices = len(dicom_files)
            print(dicom_files[0].pixel_array.shape)
            h, w = dicom_files[0].pixel_array.shape
            pixel_data = np.zeros((num_slices, h, w), dtype=dicom_files[0].pixel_array.dtype)
            slice_affines = np.zeros((num_slices, 4, 4), dtype=np.float64)

            for i, dcm in enumerate(dicom_files):
                pixel_data[i, :, :] = dcm.pixel_array
                slice_affines[i] = calculate_dicom_slice_affine(dcm)      

            stdev = np.std(pixel_data[...])
            values, counts = np.unique(pixel_data, return_counts=True)
            top_indices = np.argsort(counts)[-int(np.maximum(stdev // 10, 1)):]
            mask = ~np.isin(pixel_data, values[top_indices])
            filtered_img = pixel_data[mask]
            median = np.median(filtered_img)
            stdev = np.std(filtered_img)

            unique_orientations, counts = np.unique(
                np.round(slice_affines[:, :3, :3], decimals=3),
                axis=0,
                return_counts=True
            )

            image_position_delta = np.round(np.array(dicom_files[-1].ImagePositionPatient) - np.array(dicom_files[0].ImagePositionPatient), decimals=0)

            second_slice = dicom_files[1]

            wc_val = second_slice.get("WindowCenter")
            ww_val = second_slice.get("WindowWidth")

            final_wc = float(wc_val[0]) if isinstance(wc_val, MultiValue) else float(wc_val) if wc_val is not None else float('inf')
            final_ww = float(ww_val[0]) if isinstance(ww_val, MultiValue) else float(ww_val) if ww_val is not None else float('inf')

            return pixel_data, slice_affines, dicom_files, {
                "patient_id": str(second_slice.get("PatientID", "unknown")),
                "study_uid": str(second_slice.get("StudyInstanceUID", "unknown")),
                "series_uid": str(second_slice.get("SeriesInstanceUID", "unknown")),
                "modality": str(second_slice.get("Modality", "unknown")),
                "raw_path": str(series_path),
                "original_format": "dicom",
                "shape_d": pixel_data.shape[0],
                "shape_r": pixel_data.shape[1],
                "shape_c": pixel_data.shape[2],
                "wc": final_wc,
                "ww": final_ww,
                "min_pix": float(np.min(pixel_data)),
                "max_pix": float(np.max(pixel_data)),
                "median": median,
                "stdev": stdev,
                "num_affines": counts.shape[0],
                "delta_RL": image_position_delta[0],
                "delta_AP": image_position_delta[1],
                "delta_IS": image_position_delta[2],
            }
        except Exception:
            print(f"--- FAILED to process DICOM series: {series_path} ---")
            print(traceback.format_exc())
            return None, None, None, None
    return process_dicom_series, to_pt_coords


@app.cell
def _(itertools, nib, np, pathlib):
    def _calculate_physical_extents(data_shape: tuple, affine: np.ndarray) -> tuple[float, float, float]:
        """
        Calculates the physical extent of the volume along RAS axes.

        Args:
            data_shape (tuple): The shape of the voxel data (Depth, Height, Width).
            affine (np.ndarray): The 4x4 affine matrix mapping voxel indices to RAS coords.

        Returns:
            A tuple containing the physical size in mm along (R-L, A-P, I-S) axes.
        """
        d, h, w = data_shape
        # Define the 8 corners of the voxel grid
        corners_vox = np.array(list(itertools.product([0, d - 1], [0, h - 1], [0, w - 1])))

        # Transform voxel corners to RAS world coordinates
        corners_ras = nib.affines.apply_affine(affine, corners_vox)

        # Find the min and max RAS coordinates
        min_ras = corners_ras.min(axis=0)
        max_ras = corners_ras.max(axis=0)

        # Calculate the extent (delta) along each RAS axis
        # RAS axes are (x, y, z) which correspond to (L->R, P->A, I->S)
        delta_RL = max_ras[0] - min_ras[0]
        delta_AP = max_ras[1] - min_ras[1]
        delta_IS = max_ras[2] - min_ras[2]

        return float(delta_RL), float(delta_AP), float(delta_IS)

    def process_nifti_series(nifti_path: pathlib.Path) -> dict | None:
        """
        Processes a NIfTI file into a standard orientation for radiological viewing.

        The final pixel data will have a [depth, row, col] layout where:
        - Slicing along `depth` (axis 0) gives axial slices from Inferior to Superior.
        - An axial slice (a [row, col] plane) is oriented for "nose-up" viewing:
            - Patient's Anterior (nose) is "up" (at row index 0).
            - Patient's Right is on the "left" of the image (at column index 0).

        This is achieved by:
        1. Reorienting the NIfTI to a canonical RAS+ orientation (axes are L->R, P->A, I->S).
        2. Transposing and flipping the data axes to match the desired [I->S, A->P, R->L] layout.
        3. Correctly calculating the new affine matrix to reflect these changes.

        Args:
            nifti_path (pathlib.Path): Path to the input NIfTI file.

        Returns:
            A dictionary containing:
            - 'pixel_data': The reoriented 3D numpy array.
            - 'affine': The corresponding 4x4 affine matrix.
            - 'zooms': The voxel sizes in (depth, row, col) order.
            Returns None if the file cannot be loaded.
        """
        try:
            # 1. Load the original NIfTI image
            original_img = nib.load(nifti_path)

            # 2. Reorient to a standard canonical orientation (RAS+)
            # This is a crucial step for robustness. After this, the data axes
            # will correspond to (Left-to-Right, Posterior-to-Anterior, Inferior-to-Superior).
            canonical_img = nib.as_closest_canonical(original_img)
            canonical_data = canonical_img.get_fdata(dtype=np.float32)
            canonical_affine = canonical_img.affine

            # 3. Get the data into the desired [Depth, Row, Col] layout.
            # Current axes: (L->R, P->A, I->S)
            # Desired axes: (I->S, A->P, R->L)

            # a. Transpose to get the anatomical axes in the right order.
            # From (LR, PA, IS) to (IS, PA, LR)
            transposed_data = np.transpose(canonical_data, (2, 1, 0))

            # b. Flip the axes to get the correct direction.
            # Axis 1 (Row): P->A needs to become A->P. Flip it.
            # Axis 2 (Col): L->R needs to become R->L. Flip it.
            final_data = np.flip(transposed_data, axis=1) # Flip rows
            final_data = np.flip(final_data, axis=2) # Flip columns

            # 4. Calculate the new affine matrix that corresponds to `final_data`.
            # This is the most complex part. We start with the canonical affine and
            # apply the same transformations to it.

            # a. Account for the transpose (2, 1, 0) by swapping columns 0 and 2.
            final_affine = canonical_affine.copy()
            final_affine[:, [0, 2]] = final_affine[:, [2, 0]]

            # 5. Calculate zooms for convenience
            # The voxel dimensions are the lengths of the first three columns of the affine.
            zooms = np.linalg.norm(final_affine[:3, :3], axis=0)

            stdev = np.std(final_data)
            values, counts = np.unique(final_data, return_counts=True)
            top_indices = np.argsort(counts)[-int(np.maximum(stdev // 10, 1)):]
            mask = ~np.isin(final_data, values[top_indices])
            filtered_img = final_data[mask]
            median = np.median(filtered_img)
            stdev = np.std(filtered_img)

            delta_RL, delta_AP, delta_IS = _calculate_physical_extents(final_data.shape, final_affine)

            return final_data, final_affine, canonical_img.header, {
                "patient_id": "unknown",
                "study_uid": "unknown",
                "series_uid": "unknown",
                "modality": "unknown",
                "raw_path": str(nifti_path),
                "original_format": "nifti",
                "shape_d": final_data.shape[0],
                "shape_r": final_data.shape[1],
                "shape_c": final_data.shape[2],
                "wc": -666.0,
                "ww": -666.0,
                "min_pix": float(np.min(final_data)),
                "max_pix": float(np.max(final_data)),
                "median": median,
                "stdev": stdev,
                "num_affines": 1,
                "delta_RL": delta_RL,
                "delta_AP": delta_AP,
                "delta_IS": delta_IS,
            }

        except Exception as e:
            print(f"Error processing {nifti_path}: {e}")
            return None
    return (process_nifti_series,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
