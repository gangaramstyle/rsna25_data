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
    import nibabel as nib
    return mo, nib, np, pathlib, pydicom


@app.cell
def _(mo):
    url = mo.ui.text(
        label="URL:", 
        full_width=True
    )
    url
    return (url,)


@app.cell
def _(load, url):
    if url.value:
        processed_data = load(url.value)

    if processed_data:
        dicom, affines, dicom_headers, row = processed_data
    return affines, dicom, dicom_headers, row


@app.cell
def _(pathlib, process_nifti_series):
    import requests
    import tempfile
    import urllib.parse

    def load(url):
        response = requests.get(url)
        response.raise_for_status()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
            tmp_file.write(response.content)
            nifti_path = pathlib.Path(tmp_file.name)

        # Process the NIfTI file
        return process_nifti_series(nifti_path)
    return (load,)


@app.cell
def _(dicom, mo):
    if mo.running_in_notebook:
        POINT_SIZE = 4

        p1_d = mo.ui.slider(start=0, stop=dicom.shape[0]-1)
        p1_r = mo.ui.slider(start=POINT_SIZE, stop=dicom.shape[1]-1-POINT_SIZE)
        p1_c = mo.ui.slider(start=POINT_SIZE, stop=dicom.shape[2]-1-POINT_SIZE)

        mo.output.replace(
                mo.vstack([
                    p1_d,
                    p1_r,
                    p1_c
                ]),
        )
    return POINT_SIZE, p1_c, p1_d, p1_r


@app.cell
def _(
    POINT_SIZE,
    affines,
    dicom,
    mo,
    nifti_to_pt_coords,
    np,
    p1_c,
    p1_d,
    p1_r,
    row,
):
    if mo.running_in_notebook:
        drawn = np.repeat(dicom[:,:,:,np.newaxis], 3, axis=3)
        max_val = np.max(dicom)
        drawn[p1_d.value,p1_r.value-POINT_SIZE:p1_r.value+POINT_SIZE,p1_c.value-POINT_SIZE:p1_c.value+POINT_SIZE] = [max_val, 0, 0]
        mo.output.replace(
            mo.vstack([
                mo.image(src=drawn[p1_d.value, :, :], width=512),
                nifti_to_pt_coords(np.array([[p1_d.value, p1_r.value, p1_c.value]]), affines),
                row["zooms"],
                dicom.shape
            ])
        )
    return


@app.cell
def _(dicom_headers):
    print(dicom_headers)
    return


@app.cell
def _(np, pathlib, pydicom, traceback):
    def to_pt_coords(voxel_indices: np.ndarray, affine_matrices: np.ndarray) -> np.ndarray:
        """
        Transforms voxel coordinates (d, r, c) to patient coordinates (x, y, z)
        using a corresponding stack of 4x4 affine matrices.

        Args:
            voxel_indices (np.ndarray): An array of shape (N, 3) with [d, r, c] coordinates.
            affine_matrices (np.ndarray): The stack of affine matrices of shape (D, 4, 4),
                                          where D is the total number of slices.

        Returns:
            np.ndarray: An array of shape (N, 3) with corresponding [x, y, z] patient coordinates.
        """
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

        row_spacing, col_spacing = pixel_spacing[0], pixel_spacing[1]

        # The first is the direction cosine for rows (changes as column index increases).
        row_cosine = image_orientation[:3]
        # The second is the direction cosine for columns (changes as row index increases).
        col_cosine = image_orientation[3:]

        # This is realistically not used (theoretically could be used if using fractional slices)
        slice_cosine = np.cross(col_cosine, row_cosine)

        # TODO: Possible improvement is to make the slice spacing be the delta between consecutive slices
        slice_spacing = 0.0
        if hasattr(dcm, 'SpacingBetweenSlices'):
            slice_spacing = float(dcm.SpacingBetweenSlices)
        elif hasattr(dcm, 'SliceThickness'):
            slice_spacing = float(dcm.SliceThickness)

        affine = np.identity(4)

        # Column 0: Corresponds to a step in the 'row' direction
        affine[:3, 0] = col_cosine * row_spacing

        # Column 1: Corresponds to a step in the 'column' direction
        affine[:3, 1] = row_cosine * col_spacing

        # Column 2: Corresponds to a step in the 'slice' direction
        # Keep in mind that this isn't being used currently b/c 'slice' will be set to 0
        affine[:3, 2] = slice_cosine * slice_spacing

        # Set the translation part of the matrix (the 4th column)
        # This is the real-world coordinate of the center of the first voxel (0,0,0)
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
                "wc": float(second_slice.get("WindowCenter", float('inf'))),
                "ww": float(second_slice.get("WindowWidth", float('inf'))),
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
    return


@app.cell
def _(nib, np, pathlib):
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

            # b. Account for the flips. A flip on an axis `i` of size `N` means:
            #    - Negating the corresponding column in the affine.
            #    - Adding `(N-1) * (flipped_column)` to the translation vector (4th column).

            # # Row flip (axis 1)
            # h = final_data.shape[1] # Height (number of rows)
            # final_affine[:, 1] *= -1
            # final_affine[:3, 3] += (h - 1) * final_affine[:3, 1]

            # # Column flip (axis 2)
            # w = final_data.shape[2] # Width (number of columns)
            # final_affine[:, 2] *= -1
            # final_affine[:3, 3] += (w - 1) * final_affine[:3, 2]

            # 5. Calculate zooms for convenience
            # The voxel dimensions are the lengths of the first three columns of the affine.
            zooms = np.linalg.norm(final_affine[:3, :3], axis=0)

            return final_data, final_affine, canonical_img.header, {
                "pixel_data": final_data,
                "affine": final_affine,
                "zooms": tuple(zooms)
            }

        except Exception as e:
            print(f"Error processing {nifti_path}: {e}")
            return None

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
    return nifti_to_pt_coords, process_nifti_series


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
