import zipfile
import tempfile
from pathlib import Path

import rasterio
from rasterio.merge import merge


def reconstruct_prediction_from_zip(zip_path, out_tif, method="first"):
    """
    Reconstruct a full raster from georeferenced TIFF tiles stored in a zip file.
    Assumes tiles already contain correct CRS and transform metadata.
    """

    zip_path = Path(zip_path)
    out_tif = Path(out_tif)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract TIFFs
        with zipfile.ZipFile(zip_path, "r") as zf:
            tif_files = [f for f in zf.namelist() if f.lower().endswith(".tif")]

            print(f"Found {len(tif_files)} TIFF tiles")

            for f in tif_files:
                zf.extract(f, tmpdir)

        # Collect extracted TIFFs
        tile_paths = sorted(tmpdir.rglob("*.tif"))

        if len(tile_paths) == 0:
            raise RuntimeError("No TIFF tiles found after extraction")

        print(f"Extracted {len(tile_paths)} tiles")

        # Open datasets
        datasets = [rasterio.open(p) for p in tile_paths]

        try:
            mosaic, transform = merge(datasets, method=method)

            profile = datasets[0].profile.copy()
            profile.update(
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                transform=transform
            )

            with rasterio.open(out_tif, "w", **profile) as dst:
                dst.write(mosaic)

            print("Saved mosaic to:", out_tif)

        finally:
            for ds in datasets:
                ds.close()


if __name__ == "__main__":

    zip_path = r"..." # Path to the zip file containing the prediction TIFF tiles
    out_tif = r"..." # Path to save the reconstructed mosaic TIFF

    reconstruct_prediction_from_zip(zip_path, out_tif)