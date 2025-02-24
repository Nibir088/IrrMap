import os
import glob
import rasterio
import argparse
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from multiprocessing import Pool, cpu_count

# Root directory for Sentinel-2 data
base_path = "Sentinel-UnZip/CO/"

# Define the relevant quality masks for RGB + NIR bands
quality_masks = {
    "B02": "MSK_QUALIT_B02.jp2",
    "B03": "MSK_QUALIT_B03.jp2",
    "B04": "MSK_QUALIT_B04.jp2",
    "B08": "MSK_QUALIT_B08.jp2"
}

detector_masks = {
    "B02": "MSK_DETFOO_B02.jp2",
    "B03": "MSK_DETFOO_B03.jp2",
    "B04": "MSK_DETFOO_B04.jp2",
    "B08": "MSK_DETFOO_B08.jp2"
}

def process_qi_data(year_folder):
    """Processes all QI_DATA folders within a given year and reprojects the mask to EPSG:4326."""
    # Find all QI_DATA folders within the year
    qi_data_paths = glob.glob(os.path.join(year_folder, "*/*/GRANULE/*/QI_DATA"))

    if not qi_data_paths:
        print(f"No QI_DATA folders found in {year_folder}")
        return

    for qi_path in qi_data_paths:
        print(f"Processing: {qi_path}")

        # Derive the SAFE folder path
        safe_folder = os.path.abspath(os.path.join(qi_path, "../../../"))
        safe_name = os.path.basename(safe_folder).replace(".SAFE", "")

        # Reference 10m band for dimensions
        ref_band_files = glob.glob(os.path.join(safe_folder, "GRANULE", "*", "IMG_DATA", "R10m", f"*_B02_10m.jp2"))

        if not ref_band_files:
            print(f"No reference 10m band found for {safe_folder}, skipping.")
            continue

        ref_band_path = ref_band_files[0]  # Use first found file
        print(f"Reference Band Path: {ref_band_path}")

        # Open reference raster to get dimensions and CRS
        with rasterio.open(ref_band_path) as ref_raster:
            width, height = ref_raster.width, ref_raster.height
            src_crs = ref_raster.crs  # Source CRS
            meta = ref_raster.meta.copy()
            meta.update(dtype=rasterio.uint8, count=1)

            # Initialize quality mask (all pixels good = 1)
            qi_mask = np.ones((height, width), dtype=np.uint8)

            # Process quality masks
            for band, filename in quality_masks.items():
                mask_path = os.path.join(qi_path, filename)
                if os.path.exists(mask_path):
                    with rasterio.open(mask_path) as src:
                        mask_data = src.read(1)
                        qi_mask *= (mask_data > 0).astype(np.uint8)  # Mark bad pixels as 0

            # Process detector footprint masks
            for band, filename in detector_masks.items():
                mask_path = os.path.join(qi_path, filename)
                if os.path.exists(mask_path):
                    with rasterio.open(mask_path) as src:
                        mask_data = src.read(1)
                        qi_mask *= (mask_data > 0).astype(np.uint8)  # Mark bad pixels as 0

            # Convert CRS to EPSG:4326
            dst_crs = "EPSG:4326"
            transform, width, height = calculate_default_transform(src_crs, dst_crs, width, height, *ref_raster.bounds)

            reprojected_mask = np.empty((height, width), dtype=np.uint8)

            reproject(
                source=qi_mask,
                destination=reprojected_mask,
                src_transform=ref_raster.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest  # Nearest neighbor for binary mask
            )

            # Update metadata for the new CRS
            meta.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})

            # Save the quality mask
            output_path = os.path.join(safe_folder, f"{safe_name}_QA_PIXEL.tif")
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(reprojected_mask, 1)

            print(f"Quality mask reprojected & saved: {output_path}")

def main():
    """Uses multiprocessing to process multiple years in parallel."""
    years = glob.glob(os.path.join(base_path, "*/"))  # Find all year directories

    num_workers = min(cpu_count(), len(years))  # Use available cores but not exceed the number of years
    print(f"Using {num_workers} workers for parallel processing.")

    with Pool(processes=num_workers) as pool:
        pool.map(process_qi_data, years)  # Run process_qi_data in parallel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create QA-Band')
    parser.add_argument('-p', '--path', type=str, required=True,
                       help='Path to Downloaded Directory')
    args = parser.parse_args()
    base_path = args.path
    main()
