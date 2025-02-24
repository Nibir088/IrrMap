import os
import rasterio
import numpy as np
import glob
import logging
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
os.environ["GDAL_PAM_ENABLED"] = "NO" 

# Define bands needed (10m, 20m, 60m)
bands_10m = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR
bands_20m = ["B05", "B06", "B07", "B11", "B12"]  # Red Edge, SWIR1, SWIR2
bands_60m = ["B01", "B09"]  # Coastal, Water Vapor

# Log files
missing_bands_log = "wa_missing_bands.txt"
skipped_folders_log = "wa_skipped_folders.txt"

# Clear previous logs
open(missing_bands_log, "w").close()
open(skipped_folders_log, "w").close()

# Set up logging for errors
logging.basicConfig(
    filename='processing_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

TARGET_CRS = "EPSG:4326"  # Ensure everything is in EPSG:4326

def find_bands(sentinel_folder):
    """Finds Sentinel-2 band files in IMG_DATA/{R10m, R20m, R60m} and ensures correct resolution mapping."""
    band_files = {}

    img_data_path = sentinel_folder
    if not img_data_path:
        print(f"No IMG_DATA folder found for {sentinel_folder}")
        return band_files

    # Define band-to-folder mapping
    resolution_mapping = {
        "R10m": bands_10m, 
        "R20m": bands_20m,  
        "R60m": bands_60m 
    }

    for folder, valid_bands in resolution_mapping.items():
        folder_path = os.path.join(img_data_path, folder)
        if not os.path.exists(folder_path):
            print(f"{folder_path} is missing.")
            continue

        search_path = os.path.join(folder_path, "*.jp2")
        all_files = glob.glob(search_path)

        for file in all_files:
            for band in valid_bands:  # Only check bands that should exist in this resolution
                if f"_{band}_" in file:  # Ensure correct band naming
                    band_files[band] = file

    return band_files

def resample_to_10m(src_file, reference_raster):
    with rasterio.open(src_file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, reference_raster.crs, reference_raster.width, reference_raster.height, *reference_raster.bounds
        )
        resampled_data = np.empty((height, width), dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=resampled_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=reference_raster.crs,
            resampling=Resampling.nearest
        )

        return resampled_data
    
    
def reproject_band(src, target_crs):
    try:
            # Calculate the transformation for reprojection
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            meta = src.meta.copy()
            meta.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            # Create a reprojected array
            reprojected_data = np.zeros((height, width), dtype=meta['dtype'])
            reproject(
                source=rasterio.band(src, 1),
                destination=reprojected_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )
            return reprojected_data, meta
    except Exception as e:
        logging.error(f"Error reprojecting {src_path}: {str(e)}")
        raise
        
        

TARGET_CRS = "EPSG:4326"  # Define target CRS

def merge_bands(sentinel_path):
    """Merges RGB + NIR bands (10m) and resamples others, then transforms to EPSG:4326"""
    band_files = find_bands(sentinel_path)

    if not band_files:
        print(f"Skipping {sentinel_path}: No valid bands found.")
        with open(skipped_folders_log, "a") as sf_log:
            sf_log.write(f"{sentinel_path}\n")
        return None

    # Open 10m reference band
    ref_raster = rasterio.open(band_files["B02"]) if "B02" in band_files else None
    if not ref_raster:
        print(f"No valid 10m reference found, skipping {sentinel_path}.")
        return None

    # Merge bands
    merged_bands = []
    for band in bands_10m:
        if band in band_files:
            if band == 'B08':
                resampled_band = resample_to_10m(band_files[band], ref_raster)
                # print("NIR", resampled_band.shape)
                merged_bands.append(resampled_band)
            else:
                with rasterio.open(band_files[band]) as src:
                    data = src.read(1)

                    merged_bands.append(data)  # Direct read for 10m bands

    for band in bands_20m + bands_60m:
        if band in band_files:
            resampled_band = resample_to_10m(band_files[band], ref_raster)
            merged_bands.append(resampled_band)
          

    # Convert all bands to a numpy stack
    stacked_bands = np.stack(merged_bands, axis=-1)

    # Convert CRS to EPSG:4326
    src_crs = ref_raster.crs
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, TARGET_CRS, ref_raster.width, ref_raster.height, *ref_raster.bounds
    )

    reprojected_bands = np.empty((dst_height, dst_width, stacked_bands.shape[-1]), dtype=np.uint16)

    for i in range(stacked_bands.shape[-1]):  # Loop over each band
        reproject(
            source=stacked_bands[:, :, i],
            destination=reprojected_bands[:, :, i],
            src_transform=ref_raster.transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear  # Use bilinear for smooth transformation
        )

    # Update metadata for the new CRS
    meta = ref_raster.meta.copy()
    meta.update({
        "crs": TARGET_CRS,
        "transform": dst_transform,
        "width": dst_width,
        "height": dst_height,
        "count": stacked_bands.shape[-1],
        "dtype": rasterio.uint16
    })


    return reprojected_bands, meta



def divide_into_patches(data, meta, patch_size=224):
    """Splits the image into patches, ensuring valid patches"""
    height, width, _ = data.shape
    patches = []
    threshold = 0.7  # At least 70% valid pixels

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            # Ensure we don't go beyond image boundaries
            if y + patch_size > height or x + patch_size > width:
                continue 
            patch = data[y:y + patch_size, x:x + patch_size, :]
            non_zero_proportion = np.count_nonzero(patch) / patch.size

            if non_zero_proportion > threshold:
                window = Window(x, y, patch_size, patch_size)
                patches.append((patch, window, y, x))
    return patches


def process_image(sentinel_folder, output_dir, patch_size=224):
    """Processes an image by merging, reprojecting, and saving patches"""
    try:
        stacked_data, meta = merge_bands(sentinel_folder)
        if stacked_data is None:
            return
        
        patches = divide_into_patches(stacked_data, meta, patch_size=patch_size)
        
        for i, (patch, window, y, x) in enumerate(patches):
            output_path = os.path.join(output_dir, f'patch_{y}_{x}.TIF')
            patch_meta = meta.copy()
            patch_meta.update({
                'height': window.height,
                'width': window.width,
                'transform': rasterio.windows.transform(window, meta['transform'])
            })
            with rasterio.open(output_path, 'w', **patch_meta) as dst:
                for band in range(stacked_data.shape[-1]):
                    dst.write(patch[:, :, band], band + 1)
    
    except Exception as e:
        logging.error(f"Error processing {sentinel_folder}: {str(e)}")

def process_file(file_path, state='AZ'):
    """Prepares output directory and processes each Sentinel-2 file"""
    year = file_path.split('/')[-6]  # Fix year extraction
    fname = file_path.split('/')[-4]  # Fix dataset name extraction
    # print(file_path.split('/'),year,fname)
    output_dir = f'../patches/Sentinel/{state}/{year}/{fname}'

    if os.path.exists(output_dir):
        return f"{fname} already processed."
    
    os.makedirs(output_dir, exist_ok=True)
    process_image(file_path, output_dir)
    return f"{fname} processed successfully."

def main(state, input_dir):
    file_paths = glob.glob(f'{input_dir}/{state}/*/*/*/GRANULE/*/IMG_DATA')

    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = [executor.submit(process_file, path, state) for path in file_paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            print(future.result())
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create-Patches for Sentinel Data")
    parser.add_argument('-s', '--state', type=str, required=True, help="Two-letter state code (e.g., AZ, TX, FL)")
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="Parent Directory of Downloaded data")
    # parser.add_argument('-o', '--output_dir', type=str, required=True, help="Directory to save patches")
    
    args = parser.parse_args()
    
    main(args.state, args.input_dir)