import tarfile
import os
import shutil
import rasterio
import numpy as np
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from rasterio.windows import Window
from tqdm import tqdm
import logging
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Set up logging for errors
logging.basicConfig(
    filename='processing_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

TARGET_CRS = "EPSG:4326"  # Ensure everything is in EPSG:4326

def reproject_band(src_path, target_crs):
    try:
        with rasterio.open(src_path) as src:
            if src.crs.to_string() == target_crs:
                # If already in the target CRS, just return the data
                return src.read(1), src.meta

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

def merge_bands(b_file):
    data = []
    meta = None
    for fname in b_file:
        reprojected_data, band_meta = reproject_band(fname, TARGET_CRS)
        data.append(reprojected_data)
        meta = band_meta  # Use the latest band meta (should remain consistent)
    # print(len(data))
    stacked_data = np.stack(data, axis=-1)
    meta.update({'count': len(data), 'dtype': stacked_data.dtype})
    return stacked_data, meta

def divide_into_patches(data, meta, patch_size=224):
    height, width = data.shape[:2]
    patches = []
    threshold = 0.7
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch = data[y:y + patch_size, x:x + patch_size, :]
            non_zero_proportion = np.count_nonzero(patch) / patch.size
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size and non_zero_proportion > threshold:
                window = Window(x, y, patch_size, patch_size)
                patches.append((patch, window, y, x))
    return patches

def process_image(b_file, output_dir, patch_size=224):
    
    try:
        stacked_data, meta = merge_bands(b_file)
        patches = divide_into_patches(stacked_data, meta, patch_size=patch_size)
        # print(patches)
        for i, (patch, window, y, x) in enumerate(patches):
            output_path = os.path.join(output_dir, f'patch_{y}_{x}.TIF')
            patch_meta = meta.copy()
            patch_meta.update({
                'height': window.height,
                'width': window.width,
                'transform': rasterio.windows.transform(window, meta['transform'])
            })
            with rasterio.open(output_path, 'w', **patch_meta) as dst:
                for band in range(len(b_file)):
                    dst.write(patch[:, :, band], band + 1)
    except Exception as e:
        logging.error(f"Error processing {b_file}: {str(e)}")

def process_file(file_path, state='AZ'):
    try:
        fname = file_path.split('/')[-1]
        year = file_path.split('/')[-2]
        output_dir = f'../patches/LandSat/{state}/{year}/{fname}'
        if os.path.exists(output_dir):
            return f"{fname} already processed."
        
        b_files = [glob.glob(os.path.join(file_path, f'*_SR_B{i}.TIF'))[0] for i in range(1, 8)]
        b_files.append(glob.glob(os.path.join(file_path, '*_*_B10.TIF'))[0])
        
        os.makedirs(output_dir, exist_ok=True)
        process_image(b_files, output_dir, patch_size=224)
        return f"{fname} processed successfully."
    
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return f"Error processing {file_path}: {str(e)}"

def main(state='AZ'):
    file_paths = glob.glob(f'Extracted-File/{state}/*/*') #../landsat_downloads/newenv/LandSat-8/*/*
    with ThreadPoolExecutor(max_workers=35) as executor:
        futures = [executor.submit(process_file, path, state) for path in file_paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            print(future.result())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Patches')
    parser.add_argument('-s', '--state', type=str, required=True,
                       help='Two-letter state code (e.g., AZ, TX, FL)')
    # parser.add_argument('-d', '--directory', type=str, default='LandSat-8',
    #                    help='Base directory for extraction (default: LandSat-8)')
    
    args = parser.parse_args()
    
    main(args.state)