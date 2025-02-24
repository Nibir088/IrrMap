import os
# os.environ["GDAL_PAM_ENABLED"] = "NO"
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import mapping, box, shape
from rasterio.windows import Window
from tqdm import tqdm
from multiprocessing import Pool
from rasterio.warp import calculate_default_transform, reproject, Resampling

import sys
import glob
import argparse

TARGET_CRS = "EPSG:4326"  # Ensure everything is in EPSG:4326
import pandas as pd
crop_info = pd.read_csv('../Crop-Groups.csv')
crop_dict = {}
# for c_type,c_group in zip(crop_info['Crop Type'],crop_info['Index']):
for c_type,c_group in zip(crop_info['Group'],crop_info['Index']):
    crop_dict[c_type] = c_group
    
def reproject_band(src_path, target_crs):
    try:
        with rasterio.open(src_path) as src:
            qa_data = src.read(1)
            b_array = 1 - (qa_data & 1)
            if str(src.crs) == target_crs: #.to_string()
                bbox = box(*src.bounds)
                return b_array, src.meta, bbox
            print('Repeojection Module')
            # Reproject the data
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            meta = src.meta.copy()
            meta.update({'crs': target_crs, 'transform': transform, 'width': width, 'height': height})

            reprojected_data = np.zeros((height, width), dtype=meta['dtype'])
            reproject(
                source=b_array, destination=reprojected_data,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=target_crs,
                resampling=Resampling.bilinear
            )
            minx, miny = transform * (0, height)
            maxx, maxy = transform * (width, 0)
            bbox = box(minx, miny, maxx, maxy)

            return reprojected_data, meta, bbox
    except Exception as e:
        print(f"Error reprojecting {src_path}: {e}")
        sys.exit()

def rasterize_layer(gdf, field, img_shape, transform):
    height, width = img_shape
    shapes = [(mapping(geom), value) for geom, value in zip(gdf.geometry, gdf[field])]
    # if len(shapes)>0:
    
    shapes = sorted(shapes, key=lambda x: shape(x[0]).area)
    
    return rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype='int32')

def divide_into_patches(ras_arr, qa_band, patch_size=256):
    # print(ras_arr.shape, patch_size)
    height, width = ras_arr.shape
    patches = [
        (ras_arr[y:y + patch_size, x:x + patch_size], Window(x, y, patch_size, patch_size), y, x)
        for y in range(0, height, patch_size)
        for x in range(0, width, patch_size)
        if np.sum(ras_arr[y:y + patch_size, x:x + patch_size] > 0) > 0.0001 * patch_size**2
    ]
    # print(f"Generated {len(patches)} patches from image of shape {ras_arr.shape}")
    return patches

def process_image(b2_file, output_dir, gdf, patch_size=224):
    reprojected_data, meta, bbox = reproject_band(b2_file, TARGET_CRS)

    height, width = reprojected_data.shape
    gdf_crop = gdf[gdf.intersects(bbox)]

    if gdf_crop.empty:
        # print(f"No intersection found for {b2_file}. Skipping...")
        return
    # print(f"intersection found for {b2_file}.")
    
    rasterized_land = rasterize_layer(gdf_crop, 'LAND_TYPE', (height, width), meta['transform'])
    rasterized_crop = rasterize_layer(gdf_crop, 'CROP_TYPE', (height, width), meta['transform'])
    rasterized_irr = rasterize_layer(gdf_crop, 'IRR_TYPE', (height, width), meta['transform'])

    assert rasterized_irr.shape == reprojected_data.shape, \
        f"Shape mismatch: Rasterized {rasterized_irr.shape} vs QA mask {reprojected_data.shape}"

    masked_irr = rasterized_irr * reprojected_data

    meta.update({'count': 4, 'height': height, 'width': width, 'transform': meta['transform']})
    patches = divide_into_patches(masked_irr, reprojected_data, patch_size=patch_size)
    
    # print(output_dir, 'Max Value Print: ',np.unique(rasterized_crop), np.unique(rasterized_irr))
    if np.unique(rasterized_crop).max() >=21 or np.unique(rasterized_irr).max()>=4:
        # print('Max Value Print: ',np.unique(rasterized_crop).max(), np.unique(rasterized_irr).max())
        raise ValueError("Error: rasterized_crop contains values >= 21 or rasterized_irr contains values >= 4")
    
    for patch, window, y, x in patches:
        patch_meta = meta.copy()
        patch_meta.update({
            'height': window.height,
            'width': window.width,
            'transform': rasterio.windows.transform(window, meta['transform']),
            "driver": "GTiff"
        })
        # print(patch_meta)
        output_path = os.path.join(output_dir, f'patch_{y}_{x}.TIF')
        # print(output_path, 'Value Print: ',np.unique(rasterized_crop[y:y + patch_size, x:x + patch_size]), np.unique(rasterized_irr[y:y + patch_size, x:x + patch_size]))
        with rasterio.open(output_path, 'w', **patch_meta) as dst:
            dst.write(rasterized_irr[y:y + patch_size, x:x + patch_size], 1)
            dst.write(rasterized_land[y:y + patch_size, x:x + patch_size], 2)
            dst.write(rasterized_crop[y:y + patch_size, x:x + patch_size], 3)
            dst.write(reprojected_data[y:y + patch_size, x:x + patch_size], 4)

def process_parallel(args):
    file_path, gdf, state, year = args
    fname = os.path.basename(os.path.dirname(file_path))
    output_dir = f'../irrigation_croptypes_landuse_mask/Sentinel/{state}/{year}/{fname}'
    # if os.path.exists(output_dir):
    #         return f"{fname} already processed."
    os.makedirs(output_dir, exist_ok=True)
    process_image(file_path, output_dir, gdf)

def main(state, year):
    shape_file = f'../shape-file/{state}_polygon.geojson'
    gdf = gpd.read_file(shape_file)
    gdf['DATA_YEAR'] = gdf['DATA_YEAR'].astype(int)
    gdf = gdf[gdf['DATA_YEAR'] == year]
    # gdf['CROP_TYPE'] = gdf['CROP_TYPE'].map(crop_dict).fillna(0).astype(int)
    gdf['CROP_TYPE'] = gdf['CROP_GROUP'].map(crop_dict).fillna(0).astype(int)
    # crop_type_dict = {crop: 1+idx for idx, crop in enumerate(list(set(crop_dict.values())))}
    irr_mapping = {'Drip': 3, 'Flood': 1, 'Sprinkler': 2}
    irr_land_mapping = {'Irrigated': 1}

    gdf['IRR_TYPE'] = gdf['IRG_SYS'].map(irr_mapping).fillna(0).astype(int)
    # gdf['LAND_TYPE'] = gdf['IRG_STATUS'].map(irr_land_mapping).fillna(0).astype(int)
    gdf['LAND_TYPE'] = 1
    # gdf['CROP_TYPE'] = gdf['CROP_TYPE'].map(crop_type_dict).fillna(0).astype(int)
    print(gdf['CROP_TYPE'].unique(), gdf['IRR_TYPE'].unique())
    file_paths = glob.glob(
        f'Sentinel-UnZip/{state}/{year}/*/*/*_QA_PIXEL.tif'
        # /project/biocomplexity/wyr6fx(Nibir)/KDD-25-Data-Track/Sentinel-UnZip/AZ/2015/75e61701-af9f-4848-8fd5-efd91c9d8e5f/S2A_MSIL2A_20150814T181446_N0500_R041_T12SWC_20231017T111801.SAFE/
    )
    print(len(file_paths))
    
    args = [(file_path, gdf, state, year) for file_path in file_paths]

    with Pool(processes=8) as pool:
        list(tqdm(pool.imap(process_parallel, args), total=len(args)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raster files for a specific state and year.")
    parser.add_argument('--state', type=str, required=True, help="State to process (e.g., 'CO').")
    parser.add_argument('--year', type=int, required=True, help="Year to process (e.g., 2015).")

    args = parser.parse_args()
    main(args.state, args.year)
