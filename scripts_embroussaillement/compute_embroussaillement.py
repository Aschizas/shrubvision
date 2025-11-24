import argparse
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import binary_opening, binary_closing, grey_opening, grey_closing


def parse_args():
    parser = argparse.ArgumentParser(description="Computes vegetation height difference maps.")

    parser.add_argument("--old", required=True, help="Path to old vegetation height raster (e.g., 2012).")
    parser.add_argument("--new", required=True, help="Path to new vegetation height raster (e.g., 2023).")

    parser.add_argument("--output", required=True, help="Path to the output folder. 3 files are produced, " \
    "a forest mask, which shows the areas that were ignored because considered as forest, a raw difference map, "
    "and a cleaned difference map")
    

    return parser.parse_args()


def circular_kernel(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(np.uint8)


def export_image(img, metadata, output_path):

    with rasterio.open(output_path, "w", **metadata) as dst:
        dst.write(img, 1)


def create_base_forest_mask(img_path, output_path, height_threshold=3):
    """
    Creates a mask of where the forest was in the reference (old) vegetation height data.

    height_threshold: height over which data is considered forest, base value 3m
    """

    with rasterio.open(img_path) as src:
        data = src.read(1)
        meta = src.meta.copy()
        nodata_in = src.nodata

    mask = data > height_threshold

    # Morphology (bool → bool)
    opened = binary_opening(mask, circular_kernel(3))
    closed = binary_closing(opened, circular_kernel(5))

    # Convert bool → uint8 (0/1)
    closed_uint8 = closed.astype(np.uint8)

    # Replace original nodata with uint8 nodata
    closed_uint8[data == nodata_in] = 255   # assign new nodata value

    # Fix metadata
    meta.update({
        "dtype": "uint8",
        "nodata": 255,
        "count": 1
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(closed_uint8, 1)
    
    return closed_uint8


def compute_difference_map_raw(old_path, new_path, output_path, min_height, max_height):
    """
    Computes the difference between the old and recent vegetation height models. 
    
    min_height: ignores all vegetation under this value
    max_height: ignores all vegetation over this value
    """

    with rasterio.open(old_path) as src:
        data_old = src.read(1)
        meta = src.meta.copy()
        nodata_in = src.nodata

    with rasterio.open(new_path) as src:
        data_new = src.read(1)

    # Clip values instead of masking
    data_old_clipped = np.where(data_old < min_height, min_height, data_old)
    data_new_clipped = np.where(data_new < min_height, min_height, data_new)

    # Optionally cap at max_height
    data_old_clipped = np.where(data_old_clipped > max_height, max_height, data_old_clipped)
    data_new_clipped = np.where(data_new_clipped > max_height, max_height, data_new_clipped)

    # Compute raw difference
    diff = data_new_clipped - data_old_clipped

    # Update metadata
    meta.update({
        "dtype": "float32",
        "count": 1,
        "nodata": np.nan
    })

    # Write output
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(diff, 1)

    return diff


if __name__ == "__main__":

    args = parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Created output directory: {args.output}")

    print("Computing mask of where forest > 3m in old data...")
    forest_mask_path = f"{args.output}/mask_forest.tif"
    forest_mask = create_base_forest_mask(args.old, forest_mask_path)

    
    print("Computing difference mask of old and new vegetation height data...")
    diff_path = f"{args.output}/hdiff_raw.tif"
    diff = compute_difference_map_raw(args.old, args.new, diff_path, min_height=1.0, max_height=12)

    print("Removing forest from difference mask...")
    diff_no_forest_path = f"{args.output}/hdiff_no_forest_raw.tif"
    h_diff_no_forest = diff.copy()
    h_diff_no_forest[forest_mask != 0] = 0
    
    with rasterio.open(args.old) as src:
        data_old = src.read(1)
        meta = src.meta.copy()
        nodata_in = src.nodata
    meta.update({
        "dtype": "float32",
        "count": 1,
        "nodata": nodata_in
    })
    with rasterio.open(diff_no_forest_path, "w", **meta) as dst:
        dst.write(h_diff_no_forest.astype(np.float32), 1)

    print("Cleaning up noise in vegetation height difference...")
    diff_no_forest_path_cleaned = f"{args.output}/hdiff_no_forest_cleaned.tif"
    h_diff_no_forest_cleaned = grey_closing(h_diff_no_forest, structure=circular_kernel(2))
    h_diff_no_forest_cleaned = grey_opening(h_diff_no_forest_cleaned, structure=circular_kernel(2))
    export_image(h_diff_no_forest_cleaned, meta, diff_no_forest_path_cleaned)
    print("Final result stored in", diff_no_forest_path_cleaned)
