import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, binary_closing, grey_opening, grey_closing

def circular_kernel(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(np.uint8)


def export_image(img, metadata, output_path):

    with rasterio.open(output_path, "w", **metadata) as dst:
        dst.write(img, 1)

def create_base_forest_mask(img_path, output_path):

    with rasterio.open(img_path) as src:
        data = src.read(1)
        meta = src.meta.copy()
        nodata_in = src.nodata

    threshold_value = 3
    mask = data > threshold_value

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


def compute_difference_map_binned(old_path, new_path, output_path, bins, cutoff_height):
    with rasterio.open(old_path) as src:
        data_old = src.read(1)
        meta = src.meta.copy()
        nodata_in = src.nodata
    
    with rasterio.open(new_path) as src:
        data_new = src.read(1)
    
    # bin the data by height classes, add a cutoff height to ignore the forest areas
    binned_old = np.digitize(data_old, bins)
    binned_old = np.where(binned_old <= len(bins)-1, np.array(bins)[binned_old-1],0)
    max_value = cutoff_height
    binned_old[binned_old > max_value] = 0

    binned_new = np.digitize(data_new, bins)
    binned_new = np.where(binned_new <= len(bins)-1, np.array(bins)[binned_new-1],0)
    max_value = cutoff_height
    binned_new[binned_new > max_value] = 0

    # compute difference map
    diff = np.zeros_like(binned_old, dtype=np.int16)  
    mask_valid = (binned_old != nodata_in) & (binned_new != nodata_in)
    diff[mask_valid] = binned_new[mask_valid] - binned_old[mask_valid]
    diff[~mask_valid] = -32768

    print("binned_old unique values:", np.unique(binned_old))
    print("binned_new unique values:", np.unique(binned_new))
    print("diff unique values:", np.unique(diff))

    meta.update({
        "dtype": "float32",
        "count": 1,
        "nodata": nodata_in
    })

    # Write output GeoTIFF
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(diff.astype(np.float32), 1)

    return diff

def compute_difference_map_raw(old_path, new_path, output_path, min_height, max_height):
    import rasterio
    import numpy as np

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

    print("diff stats: min =", np.nanmin(diff), "max =", np.nanmax(diff))

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

    old_path = r"C:\Users\Admin\Desktop\embroussaillement_hongrin\hauteur_vegetation_hongrin_2012.tif"
    new_path = r"C:\Users\Admin\Desktop\embroussaillement_hongrin\hauteur_vegetation_hongrin_2023.tif"

    output_path = r"C:\Users\Admin\Desktop\embroussaillement_hongrin\mask_forest_2012.tif"
    forest_mask = create_base_forest_mask(old_path, output_path)

    output_path = r"C:\Users\Admin\Desktop\embroussaillement_hongrin\2012_binned.tif"
    # diff = compute_difference_map_binned(old_path, new_path, output_path, bins=[0, 2, 4, 6, 8, 12], cutoff_height=12)
    diff = compute_difference_map_raw(old_path, new_path, output_path, min_height=1.0, max_height=12)

    h_diff_no_forest = diff.copy()
    h_diff_no_forest[forest_mask != 0] = 0
    print("hdiff no forest unique values:", np.unique(h_diff_no_forest))

    output_path = r"C:\Users\Admin\Desktop\embroussaillement_hongrin\hdiff_no_forest_raw.tif"
    with rasterio.open(old_path) as src:
        data_old = src.read(1)
        meta = src.meta.copy()
        nodata_in = src.nodata
    meta.update({
        "dtype": "float32",
        "count": 1,
        "nodata": nodata_in
    })
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(h_diff_no_forest.astype(np.float32), 1)

    h_diff_no_forest_cleaned = grey_closing(h_diff_no_forest, structure=circular_kernel(2))
    h_diff_no_forest_cleaned = grey_opening(h_diff_no_forest_cleaned, structure=circular_kernel(2))
    
    print("hdiff no forest cleaned unique values:", np.unique(h_diff_no_forest_cleaned))
    output_path = r"C:\Users\Admin\Desktop\embroussaillement_hongrin\hdiff_no_forest_cleaned_raw.tif"
    export_image(h_diff_no_forest_cleaned, meta, output_path)

# SIMPLE VERSION
    # create forest mask
    # create map forest > n meters for new and old
    # new - old mask
    # separate in + and -, apply forest mask
    # morphological cleanup
    # put back together

# BINS VERSION
    # create forest mask
    # bin map in height slices , ignore highest values
    # compute new - old mask
    # apply forest mask to difference result image
    # morphological cleanup
    # put back together