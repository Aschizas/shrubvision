import ee
import geemap
import matplotlib.pyplot as plt

ee.Initialize(project="embroussaillement")

# --- Cloud masking
def maskS2sr(image):
    qa = image.select('QA60')
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask).divide(10000)

# --- Define ROI
roi = ee.Geometry.Polygon([[  
    [7.000926465483945, 46.40080672313582],
    [7.099803418608945, 46.40080672313582],
    [7.099803418608945, 46.43820175527275],
    [7.000926465483945, 46.43820175527275],
    [7.000926465483945, 46.40080672313582]
]])

# --- Load Sentinel-2 collections for two years
s2_old = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(roi) \
    .filterDate('2018-04-01', '2018-10-30') \
    .map(maskS2sr)

s2_recent = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(roi) \
    .filterDate('2024-04-01', '2024-10-30') \
    .map(maskS2sr)

# --- Median composites
image_old = s2_old.median().clip(roi)
image_recent = s2_recent.median().clip(roi)

# --- Compute NDVI
ndvi_old = image_old.normalizedDifference(['B8','B4']).rename('NDVI_old')
ndvi_recent = image_recent.normalizedDifference(['B8','B4']).rename('NDVI_recent')
ndvi_diff = ndvi_recent.subtract(ndvi_old).rename('NDVI_diff')

# --- Convert to NumPy arrays
rgb_old_array = geemap.ee_to_numpy(image_old.select(['B4','B3','B2']), region=roi, scale=10)
rgb_recent_array = geemap.ee_to_numpy(image_recent.select(['B4','B3','B2']), region=roi, scale=10)
ndvi_old_array = geemap.ee_to_numpy(ndvi_old, region=roi, scale=10)
ndvi_recent_array = geemap.ee_to_numpy(ndvi_recent, region=roi, scale=10)
ndvi_diff_array = geemap.ee_to_numpy(ndvi_diff, region=roi, scale=10)

# --- Plot 2x3 grid: RGB top, NDVI bottom
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Top row: RGB
axes[0,0].imshow(rgb_old_array)
axes[0,0].set_title("Old RGB (2018)")
axes[0,0].axis('off')

axes[0,1].imshow(rgb_recent_array)
axes[0,1].set_title("Recent RGB (2024)")
axes[0,1].axis('off')

im_diff_rgb = axes[0,2].imshow(ndvi_diff_array, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
axes[0,2].set_title("NDVI Difference (2024 - 2018)")
axes[0,2].axis('off')
cbar_rgb = fig.colorbar(im_diff_rgb, ax=axes[0,2], fraction=0.046, pad=0.04)
cbar_rgb.set_label("NDVI Change")

# Bottom row: NDVI for old, recent, difference
im_ndvi_old = axes[1,0].imshow(ndvi_old_array, cmap='RdYlGn', vmin=-0.2, vmax=1)
axes[1,0].set_title("NDVI Old (2018)")
axes[1,0].axis('off')

im_ndvi_recent = axes[1,1].imshow(ndvi_recent_array, cmap='RdYlGn', vmin=-0.2, vmax=1)
axes[1,1].set_title("NDVI Recent (2024)")
axes[1,1].axis('off')

im_ndvi_diff = axes[1,2].imshow(ndvi_diff_array, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
axes[1,2].set_title("NDVI Difference")
axes[1,2].axis('off')
cbar_ndvi = fig.colorbar(im_ndvi_diff, ax=axes[1,2], fraction=0.046, pad=0.04)
cbar_ndvi.set_label("NDVI Change")

plt.tight_layout()
plt.show()
