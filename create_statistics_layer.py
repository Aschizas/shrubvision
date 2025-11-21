import os
import argparse
import numpy as np
import rasterio
import geopandas as gpd
from rasterstats import zonal_stats

def process_geospatial_data(embrouss_path, milieux_path, region_path, output_dir="output"):
    """
    Performs image processing, vector clipping, and zonal statistics 
    purely in Python using rasterio, geopandas, and rasterstats.

    Args:
        embrouss_path (str): Path to the Embroussaillement raster (TIF).
        milieux_path (str): Path to the Carte des milieux vector layer (e.g., SHP, GeoJSON).
        region_path (str): Path to the Region d'intérêt vector layer (e.g., SHP, GeoJSON).
        output_dir (str): Directory where the final results will be saved.
    """
    print("--- Starting Geospatial Processing ---")
    
    # 1. Setup Output Directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 2. Load Vector Data (geopandas)
    try:
        carte_milieux = gpd.read_file(milieux_path)
        region_interet = gpd.read_file(region_path)
        print(f"Loaded vector layers (CRS: {carte_milieux.crs})")
    except Exception as e:
        print(f"Error loading vector files. Check paths and formats. Error: {e}")
        return

    # 3. Raster Calculator: Create Mask (embrouss > 0.5)
    mask_output_path = os.path.join(output_dir, "embroussaillement_mask.tif")
    
    with rasterio.open(embrouss_path) as src:
        embrouss_array = src.read(1)
        embrouss_meta = src.meta.copy()

        # Calculation: (A > 0.5)
        embroussaillement_mask_array = (embrouss_array > 0.5).astype(np.uint8)

        # Update metadata for the new mask file
        embrouss_meta.update({
            'dtype': rasterio.uint8,
            'nodata': None, 
            'count': 1
        })
        
        # Save the mask locally
        with rasterio.open(mask_output_path, 'w', **embrouss_meta) as dst:
            dst.write(embroussaillement_mask_array, 1)

    print(f"Created raster mask: {mask_output_path}")

    # 4. Buffer: 200m around region_interet 
    if region_interet.crs and region_interet.crs.is_geographic:
        # Project to UTM for buffer (meters)
        projected_region = region_interet.to_crs(epsg=2056)
        tampon_200m = projected_region.buffer(200).to_crs(region_interet.crs)
        tampon_200m_crs = tampon_200m.crs
    else:
        # Assume input is already projected in meters
        tampon_200m = region_interet.buffer(200)
        tampon_200m_crs = tampon_200m.crs

    print("Created 200m buffer around region d'intérêt.")
    tampon_200m_geom = tampon_200m.union_all()
    tampon_200m = gpd.GeoDataFrame(geometry=[tampon_200m_geom], crs=tampon_200m_crs)

    # 5. Intersection/Clip: Crop carte_milieux to tampon_200m
    # geopandas.overlay is equivalent to QGIS's "native:intersection"
    milieux_hongrin = gpd.overlay(carte_milieux, tampon_200m, how='intersection')

    print(f"Clipped Carte des milieux to the 200m buffer zone. ({len(milieux_hongrin)} features remaining)")

    # 6. Zonal Statistics (rasterstats)
    print("Calculating Zonal Statistics (Mean, Sum, Count)...")
    
    # Get the geometry objects from the GeoDataFrame for rasterstats
    zone_geometries = milieux_hongrin.geometry
    
    stats = zonal_stats(
        zone_geometries,            
        mask_output_path,             
        affine=embrouss_meta['transform'], # Use the transform from the mask's metadata
        stats=['mean', 'sum', 'count'], 
        nodata=None                     
    )

    # 7. Add Results to the GeoDataFrame
    milieux_hongrin['embrouss_mean'] = [s['mean'] for s in stats]
    milieux_hongrin['embrouss_sum'] = [s['sum'] for s in stats]
    milieux_hongrin['embrouss_count'] = [s['count'] for s in stats]

    # 8. Save the Final Output
    final_output_path = os.path.join(output_dir, "milieux_hongrin_stats.gpkg")
    
    # Using GeoPackage (.gpkg) as it's a single file format, 
    # much cleaner than a multi-file Shapefile (.shp)
    milieux_hongrin.to_file(final_output_path, driver='GPKG')

    print("\n--- Processing Finished ---")
    print(f"Final GeoPackage with statistics saved to: {final_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform geospatial processing (raster masking, vector clipping, and zonal statistics)."
    )
    parser.add_argument(
        "embrouss_path",
        type=str,
        help="Path to the Embroussaillement raster file (*.tif)."
    )
    parser.add_argument(
        "milieux_path",
        type=str,
        help="Path to the Carte des milieux vector file (*.shp, *.geojson, *.gpkg)."
    )
    parser.add_argument(
        "region_path",
        type=str,
        help="Path to the Region d'intérêt vector file (*.shp, *.geojson, *.gpkg)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="output",
        help="Directory to save the resulting files. Defaults to 'output'."
    )

    args = parser.parse_args()

    process_geospatial_data(
        args.embrouss_path,
        args.milieux_path,
        args.region_path,
        args.output_dir
    )