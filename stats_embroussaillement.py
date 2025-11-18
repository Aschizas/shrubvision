import geopandas as gpd
import fiona
import matplotlib.pyplot as plt

MILIEUX = [21, 221, 222, 223, 231, 41, 422, 424, 431, 432, 433, 451, 452, 453, 454]
gpkg_path = r"C:\Users\Admin\Desktop\embroussaillement_hongrin\stats_05m_full.gpkg"
milieux_hongrin_path = r"C:\Users\Admin\Desktop\embroussaillement_hongrin\habitats_hongrin_raw.gpkg"

layers = fiona.listlayers(milieux_hongrin_path)
layer_name = layers[0]   # replace if needed
print(f"\nLoading layer: {layer_name}")
milieux_hongrin = gpd.read_file(milieux_hongrin_path, layer=layer_name)
milieux_critiques = milieux_hongrin[milieux_hongrin["TypoCH_NUM"].isin(MILIEUX)]
print(milieux_critiques)

surfaces_milieux = milieux_critiques.groupby("TypoCH_NUM")["Shape_Area"].sum().reset_index()
print(surfaces_milieux)
# Boxplot of _mean for selected classes
surfaces_milieux.boxplot(column="Shape_Area", by="TypoCH_NUM", figsize=(10,6), showfliers=False)
plt.title(f"Distribution de la surface des milieux présents sur la place de l'Hongrin")
plt.xlabel("Milieu TypoCH_NUM")
plt.ylabel(f"Surface m^2")
plt.suptitle("")
plt.show()

layers = fiona.listlayers(gpkg_path)
layer_name = layers[0]   
gdf = gpd.read_file(gpkg_path, layer=layer_name)
milieux_critiques = gdf[gdf["TypoCH_NUM"].isin(MILIEUX)]

# Aggregate _sum stats per class
stats_milieux_critiques = milieux_critiques.groupby("TypoCH_NUM")["_mean"].mean().reset_index()
print(stats_milieux_critiques)

# Boxplot of _mean for selected classes
milieux_critiques.boxplot(column="_mean", by="TypoCH_NUM", figsize=(10,6), showfliers=False)
plt.title(f"Distribution de l'embroussaillement (> 0.5m) en % surface par milieu TypoCH_NUM")
plt.xlabel("Milieu TypoCH_NUM")
plt.ylabel(f"_mean (% surface)")
plt.suptitle("")
plt.show()

