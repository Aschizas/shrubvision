import geopandas as gpd
import fiona
import matplotlib.pyplot as plt
import pandas as pd

MILIEUX = [21, 221, 222, 223, 231, 41, 422, 424, 431, 432, 433, 451, 452, 453, 454]
gpkg_path = r"C:\Users\Admin\Desktop\embroussaillement_hongrin\stats_05m_full.gpkg"
milieux_hongrin_path = r"C:\Users\Admin\Desktop\embroussaillement_hongrin\habitats_hongrin_raw.gpkg"

# --- Dynamic Area Formatting Function ---
def format_area_dynamic(area_sq_m):
    SQUARE_KM_THRESHOLD = 1_000_000
    
    if area_sq_m >= SQUARE_KM_THRESHOLD:
        area_sq_km = area_sq_m / SQUARE_KM_THRESHOLD
        return f"{area_sq_km:,.1f} km²"
    else:
        return f"{area_sq_m:,.0f} m²"
    
# ----------------------------------------------------------------------
# --- Part 1: Calculate and Sort Total Surface Area ---
# ----------------------------------------------------------------------

layers = fiona.listlayers(milieux_hongrin_path)
layer_name = layers[0]
print(f"\nLoading layer: {layer_name} for surface calculation")
milieux_hongrin = gpd.read_file(milieux_hongrin_path, layer=layer_name)
milieux_critiques_area = milieux_hongrin[milieux_hongrin["TypoCH_NUM"].isin(MILIEUX)]

surfaces_milieux = milieux_critiques_area.groupby("TypoCH_NUM")["Shape_Area"].sum().reset_index()
surfaces_milieux = surfaces_milieux.rename(columns={'Shape_Area': 'Total_Area'})
print("\nTotal Surface Area per Milieu:\n", surfaces_milieux)

sorted_milieux = surfaces_milieux.sort_values(by="Total_Area", ascending=False)
plot_order = sorted_milieux["TypoCH_NUM"].astype(str).tolist()

# ----------------------------------------------------------------------
# --- Part 2: Prepare Embroussaillement Data ---
# ----------------------------------------------------------------------

layers = fiona.listlayers(gpkg_path)
layer_name = layers[0]
print(f"\nLoading layer: {layer_name} for embroussaillement stats")
gdf = gpd.read_file(gpkg_path, layer=layer_name)
milieux_critiques_stats = gdf[gdf["TypoCH_NUM"].isin(MILIEUX)]

milieux_critiques_stats["TypoCH_NUM"] = milieux_critiques_stats["TypoCH_NUM"].astype(str)

# ----------------------------------------------------------------------
# --- Part 3: Generate Sorted Boxplot with Custom Labels ---
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 6))

data_to_plot = [milieux_critiques_stats[milieux_critiques_stats["TypoCH_NUM"] == cat]["_mean"] for cat in plot_order]

ax.boxplot(data_to_plot, showfliers=False)

ax.set_xticks(range(1, len(plot_order) + 1))
ax.set_xticklabels(plot_order, ha="right")

# --- Custom Label Positioning ---
y_min = ax.get_ylim()[0]
y_label_pos = y_min - 0.15 * (ax.get_ylim()[1] - y_min) 

for i, typo_ch_num_str in enumerate(plot_order):
    typo_ch_num = int(typo_ch_num_str)
    
    area = sorted_milieux[sorted_milieux["TypoCH_NUM"] == typo_ch_num]["Total_Area"].iloc[0]
    area_formatted = f"Total: {format_area_dynamic(area)}"
    
    ax.text(
        i + 1,
        y_label_pos,
        area_formatted,
        ha='center',
        rotation=0, 
        color='blue',
        fontsize=8
    )

ax.set_title(f"Distribution de l'embroussaillement (> 0.5m) en % de surface par milieu TypoCH_NUM\n(Trié par surface totale du milieu)")
ax.set_xlabel("Milieu TypoCH_NUM")
ax.set_ylabel(f"Embroussaillement moyen en % surface")

plt.subplots_adjust(bottom=0.25) 
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()