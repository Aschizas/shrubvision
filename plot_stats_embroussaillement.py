import geopandas as gpd
import fiona
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.lines as mlines 
import seaborn as sns # Used for both plots

# --- Configuration ---
MILIEUX = [21, 221, 222, 223, 231, 41, 422, 424, 431, 432, 433, 451, 452, 453, 454]
gpkg_path = r"C:\Users\Admin\Desktop\embroussaillement_hongrin\stats_05m_full.gpkg"
gpkg_path = r"C:\Users\Admin\Desktop\test.gpkg"

# --- Dynamic Area Formatting Function ---
def format_area_dynamic(area_sq_m):
    SQUARE_KM_THRESHOLD = 1_000_000
    
    if area_sq_m >= SQUARE_KM_THRESHOLD:
        area_sq_km = area_sq_m / SQUARE_KM_THRESHOLD
        return f"{area_sq_km:.1f} km²"
    else:
        return f"{area_sq_m:.0f} m²"

# ----------------------------------------------------------------------
# --- Part 1: Load Data and Calculate Sorting Metric (Total Count/Sum) ---
# ----------------------------------------------------------------------

layers = fiona.listlayers(gpkg_path)
layer_name = layers[0]
print(f"\nLoading layer: {layer_name} from {gpkg_path}")

gdf = gpd.read_file(gpkg_path, layer=layer_name)
milieux_critiques = gdf[gdf["TypoCH_NUM"].isin(MILIEUX)]

# Aggregate both '_count' (total area proxy) and '_sum' (embroussaillement area)
grouped_stats = milieux_critiques.groupby("TypoCH_NUM").agg(
    Total_Count=('embrouss_count', 'sum'),
    Total_Embroussaillement=('embrouss_sum', 'sum')
).reset_index()

# Use Total_Count for sorting
sorted_milieux = grouped_stats.sort_values(by="Total_Count", ascending=False)
plot_order = sorted_milieux["TypoCH_NUM"].astype(str).tolist()

# Prepare the main plotting data (TypoCH_NUM as string for consistent plotting)
milieux_critiques["TypoCH_NUM"] = milieux_critiques["TypoCH_NUM"].astype(str)
milieux_critiques_stats = milieux_critiques

# ----------------------------------------------------------------------
# --- Part 2: Generate Sorted BOXPLOT (Seaborn) with Custom Labels ---
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 6))

# --- SEABORN BOXPLOT ---
sns.boxplot(
    data=milieux_critiques_stats,
    x="TypoCH_NUM",
    y="embrouss_mean",
    order=plot_order, # Use the defined sort order
    showfliers=False, # Hide outliers, matching previous plot
    ax=ax
)
# -----------------------

# Set X-axis labels (must come AFTER the seaborn plot)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

# --- Custom Label Positioning (Two Lines, Centered) ---
y_min = ax.get_ylim()[0]

# Define two vertical positions for the labels
# Note: These positions are relative to the final y-limits set by Seaborn
y_embroussaillement_pos = y_min - 0.15 * (ax.get_ylim()[1] - y_min) 
y_total_count_pos = y_min - 0.22 * (ax.get_ylim()[1] - y_min)        

for i, typo_ch_num_str in enumerate(plot_order):
    typo_ch_num = int(typo_ch_num_str)
    
    # Fetch both metrics
    row = sorted_milieux[sorted_milieux["TypoCH_NUM"] == typo_ch_num].iloc[0]
    total_count = row["Total_Count"]
    total_embroussaillement = row["Total_Embroussaillement"]
    
    # Format the values
    embroussaillement_formatted = format_area_dynamic(total_embroussaillement)
    total_count_formatted = format_area_dynamic(total_count)
    
    # Line 1: Embroussaillement (Above, Colored Red)
    ax.text(
        i, # Use index 'i' for x-position in Seaborn plots
        y_embroussaillement_pos,
        f"{embroussaillement_formatted}",
        ha='center',
        rotation=0, 
        color='red',
        fontsize=8
    )

    # Line 2: Total Count (Below, Colored Blue)
    ax.text(
        i, # Use index 'i' for x-position in Seaborn plots
        y_total_count_pos,
        f"{total_count_formatted}",
        ha='center',
        rotation=0, 
        color='blue',
        fontsize=8
    )

# --- Add Legend Box ---
red_line = mlines.Line2D([], [], color='red', marker='s', linestyle='None', markersize=5, label='Embroussaillement')
blue_line = mlines.Line2D([], [], color='blue', marker='s', linestyle='None', markersize=5, label='Surface totale du milieu')

ax.legend(handles=[red_line, blue_line], 
          loc='upper right', 
          fontsize=9,
          frameon=True,
          fancybox=True,
          shadow=True)

ax.set_title(f"Distribution de l'embroussaillement (> 0.5m) en % surface par milieu TypoCH_NUM\n(Trié par nombre total de caractéristiques)")
ax.set_xlabel("Milieu TypoCH_NUM")
ax.set_ylabel(f"_mean (% surface)")

plt.subplots_adjust(bottom=0.30) 
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

# ----------------------------------------------------------------------
# --- Part 3: Generate Sorted Violin Plot (Seaborn) ---
# ----------------------------------------------------------------------

plt.figure(figsize=(12, 6))

sns.violinplot(
    data=milieux_critiques_stats,
    x="TypoCH_NUM",
    y="embrouss_mean",
    order=plot_order,
    inner='quartile', 
    palette="pastel"
)

plt.title(f"Distribution de l'embroussaillement (> 0.5m) en % surface par milieu TypoCH_NUM (Violin Plot)\n(Trié par nombre total de caractéristiques)")
plt.xlabel("Milieu TypoCH_NUM")
plt.ylabel(f"_mean (% surface)")
plt.xticks(rotation=45, ha="right") 

plt.tight_layout()
plt.show()