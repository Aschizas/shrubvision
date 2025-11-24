# shrubvision
Small project to compute woody plant encroachment (embroussaillement) and analyze its evolution over time.

## Overview
The goal of this project was to monitor how woody plants (trees and shrubs) expand into grassy ecosystems. This data can be useful to highligh vulnerable regions where the expansion of woody plants could have negative effects on biodiversity.

You can find the scripts needed in /scripts_embroussaillement. These scripts are based on the [Vegetation Height Model NFI for Switzerland](https://www.envidat.ch/dataset/vegetation-height-model-nfi) data.
The full pipeline consists of 3 scripts: \
compute_embroussaillement.py \
stats_embroussaillement_milieux.py \
plot_stats_embroussaillement.py \

The first script computes the encroachment / embroussaillement between two sets of vegetation height data for different years. Note that in our case, the data was cropped manually to the region that interested us. \
The second script computes statistics of the encroachment / embroussaillement using a map of each existing natural environment. This script also exists as a QGIS-compatible version, that can be launched directly in QGIS. \
The third scripts creates plots for these statistics (boxplots and violinplot)


The /other_scripts folder contains experiments with other sets of data, such as a computer-vision based approach on aerial images, or the NDVI from satellite data (sentinel-2)

## How to use
Install dependencies: it is recommended to do this in a virtual environment.\
`python -m venv .venv` 

Activate the virtual environment: \
`.\venv\Scripts\activate`

Install the dependencies \
`pip install -r requirements.txt`

Run the first script: 
`python .\compute_embroussaillement.py --old "path_to_old_data"  --old "path_to_new_data" --output "path_to_output_folder"`

Run the second script: 
`python .\stats_embroussaillement_milieux.py "path_to_embroussaillement_file"  "path_to_milieux_file" "path_to_region_of_interest_file" -o "path_to_output_folder"`

Run the third script: 
`python .\plot_stats_embroussaillement.py "path_to_statistics_file"`



