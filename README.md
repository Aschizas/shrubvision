# shrubvision
Estimate tree/shrub evolution using CV on aerial photos

# activate venv
.\venv\Scripts\activate 

# install libs while in venv, save reqs with pip freeze > requirements.txt
deactivate


# TODO GENERAL: 

1. combine vegetation (hue) and feature (otsu thresholding) masks in intelligent order
5. adapt and tune parameters, if possible make them dependent of image size
1. create roi-selector tool
2. create more robust hue detector: mix finding local maxima with expected green hue (problem to solve: detecting lake)
3. clean-up code, have full pipeline to run on roi and output vegetation detected
4. start investigating how to extract part of an image from qgis: 
    -> create polygon
    -> extract image of whats in the polygon
5. move bombina variegata code to repo
6. experiment with other vegetation indexes (not just hues)




# TODO DAY 1:
1. extract forest only -> otsu, clustering, closing.
2. extract individual trees, combine with previous
3. create mask for only green hues
4. combine masks in intelligent order
5. adapt and tune parameters, if possible make them dependent of image size

# TODO DAY 2:
1. create roi-selector tool
2. create more robust hue detector: mix finding local maxima with expected green hue (problem to solve: detecting lake)
3. clean-up code, have full pipeline to run on roi and output vegetation detected
4. start investigating how to extract part of an image from qgis: 
    -> create polygon
    -> extract image of whats in the polygon
5. move bombina variegata code to repo


# 2010 - 2015- 2025

# TODO DAY 3:
1. try to find sat images (copernicus vhr?)
2. search what lidar data are available ()

# TODO DAY 5
1. apply forest location 2012 as a mask (clean up the mask?)

# TODO DAY 6
boxplot % variation for each milieu