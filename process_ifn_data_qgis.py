from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer, QgsFeatureRequest
from qgis.analysis import QgsZonalStatistics
from typing import Literal
import processing
import tempfile
import os


def load_layer(input_path, layer_type: Literal["raster", "vector"], name):

    # load layer as raster or vector layer depending on type
    if layer_type == "raster":
        layer = QgsRasterLayer(input_path, name)
    elif layer_type == "vector":
        layer = QgsVectorLayer(input_path, name, "ogr")

    # Check if loaded correctly
    if not layer.isValid():
        print("Failed to load the Layer.")
    else:
        # Add the layer to the current project
        QgsProject.instance().addMapLayer(layer)
        print("Layer added successfully!")

    return layer

# ------------------------------------- SCRIPT STARTS HERE -------------------------------------
# path to input file
input_path = r"C:\Users\Admin\Desktop\embroussaillement_hongrin\hdiff_no_forest_cleaned_raw.tif"
raster_layer = load_layer(input_path, "raster", "input embroussaillement")
# Output path for the new raster
output_path = r"C:\Users\Admin\Desktop\test.tif"

# Binary layer, embroussaillement > 0.5m 
processing.run("gdal:rastercalculator", {
    'INPUT_A': raster_layer.dataProvider().dataSourceUri(),
    'BAND_A': 1,
    'FORMULA': '(A > 0.5)',
    'OUTPUT': output_path,
    'RTYPE': 5  # Float32
})

embroussaillement_mask = QgsRasterLayer(output_path, "Embroussaillement > 0.5")
QgsProject.instance().addMapLayer(embroussaillement_mask)


# crée un tampon +200m de la place de l'hongrin
place_darme_hongrin_path = r"D:\Dropbox\n+p\Mandats\2601 NPA\NPA Petit Hongrin\6. QGIS\NPA Petit Hongrin.gpkg"
tampon_200m = processing.run("native:buffer", {
    "INPUT": place_darme_hongrin_path,
    "DISTANCE": 200,
    "SEGMENTS": 5,
    "DISSOLVE": False,
    "OUTPUT": "memory:"
})["OUTPUT"]

carte_milieux_path = r"D:\Dropbox\n+p\QData\Données fédérales\HabitatMap_CH\HabitatCH_VD_2022.gpkg"
milieux = QgsVectorLayer(carte_milieux_path, "carte des milieux", "ogr")
if not milieux.isValid():
    raise Exception("Failed to load the source layer.")

# crop la carte des milieux à l'hongrin
milieux_hongrin = processing.run("native:intersection", {
    "INPUT": milieux,
    "OVERLAY": tampon_200m,
    "INPUT_FIELDS": [],
    "OVERLAY_FIELDS": [],
    "OVERLAY_FIELDS_PREFIX": "",
    "OUTPUT": "memory:"
})["OUTPUT"]

# exporte le style de la carte des milieux à la nouvelle carte des milieux croppée à l'hongrin
with tempfile.NamedTemporaryFile(suffix=".qml", delete=False) as tmp_qml:
    qml_path = tmp_qml.name
milieux.saveNamedStyle(qml_path)            
ok, msg = milieux_hongrin.loadNamedStyle(qml_path)  
if not ok:
    print("Failed to load style:", msg)
milieux_hongrin.triggerRepaint()
os.remove(qml_path)                       

milieux_hongrin.setName("milieux_hongrin")
QgsProject.instance().addMapLayer(milieux_hongrin)

# stats entre layer binaire d'embroussaillement et carte des milieux
prefix = "embrouss_"
# Create QgsZonalStatistics object
stats = QgsZonalStatistics(
    milieux_hongrin,     
    embroussaillement_mask,     
    prefix,          
    1,                # band number 1
    QgsZonalStatistics.Mean | QgsZonalStatistics.Sum | QgsZonalStatistics.Count
)

result = stats.calculateStatistics(None)
print("Zonal stats result:", result)


# hdiff final > 0.5m -> binary
# place d'arme hongrin -> tampon 200m 
# import habitats -> crop to tampon
# stats vectorial + raster. 

# jupyter logbook for plots
