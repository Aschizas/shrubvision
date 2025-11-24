from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer
from qgis.analysis import QgsZonalStatistics
from qgis.PyQt.QtWidgets import QDialog, QPushButton, QLineEdit, QLabel, QVBoxLayout, QFileDialog
from typing import Literal
import os
import processing
import tempfile

# --- Minimal load_layer function ---
def load_layer(input_path, layer_type: Literal["raster", "vector"], name):
    if layer_type == "raster":
        layer = QgsRasterLayer(input_path, name)
    elif layer_type == "vector":
        layer = QgsVectorLayer(input_path, name, "ogr")

    if not layer.isValid():
        print("Failed to load the Layer:", input_path)
    else:
        QgsProject.instance().addMapLayer(layer)
        print("Layer added successfully:", name)
    return layer

# --- CODE FENÊTRE ---
class MultiFileDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Input Layers")
        self.selected_files = {"embroussaillement": "", "carte_milieux": "", "region_interet": ""}

        layout = QVBoxLayout()

        # Embroussaillement raster
        self.embrouss_line = QLineEdit()
        btn_a = QPushButton("charger fichier embroussaillement")
        btn_a.clicked.connect(lambda: self.select_file("embroussaillement", self.embrouss_line, "Raster files (*.tif)"))
        layout.addWidget(QLabel("Embroussaillement (raster)"))
        layout.addWidget(self.embrouss_line)
        layout.addWidget(btn_a)

        # Carte des milieux vector
        self.carte_line = QLineEdit()
        btn_b = QPushButton("charger la carte des milieux")
        btn_b.clicked.connect(lambda: self.select_file("carte_milieux", self.carte_line, "Vector files (*.shp *.geojson *.gpkg)"))
        layout.addWidget(QLabel("Carte des milieux (vector)"))
        layout.addWidget(self.carte_line)
        layout.addWidget(btn_b)

        # Region d'intérêt vector
        self.region_line = QLineEdit()
        btn_v = QPushButton("charger le fichier de region d'intérêt")
        btn_v.clicked.connect(lambda: self.select_file("region_interet", self.region_line, "Vector files (*.shp *.geojson *.gpkg)"))
        layout.addWidget(QLabel("Region d'intérêt (vector)"))
        layout.addWidget(self.region_line)
        layout.addWidget(btn_v)

        # Continue button
        self.btn_continue = QPushButton("Continue")
        self.btn_continue.setEnabled(False)
        self.btn_continue.clicked.connect(self.accept)
        layout.addWidget(self.btn_continue)

        self.setLayout(layout)

    def select_file(self, key, line_edit, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, f"Select {key}", "", filter_str)
        if path:
            self.selected_files[key] = path
            line_edit.setText(path)
        self.btn_continue.setEnabled(all(self.selected_files.values()))


# ------------------------------------- SCRIPT STARTS HERE -------------------------------------
dialog = MultiFileDialog()
if dialog.exec_():
    paths = dialog.selected_files
    embroussaillement_path = paths["embroussaillement"]
    carte_milieux_path = paths["carte_milieux"]
    region_interet_path = paths["region_interet"]

    # Load layers
    embroussaillement = load_layer(embroussaillement_path, "raster", "Embroussaillement")
    # carte_milieux = load_layer(carte_milieux_path, "vector", "Carte des milieux")
    # region_interet = load_layer(region_interet_path, "vector", "Region d'intérêt")

    result = processing.run("gdal:rastercalculator", {
    'INPUT_A': embroussaillement.dataProvider().dataSourceUri(),
    'BAND_A': 1,
    'FORMULA': '(A > 0.5)',
    'OUTPUT': 'TEMPORARY_OUTPUT',  # instead of memory
    'RTYPE': 5
    })

    embroussaillement_mask = QgsRasterLayer(result['OUTPUT'], "Embroussaillement > 0.5m")
    QgsProject.instance().addMapLayer(embroussaillement_mask)

    # Buffer 200m around region d'intérêt (memory layer)
    tampon_200m = processing.run("native:buffer", {
        "INPUT": region_interet_path,
        "DISTANCE": 200,
        "SEGMENTS": 5,
        "DISSOLVE": False,
        "OUTPUT": "memory:"
    })["OUTPUT"]

    # Crop carte_milieux to tampon_200m
    milieux_hongrin = processing.run("native:intersection", {
        "INPUT": carte_milieux_path,
        "OVERLAY": tampon_200m,
        "INPUT_FIELDS": [],
        "OVERLAY_FIELDS": [],
        "OVERLAY_FIELDS_PREFIX": "",
        "OUTPUT": "memory:"
    })["OUTPUT"]

    # exporte le style de la carte des milieux à la nouvelle carte des milieux croppée à l'hongrin$
    milieux = QgsVectorLayer(carte_milieux_path, "carte des milieux", "ogr")
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

    # Zonal statistics
    stats = QgsZonalStatistics(
        milieux_hongrin,     
        embroussaillement_mask,     
        "embrouss_",          
        1,                
        QgsZonalStatistics.Mean | QgsZonalStatistics.Sum | QgsZonalStatistics.Count
    )
    result = stats.calculateStatistics(None)

    print("Processing finished, all layers loaded as temporary memory layers in QGIS.")
