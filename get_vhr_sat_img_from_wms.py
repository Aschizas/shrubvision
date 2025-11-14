import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# WMS endpoint
WMS_URL = "https://image.discomap.eea.europa.eu/arcgis/services/GioLand/VHR_2021_LAEA/ImageServer/WMSServer?"

# Layer name (from <Name> tag)
LAYER = "VHR_2021_LAEA"

# Example bounding box over Switzerland (in EPSG:4326)
# Order is miny, minx, maxy, maxx for WMS 1.3.0 with EPSG:4326
bbox = (46.2, 6.0, 47.2, 7.5)

params = {
    "SERVICE": "WMS",
    "VERSION": "1.3.0",
    "REQUEST": "GetMap",
    "FORMAT": "image/jpeg",
    "TRANSPARENT": "TRUE",
    "LAYERS": LAYER,
    "CRS": "EPSG:4326",
    "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
    "WIDTH": 1024,
    "HEIGHT": 1024
}

response = requests.get(WMS_URL, params=params)
img = Image.open(BytesIO(response.content))

plt.imshow(img)
plt.title("Copernicus VHR 2021 Mosaic (Switzerland)")
plt.axis('off')
plt.show()
