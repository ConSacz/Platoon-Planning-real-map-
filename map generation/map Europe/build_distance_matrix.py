import requests
import numpy as np
import os

from locations import locations

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------------
# Build coordinate string
# --------------------------------------------------

coords = []

for info in locations.values():

    lat, lon = info["coord"]

    coords.append(f"{lon},{lat}")

coord_string = ";".join(coords)

url = (
    "https://router.project-osrm.org/table/v1/driving/"
    + coord_string
    + "?annotations=distance,duration"
)

print("Requesting OSRM...")

response = requests.get(url, timeout=60)

response.raise_for_status()

data = response.json()

distance = np.array(data["distances"]) / 1000.0

duration = np.array(data["durations"]) / 3600.0

np.save("data/distance_matrix.npy", distance)

np.save("data/travel_time_matrix.npy", duration)

print("Saved distance matrix:", distance.shape)
print("Saved travel time matrix:", duration.shape)