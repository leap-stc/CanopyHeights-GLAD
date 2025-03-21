import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd
import requests
import re
import os
from datetime import datetime
from rasterio.io import MemoryFile
import matplotlib.pyplot as plt
import warnings
from dask.distributed import Client
import s3fs

# Suppress warnings
warnings.filterwarnings("ignore")

# Start Dask client
client = Client()
print(client)

# ───────────────────────────────────────────────
# 1. Setup Remote Store and Metadata
# ───────────────────────────────────────────────
base_url = "https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=/"
product_name = "CanopyHeights-GLAD"
root_dir = "leap-pangeo-pipeline"
zarr_store_path = os.join.path(root_dir,product_name,f"{product_name}.zarr")
base_dir = "https://nyu1.osn.mghpcc.org"

# S3 Zarr mapper
fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://nyu1.osn.mghpcc.org"})
store = fs.get_mapper(zarr_store_path)

# ───────────────────────────────────────────────
# 2. Retrieve All File Names from VRT File
# ───────────────────────────────────────────────
vrt_file_url = base_url + "ETH_GlobalCanopyHeight_10m_2020_mosaic_Map.vrt"
response = requests.get(vrt_file_url)
file_names = re.findall(r'3deg_cogs/ETH_GlobalCanopyHeight_10m_2020_[NS]\d{2}[EW]\d{3}_Map\.tif', response.text)

# ───────────────────────────────────────────────
# 3. Define Function to Open and Process Each File
# ───────────────────────────────────────────────
def read_file(file_name: str, base_url: str) -> xr.Dataset:
    std_file_name = file_name.replace("_Map.tif", "_Map_SD.tif")
    mean_url = base_url + file_name
    std_url = base_url + std_file_name

    response_mean = requests.get(mean_url, stream=True)
    response_std = requests.get(std_url, stream=True)

    if response_mean.status_code == 200 and response_std.status_code == 200:
        with MemoryFile(response_mean.content) as mem_mean, MemoryFile(response_std.content) as mem_std:
            with mem_mean.open() as src_mean, mem_std.open() as src_std:
                da_mean = rxr.open_rasterio(src_mean).squeeze()
                da_std = rxr.open_rasterio(src_std).squeeze()

                ch = np.where(da_mean.values == 255, np.nan, da_mean.values)
                std = np.where(da_std.values == 255, np.nan, da_std.values)

                lon, lat = np.meshgrid(da_mean.x.values, da_mean.y.values)
                date = pd.to_datetime("2020-01-01")

                ds = xr.Dataset({
                    "canopy_height": (("y", "x"), ch),
                    "std": (("y", "x"), std)
                }, coords={
                    "time": [date],
                    "lat": (("y", "x"), lat),
                    "lon": (("y", "x"), lon)
                })
                return ds
    else:
        print(f"❌ Failed to fetch {file_name}")
        return None

# ───────────────────────────────────────────────
# 4. Write to Zarr Incrementally
# ───────────────────────────────────────────────
first_written = False

for i, file_name in enumerate(file_names):
    if i % 100 == 0:
        print(f"🌲 Processing tile {i+1} of {len(file_names)}")

    ds = read_file(file_name, base_url)
    if ds:
        ds = ds.chunk({"time": 1, "y": 10800, "x": 10800})

        if not first_written:
            ds.to_zarr(store, mode="w", consolidated=False)
            first_written = True
        else:
            ds.to_zarr(store, mode="a", consolidated=False, append_dim="time")

# ───────────────────────────────────────────────
# 5. Visualise First Tile from Zarr
# ───────────────────────────────────────────────
ds = xr.open_dataset(store, engine="zarr", chunks={})
ds.isel(time=0).canopy_height.plot(cmap="YlGn")
plt.title("GLAD Global Canopy Height (10m, 2020)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
