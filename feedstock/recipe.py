# ───────────────────────────────────────────────
# 1. Import Libraries and Start Dask
# ───────────────────────────────────────────────
import xarray as xr
import fsspec
import earthaccess
import os
import requests
import re
import numpy as np
import rioxarray as rxr
import matplotlib.pyplot as plt
import pandas as pd
import s3fs
from dask.distributed import Client

client = Client()
print(client)

# ───────────────────────────────────────────────
# 2. Define Paths and VRT URL
# ───────────────────────────────────────────────
base_dir = "https://nyu1.osn.mghpcc.org"
root_dir = "leap-pangeo-pipeline"
product_name = "CanopyHeights-GLAD"
zarr_store_path = os.path.join(root_dir, product_name, f"{product_name}.zarr")

# VRT file used to list the tiles
vrt_url = "https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=/ETH_GlobalCanopyHeight_10m_2020_mosaic_Map.vrt"
vrt_response = requests.get(vrt_url)
file_names = re.findall(r'3deg_cogs/ETH_GlobalCanopyHeight_10m_2020_[NS]\d{2}[EW]\d{3}_Map\.tif', vrt_response.text)

# ───────────────────────────────────────────────
# 3. Read Remote TIFF Files into Xarray
# ───────────────────────────────────────────────
def read_canopy_file(file_name: str, base_url: str) -> xr.Dataset:
    std_file_name = file_name.replace("_Map.tif", "_Map_SD.tif")
    mean_url = base_url + file_name
    std_url = base_url + std_file_name

    resp_mean = requests.get(mean_url, stream=True)
    resp_std = requests.get(std_url, stream=True)

    if resp_mean.status_code == 200 and resp_std.status_code == 200:
        with rxr.open_rasterio(mean_url).squeeze() as da_mean, \
             rxr.open_rasterio(std_url).squeeze() as da_std:

            ch = da_mean.values
            std = da_std.values

            ch = np.where(ch == 255, np.nan, ch)
            std = np.where(std == 255, np.nan, std)

            lon, lat = np.meshgrid(da_mean.x.values, da_mean.y.values)
            date = pd.to_datetime("2020-01-01")  # Single static timestamp

            return xr.Dataset(
                {
                    "canopy_height": ("y", "x", ch),
                    "std": ("y", "x", std)
                },
                coords={
                    "time": date,
                    "lat": (("y", "x"), lat),
                    "lon": (("y", "x"), lon)
                }
            )
    else:
        print(f"❌ Failed to fetch: {file_name}")
        return None
# ───────────────────────────────────────────────
# 5. Process and Write All Tiles to Zarr
# ───────────────────────────────────────────────
fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": base_dir})
mapper = fs.get_mapper(zarr_store_path)

datasets = []
for i, file_name in enumerate(file_names):
    if i % 100 == 0:
        print(f"🌿 Processing tile {i} of {len(file_names)}")
    ds = read_canopy_file(file_name, base_url="https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=/")
    if ds:
        ds = ds.chunk({"time": 1, "y": 3600, "x": 3600})
        datasets.append(ds)

if datasets:
    cds = xr.concat(datasets, dim="time")
    cds.to_zarr(mapper, mode="w", consolidated=True)

# ───────────────────────────────────────────────
# 6. Load and Visualise from Zarr Store
# ───────────────────────────────────────────────
ds = xr.open_dataset(mapper, engine="zarr", chunks={})
ds.isel(time=0).canopy_height.plot(cmap="viridis")
plt.title("Global Canopy Height - GLAD 2020")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
