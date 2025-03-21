# ───────────────────────────────────────────────
# 1. Import Libraries and Start Dask
# ───────────────────────────────────────────────
import xarray as xr
import numpy as np
import rioxarray as rxr
import pandas as pd
import matplotlib.pyplot as plt
import s3fs
import re
import requests
import os
from dask.distributed import Client

client = Client()
print(client)

# ───────────────────────────────────────────────
# 2. Define Paths and Remote URLs
# ───────────────────────────────────────────────
base_dir = "https://nyu1.osn.mghpcc.org"
root_dir = "leap-pangeo-pipeline"
product_name = "CanopyHeights-GLAD"

zarr_store_path = os.path.join(root_dir, product_name, f"{product_name}.zarr")
print(zarr_store_path)
mapper_path = os.path.join(root_dir, f"{product_name}.zarr")
# Base URL for GLAD canopy data
base_data_url = "https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=/"

# Retrieve VRT file to extract available tile names
vrt_url = base_data_url + "ETH_GlobalCanopyHeight_10m_2020_mosaic_Map.vrt"
vrt_response = requests.get(vrt_url)
file_names = re.findall(r'3deg_cogs/ETH_GlobalCanopyHeight_10m_2020_[NS]\d{2}[EW]\d{3}_Map\.tif', vrt_response.text)

# ───────────────────────────────────────────────
# 3. Function to Read a Single Tile as Xarray Dataset
# ───────────────────────────────────────────────
def read_canopy_file(file_name: str, base_url: str) -> xr.Dataset:
    std_file_name = file_name.replace("_Map.tif", "_Map_SD.tif")
    mean_url = base_url + file_name
    std_url = base_url + std_file_name

    try:
        da_mean = rxr.open_rasterio(mean_url, masked=True, chunks={}).squeeze()
        da_std = rxr.open_rasterio(std_url, masked=True, chunks={}).squeeze()

        ch = da_mean.where(da_mean != 255)
        std = da_std.where(da_std != 255)

        lon, lat = np.meshgrid(da_mean.x.values, da_mean.y.values)
        date = pd.to_datetime("2020-01-01")  # Static timestamp

        ds = xr.Dataset(
            {
                "canopy_height": (("y", "x"), ch.data),
                "std": (("y", "x"), std.data)
            },
            coords={
                "time": date,
                "lat": (("y", "x"), lat),
                "lon": (("y", "x"), lon)
            }
        )

        return ds

    except Exception as e:
        print(f"❌ Error reading {file_name}: {e}")
        return None

# ───────────────────────────────────────────────
# 4. Set Up Zarr Store and FileSystem
# ───────────────────────────────────────────────
fs = s3fs.S3FileSystem(
    key="", secret="", client_kwargs={"endpoint_url": base_dir}
)
mapper = fs.get_mapper(mapper_path)

# ───────────────────────────────────────────────
# 5. Loop Through Tiles and Write to Zarr One-by-One
# ───────────────────────────────────────────────
first_write = True

for i, file_name in enumerate(file_names):
    print(f"🌿 Processing tile {i+1} of {len(file_names)}: {file_name}")
    ds = read_canopy_file(file_name, base_data_url)

    if ds is not None:
        ds = ds.expand_dims("time")  # Make time a dimension
        ds = ds.chunk({"time": 1, "y": 3600, "x": 3600})  # Adjust chunk size if needed

        if first_write:
            ds.to_zarr(mapper, mode="w", consolidated=True)  # Initialize store
            first_write = False
        else:
            ds.to_zarr(mapper, mode="a", append_dim="time", consolidated=True)  # Append

# ───────────────────────────────────────────────
# 6. Load from Zarr Store and Visualise One Tile
# ───────────────────────────────────────────────
print("✅ Finished writing to Zarr. Loading dataset...")
ds_zarr = xr.open_dataset(mapper, engine="zarr", chunks={})
ds_zarr.isel(time=0).canopy_height.plot(cmap="viridis")
plt.title("Global Canopy Height - GLAD 2020 (Tile 0)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
