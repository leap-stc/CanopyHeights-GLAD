# —— 1. Import Libraries and Start Dask ——
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
from dask.diagnostics import ProgressBar

client = Client(n_workers=4, threads_per_worker=4, memory_limit="4GB")
print(client)

# —— 2. Define Paths and Remote URLs ——
base_dir = "https://nyu1.osn.mghpcc.org"
root_dir = "leap-pangeo-pipeline"
product_name = "CanopyHeights-GLAD"
zarr_store_path = os.path.join(root_dir, product_name, f"{product_name}.zarr")
mapper_path = os.path.join(root_dir, f"{product_name}.zarr")
base_data_url = "https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=/"

# —— 3. Retrieve tile names from VRT file ——
vrt_url = base_data_url + "ETH_GlobalCanopyHeight_10m_2020_mosaic_Map.vrt"
print("requesting access")
vrt_response = requests.get(vrt_url)
print("finding files access")
file_names = re.findall(r'3deg_cogs/ETH_GlobalCanopyHeight_10m_2020_[NS]\d{2}[EW]\d{3}_Map\\.tif', vrt_response.text)
print(f"{len(file_names)} filenames are found")

# —— 4. Read a Tile and Build Dataset ——
def read_canopy_file(file_name: str, base_url: str) -> xr.Dataset:
    try:
        std_file_name = file_name.replace("_Map.tif", "_Map_SD.tif")
        mean_url = base_url + file_name
        std_url = base_url + std_file_name

        da_mean = rxr.open_rasterio(mean_url, masked=True, chunks={}).squeeze()
        da_std = rxr.open_rasterio(std_url, masked=True, chunks={}).squeeze()

        ch = da_mean.where(da_mean != 255)
        std = da_std.where(da_std != 255)

        lon, lat = np.meshgrid(da_mean.x.values, da_mean.y.values)
        date = pd.to_datetime("2020-01-01")

        tile_id_str = re.search(r'[NS]\d{2}[EW]\d{3}', file_name).group(0)
        tile_id_array = np.full(ch.shape, tile_id_str, dtype=object)

        ds = xr.Dataset(
            {
                "canopy_height": ("y", "x", ch.data),
                "std": ("y", "x", std.data),
                "tile_id": ("y", "x", tile_id_array)
            },
            coords={
                "time": date,
                "lat": ("y", "x", lat),
                "lon": ("y", "x", lon)
            }
        )
        return ds

    except Exception as e:
        print(f"⚠️ Error reading file {file_name}: {e}")
        return None

# —— 5. Create Global Grid Dimensions ——
lat_vals = np.arange(-60, 90, 10 / 3600)
lon_vals = np.arange(-180, 180, 10 / 3600)
y_dim = len(lat_vals)
x_dim = len(lon_vals)
time_val = pd.to_datetime("2020-01-01")

# —— 6. Initialize Empty Zarr Dataset ——
init_ds = xr.Dataset(
    {
        "canopy_height": ("time", "y", "x", np.empty((1, y_dim, x_dim), dtype=np.float32)),
        "std": ("time", "y", "x", np.empty((1, y_dim, x_dim), dtype=np.float32)),
        "tile_id": ("time", "y", "x", np.empty((1, y_dim, x_dim), dtype=object))
    },
    coords={
        "time": [time_val],
        "y": lat_vals,
        "x": lon_vals
    }
)
init_ds = init_ds.chunk({"time": 1, "y": 1000, "x": 1000})
init_ds.to_zarr(zarr_store_path, mode="w", consolidated=False, compute=True)
print("Zarr initialized")

# —— 7. Write Each Tile to the Zarr Region ——
fs = s3fs.S3FileSystem(key="", secret="", client_kwargs={"endpoint_url": base_dir})
mapper = fs.get_mapper(mapper_path)

for i, file_name in enumerate(file_names):
    print(f"🌿 Processing tile {i + 1} of {len(file_names)}: {file_name}")

    ds = read_canopy_file(file_name, base_data_url)
    if ds is None:
        continue

    ds = ds.expand_dims("time")
    ds = ds.rename({"lat": "y", "lon": "x"})
    ds = ds.chunk({"time": 1, "y": 1000, "x": 1000})

    y_idx_start = np.searchsorted(lat_vals, ds.y.values.min())
    y_idx_end = y_idx_start + ds.dims['y']
    x_idx_start = np.searchsorted(lon_vals, ds.x.values.min())
    x_idx_end = x_idx_start + ds.dims['x']

    try:
        with ProgressBar():
            ds.to_zarr(
                mapper,
                region={"time": slice(0, 1), "y": slice(y_idx_start, y_idx_end), "x": slice(x_idx_start, x_idx_end)},
                compute=True
            )
    except Exception as e:
        print(f"⚠️ Failed to write {file_name}: {e}")
    del ds

print("All tiles processed")

# —— 8. Visual Check ——
ds_zarr = xr.open_dataset(mapper, engine="zarr", chunks={})
subset = ds_zarr.isel(x=slice(0, 1000), y=slice(0, 1000)).canopy_height.isel(time=0)
subset.plot(cmap="viridis")
plt.title("Global Canopy Height - GLAD 2020 (Subset)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
