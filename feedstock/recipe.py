# ───────────────────────────────────────────────
# 1. Import Libraries
# ───────────────────────────────────────────────
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os
import re
import rioxarray as rxr
from datetime import datetime
import warnings
import s3fs
warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────
# 2. Define Paths and Remote Access
# ───────────────────────────────────────────────
root_dir = "/Users/mitraasadollahi/Projects/CliMA/dummy/"
product_name = "CanopyHeights-GLAD"
zarr_path = os.path.join(base_dir,root_dir, f"{product_name}.zarr")
os.makedirs(root_dir, exist_ok=True)

base_url = "https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=/"
vrt_file_url = base_url + "ETH_GlobalCanopyHeight_10m_2020_mosaic_Map.vrt"
# Use s3fs for writing
mapper_path = os.path.join(root_dir, product_name, f"{product_name}.zarr")
fs = s3fs.S3FileSystem(
    key="", secret="", client_kwargs={"endpoint_url": base_dir}
)
store = fs.get_mapper(mapper_path)

# ───────────────────────────────────────────────
# 3. Discover Available Tiles
# ───────────────────────────────────────────────
print("🔍 Discovering available canopy height tiles...")
response = requests.get(vrt_file_url)
file_names = re.findall(r'3deg_cogs/ETH_GlobalCanopyHeight_10m_2020_[NS]\d{2}[EW]\d{3}_Map\.tif', response.text)
print(f"✅ Found {len(file_names)} canopy height tiles.")

# ───────────────────────────────────────────────
# 4. Define Tile Reader
# ───────────────────────────────────────────────
from rasterio.io import MemoryFile

def read_canopy_file(file_name: str, base_url: str, i: int) -> xr.Dataset:
    try:
        std_file_name = file_name.replace("_Map.tif", "_Map_SD.tif")
        mean_url = base_url + file_name
        std_url = base_url + std_file_name

        print(f"⏱️ Streaming and reading: {file_name}")

        response_mean = requests.get(mean_url, stream=True)
        response_std = requests.get(std_url, stream=True)

        if response_mean.status_code == 200 and response_std.status_code == 200:
            with MemoryFile(response_mean.content) as memfile_mean, MemoryFile(response_std.content) as memfile_std:
                with memfile_mean.open() as src_mean, memfile_std.open() as src_std:
                    da_mean = rxr.open_rasterio(src_mean).squeeze()
                    da_std = rxr.open_rasterio(src_std).squeeze()

                    # Clean out fill values (255 = no data)
                    ch = da_mean.where(da_mean != 255)
                    std = da_std.where(da_std != 255)

                    tile_id_str = re.search(r'[NS]\d{2}[EW]\d{3}', file_name).group(0)
                    time = datetime(2020, 1, 1)

                    return xr.Dataset(
                        {
                            "canopy_height": (["tile_id", "lat", "lon"], ch.data[np.newaxis, :, :]),
                            "std": (["tile_id", "lat", "lon"], std.data[np.newaxis, :, :])
                        },
                        coords={
                            "tile_id": [tile_id_str],
                            "time": [time],
                            "lat": da_mean.y.values,
                            "lon": da_mean.x.values
                        }
                    )
        else:
            print(f"❌ Could not fetch {file_name}. Status codes: mean={response_mean.status_code}, std={response_std.status_code}")
            return None

    except Exception as e:
        print(f"⚠️ Error reading file {file_name}: {e}")
        return None


# ───────────────────────────────────────────────
# 5. Iterate: Write first one as init, then append
# ───────────────────────────────────────────────
first_written = False

for i, file_name in enumerate(file_names[:100]):
    if i % 10 == 0:
        print(f"🌿 Processing tile {i + 1} of {len(file_names)}")

    ds = read_canopy_file(file_name, base_url)
    if ds is None:
        continue

    ds = ds.chunk({"tile_id": 1, "time": 1, "lat": 1000, "lon": 1000})

    if not first_written:
        # Initialise the Zarr store
        ds.to_zarr(store, mode="w", consolidated=False)
        first_written = True
    else:
        # Append subsequent tiles
        ds.to_zarr(store, mode="a", consolidated=False, append_dim="tile_id")

# ───────────────────────────────────────────────
# 6. Open and Plot Final Dataset
# ───────────────────────────────────────────────
print("🖼️ Plotting a subset for verification...")
ds_zarr = xr.open_dataset(zarr_path, engine="zarr", chunks={})
ds_zarr.isel(time=0).canopy_height.plot(cmap="viridis")
plt.title("Global Canopy Height - GLAD 2020 (Subset)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
