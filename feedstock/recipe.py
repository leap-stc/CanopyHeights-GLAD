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
warnings.filterwarnings("ignore")
import pandas as pd
from dask.distributed import Client
from rasterio.io import MemoryFile


# ───────────────────────────────────────────────
# 2. functions
# ───────────────────────────────────────────────




def read_canopy_file(file_name: str, base_url: str, i: int) -> xr.Dataset:
    """
    Streams and reads a canopy height tile from ETHZ GLAD dataset.

    Parameters:
        file_name (str): Name of the tile file (e.g. '..._N00E006_Map.tif')
        base_url (str): URL base for downloading tiles
        i (int): Unique ID to assign to this tile in the dataset

    Returns:
        xr.Dataset or None if download/parsing failed
    """
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

                    # Mask out invalid values (e.g., 255 is often used as no-data)
                    ch = da_mean.where(da_mean != 255)
                    std = da_std.where(da_std != 255)

                    return xr.Dataset(
                        {
                            "canopy_height": (["lat", "lon"], ch.data),
                            "std": (["lat", "lon"], std.data),
                        },
                        coords={
                            "time": [datetime(2020, 1, 1)],
                            "tile_id": [i],  # Must be unique
                            "lat": da_mean.y.values,
                            "lon": da_mean.x.values,
                        }
                    )
        else:
            print(f"❌ Could not fetch {file_name} (mean: {response_mean.status_code}, std: {response_std.status_code})")
            return None

    except Exception as e:
        print(f"⚠️ Error reading file {file_name}: {e}")
        return None

def get_canopy_tile_list(vrt_url: str) -> list:
    print("🔍 Requesting tile list from VRT...")
    try:
        response = requests.get(vrt_url)
        response.raise_for_status()  # raises error if status code is not 200
        file_names = re.findall(
            r'3deg_cogs/ETH_GlobalCanopyHeight_10m_2020_[NS]\d{2}[EW]\d{3}_Map\.tif',
            response.text
        )
        print(f"Found {len(file_names)} canopy height tiles.")
        return file_names
    except Exception as e:
        print(f"❌ Failed to fetch or parse tile list: {e}")
        return []

def plot_canopy_height_subset(store_mapper, tile_id=0, coarsen_factor=100):
    """
    Opens the Zarr store and plots a downsampled canopy height map for a single tile.

    Parameters:
        store_mapper: Zarr-compatible store (e.g. fsspec mapper or string path)
        tile_id (int): Which tile to plot
        coarsen_factor (int): Degree of downsampling for visualization

    Returns:
        None
    """
    print("🖼️ Opening Zarr store to plot a subset for verification...")

    try:
        ds_zarr = xr.open_dataset(store_mapper, engine="zarr", chunks={})
        tile = ds_zarr.isel(tile_id=tile_id, time=0)

        # Optional coarsening to speed up plotting
        plot_data = tile.canopy_height.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary="trim").mean()

        # Plot
        plot_data.plot()
        plt.title(f"🌍 Global Canopy Height - Tile {tile_id} (Downsampled x{coarsen_factor})")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"⚠️ Could not open or plot dataset: {e}")

def write_tiles_to_zarr(file_names, base_url, mapper, max_tiles=2):
    """
    Iterates over canopy height tile names, reads each one, and writes to a Zarr store.

    Parameters:
        file_names (list): List of .tif file names to process
        base_url (str): URL prefix for fetching files
        mapper: s3fs or fsspec mapper for Zarr storage
        max_tiles (int): Limit number of tiles to process (for testing)

    Returns:
        None
    """
    first_written = False
    total_tiles = len(file_names[:max_tiles])

    for i, file_name in enumerate(file_names[:max_tiles]):
        if i % 10 == 0 or i == 0:
            print(f"🌿 [{i+1}/{total_tiles}] Processing tile: {file_name}")

        ds = read_canopy_file(file_name, base_url, tile_id=i)
        if ds is None:
            print(f"⚠️ Skipping tile {i} due to read error.")
            continue

        # Chunk dataset for scalable I/O
        ds = ds.chunk({"tile_id": 1, "time": 1, "lat": 500, "lon": 500})

        # Write or append to Zarr
        try:
            if not first_written:
                print(f"Initializing Zarr store at: {mapper.root}")
                ds.to_zarr(mapper, mode="w", consolidated=False)
                first_written = True
                print("first file written")
            else:
                print("writing in already existing files")
                ds.to_zarr(mapper, mode="a", consolidated=False, append_dim="tile_id")
        except Exception as e:
            print(f"❌ Failed to write tile {i}: {e}")
# ───────────────────────────────────────────────
# 2. main
# ───────────────────────────────────────────────
def main():
    # Start Dask client
    client = Client(
    )
    print("Dask client started")

    # Define storage and dataset parameters
    base_dir = "https://nyu1.osn.mghpcc.org"
    root_dir = "gs://leap-persistant" #"leap-pangeo-pipeline"
    product_name = "CanopyHeights-GLAD"
    zarr_path = f"{root_dir}/{product_name}/{product_name}.zarr"


    # Define canopy dataset source
    base_url = "https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=/"
    vrt_file_url = base_url + "ETH_GlobalCanopyHeight_10m_2020_mosaic_Map.vrt"

    # Get tile list
    file_names = get_canopy_tile_list(vrt_file_url)
    if not file_names:
        client.close()
        print("❌ No files found. Exiting.")
        return

    # Write tiles using original read_canopy_file
    first_written = False
    for i, file_name in enumerate(file_names[:2]):
        print(f"🌿 [{i+1}/{len(file_names[:2])}] Processing tile: {file_name}")
        ds = read_canopy_file(file_name, base_url, i)

        if ds is None:
            continue

        ds = ds.chunk({"tile_id": 1, "time": 1, "lat": 500, "lon": 500})

        try:
            if not first_written:
                ds.to_zarr(zarr_path, mode="w", consolidated=False)
                first_written = True
                print("first write was successful")
            else:
                ds.to_zarr(zarr_path, mode="a", consolidated=False, append_dim="tile_id")
        except Exception as e:
            print(f"❌ Failed to write tile {i}: {e}")

    # Optional: Plot one tile
    plot_canopy_height_subset(zarr_path, tile_id=0, coarsen_factor=100)

    # Cleanup
    print("Workflow complete")
    client.close()
    print("Dask client closed")


if __name__ == "__main__":
    main()

