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
from rasterio.io import MemoryFile
import logging
import time
import gc
import sys
import psutil

def report_memory_usage(when):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / 1024 ** 2
    print(f"\n📦 RSS at{when} is: {rss_mb/1000:.2f} GB")

# ───────────────────────────────────────────────
# Configure Logging
# ───────────────────────────────────────────────


def main(N,M,restart):
    log_file = "canopy_height_processing.log"
    png_file = "GLAD_mean.png"
    skipped_file = "skipped_tiles.txt"
    file_name_files = "file_names.txt"
    if restart==True:
        for path in [log_file, png_file,skipped_file]:
            if os.path.exists(path):
                os.remove(path)
                print(f"🧹 Removed existing file: {path}")
    log_file = "canopy_height_processing.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    base_url = "https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=/"
    vrt_file_url = base_url + "ETH_GlobalCanopyHeight_10m_2020_mosaic_Map.vrt"
    # ───────────────────────────────────────────────
    # 3. Discover Available Tiles
    # ───────────────────────────────────────────────
    file_names_path = "file_names.txt"

    if restart:
        print("🔍 Discovering available canopy height tiles...")
        response = requests.get(vrt_file_url)
        file_names = re.findall(
            r'3deg_cogs/ETH_GlobalCanopyHeight_10m_2020_[NS]\d{2}[EW]\d{3}_Map\.tif',
            response.text
        )
        with open(file_names_path, "w") as f:
            for name in file_names:
                f.write(name + "\n")
        logger.info(f"✅ Found and saved {len(file_names)} tile names to file_names.txt.")
    else:
        with open(file_names_path, "r") as f:
            file_names = [line.strip() for line in f.readlines()]
        logger.info(f"📂 Loaded {len(file_names)} tile names from file_names.txt.")

    # ───────────────────────────────────────────────
    # 2. Define Paths and Remote Access
    # ───────────────────────────────────────────────
    #root_dir = "leap-scratch/mitraa90/"
    product_name = "CanopyHeights-GLAD"
    store = f"gs://leap-persistent/data-library/{product_name}/{product_name}.zarr" 
    #os.makedirs(root_dir, exist_ok=True)
    base_url = "https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=/"
    vrt_file_url = base_url + "ETH_GlobalCanopyHeight_10m_2020_mosaic_Map.vrt"
    # ───────────────────────────────────────────────
    # 3. Discover Available Tiles
    # ───────────────────────────────────────────────
    print("🔍 Discovering available canopy height tiles...")
    response = requests.get(vrt_file_url)
    #file_names = re.findall(r'3deg_cogs/ETH_GlobalCanopyHeight_10m_2020_[NS]\d{2}[EW]\d{3}_Map\.tif', response.text)
    logger.info(f"✅ Found {len(file_names)} canopy height tiles.")
    # ───────────────────────────────────────────────
    # 4. Define Tile Reader
    # ───────────────────────────────────────────────
    def read_canopy_file(file_name: str, base_url: str, i: int,skipped_tiles: list) -> xr.Dataset:
        mean_url = base_url + file_name
        r = requests.get(mean_url, stream=True, timeout=10)
        print(f"→ {file_name}: status={r.status_code}, content-type={r.headers.get('Content-Type')}")
        for attempt in range(5):
            logger.info(f"🔄 Attempt {attempt + 1}/5 to read tile {i}: {file_name}")
            try:
                std_file_name = file_name.replace("_Map.tif", "_Map_SD.tif")
                std_url = base_url + std_file_name
                response_mean = requests.get(mean_url, stream=True, timeout=10)
                response_std = requests.get(std_url, stream=True, timeout=10)
                if response_mean.status_code == 200 and response_std.status_code == 200:
                    with MemoryFile(response_mean.content) as memfile_mean, MemoryFile(response_std.content) as memfile_std:
                        with memfile_mean.open() as src_mean, memfile_std.open() as src_std:
                            da_mean = rxr.open_rasterio(src_mean).squeeze().rename({'x': 'lon', 'y': 'lat'})
                            da_std = rxr.open_rasterio(src_std).squeeze().rename({'x': 'lon', 'y': 'lat'})
                            ch = da_mean.where(da_mean != 255)
                            std = da_std.where(da_std != 255)
                            if np.isnan(ch.values).all():
                                logger.warning(f"⚠️ Skipping tile {i} ({file_name}) — all values are NaN.")
                                skipped_tiles.append((i, file_name, "All NaNs"))
                                return None  
                            min_val = float(np.nanmin(ch.values))
                            max_val = float(np.nanmax(ch.values))
                            logger.info(f"   🌍 Tile {i} extent: lat=({da_mean.lat.min().item():.4f}, {da_mean.lat.max().item():.4f}), "
                                        f"lon=({da_mean.lon.min().item():.4f}, {da_mean.lon.max().item():.4f}) and ch.lat.shape: {ch.lat.shape}")
                            logger.info(f"   ✅ Tile {i} canopy height range: min={min_val:.2f}, max={max_val:.2f}")
        
                            ch = ch.expand_dims(tile_id=[i]).compute()
                            std = std.expand_dims(tile_id=[i]).compute()
                            lat2d = ch.lat.values[np.newaxis, :]    # shape (1, n_lat)
                            lon2d = ch.lon.values[np.newaxis, :]    # shape (1, n_lon)
                            ds =  xr.Dataset(
                                  {
                                    "canopy_height": (["tile_id","y","x"], ch.values),
                                    "std": (["tile_id", "lat", "lon"], std.values) ,
                                    "lat":       (["tile_id","lat"],  lat2d),
                                    "lon":       (["tile_id","lon"],  lon2d),
                                  },
                                  coords={"tile_id":[i],
                                  "time": [datetime(2020, 1, 1)],}
                                )
                            # assume ds is your xarray Dataset or DataArray
                            if {"x", "y"}.issubset(ds.dims):
                                ds = ds.rename({"x": "lon", "y": "lat"})
                            tile_lat = ds["lat"].squeeze("tile_id")  
                            tile_lon = ds["lon"].squeeze("tile_id")
                            lat_min, lat_max = float(tile_lat.min()), float(tile_lat.max())
                            lon_min, lon_max = float(tile_lon.min()), float(tile_lon.max())
                            print(f"Tile {i} lat = ({lat_min:.4f}, {lat_max:.4f})")
                            print(f"Tile {i} lon = ({lon_min:.4f}, {lon_max:.4f})")
                            return ds
                else:
                    logger.info(f"❌ HTTP error: mean={response_mean.status_code}, std={response_std.status_code}")
                    skipped_tiles.append((i, file_name, f"HTTP {response_mean.status_code}/{response_std.status_code}"))
            except Exception as e:
                logger.warning(f"⚠️ Error reading tile {i} on attempt {attempt + 1}: {e}")
                time.sleep(2)
            finally:
                # Only close if defined
                try:
                    response_mean.close()
                except:
                    pass
                try:
                    response_std.close()
                except:
                    pass
    # ───────────────────────────────────────────────
    # 5. Iterate: Write first one as init, then append
    # ───────────────────────────────────────────────
    skipped_file = "skipped_tiles.txt"
    if N==0:
        first_written = False
        first_skipped=False
    else:
        first_written = True
    for i, file_name in enumerate(file_names[N:M], start=N):
        report_memory_usage(f"start of the loop {i}")
        skipped_tiles = []
        ds = read_canopy_file(file_name, base_url,i,skipped_tiles)
        if skipped_tiles:
            if first_skipped==False:
                mode="w"
            else:
                mode="a"
            with open("skipped_tiles.txt", mode) as f:
                if mode=="w":
                    first_skipped=true
                for entry in skipped_tiles:
                    tid, fname = entry[:2]
                    reason = entry[2] if len(entry) > 2 else "unknown"
                    f.write(f"{tid},{fname},{reason}\n")
                    # Mask nodata and add tile_id dimension
            logger.info(f"📝 Saved list of {len(skipped_tiles)} skipped tiles to skipped_tiles.txt")
        if ds is None:
            continue
        ds = ds.chunk({"tile_id": 1, "time": 1, "lat": 6000, "lon": 4000})
        print(f"N is {N},  first_written is {first_written}")
        if first_written == False:
            ds.to_zarr(store, mode="w", consolidated=True)
            first_written=True
        else:
            ds.to_zarr(store, mode="a",  append_dim="tile_id")
            #logger.info(f"there is no overlap file is written") 
        ds.close()
        del ds
        gc.collect()
        if i==M-1:
            logger.info("last file is also written to zarr")

if __name__ == "__main__":
    t0 = time.time()
    restart=False
    end=2651
    start=0
    main(start, end,restart)
    elapsed = time.time() - t0
    print(f"✅ Batch {start}-{end-1} completed in {elapsed:.2f} seconds")