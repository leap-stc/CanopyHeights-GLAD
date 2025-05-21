# ───────────────────────────────────────────────
    # 6. Open and Plot Final Dataset
    # ───────────────────────────────────────────────
    print("🖼️ Plotting a subset for verification...")
    ds_zarr = xr.open_dataset(store, engine="zarr", chunks={})
    # --- new: print out the chunk layout for the canopy_height variable ---
    ch = ds_zarr["canopy_height"].chunks

    # --- new: print out the chunk layout for the canopy_height variable ---
    print("\n📦 Variables:")
    for var in ds_zarr.data_vars:
        print(f"  • {var}: {ds_zarr[var].dims} shape={ds_zarr[var].shape} dtype={ds_zarr[var].dtype}")
    print("\n📐 Dimensions:")
    for dim, size in ds_zarr.sizes.items():
        print(f"  • {dim}: {size}")
    print("\n🧭 Coordinate Ranges:")
    for coord in ["lat", "lon", "tile_id", "time"]:
        if coord in ds_zarr.coords:
            values = ds_zarr[coord].values
            if np.issubdtype(values.dtype, np.number):
                print(f"  • {coord}: min={values.min():.2f}, max={values.max():.2f}, len={len(values)}")
            else:
                print(f"  • {coord}: {values} (len={len(values)})")
    # ───────────────────────────────────────────────
    # Estimate memory size of first chunk
    # ───────────────────────────────────────────────
    ch = ds_zarr["canopy_height"].chunks
    print(f"\n🧠 Chunk layout for canopy_height: {ch!r}")
    dtype_bytes = ds_zarr["canopy_height"].dtype.itemsize
    first_chunk = tuple(c[0] for c in ch)  # get size of first chunk
    chunk_elems = np.prod(first_chunk)
    chunk_MiB = chunk_elems * dtype_bytes / (1024 ** 2)
    logger.info(f"Approximate memory per chunk: {chunk_MiB:.2f} MiB")
    print("Unique tile_ids:", ds_zarr.tile_id.values)
    print("Number of unique tile_ids:", len(np.unique(ds_zarr.tile_id.values)))
    print("Total tiles in file:", ds_zarr.sizes['tile_id'])

    # ───────────────────────────────────────────────
    # Plot a coarsened global view from first tile
    # ───────────────────────────────────────────────
    try:
        #ds_zarr.canopy_height.mean(dim="tile_id").coarsen(lat=100, lon=100).mean().plot()
        # Open your Zarr dataset
        # Let's assume the variable name is 'canopy_height'
        tiles = ds_zarr['canopy_height']  # shape: (tile, lat, lon)
        # Assign artificial lat/lon for plotting
        n_tiles, n_lat, n_lon = tiles.shape
        # Determine coarsening factor — e.g., reduce to ~360 x 360 pixels per tile
        coarsen_factor = 100  # adjust to taste
        # Coarsen and store in list
        coarsened_tiles = []
        for i in range(tiles.shape[0]):
            tile = tiles.isel(tile_id=i)
            tile_coarse = tile.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
            coarsened_tiles.append(tile_coarse)
        # Stitch tiles along longitude
        stitched = xr.concat(coarsened_tiles, dim="lon")
        # Adjust lon coordinates to reflect stitched layout
        tile_lon_range = ds['lon']
        lon_per_tile = tile_lon_range.values
        lon_step = (lon_per_tile[-1] - lon_per_tile[0]) / (len(lon_per_tile) - 1)
        tile_width_deg = (lon_per_tile[-1] - lon_per_tile[0]) + lon_step
        stitched = stitched.assign_coords(
            lon=np.linspace(
                float(lon_per_tile[0]),
                float(lon_per_tile[0]) + tile_width_deg * tiles.shape[0],
                stitched.sizes['lon']
            )
        )
        #print(stitched, "stitched before plotting")
        # Plot
        plt.figure(figsize=(12, 6))
        im = stitched.plot(x='lon', y='lat', cmap='viridis', add_colorbar=True)
        plt.title("Global Canopy Height - GLAD 2020 (Subset, Coarsened)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        outfn = os.path.join(os.getcwd(), f"GLAD_mean_{N}_to_{M}.png")
        plt.savefig(outfn, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Plot saved to {outfn}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to plot coarsened canopy height: {e}")




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
import dask
from dask.distributed import Client
from rasterio.io import MemoryFile
import warnings
import logging
import time
warnings.filterwarnings("ignore")
import zarr

import tracemalloc
tracemalloc.start()

import gc
from dask import config
config.set({
    "distributed.worker.memory.target": 0.6,   # spill to disk once 60% of the limit is used
    "distributed.worker.memory.spill": 0.7,    # allow spilling until 70% is used
    "distributed.worker.memory.pause": 0.8,    # pause worker at 80% (backpressure)
})
config.set(scheduler="synchronous")
import psutil
import  dask.array as da
def report_mem(when):
    p = psutil.Process()
    rss = p.memory_info().rss / (1024**3)
    logger.info(f"[{when}] RSS = {rss:.1f} GiB")
# ───────────────────────────────────────────────
# Configure Logging
# ───────────────────────────────────────────────
log_file = "canopy_height_processing.log"
png_file = "GLAD_mean.png"

skipped_file = "skipped_tiles.txt"

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

def main():
    def debug_memory():
        proc = psutil.Process()
        print(f"→ RSS = {proc.memory_info().rss/1024**3:.2f} GiB")
        gc.collect()
        cnt, total = 0, 0
        for obj in gc.get_objects():
            try:
                if isinstance(obj, np.ndarray):
                    cnt += 1
                    total += obj.nbytes
            except ReferenceError:
                # some objects gone during iteration
                continue
        print(f"→ {cnt} NumPy arrays totalling {total/1024**3:.2f} GiB")

    # ───────────────────────────────────────────────
    # 2. Define Paths and Remote Access
    # ───────────────────────────────────────────────
    product_name = "CanopyHeights-GLAD"
    store = f"gs://leap-persistent/data-library/{product_name}/{product_name}.zarr" 
    base_url = "https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=/"
    vrt_file_url = base_url + "ETH_GlobalCanopyHeight_10m_2020_mosaic_Map.vrt"
    # ───────────────────────────────────────────────
    # 3. Discover Available Tiles
    # ───────────────────────────────────────────────
    print("🔍 Discovering available canopy height tiles...")
    response = requests.get(vrt_file_url)


    file_names = re.findall(r'3deg_cogs/ETH_GlobalCanopyHeight_10m_2020_[NS]\d{2}[EW]\d{3}_Map\.tif', response.text)
    logger.info(f"✅ Found {len(file_names)} canopy height tiles.")

    vrt = rxr.open_rasterio(vrt_file_url).squeeze()
    global_lat = vrt.y.values
    global_lon = vrt.x.values
    Ntiles=len(file_names)
     
    # ───────────────────────────────────────────────
    # Pre‑allocate an empty dataset with nan fill
    # ───────────────────────────────────────────────
    # after you’ve fetched global_lat, global_lon, and computed Ntiles …

    # build a Zarr‐side compressor (this *is* a BytesBytesCodec)
    
    from numcodecs import Blosc
    zcomp = Blosc(cname="zstd", clevel=1, shuffle=2)
    
    encoding = {
    "canopy_height": {
        "compressor": {
            "name": "blosc",
            "configuration": {
                "cname": "zstd",
                "clevel": 1,
                "shuffle": 2
            }
        },
        "chunks": (1, 4500, 6000)
    },
    "std": {
        "compressor": {
            "name": "blosc",
            "configuration": {
                "cname": "zstd",
                "clevel": 1,
                "shuffle": 2
            }
        },
        "chunks": (1, 4500, 6000)
    }
}


    empty = da.full(
    shape=(Ntiles, len(global_lat), len(global_lon)),
    fill_value=np.nan,
    chunks=(1, 4500, 6000),
    dtype="float32")
    ds0 = xr.Dataset(
        {
            "canopy_height": (("tile_id", "lat", "lon"), empty),
            "std":           (("tile_id", "lat", "lon"), empty),
        },
        coords={
            "tile_id": np.arange(Ntiles),
            "lat":      global_lat,
            "lon":      global_lon,
        },)

    ds0.to_zarr(
            store,
            mode="w",
            consolidated=False
        )


    # ───────────────────────────────────────────────
    # 4. Define Tile Reader
    # ───────────────────────────────────────────────
    def read_canopy_file(file_name: str, base_url: str, i: int) -> xr.Dataset:
        import rioxarray as rxr

        
        mean_url = base_url + file_name
        r = requests.get(mean_url, stream=True, timeout=10)
        print(f"→ {file_name}: status={r.status_code}, content-type={r.headers.get('Content-Type')}")
        for attempt in range(3):
            logger.info(f"🔄 Attempt {attempt + 1}/3 to read tile {i}: {file_name}")
            try:
                std_file_name = file_name.replace("_Map.tif", "_Map_SD.tif")
                std_url = base_url + std_file_name
                response_mean = requests.get(mean_url, stream=True)
                response_std = requests.get(std_url, stream=True)
                if response_mean.status_code == 200 and response_std.status_code == 200:
                    with MemoryFile(response_mean.content) as memfile_mean, MemoryFile(response_std.content) as memfile_std:
                        with memfile_mean.open() as src_mean, memfile_std.open() as src_std:
                            da_mean = rxr.open_rasterio(src_mean, chunks={"band":1,"x":4500,"y":6000}).squeeze()  # CHANGED
                            da_std  = rxr.open_rasterio(src_std,  chunks={"band":1,"x":4500,"y":6000}).squeeze()  # CHANGED
                            da_mean = da_mean.rename({"y": "lat", "x": "lon"})
                            da_std  = da_std. rename({"y": "lat", "x": "lon"})
                            # Mask and downcast to float32 lazily
                            da_mean = da_mean.where(da_mean != 255).astype("float32")  # CHANGED
                            da_std  = da_std.where(da_std  != 255).astype("float32")  # CHANGED
                
                            
                            ch = da_mean.where(da_mean != 255).expand_dims(tile_id=[i]).compute()
                            std = da_std.where(da_std != 255).expand_dims(tile_id=[i]).compute()
                            # Log min/max (computed in small chunks)
                            min_val = float(np.nanmin(ch.values))
                            max_val = float(np.nanmax(ch.values))
                            lat_min, lat_max = da_mean.lat.min(), da_mean.lat.max()
                            lon_min, lon_max = da_mean.lon.min(), da_mean.lon.max()
                            logger.info(f"   🌍 Tile {i} extent: lat=({lat_min:.4f}, {lat_max:.4f}), lon=({lon_min:.4f},{lon_max:.4f})")
                            logger.info(f"   ✅ Tile {i} canopy height range: min={min_val:.2f}, max={max_val:.2f}")
                            arr_ch   = da_mean.values
                            arr_std  = da_std.values
                            tlat     = da_mean.lat.values
                            tlon     = da_mean.lon.values

                            return arr_ch, arr_std, tlat, tlon
                else:
                    logger.info(f"❌ HTTP error: mean={response_mean.status_code}, std={response_std.status_code}")
                    
            except Exception as e:
                logger.info(f"⚠️ Error reading tile {i} on attempt {attempt + 1}: {e}")
                time.sleep(2)

    # ───────────────────────────────────────────────
    # 5. Iterate: Write first one as init, then append
    # ───────────────────────────────────────────────

    
    first_write = True
    N=0
    M=2#21#Ntiles
    for i, file_name in enumerate(file_names[N:M],start=N):
        report_mem("start loop")
        #debug_memory()
        if i != 0 and i % 10 == 0 and i !=2653 :
            print(f"🌿 Processing tile {i + 1} and printing zarr file statistics")
            ds_zarr = xr.open_dataset(store, engine="zarr", chunks={})        
            print("sored zarr has \n📊 variable Value Ranges:")
            for var in ["canopy_height", "std"]:
                if var in ds_zarr:
                    arr = ds_zarr[var]
                    try:
                        # Select the first tile_id and time if available
                        sample = arr
                        min_val = float(sample.min(skipna=True).compute().values)
                        max_val = float(sample.max(skipna=True).compute().values)
                        print(f"  • {var}: min={min_val:.2f}, max={max_val:.2f}")
                    except Exception as e:
                        print(f"  ⚠️ Could not compute value range for {var}: {e}")
            ds_zarr.close()
            del ds_zarr 
            gc.collect()
        #skipped_tiles = []
        
        ch_arr, std_arr, tlat, tlon =  read_canopy_file(file_name, base_url,i)#,skipped_tiles)
        i0 = np.searchsorted(global_lat, tlat.min(),  'left')
        i1 = np.searchsorted(global_lat, tlat.max(),  'right')
        j0 = np.searchsorted(global_lon, tlon.min(), 'left')
        j1 = np.searchsorted(global_lon, tlon.max(), 'right')
        

        if np.isnan(ch_arr).all():
            print(f"Tile {i+1} is all NaN, skipping write")
            continue
        # write just that slice
        root["canopy_height"][i, i0:i1, j0:j1] = ch_arr
        root["std"          ][i, i0:i1, j0:j1] = std_arr    
        
        del ch_arr, std_arr, tlat, tlon
        #client.restart()  # clear memory completely
        gc.collect()
            
        if i==M-1:
            logger.info("last file is also written to zarr")
    # ───────────────────────────────────────────────
    # 6. Open and Plot Final Dataset
    # ───────────────────────────────────────────────
    print("🖼️ Plotting a subset for verification...")
    ds_zarr = xr.open_dataset(store, engine="zarr", chunks={})
    # --- new: print out the chunk layout for the canopy_height variable ---
    ch = ds_zarr["canopy_height"].chunks

    # --- new: print out the chunk layout for the canopy_height variable ---
    print("\n📦 Variables:")
    for var in ds_zarr.data_vars:
        print(f"  • {var}: {ds_zarr[var].dims} shape={ds_zarr[var].shape} dtype={ds_zarr[var].dtype}")
    
    print("\n📐 Dimensions:")
    for dim, size in ds_zarr.sizes.items():
        print(f"  • {dim}: {size}")
    
    print("\n🧭 Coordinate Ranges:")
    for coord in ["lat", "lon", "tile_id", "time"]:
        if coord in ds_zarr.coords:
            values = ds_zarr[coord].values
            if np.issubdtype(values.dtype, np.number):
                print(f"  • {coord}: min={values.min():.2f}, max={values.max():.2f}, len={len(values)}")
            else:
                print(f"  • {coord}: {values} (len={len(values)})")
    
    print("\n📊 Variable Value Ranges:")
    for var in ["canopy_height", "std"]:
        if var in ds_zarr:
            arr = ds_zarr[var]
            try:
                sample = arr
                min_val = float(sample.min(skipna=True).compute().values)
                max_val = float(sample.max(skipna=True).compute().values)
                print(f"  • {var}: min={min_val:.2f}, max={max_val:.2f}")
            except Exception as e:
                print(f"  ⚠️ Could not compute range for {var}: {e}")
    
    # ───────────────────────────────────────────────
    # Estimate memory size of first chunk
    # ───────────────────────────────────────────────
    ch = ds_zarr["canopy_height"].chunks
    print(f"\n🧠 Chunk layout for canopy_height: {ch!r}")
    dtype_bytes = ds_zarr["canopy_height"].dtype.itemsize
    first_chunk = tuple(c[0] for c in ch)  # get size of first chunk
    chunk_elems = np.prod(first_chunk)
    chunk_MiB = chunk_elems * dtype_bytes / (1024 ** 2)
    logger.info(f"Approximate memory per chunk: {chunk_MiB:.2f} MiB")
    print("Unique tile_ids:", ds_zarr.tile_id.values)
    print("Number of unique tile_ids:", len(np.unique(ds_zarr.tile_id.values)))
    print("Total tiles in file:", ds_zarr.sizes['tile_id'])

    # ───────────────────────────────────────────────
    # Plot a coarsened global view from first tile
    # ───────────────────────────────────────────────
    try:
        ds_zarr.canopy_height.mean(dim="tile_id").coarsen(lat=100, lon=100).mean().plot()

        plt.title("Global Canopy Height - GLAD 2020 (Subset)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        outfn = os.path.join(os.getcwd(), "GLAD_mean.png")
        plt.savefig(outfn, dpi=150, bbox_inches="tight")
        logger.info(f"✅ Saved mean‑Canopy Height plot to: {outfn}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to plot coarsened canopy height: {e}")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    elapsed_minutes = (end - start) / 60
    print(f"\n⏱️ Total execution time: {elapsed_minutes:.2f} minutes")
    logger.info(f"⏱️ Total execution time: {elapsed_minutes:.2f} minutes")

        