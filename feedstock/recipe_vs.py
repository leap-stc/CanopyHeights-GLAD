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
import warnings
import logging
import time
warnings.filterwarnings("ignore")

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


    file_names = re.findall(r'3deg_cogs/ETH_GlobalCanopyHeight_10m_2020_[NS]\d{2}[EW]\d{3}_Map\.tif', response.text)
    logger.info(f"✅ Found {len(file_names)} canopy height tiles.")
    
    # ───────────────────────────────────────────────
    # 4. Define Tile Reader
    # ───────────────────────────────────────────────
    def read_canopy_file(file_name: str, base_url: str, i: int,skipped_tiles: list) -> xr.Dataset:
        
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
                            da_mean = rxr.open_rasterio(src_mean).squeeze()
                            da_std = rxr.open_rasterio(src_std).squeeze()
                            ch = da_mean.where(da_mean != 255)
                            std = da_std.where(da_std != 255)
                            
                            lat_vals = da_mean.y.values
                            lon_vals = da_mean.x.values

                            min_val = float(np.nanmin(ch.values))
                            max_val = float(np.nanmax(ch.values))
                            lat_min, lat_max = lat_vals.min(), lat_vals.max()
                            lon_min, lon_max = lon_vals.min(), lon_vals.max()
                            logger.info(f"   🌍 Tile {i} extent: lat=({lat_min:.4f}, {lat_max:.4f}), lon=({lon_min:.4f},{lon_max:.4f})")
                            
                            logger.info(f"   ✅ Tile {i} canopy height range: min={min_val:.2f}, max={max_val:.2f}")
                            if np.isnan(ch.values).all():
                                logger.warning(f"⚠️ Skipping tile {i} ({file_name}) — all values are NaN after masking.")
                                skipped_tiles.append((i, file_name, "All NaNs"))

                                return None
                            else:
                                # Mask nodata and add tile_id dimension
                                ch = da_mean.where(da_mean != 255).expand_dims(tile_id=[i]).compute()
                                std = da_std.where(da_std != 255).expand_dims(tile_id=[i]).compute()
                                
                                return xr.Dataset(
                                        {
                                            "canopy_height": (["tile_id", "lat", "lon"], ch.values),
                                            "std": (["tile_id", "lat", "lon"], std.values),
                                        },
                                        coords={
                                            "time": [datetime(2020, 1, 1)],
                                            "tile_id": [i],
                                            "lat": lat_vals,
                                            "lon": lon_vals
                                        }
                                    )
                else:
                    logger.info(f"❌ HTTP error: mean={response_mean.status_code}, std={response_std.status_code}")
                    skipped_tiles.append((i, file_name, f"HTTP {response_mean.status_code}/{response_std.status_code}"))

            except Exception as e:
                logger.info(f"⚠️ Error reading tile {i} on attempt {attempt + 1}: {e}")
                skipped_tiles.append((i, file_name, f"Exception: {str(e)}"))

                time.sleep(2)

    # ───────────────────────────────────────────────
    # 5. Iterate: Write first one as init, then append
    # ───────────────────────────────────────────────

    
    skipped_file = "skipped_tiles.txt"
    first_written = False
    N=0
    M=3#len(file_names)
    for i, file_name in enumerate(file_names[N:M],start=N):
        skipped_tiles = []
        ds = read_canopy_file(file_name, base_url,i,skipped_tiles)
        if skipped_tiles:
            with open("skipped_tiles.txt", "w") as f:
                for entry in skipped_tiles:
                    tid, fname = entry[:2]
                    reason = entry[2] if len(entry) > 2 else "unknown"
                    f.write(f"{tid},{fname},{reason}\n")
                    # Mask nodata and add tile_id dimension
            logger.info(f"📝 Saved list of {len(skipped_tiles)} skipped tiles to skipped_tiles.txt")
        if ds is None:
            continue
        ds = ds.chunk({"tile_id": 1, "time": 1, "lat": 4500, "lon": 6000})
        if first_written == False:
            ds.to_zarr(store, mode="w", consolidated=False)
            first_written=True
            lat_vals = ds.coords["lat"].values
            lon_vals = ds.coords["lon"].values
            lat_min, lat_max = lat_vals.min(), lat_vals.max()
            lon_min, lon_max = lon_vals.min(), lon_vals.max()
            written_bounds = [] 
            written_bounds.append((lat_min, lat_max, lon_min, lon_max))
        else:
            lat_vals = ds.coords["lat"].values
            lon_vals = ds.coords["lon"].values
            lat_min, lat_max = lat_vals.min(), lat_vals.max()
            lon_min, lon_max = lon_vals.min(), lon_vals.max()
    
            #overlap = False

            ds.to_zarr(store, mode="a", consolidated=False, append_dim="tile_id")
            #logger.info(f"there is no overlap file is written")
            written_bounds.append((lat_min, lat_max, lon_min, lon_max))

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
        print(stitched, "stitched before plotting")
        # Plot
        plt.figure(figsize=(12, 6))
        im = stitched.plot(x='lon', y='lat', cmap='viridis', add_colorbar=True)
        
        plt.title("Global Canopy Height - GLAD 2020 (Subset, Coarsened)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
        outfn = os.path.join(os.getcwd(), "GLAD_mean.png")
        plt.savefig(outfn, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Plot saved to {outfn}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to plot coarsened canopy height: {e}")


if __name__ == "__main__":
    main()