import time
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

def stitch_and_plot_tiles(ds_zarr, N, M, coarsen_factor=100, outdir=None):
    """
    Coarsen tiles N through M-1, stitch them along their real lat/lon,
    and plot a single global mosaic.
    """
    t0 = time.time()
    arr = ds_zarr["canopy_height"]

    # If dims are still ('tile_id','y','x'), rename to ('tile_id','lat','lon')
    if {"y","x"}.issubset(arr.dims):
        arr = arr.rename({"y": "lat", "x": "lon"})

    coarsened = []

    print("🧩 Coarsening tiles and logging extents…")
    for i in range(N, M):
        # pull just tile i (data + coords) into memory; dims → (lat, lon)
        tile = arr.isel(tile_id=i).compute()

        # log its native bounds
        lat0, lat1 = float(tile.lat.min()), float(tile.lat.max())
        lon0, lon1 = float(tile.lon.min()), float(tile.lon.max())
        if i%100==0:
            print(f" Tile {i}: lat=({lat0:.4f},{lat1:.4f}), lon=({lon0:.4f},{lon1:.4f})")

        # coarsen in place, skipping NaNs
        small = tile.coarsen(
            lat=coarsen_factor,
            lon=coarsen_factor,
            boundary="trim"
        ).mean(skipna=True)

        coarsened.append(small)

    print("\n🧵 Stitching all coarsened tiles into one mosaic…")
    ds_mosaic = xr.combine_by_coords(coarsened, combine_attrs="override")
    if isinstance(ds_mosaic, xr.Dataset):
        # replace 'canopy_height' with the actual var name if different
        mosaic = ds_mosaic["canopy_height"]
    else:
        mosaic = ds_mosaic

    print(f"📏 Final shape: {mosaic.shape}")
    print(f"🗺  lat: {float(mosaic.lat.min()):.2f} → {float(mosaic.lat.max()):.2f}")
    print(f"🗺  lon: {float(mosaic.lon.min()):.2f} → {float(mosaic.lon.max()):.2f}\n")

    # plot
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    im = ax.imshow(
        mosaic.values,
        origin="lower",
        extent=[
            float(mosaic.lon.min()), float(mosaic.lon.max()),
            float(mosaic.lat.min()), float(mosaic.lat.max())
        ],
        aspect="auto"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Canopy Height Tiles {N}–{M-1} (coarsened ×{coarsen_factor})")
    fig.colorbar(im, ax=ax, label="Canopy height")

    # save
    fname = f"GLAD_mosaic_{N}_{M}.png"
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fname = os.path.join(outdir, fname)
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved plot: {fname}")
    print(f"⏱ Total time: {(time.time() - t0)/60:.2f} min")
# ──────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────
def extract_coord(ds_zarr, i):
    """
    Print and return the lat/lon bounds for tile i
    in a ds_zarr whose 'canopy_height' has dims (tile_id,lat,lon).
    """
    # pull in tile i (data + coords) into memory
    arr = ds_zarr["canopy_height"].isel(tile_id=i).compute()

    # coordinate bounds
    lat0 = arr.lat.min().item()
    lat1 = arr.lat.max().item()
    lon0 = arr.lon.min().item()
    lon1 = arr.lon.max().item()

    print(f"Tile {i} — lat = ({lat0:.4f}, {lat1:.4f}), "
          f"lon = ({lon0:.4f}, {lon1:.4f})")

    return lat0, lat1, lon0, lon1


if __name__ == "__main__":
    store = "gs://leap-persistent/data-library/CanopyHeights-GLAD/CanopyHeights-GLAD.zarr"

    # Open once, with consolidated metadata and single‑tile chunks
    ds_zarr = (
        xr.open_zarr(store, consolidated=True)
          .chunk({"tile_id": 1})
    )
    print("ds_zarr.dims",ds_zarr.dims)
    stitch_and_plot_tiles(ds_zarr, N=0, M=2509, coarsen_factor=100)
   


