"""Entry helpers. Data download: scripts/download_era5_0p5.py (needs proxyon)."""

# Example after download finishes:
#   import xarray as xr
#   train = xr.open_zarr("data/era5_28ch_0p5_6h.zarr/train.zarr")
#   print(train.fields.shape)  # (512, 28, 360, 720)

if __name__ == "__main__":
    print("Use: python scripts/download_era5_0p5.py --train-n 512")
    print("Log:  logs/download_0p5_n512.log")
