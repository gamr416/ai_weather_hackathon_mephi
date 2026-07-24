#!/usr/bin/env python3
"""Download seasonally-balanced ERA5 28ch subset, remap 0.25° → 0.5°, save local Zarr.

Years (TZ): train 2014–2019, val 2020, test 2021.
Requires HTTP(S)_PROXY to GCS (e.g. proxyon → 127.0.0.1:12334).
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
    os.environ.setdefault(k, "http://127.0.0.1:12334")

import gcsfs

warnings.filterwarnings("ignore", category=UserWarning)

WB2_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)

SURFACE = {
    "t2m": "2m_temperature",
    "mslp": "mean_sea_level_pressure",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "tp6h": "total_precipitation_6hr",
    "sst": "sea_surface_temperature",
    "tcwv": "total_column_water_vapour",
    "tcc": "total_cloud_cover",
}
PRESSURE_VARS = {
    "T": "temperature",
    "U": "u_component_of_wind",
    "V": "v_component_of_wind",
    "Z": "geopotential",
    "Q": "specific_humidity",
}
LEVELS = [1000, 925, 850, 700]
CHANNEL_ORDER = [
    "t2m", "mslp", "u10", "v10", "tp6h", "sst", "tcwv", "tcc",
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Z1000", "Z925", "Z850", "Z700",
    "Q1000", "Q925", "Q850", "Q700",
]

TARGET_LAT = np.arange(-89.75, 90.0, 0.5, dtype=np.float64)
TARGET_LON = np.arange(0.0, 360.0, 0.5, dtype=np.float64)

TIME_ENCODING = {
    "time": {"units": "hours since 1970-01-01T00:00:00", "dtype": "float64", "calendar": "proleptic_gregorian"},
}


def open_wb2() -> xr.Dataset:
    fs = gcsfs.GCSFileSystem(token="anon", session_kwargs={"trust_env": True})
    return xr.open_zarr(fs.get_mapper(WB2_URL), consolidated=True)


def season_id(ts: pd.Timestamp) -> int:
    m = ts.month
    if m in (12, 1, 2):
        return 0
    if m in (3, 4, 5):
        return 1
    if m in (6, 7, 8):
        return 2
    return 3


def pick_balanced(times: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = pd.to_datetime(times)
    by_season: list[list[np.datetime64]] = [[], [], [], []]
    for ti in t:
        by_season[season_id(ti)].append(np.datetime64(ti.to_datetime64()))
    for s in range(4):
        rng.shuffle(by_season[s])
    per, rem = divmod(n, 4)
    chosen: list[np.datetime64] = []
    for s in range(4):
        chosen.extend(by_season[s][: per + (1 if s < rem else 0)])
    out = np.array(sorted(chosen), dtype="datetime64[ns]")
    if len(out) < n:
        raise RuntimeError(f"Need {n} times, got {len(out)}")
    return out[:n]


def remap_0p5(da: xr.DataArray) -> xr.DataArray:
    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"
    out = da.coarsen({lon_name: 2}, boundary="trim").mean()
    out = out.interp({lat_name: TARGET_LAT, lon_name: TARGET_LON}, method="linear")
    return out.rename({lat_name: "lat", lon_name: "lon"})


def stack_frame(ds: xr.Dataset, t) -> xr.DataArray:
    """Fetch one timestep once, remap locally → (time=1, channel=28, lat, lon)."""
    import time as time_mod

    surf_names = list(SURFACE.values())
    pres_names = list(PRESSURE_VARS.values())
    last_err: Exception | None = None
    surf = pres = None
    for attempt in range(1, 8):
        try:
            surf = ds[surf_names].sel(time=t).load()
            pres = ds[pres_names].sel(time=t, level=LEVELS).load()
            break
        except Exception as e:  # noqa: BLE001 — retry flaky GCS/proxy
            last_err = e
            wait = min(60, 2 ** attempt)
            print(f"  retry {attempt}/7 after {type(e).__name__}: sleep {wait}s", flush=True)
            time_mod.sleep(wait)
    else:
        raise RuntimeError(f"Failed to fetch time={t}") from last_err

    assert surf is not None and pres is not None
    slices: list[xr.DataArray] = []
    for name in surf_names:
        da = remap_0p5(surf[name]).astype(np.float32)
        slices.append(da.reset_coords(drop=True))
    for name in pres_names:
        for lev in LEVELS:
            da = remap_0p5(pres[name].sel(level=lev)).astype(np.float32)
            slices.append(da.reset_coords(drop=True))

    ch = xr.concat(slices, dim=pd.Index(CHANNEL_ORDER, name="channel"))
    return ch.expand_dims(time=[np.datetime64(t, "ns")]).rename("fields")


def download_split(ds: xr.Dataset, times: np.ndarray, out_path: Path, split: str, batch: int) -> None:
    del batch
    if out_path.exists():
        raise SystemExit(f"Exists: {out_path}")

    import dask.array as da

    n = len(times)
    template = xr.Dataset(
        {
            "fields": (
                ("time", "channel", "lat", "lon"),
                da.zeros((n, 28, len(TARGET_LAT), len(TARGET_LON)), chunks=(1, 28, 360, 720), dtype=np.float32),
            )
        },
        coords={
            "time": ("time", times.astype("datetime64[ns]")),
            "channel": ("channel", np.array(CHANNEL_ORDER, dtype="U8")),
            "lat": ("lat", TARGET_LAT),
            "lon": ("lon", TARGET_LON),
        },
    )
    template.to_zarr(
        out_path,
        mode="w",
        compute=False,
        encoding={
            **TIME_ENCODING,
            "fields": {"chunks": (1, 28, 360, 720), "dtype": "float32"},
        },
        consolidated=True,
    )

    for i, t in enumerate(times):
        print(f"[{split}] {i+1}/{n} {np.datetime_as_string(t, unit='h')}", flush=True)
        frame = stack_frame(ds, t).to_dataset()
        # region write: only time-aligned data vars; drop non-region coords
        frame = frame.drop_vars([c for c in ("channel", "lat", "lon") if c in frame.coords])
        frame.to_zarr(out_path, region={"time": slice(i, i + 1)})

    opened = xr.open_zarr(out_path)
    assert opened.fields.shape == (n, 28, 360, 720)
    assert np.array_equal(opened.time.values, times.astype("datetime64[ns]"))
    opened.close()
    print(f"[{split}] wrote {out_path} shape={n}x28x360x720", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-n", type=int, default=512)
    p.add_argument("--val-n", type=int, default=128)
    p.add_argument("--test-n", type=int, default=128)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("data/era5_28ch_0p5_6h.zarr"))
    args = p.parse_args()

    print("Opening WB2…", flush=True)
    ds = open_wb2()
    times = ds["time"].values

    train_t = pick_balanced(
        times[(times >= np.datetime64("2014-01-01")) & (times < np.datetime64("2020-01-01"))],
        args.train_n,
        args.seed,
    )
    val_t = pick_balanced(
        times[(times >= np.datetime64("2020-01-01")) & (times < np.datetime64("2021-01-01"))],
        args.val_n,
        args.seed + 1,
    )
    test_t = pick_balanced(
        times[(times >= np.datetime64("2021-01-01")) & (times < np.datetime64("2022-01-01"))],
        args.test_n,
        args.seed + 2,
    )

    root = args.out
    if root.exists():
        raise SystemExit(f"Refusing to overwrite existing {root}")
    root.mkdir(parents=True)

    meta = {
        "grid": "0.5deg",
        "shape": [28, 360, 720],
        "channels": CHANNEL_ORDER,
        "years": {"train": "2014-2019", "val": "2020", "test": "2021"},
        "n": {"train": args.train_n, "val": args.val_n, "test": args.test_n},
        "seed": args.seed,
        "remap": "lon coarsen-2 mean + lat/lon linear interp (no xesmf)",
        "source": WB2_URL,
        "times": {
            "train": [np.datetime_as_string(t) for t in train_t],
            "val": [np.datetime_as_string(t) for t in val_t],
            "test": [np.datetime_as_string(t) for t in test_t],
        },
    }
    (root / "manifest.json").write_text(json.dumps(meta, indent=2))

    download_split(ds, train_t, root / "train.zarr", "train", args.batch)
    download_split(ds, val_t, root / "val.zarr", "val", args.batch)
    download_split(ds, test_t, root / "test.zarr", "test", args.batch)

    print("static…", flush=True)
    static_vars = {}
    for key in ("land_sea_mask", "geopotential_at_surface"):
        if key in ds:
            static_vars[key] = remap_0p5(ds[key].load()).astype(np.float32)
    if static_vars:
        xr.Dataset(static_vars).to_zarr(root / "static.zarr", mode="w", consolidated=True)

    # verify times
    tr = xr.open_zarr(root / "train.zarr")
    print("train times:", tr.time.values[0], "…", tr.time.values[-1], "n=", tr.sizes["time"])
    print("Done:", root)


if __name__ == "__main__":
    main()
