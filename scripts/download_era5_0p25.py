
"""Download seasonally-balanced ERA5 28ch subset at native 0.25°, save local Zarr.

Years (TZ): train 2014–2019, val 2020, test 2021.
Grid: 721 × 1440 (no remapping). Downloads train / val / test / static.

Default: both N=64 and N=128 (val/test = N/2), matching existing 0.5° subsets.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time as time_mod
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# GCS from RU typically needs local proxy (same as 0.5° script).
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

NLAT, NLON = 721, 1440
CHUNKS = (1, 28, NLAT, NLON)

TIME_ENCODING = {
    "time": {
        "units": "hours since 1970-01-01T00:00:00",
        "dtype": "float64",
        "calendar": "proleptic_gregorian",
    },
}

LOG = logging.getLogger("download_era5_0p25")


def setup_logging(log_path: Path | None) -> Path:
    """Console + file logging. Returns path to the log file."""
    root = Path("logs")
    root.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = root / f"download_era5_0p25_{stamp}.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.handlers.clear()
    LOG.setLevel(logging.INFO)
    LOG.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    LOG.addHandler(fh)
    LOG.addHandler(sh)
    return log_path


def open_wb2() -> xr.Dataset:
    LOG.info("Opening WeatherBench2 Zarr (anon GCS): %s", WB2_URL)
    fs = gcsfs.GCSFileSystem(token="anon", session_kwargs={"trust_env": True})
    ds = xr.open_zarr(fs.get_mapper(WB2_URL), consolidated=True)
    LOG.info(
        "Opened WB2: time=%s lat=%s lon=%s n_vars=%d",
        ds.sizes.get("time"),
        ds.sizes.get("latitude") or ds.sizes.get("lat"),
        ds.sizes.get("longitude") or ds.sizes.get("lon"),
        len(ds.data_vars),
    )
    return ds


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
        take = per + (1 if s < rem else 0)
        if len(by_season[s]) < take:
            raise RuntimeError(
                f"Season {s}: need {take} times, have {len(by_season[s])}"
            )
        chosen.extend(by_season[s][:take])
    out = np.array(sorted(chosen), dtype="datetime64[ns]")
    if len(out) < n:
        raise RuntimeError(f"Need {n} times, got {len(out)}")
    return out[:n]


def rename_latlon(da: xr.DataArray) -> xr.DataArray:
    rename: dict[str, str] = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    return da.rename(rename) if rename else da


def stack_frame(ds: xr.Dataset, t) -> xr.DataArray:
    """Fetch one timestep → (time=1, channel=28, lat=721, lon=1440)."""
    surf_names = list(SURFACE.values())
    pres_names = list(PRESSURE_VARS.values())
    last_err: Exception | None = None
    surf = pres = None
    for attempt in range(1, 8):
        try:
            surf = ds[surf_names].sel(time=t).load()
            pres = ds[pres_names].sel(time=t, level=LEVELS).load()
            break
        except Exception as e:  # noqa: BLE001 — retry flaky GCS
            last_err = e
            wait = min(60, 2**attempt)
            LOG.warning(
                "retry %d/7 after %s: sleep %ds (time=%s)",
                attempt,
                type(e).__name__,
                wait,
                np.datetime_as_string(np.datetime64(t), unit="h"),
            )
            time_mod.sleep(wait)
    else:
        raise RuntimeError(f"Failed to fetch time={t}") from last_err

    assert surf is not None and pres is not None
    slices: list[xr.DataArray] = []
    for name in surf_names:
        da = rename_latlon(surf[name]).astype(np.float32)
        slices.append(da.reset_coords(drop=True))
    for name in pres_names:
        for lev in LEVELS:
            da = rename_latlon(pres[name].sel(level=lev)).astype(np.float32)
            slices.append(da.reset_coords(drop=True))

    # Avoid pd.Index(...)/StringDtype — breaks on newer pandas+xarray.
    ch = xr.concat(slices, dim="channel")
    ch = ch.assign_coords(channel=("channel", np.array(CHANNEL_ORDER, dtype="U16")))
    if ch.sizes.get("lat") != NLAT or ch.sizes.get("lon") != NLON:
        raise RuntimeError(
            f"Unexpected spatial size {dict(ch.sizes)}, expected lat={NLAT} lon={NLON}"
        )
    return ch.expand_dims(time=[np.datetime64(t, "ns")]).rename("fields")


def download_split(
    ds: xr.Dataset,
    times: np.ndarray,
    out_path: Path,
    split: str,
) -> None:
    if out_path.exists():
        raise SystemExit(f"Exists: {out_path}")

    import dask.array as da

    n = len(times)
    lat = ds["latitude"].values.astype(np.float64) if "latitude" in ds.coords else ds["lat"].values
    lon = ds["longitude"].values.astype(np.float64) if "longitude" in ds.coords else ds["lon"].values
    if len(lat) != NLAT or len(lon) != NLON:
        raise RuntimeError(f"Bad grid: lat={len(lat)} lon={len(lon)}")

    LOG.info("[%s] creating template %s  n=%d  shape=(%d,28,%d,%d)", split, out_path, n, n, NLAT, NLON)
    template = xr.Dataset(
        {
            "fields": (
                ("time", "channel", "lat", "lon"),
                da.zeros((n, 28, NLAT, NLON), chunks=CHUNKS, dtype=np.float32),
            )
        },
        coords={
            "time": ("time", times.astype("datetime64[ns]")),
            "channel": ("channel", np.array(CHANNEL_ORDER, dtype="U8")),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
    )
    template.to_zarr(
        out_path,
        mode="w",
        compute=False,
        encoding={
            **TIME_ENCODING,
            "fields": {"chunks": list(CHUNKS), "dtype": "float32"},
        },
        consolidated=True,
    )

    t0 = time_mod.time()
    for i, t in enumerate(times):
        ts = np.datetime_as_string(t, unit="h")
        LOG.info("[%s] %d/%d %s", split, i + 1, n, ts)
        frame = stack_frame(ds, t).to_dataset()
        frame = frame.drop_vars([c for c in ("channel", "lat", "lon") if c in frame.coords])
        frame.to_zarr(out_path, region={"time": slice(i, i + 1)})
        if (i + 1) % 8 == 0 or i + 1 == n:
            elapsed = time_mod.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (n - i - 1) / max(rate, 1e-6)
            LOG.info(
                "[%s] progress %d/%d  %.2f frame/s  elapsed=%.0fs  eta=%.0fs",
                split,
                i + 1,
                n,
                rate,
                elapsed,
                eta,
            )

    opened = xr.open_zarr(out_path)
    assert opened.fields.shape == (n, 28, NLAT, NLON)
    assert np.array_equal(opened.time.values, times.astype("datetime64[ns]"))
    opened.close()
    LOG.info("[%s] wrote %s shape=%dx28x%dx%d", split, out_path, n, NLAT, NLON)


def download_static(ds: xr.Dataset, out_path: Path) -> None:
    if out_path.exists():
        raise SystemExit(f"Exists: {out_path}")
    LOG.info("Downloading static → %s", out_path)
    static_vars: dict[str, xr.DataArray] = {}
    for key in ("land_sea_mask", "geopotential_at_surface"):
        if key not in ds:
            LOG.warning("static var missing in WB2: %s", key)
            continue
        da = rename_latlon(ds[key].load()).astype(np.float32)
        static_vars[key] = da
        LOG.info("  %s shape=%s", key, tuple(da.shape))
    if not static_vars:
        raise RuntimeError("No static variables found in WB2")
    xr.Dataset(static_vars).to_zarr(out_path, mode="w", consolidated=True)
    LOG.info("static wrote %s", out_path)


def run_one(
    ds: xr.Dataset,
    *,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
    out: Path,
) -> None:
    LOG.info("=" * 60)
    LOG.info(
        "Dataset out=%s  train_n=%d val_n=%d test_n=%d seed=%d",
        out,
        train_n,
        val_n,
        test_n,
        seed,
    )
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing {out}")

    times = ds["time"].values
    train_t = pick_balanced(
        times[(times >= np.datetime64("2014-01-01")) & (times < np.datetime64("2020-01-01"))],
        train_n,
        seed,
    )
    val_t = pick_balanced(
        times[(times >= np.datetime64("2020-01-01")) & (times < np.datetime64("2021-01-01"))],
        val_n,
        seed + 1,
    )
    test_t = pick_balanced(
        times[(times >= np.datetime64("2021-01-01")) & (times < np.datetime64("2022-01-01"))],
        test_n,
        seed + 2,
    )
    LOG.info(
        "Picked times: train=%d (%s … %s)  val=%d  test=%d",
        len(train_t),
        np.datetime_as_string(train_t[0], unit="h"),
        np.datetime_as_string(train_t[-1], unit="h"),
        len(val_t),
        len(test_t),
    )

    out.mkdir(parents=True)
    meta = {
        "grid": "0.25deg",
        "shape": [28, NLAT, NLON],
        "channels": CHANNEL_ORDER,
        "years": {"train": "2014-2019", "val": "2020", "test": "2021"},
        "n": {"train": train_n, "val": val_n, "test": test_n},
        "seed": seed,
        "remap": "none (native WB2 0.25°)",
        "source": WB2_URL,
        "times": {
            "train": [np.datetime_as_string(t) for t in train_t],
            "val": [np.datetime_as_string(t) for t in val_t],
            "test": [np.datetime_as_string(t) for t in test_t],
        },
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    LOG.info("Wrote manifest %s", out / "manifest.json")

    download_split(ds, train_t, out / "train.zarr", "train")
    download_split(ds, val_t, out / "val.zarr", "val")
    download_split(ds, test_t, out / "test.zarr", "test")
    download_static(ds, out / "static.zarr")

    for split in ("train", "val", "test"):
        z = xr.open_zarr(out / f"{split}.zarr")
        LOG.info(
            "verify %s: shape=%s  time0=%s  timeN=%s",
            split,
            z.fields.shape,
            z.time.values[0],
            z.time.values[-1],
        )
        z.close()
    LOG.info("Done: %s", out)


def default_val_test(train_n: int) -> tuple[int, int]:
    """Match existing 0.5° subsets: N=64→32/32, N=128→64/64, else max(32, N//2)."""
    if train_n == 64:
        return 32, 32
    if train_n == 128:
        return 64, 64
    half = max(32, train_n // 2)
    return half, half


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download ERA5 28ch at 0.25° (train/val/test/static)."
    )
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[64, 128],
        help="Train sizes to download (default: 64 128). val/test = N/2 convention.",
    )
    p.add_argument("--train-n", type=int, default=None, help="Override: single train N.")
    p.add_argument("--val-n", type=int, default=None)
    p.add_argument("--test-n", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-template",
        type=str,
        default="data/era5_28ch_0p25_6h_n{n}.zarr",
        help="Output path template; {n} replaced by train size.",
    )
    p.add_argument("--log", type=Path, default=None, help="Log file path (default: logs/...).")
    args = p.parse_args()

    log_path = setup_logging(args.log)
    LOG.info("Log file: %s", log_path.resolve())
    LOG.info("Args: %s", vars(args))

    if args.train_n is not None:
        jobs = [(args.train_n, args.val_n, args.test_n)]
    else:
        jobs = [(n, None, None) for n in args.sizes]

    t_all = time_mod.time()
    ds = open_wb2()

    for train_n, val_n, test_n in jobs:
        if val_n is None or test_n is None:
            vn, tn = default_val_test(train_n)
            val_n = val_n if val_n is not None else vn
            test_n = test_n if test_n is not None else tn
        out = Path(args.out_template.format(n=train_n))
        run_one(
            ds,
            train_n=train_n,
            val_n=val_n,
            test_n=test_n,
            seed=args.seed,
            out=out,
        )

    LOG.info("All jobs finished in %.0fs", time_mod.time() - t_all)


if __name__ == "__main__":
    main()
