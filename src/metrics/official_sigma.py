"""Official per-channel σ for NRMSE (LadCast + train-pool gaps)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.era5 import CHANNEL_ORDER

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIGMA_PATH = ROOT / "data" / "ref_stats" / "sigma_official_28ch.npz"
DEFAULT_LADCAST_JSON = ROOT / "data" / "ref_stats" / "ERA5_normal_1979_2017.json"

# Our channel → (LadCast key, optional pressure level)
_LADCAST_MAP: dict[str, tuple[str, int | None]] = {
    "t2m": ("2m_temperature", None),
    "mslp": ("mean_sea_level_pressure", None),
    "u10": ("10m_u_component_of_wind", None),
    "v10": ("10m_v_component_of_wind", None),
    "tp6h": ("total_precipitation_6hr", None),
    "sst": ("sea_surface_temperature", None),
    "T1000": ("temperature", 1000),
    "T925": ("temperature", 925),
    "T850": ("temperature", 850),
    "T700": ("temperature", 700),
    "U1000": ("u_component_of_wind", 1000),
    "U925": ("u_component_of_wind", 925),
    "U850": ("u_component_of_wind", 850),
    "U700": ("u_component_of_wind", 700),
    "V1000": ("v_component_of_wind", 1000),
    "V925": ("v_component_of_wind", 925),
    "V850": ("v_component_of_wind", 850),
    "V700": ("v_component_of_wind", 700),
    "Z1000": ("geopotential", 1000),
    "Z925": ("geopotential", 925),
    "Z850": ("geopotential", 850),
    "Z700": ("geopotential", 700),
    "Q1000": ("specific_humidity", 1000),
    "Q925": ("specific_humidity", 925),
    "Q850": ("specific_humidity", 850),
    "Q700": ("specific_humidity", 700),
}

# Not present in LadCast static JSON — fill from train pool
_TRAIN_POOL_ONLY = ("tcwv", "tcc")


def _pick_level(stat: dict | float, level: int | None) -> float:
    if level is None:
        if isinstance(stat, dict):
            raise ValueError("expected scalar mean/std for surface variable")
        return float(stat)
    if not isinstance(stat, dict):
        raise ValueError(f"expected per-level dict for level={level}")
    if level in stat:
        return float(stat[level])
    if str(level) in stat:
        return float(stat[str(level)])
    raise KeyError(f"level {level} missing in LadCast stats")


def load_ladcast_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def ladcast_mean_std(ladcast: dict, channel: str) -> tuple[float, float]:
    """Return (mean, std) in physical units for a hackathon channel from LadCast JSON."""
    if channel not in _LADCAST_MAP:
        raise KeyError(f"{channel} is not mapped from LadCast (use train_pool)")
    key, level = _LADCAST_MAP[channel]
    if key not in ladcast:
        raise KeyError(f"LadCast JSON missing key {key}")
    entry = ladcast[key]
    return _pick_level(entry["mean"], level), _pick_level(entry["std"], level)


def lat_weights(n_lat: int) -> np.ndarray:
    """Cosφ weights for equiangular 0.5° grid approx (lat from +90→−90 exclusive poles)."""
    # Match Era5ZarrDataset: cos(lat) with lat centers −89.75…89.75
    lat = np.linspace(89.75, -89.75, n_lat, dtype=np.float64)
    w = np.cos(np.deg2rad(lat))
    w = np.maximum(w, 0.0)
    return w / w.mean()


def train_pool_mean_std(
    train_zarr: Path | str,
    channel_indices: list[int],
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Latitude-weighted mean/std over train.zarr for selected channel indices."""
    import xarray as xr

    path = Path(train_zarr)
    ds = xr.open_zarr(path)
    fields = ds["fields"]  # (time, channel, lat, lon)
    n_time = int(fields.sizes["time"])
    n_use = n_time if max_frames is None else min(n_time, max_frames)
    n_lat = int(fields.sizes["lat"])
    w = lat_weights(n_lat).reshape(1, n_lat, 1)  # broadcast over time-slice, lon

    means = np.zeros(len(channel_indices), dtype=np.float64)
    stds = np.zeros(len(channel_indices), dtype=np.float64)

    for j, ci in enumerate(channel_indices):
        # accumulate in chunks to limit RAM
        acc = 0.0
        acc2 = 0.0
        wsum = 0.0
        chunk = 32
        for t0 in range(0, n_use, chunk):
            t1 = min(n_use, t0 + chunk)
            x = fields.isel(time=slice(t0, t1), channel=ci).values.astype(np.float64)
            # x: (T, H, W)
            finite = np.isfinite(x)
            ww = np.broadcast_to(w, x.shape).copy()
            ww = np.where(finite, ww, 0.0)
            x0 = np.where(finite, x, 0.0)
            acc += float((x0 * ww).sum())
            acc2 += float((x0 * x0 * ww).sum())
            wsum += float(ww.sum())
        mean = acc / max(wsum, 1e-12)
        var = acc2 / max(wsum, 1e-12) - mean * mean
        means[j] = mean
        stds[j] = float(np.sqrt(max(var, 1e-12)))
    return means.astype(np.float32), stds.astype(np.float32)


def build_official_sigma(
    ladcast_json: Path | str,
    train_zarr: Path | str | None = None,
    max_frames: int | None = None,
) -> dict:
    """Build 28-channel mean/std/source arrays."""
    ladcast = load_ladcast_json(ladcast_json)
    mean = np.zeros(28, dtype=np.float32)
    std = np.zeros(28, dtype=np.float32)
    source = np.array(["ladcast"] * 28, dtype=object)

    gap_idx: list[int] = []
    for i, ch in enumerate(CHANNEL_ORDER):
        if ch in _TRAIN_POOL_ONLY:
            gap_idx.append(i)
            source[i] = "train_pool"
            continue
        m, s = ladcast_mean_std(ladcast, ch)
        mean[i] = m
        std[i] = max(float(s), 1e-8)

    if gap_idx:
        if train_zarr is None:
            raise ValueError(f"Need --train-zarr to fill channels {[CHANNEL_ORDER[i] for i in gap_idx]}")
        m_gap, s_gap = train_pool_mean_std(train_zarr, gap_idx, max_frames=max_frames)
        for j, i in enumerate(gap_idx):
            mean[i] = m_gap[j]
            std[i] = max(float(s_gap[j]), 1e-8)

    return {
        "mean": mean,
        "std": std,
        "source": source,
        "channels": np.array(CHANNEL_ORDER),
    }


def save_official_sigma(payload: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=payload["mean"],
        std=payload["std"],
        source=payload["source"],
        channels=payload["channels"],
    )


def load_official_sigma(path: Path | str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, std, source) arrays of shape (28,)."""
    p = Path(path) if path else DEFAULT_SIGMA_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Missing official sigma at {p}. Run scripts/metrics/build_official_sigma.py first."
        )
    z = np.load(p, allow_pickle=True)
    mean = np.asarray(z["mean"], dtype=np.float64).reshape(28)
    std = np.asarray(z["std"], dtype=np.float64).reshape(28)
    std = np.maximum(std, 1e-8)
    source = np.asarray(z["source"], dtype=object).reshape(28)
    ch = list(z["channels"])
    if list(ch) != list(CHANNEL_ORDER):
        raise ValueError(f"sigma channel order mismatch: {ch} vs {CHANNEL_ORDER}")
    return mean, std, source
