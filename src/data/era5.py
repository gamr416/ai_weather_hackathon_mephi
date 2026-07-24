"""ERA5 28-channel 0.5° dataset from local zarr produced by scripts/download_era5_0p5.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

CHANNEL_ORDER = [
    "t2m", "mslp", "u10", "v10", "tp6h", "sst", "tcwv", "tcc",
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Z1000", "Z925", "Z850", "Z700",
    "Q1000", "Q925", "Q850", "Q700",
]

# Default relative channel weights for reconstruction loss
DEFAULT_CHANNEL_WEIGHTS = {
    "t2m": 1.0,
    "mslp": 0.5,
    "u10": 1.5,
    "v10": 1.5,
    "tp6h": 5.0,
    "sst": 1.0,
    "tcwv": 1.5,
    "tcc": 3.0,
    "T1000": 1.0,
    "T925": 1.0,
    "T850": 1.0,
    "T700": 1.0,
    "U1000": 1.5,
    "U925": 1.5,
    "U850": 1.5,
    "U700": 1.5,
    "V1000": 1.5,
    "V925": 1.5,
    "V850": 1.5,
    "V700": 1.5,
    "Z1000": 0.5,
    "Z925": 0.5,
    "Z850": 0.5,
    "Z700": 0.5,
    "Q1000": 3.0,
    "Q925": 3.0,
    "Q850": 3.0,
    "Q700": 3.0,
}


def channel_weight_tensor(weights: dict[str, float] | None = None) -> torch.Tensor:
    w = weights or DEFAULT_CHANNEL_WEIGHTS
    return torch.tensor([float(w.get(c, 1.0)) for c in CHANNEL_ORDER], dtype=torch.float32)


class Era5ZarrDataset(Dataset):
    """Lazy frame loader: fields (time, channel, lat, lon) + optional static conditioning."""

    def __init__(
        self,
        zarr_path: str | Path,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        crop_size: int | None = None,
        augment: bool = False,
        seed: int = 0,
        static_path: str | Path | None = None,
    ) -> None:
        self.path = Path(zarr_path)
        try:
            self.ds = xr.open_zarr(self.path, consolidated=True)
        except Exception:
            self.ds = xr.open_zarr(self.path, consolidated=False)
        self.fields = self.ds["fields"]  # lazy
        self.n = int(self.fields.sizes["time"])
        if self.n < 1:
            raise RuntimeError(f"Empty dataset: {self.path}")
        self.crop_size = crop_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        if mean is None or std is None:
            mean, std = self._estimate_stats(max_frames=min(64, self.n))
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(28, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(28, 1, 1).clamp_min(1e-4)

        lat = self.ds["lat"].values.astype(np.float64)
        lon = self.ds["lon"].values.astype(np.float64)
        self.lat_weights = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32)
        self.lat_weights = self.lat_weights / self.lat_weights.mean()

        self.static: torch.Tensor | None = None  # (4, H, W): lsm, orog_norm, sinφ, cosφ
        sp = Path(static_path) if static_path is not None else self.path.parent / "static.zarr"
        self._load_static(sp, lat, lon)

    def _load_static(self, static_path: Path, lat: np.ndarray, lon: np.ndarray) -> None:
        h, w = len(lat), len(lon)
        phi = np.deg2rad(lat)[:, None]
        sin_phi = np.broadcast_to(np.sin(phi), (h, w)).astype(np.float32)
        cos_phi = np.broadcast_to(np.cos(phi), (h, w)).astype(np.float32)

        lsm = np.zeros((h, w), dtype=np.float32)
        orog = np.zeros((h, w), dtype=np.float32)
        if static_path.exists():
            try:
                st = xr.open_zarr(static_path, consolidated=True)
            except Exception:
                st = xr.open_zarr(static_path, consolidated=False)
            if "land_sea_mask" in st:
                lsm = np.nan_to_num(st["land_sea_mask"].values.astype(np.float32), nan=0.0)
            key = "geopotential_at_surface" if "geopotential_at_surface" in st else None
            if key is not None:
                orog = np.nan_to_num(st[key].values.astype(np.float32), nan=0.0)
                # normalize orography to ~unit scale
                std = float(orog.std()) or 1.0
                orog = (orog - float(orog.mean())) / std

        self.static = torch.from_numpy(
            np.stack([lsm, orog, sin_phi, cos_phi], axis=0)
        )  # 4,H,W

    def _estimate_stats(self, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.linspace(0, self.n - 1, num=max_frames, dtype=int)
        acc = np.zeros(28, dtype=np.float64)
        acc2 = np.zeros(28, dtype=np.float64)
        count = np.zeros(28, dtype=np.float64)
        for i in idx:
            x = self.fields.isel(time=int(i)).values.astype(np.float64)
            flat = x.reshape(28, -1)
            valid = np.isfinite(flat)
            flat0 = np.where(valid, flat, 0.0)
            acc += flat0.sum(axis=1)
            acc2 += (flat0**2).sum(axis=1)
            count += valid.sum(axis=1)
        count = np.maximum(count, 1.0)
        mean = acc / count
        var = acc2 / count - mean**2
        std = np.sqrt(np.maximum(var, 1e-8))
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise RuntimeError("Failed to estimate finite mean/std — check data for all-NaN channels")
        return mean.astype(np.float32), std.astype(np.float32)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        x = torch.from_numpy(self.fields.isel(time=i).values.astype(np.float32))  # C,H,W
        static = self.static
        assert static is not None
        _, h, w = x.shape

        if self.crop_size is not None:
            cs = self.crop_size
            if h < cs or w < cs:
                raise ValueError(f"crop_size {cs} > frame {h}x{w}")
            top = int(self.rng.integers(0, h - cs + 1)) if self.augment else (h - cs) // 2
            left = int(self.rng.integers(0, w - cs + 1)) if self.augment else (w - cs) // 2
            x = x[:, top : top + cs, left : left + cs]
            static = static[:, top : top + cs, left : left + cs]
            w_lat = self.lat_weights[top : top + cs]
        else:
            w_lat = self.lat_weights

        if self.augment and self.rng.random() < 0.5:
            x = torch.flip(x, dims=[-1])
            static = torch.flip(static, dims=[-1])

        x = torch.where(torch.isfinite(x), x, self.mean.expand_as(x))
        x_n = (x - self.mean) / self.std
        return {
            "x": x_n,
            "static": static,
            "lat_weight": w_lat,
            "index": torch.tensor(i, dtype=torch.int64),
        }


def save_stats(mean: np.ndarray, std: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, std=std, channels=np.array(CHANNEL_ORDER))


def load_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return z["mean"], z["std"]
