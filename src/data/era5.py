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


class Era5ZarrDataset(Dataset):
    """Lazy frame loader: fields (time, channel, lat, lon)."""

    def __init__(
        self,
        zarr_path: str | Path,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        crop_size: int | None = None,
        augment: bool = False,
        seed: int = 0,
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

        lat = self.ds["lat"].values.astype(np.float32)
        self.lat_weights = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32)
        self.lat_weights = self.lat_weights / self.lat_weights.mean()

    def _estimate_stats(self, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.linspace(0, self.n - 1, num=max_frames, dtype=int)
        # nan-aware: SST is NaN over land; poles/interp can add sparse NaNs
        acc = np.zeros(28, dtype=np.float64)
        acc2 = np.zeros(28, dtype=np.float64)
        count = np.zeros(28, dtype=np.float64)
        for i in idx:
            x = self.fields.isel(time=int(i)).values.astype(np.float64)  # (C,H,W)
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
        c, h, w = x.shape
        if self.crop_size is not None:
            cs = self.crop_size
            if h < cs or w < cs:
                raise ValueError(f"crop_size {cs} > frame {h}x{w}")
            top = int(self.rng.integers(0, h - cs + 1)) if self.augment else (h - cs) // 2
            left = int(self.rng.integers(0, w - cs + 1)) if self.augment else (w - cs) // 2
            x = x[:, top : top + cs, left : left + cs]
            w_lat = self.lat_weights[top : top + cs]
        else:
            w_lat = self.lat_weights

        if self.augment and self.rng.random() < 0.5:
            x = torch.flip(x, dims=[-1])  # lon flip

        # NaN → channel mean before norm ⇒ 0 after norm (TZ: SST land = 0)
        x = torch.where(torch.isfinite(x), x, self.mean.expand_as(x))
        x_n = (x - self.mean) / self.std
        return {
            "x": x_n,
            "lat_weight": w_lat,  # (H,)
            "index": torch.tensor(i, dtype=torch.int64),
        }


def save_stats(mean: np.ndarray, std: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, std=std, channels=np.array(CHANNEL_ORDER))


def load_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return z["mean"], z["std"]
