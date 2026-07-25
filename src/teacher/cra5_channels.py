"""Channel / grid mapping between our 28ch ERA5 and CRA5 VAEformer (268 vars)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.era5 import CHANNEL_ORDER

# CRA5 268v layout (cra5_268v_config.py)
CRA5_PRESSURE_VARS = ["z", "q", "u", "v", "t", "r", "w"]
CRA5_SINGLE_VARS = ["v10", "u10", "v100", "u100", "t2m", "tcc", "sp", "tp", "msl"]
CRA5_LEVELS = [
    1000.0, 975.0, 950.0, 925.0, 900.0, 875.0, 850.0, 825.0, 800.0,
    775.0, 750.0, 700.0, 650.0, 600.0, 550.0, 500.0, 450.0, 400.0,
    350.0, 300.0, 250.0, 225.0, 200.0, 175.0, 150.0, 125.0, 100.0,
    70.0, 50.0, 30.0, 20.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0,
]
CRA5_H, CRA5_W = 721, 1440
OUR_H, OUR_W = 360, 720

# Our name → CRA5 name (pressure: prefix_level)
# tp6h (m/6h) → CRA5 tp stored as mm (×1000) in their loader
_OUR_TO_CRA5: dict[str, str] = {
    "t2m": "t2m",
    "mslp": "msl",
    "u10": "u10",
    "v10": "v10",
    "tp6h": "tp",
    "tcc": "tcc",
    "T1000": "t_1000",
    "T925": "t_925",
    "T850": "t_850",
    "T700": "t_700",
    "U1000": "u_1000",
    "U925": "u_925",
    "U850": "u_850",
    "U700": "u_700",
    "V1000": "v_1000",
    "V925": "v_925",
    "V850": "v_850",
    "V700": "v_700",
    "Z1000": "z_1000",
    "Z925": "z_925",
    "Z850": "z_850",
    "Z700": "z_700",
    "Q1000": "q_1000",
    "Q925": "q_925",
    "Q850": "q_850",
    "Q700": "q_700",
}

# No CRA5 teacher: sst, tcwv
NO_TEACHER_CHANNELS = ("sst", "tcwv")


def cra5_channel_names() -> list[str]:
    names: list[str] = []
    for v in CRA5_PRESSURE_VARS:
        for level in CRA5_LEVELS:
            names.append(f"{v}_{int(level)}")
    names.extend(CRA5_SINGLE_VARS)
    assert len(names) == 268
    return names


def build_overlap_maps() -> tuple[list[int], list[int], list[str]]:
    """Return (our_indices, cra5_indices, names) for overlapping channels."""
    cra5_names = cra5_channel_names()
    name_to_cra5 = {n: i for i, n in enumerate(cra5_names)}
    our_idx: list[int] = []
    cra5_idx: list[int] = []
    names: list[str] = []
    for i, ch in enumerate(CHANNEL_ORDER):
        cra5_name = _OUR_TO_CRA5.get(ch)
        if cra5_name is None:
            continue
        if cra5_name not in name_to_cra5:
            raise KeyError(f"CRA5 missing mapped channel {cra5_name} for {ch}")
        our_idx.append(i)
        cra5_idx.append(name_to_cra5[cra5_name])
        names.append(ch)
    return our_idx, cra5_idx, names


def load_cra5_mean_std(api_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """CRA5 per-channel mean/std for 268 variables (physical units)."""
    if api_dir is None:
        api_dir = Path(__file__).resolve().parents[2] / "third_party" / "CRA5" / "cra5" / "api"
    with (api_dir / "mean_std.json").open() as f:
        mean_std = json.load(f)
    with (api_dir / "mean_std_single.json").open() as f:
        mean_std_single = json.load(f)

    level_mapping = list(range(len(CRA5_LEVELS)))  # all levels
    mean_list: list[float] = []
    std_list: list[float] = []
    for vname in CRA5_PRESSURE_VARS:
        mean_list += [mean_std["mean"][vname][idx] for idx in level_mapping]
        std_list += [mean_std["std"][vname][idx] for idx in level_mapping]
    for vname in CRA5_SINGLE_VARS:
        mean_list.append(mean_std_single["mean"][vname])
        std_list.append(mean_std_single["std"][vname])
    return np.asarray(mean_list, dtype=np.float32), np.asarray(std_list, dtype=np.float32)


def upsample_to_cra5(x: torch.Tensor) -> torch.Tensor:
    """(B,C,360,720) → (B,C,721,1440) bilinear."""
    return F.interpolate(x, size=(CRA5_H, CRA5_W), mode="bilinear", align_corners=False)


def downsample_to_ours(x: torch.Tensor) -> torch.Tensor:
    """(B,C,721,1440) → (B,C,360,720)."""
    return F.interpolate(x, size=(OUR_H, OUR_W), mode="bilinear", align_corners=False)


def pack_to_cra5_physical(
    x_phys_28: torch.Tensor,
    our_idx: list[int],
    cra5_idx: list[int],
    cra5_mean: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    """
    Pack our physical 28ch frame(s) into CRA5 268ch tensor.
    Missing CRA5 channels are filled with CRA5 climatological mean (→ ~0 after norm),
    not physical zeros (those become huge anomalies and blow up VAEformer).
    tp6h (m) → CRA5 tp (mm) via ×1000.
    """
    b = x_phys_28.shape[0]
    x_up = upsample_to_cra5(x_phys_28)
    if cra5_mean is None:
        mean_np, _ = load_cra5_mean_std()
        cra5_mean = mean_np
    if isinstance(cra5_mean, np.ndarray):
        mean_t = torch.as_tensor(cra5_mean, device=x_up.device, dtype=x_up.dtype).view(1, 268, 1, 1)
    else:
        mean_t = cra5_mean.to(device=x_up.device, dtype=x_up.dtype).view(1, 268, 1, 1)
    out = mean_t.expand(b, -1, CRA5_H, CRA5_W).clone()
    for oi, ci in zip(our_idx, cra5_idx):
        ch = x_up[:, oi]
        # replace non-finite with mean for that channel
        ch = torch.where(torch.isfinite(ch), ch, mean_t[:, ci])
        if CHANNEL_ORDER[oi] == "tp6h":
            ch = ch * 1000.0  # m → mm
        out[:, ci] = ch
    return out


def unpack_from_cra5_physical(
    x_cra5: torch.Tensor,
    our_idx: list[int],
    cra5_idx: list[int],
) -> torch.Tensor:
    """Extract overlapping channels from CRA5 physical 268ch → our 28ch grid (zeros elsewhere)."""
    b = x_cra5.shape[0]
    overlap = x_cra5.new_zeros(b, len(our_idx), CRA5_H, CRA5_W)
    for j, (oi, ci) in enumerate(zip(our_idx, cra5_idx)):
        ch = x_cra5[:, ci]
        if CHANNEL_ORDER[oi] == "tp6h":
            ch = ch / 1000.0  # mm → m
        overlap[:, j] = ch
    overlap = downsample_to_ours(overlap)
    out = x_cra5.new_zeros(b, 28, OUR_H, OUR_W)
    for j, oi in enumerate(our_idx):
        out[:, oi] = overlap[:, j]
    return out
