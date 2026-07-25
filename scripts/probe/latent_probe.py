"""Latent +6h probe on frozen Residual-FSQ encoder (≤2M params, ≤5k steps, 1024 pairs)."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.era5 import Era5ZarrDataset, load_stats
from scripts.residual_fsq.train import build_model


class LatentPairDataset(Dataset):
    """Pairs (z_t, z_{t+1}) from consecutive frames in zarr (6h step assumed)."""

    def __init__(self, z_cache: np.ndarray, max_pairs: int = 1024):
        # z_cache: (T, D)
        self.z = z_cache
        n = min(max_pairs, len(z_cache) - 1)
        self.idx = np.arange(n)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        t = int(self.idx[i])
        return {
            "z0": torch.from_numpy(self.z[t]),
            "z1": torch.from_numpy(self.z[t + 1]),
        }


class TinyLatentMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d = dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers.append(nn.Linear(d, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@torch.no_grad()
def encode_latents(model, ds: Era5ZarrDataset, device, max_frames: int) -> np.ndarray:
    """Pool spatial tokens → compact latent vector (mean over tokens per channel)."""
    model.eval()
    zs = []
    n = min(max_frames, len(ds))
    for i in tqdm(range(n), desc="encode"):
        b = ds[i]
        x = b["x"].unsqueeze(0).to(device)
        st = b["static"].unsqueeze(0).to(device)
        out = model(x, st)
        # z_c: B,Nc,Cc → mean over tokens → Cc; same for fine
        zc = out["z_c"].float().mean(dim=1)  # B, Cc
        zf = out["z_f"].float().mean(dim=1)  # B, Cf
        z = torch.cat([zc, zf], dim=-1).cpu().numpy()[0]
        zs.append(z)
    return np.stack(zs, axis=0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--pairs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg") or yaml.safe_load((args.ckpt.parent / "config.yaml").read_text())
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    mean, std = load_stats(args.ckpt.parent / "norm_stats.npz")
    root = args.data
    # need consecutive frames — use full split without crop
    n_need = args.pairs + 1
    ds = Era5ZarrDataset(
        root / f"{args.split}.zarr",
        mean=mean,
        std=std,
        crop_size=None,
        augment=False,
        static_path=root / "static.zarr" if (root / "static.zarr").exists() else None,
    )
    z_cache = encode_latents(model, ds, device, n_need)
    dim = z_cache.shape[1]
    probe = TinyLatentMLP(dim, hidden=min(512, max(128, dim // 8)), depth=3).to(device)
    nparams = probe.num_parameters()
    print(f"probe params={nparams/1e6:.3f}M (limit 2M) dim={dim}")
    if nparams > 2_000_000:
        raise SystemExit("probe > 2M params")

    pair_ds = LatentPairDataset(z_cache, max_pairs=args.pairs)
    loader = DataLoader(pair_ds, batch_size=32, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
    hist = []
    step = 0
    it = iter(loader)
    pbar = tqdm(total=args.steps, desc="probe")
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        z0 = batch["z0"].to(device)
        z1 = batch["z1"].to(device)
        pred = probe(z0)
        loss = (pred - z1).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        pbar.update(1)
        if step % 100 == 0:
            # persistence baseline
            pers = (z0 - z1).pow(2).mean()
            hist.append({"step": step, "mse": float(loss.detach()), "persist_mse": float(pers.detach())})
            pbar.set_postfix(mse=float(loss), pers=float(pers))
    pbar.close()

    out = args.out or (args.ckpt.parent / "probe_6h")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"probe": probe.state_dict(), "dim": dim, "nparams": nparams}, out / "probe.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    meta = {
        "ckpt": str(args.ckpt),
        "pairs": args.pairs,
        "steps": args.steps,
        "nparams": nparams,
        "dim": dim,
        "final": hist[-1] if hist else None,
        "metric_space": "latent_mse",
        "note": (
            "Probe metrics are MSE in pooled latent space vs persistence; "
            "not physical-field NRMSE / σ_official. Reconstruction quality uses "
            "scripts/residual_fsq/evaluate.py with data/ref_stats/sigma_official_28ch.npz."
        ),
    }
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"done → {out}")


if __name__ == "__main__":
    main()
