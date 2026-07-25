"""Smoke-test CRA5 VAEformer teacher on a few ERA5 frames."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.era5 import CHANNEL_ORDER, load_stats
from src.teacher.cra5_channels import build_overlap_maps
from src.teacher.cra5_teacher import CRA5Teacher


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data" / "era5_28ch_0p5_6h_n128.zarr")
    p.add_argument("--ckpt", type=Path, default=ROOT / "third_party" / "checkpoints" / "cra5_268v_300k.pth")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "teacher_smoke")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    zpath = args.data / f"{args.split}.zarr"
    try:
        ds = xr.open_zarr(zpath, consolidated=True)
    except Exception:
        ds = xr.open_zarr(zpath, consolidated=False)

    # stats from run control or estimate from train
    stats_path = args.data / "norm_stats.npz"
    if not stats_path.exists():
        # fall back to control run stats
        stats_path = ROOT / "runs" / "residual_fsq_0p5" / "20260725_010045_n128s9k" / "norm_stats.npz"
    mean, std = load_stats(stats_path) if stats_path.exists() else (None, None)
    if mean is None:
        raise SystemExit(f"Need norm_stats at {stats_path}")

    our_idx, _, names = build_overlap_maps()
    print(f"overlap channels ({len(names)}): {names}")
    print(f"loading teacher on {args.device}…")
    t0 = time.time()
    teacher = CRA5Teacher(ckpt=args.ckpt, device=args.device, use_quantized=False)
    print(f"teacher ready in {time.time()-t0:.1f}s")

    metrics = []
    n = min(args.n, int(ds.sizes["time"]))
    for i in range(n):
        x_phys = torch.from_numpy(ds["fields"].isel(time=i).values.astype(np.float32)).unsqueeze(0)
        t1 = time.time()
        rec = teacher.reconstruct_physical(x_phys).cpu()
        dt = time.time() - t1
        err = {}
        for oi, name in zip(our_idx, names):
            diff = (rec[0, oi] - x_phys[0, oi]).numpy()
            err[name] = float(np.sqrt(np.nanmean(diff**2)))
        mean_rmse = float(np.mean(list(err.values())))
        print(f"frame {i}: {dt:.1f}s  mean_overlap_RMSE={mean_rmse:.4g}")
        metrics.append({"frame": i, "seconds": dt, "rmse": err, "mean_rmse": mean_rmse})

        # plot a few channels
        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        for ax, ch in zip(axs.flat, ["t2m", "u10", "Z500" if False else "Z850", "T850", "Q700", "tcc"]):
            if ch not in CHANNEL_ORDER:
                ax.axis("off")
                continue
            ci = CHANNEL_ORDER.index(ch)
            im = ax.imshow(rec[0, ci].numpy() - x_phys[0, ci].numpy(), cmap="RdBu_r")
            ax.set_title(f"{ch} err")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"CRA5 teacher residual frame {i}")
        fig.tight_layout()
        fig.savefig(args.out / f"teacher_residual_f{i}.png", dpi=120)
        plt.close(fig)

    (args.out / "metrics.json").write_text(json.dumps({"overlap": names, "frames": metrics}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
