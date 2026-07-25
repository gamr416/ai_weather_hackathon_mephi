"""Offline CRA5 teacher cache: recon_overlap in our normalized space."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.era5 import CHANNEL_ORDER, load_stats
from src.teacher.cra5_channels import build_overlap_maps
from src.teacher.cra5_teacher import CRA5Teacher


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build CRA5 teacher recon cache")
    p.add_argument("--data", type=Path, required=True, help="era5 zarr root with train/val.zarr")
    p.add_argument("--stats", type=Path, default=None, help="norm_stats.npz (default: control run)")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "teacher_cra5")
    p.add_argument("--ckpt", type=Path, default=ROOT / "third_party" / "checkpoints" / "cra5_268v_300k.pth")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--splits", type=str, default="train,val")
    p.add_argument("--max-frames", type=int, default=None, help="Cap frames per split (smoke/partial)")
    p.add_argument("--start", type=int, default=0)
    return p.parse_args()


def open_fields(path: Path):
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:
        return xr.open_zarr(path, consolidated=False)


def main() -> None:
    args = parse_args()
    stats_path = args.stats or (
        ROOT / "runs" / "residual_fsq_0p5" / "20260725_010045_n128s9k" / "norm_stats.npz"
    )
    mean, std = load_stats(stats_path)
    mean_t = torch.from_numpy(mean).view(1, 28, 1, 1)
    std_t = torch.from_numpy(std).view(1, 28, 1, 1).clamp_min(1e-4)

    our_idx, _, names = build_overlap_maps()
    mask = np.zeros(28, dtype=np.float32)
    for i in our_idx:
        mask[i] = 1.0

    print(f"loading teacher on {args.device}…")
    teacher = CRA5Teacher(ckpt=args.ckpt, device=args.device, use_quantized=False)

    meta = {
        "stats": str(stats_path),
        "ckpt": str(args.ckpt),
        "overlap_names": names,
        "overlap_indices": our_idx,
        "channels": CHANNEL_ORDER,
        "note": "recon is in our normalized space; non-overlap channels are zero",
        "splits": {},
    }

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        zpath = args.data / f"{split}.zarr"
        ds = open_fields(zpath)
        n_all = int(ds.sizes["time"])
        end = n_all if args.max_frames is None else min(n_all, args.start + args.max_frames)
        start = min(args.start, n_all)
        n = end - start
        out_split = args.out / split
        out_split.mkdir(parents=True, exist_ok=True)
        # memmap to avoid RAM blowup
        recon_path = out_split / "recon_norm.npy"
        shape = (n, 28, 360, 720)
        recon_mm = np.lib.format.open_memmap(recon_path, mode="w+", dtype=np.float32, shape=shape)
        times = []
        t_list = []
        for j, ti in enumerate(tqdm(range(start, end), desc=f"teacher:{split}")):
            x_phys = torch.from_numpy(ds["fields"].isel(time=ti).values.astype(np.float32)).unsqueeze(0)
            t0 = time.time()
            rec_phys = teacher.reconstruct_physical(x_phys).cpu()
            rec_norm = (rec_phys - mean_t) / std_t
            # zero non-overlap
            rec_norm = rec_norm * torch.from_numpy(mask).view(1, 28, 1, 1)
            recon_mm[j] = rec_norm.numpy()[0]
            recon_mm.flush()
            dt = time.time() - t0
            times.append(dt)
            t_list.append(str(ds["time"].values[ti]))
            if j == 0:
                print(f"  first frame {dt:.1f}s")
        meta["splits"][split] = {
            "n": n,
            "start": start,
            "end": end,
            "recon": str(recon_path.relative_to(ROOT)) if recon_path.is_relative_to(ROOT) else str(recon_path),
            "mean_seconds": float(np.mean(times)) if times else None,
            "times": t_list,
        }
        print(f"{split}: wrote {recon_path} shape={shape}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    np.savez(args.out / "overlap_mask.npz", mask=mask, names=np.array(names), indices=np.array(our_idx))
    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
