"""Evaluate Residual-FSQ official bitstream CR + exact index roundtrip."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.codec.bitstream import official_cr
from src.data.era5 import Era5ZarrDataset, load_stats
from src.models.residual_fsq import ResidualFSQAE
from scripts.residual_fsq.train import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--max-frames", type=int, default=8)
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
    model.eval()

    stats_path = args.ckpt.parent / "norm_stats.npz"
    mean, std = load_stats(stats_path)
    root = args.data
    ds = Era5ZarrDataset(
        root / f"{args.split}.zarr",
        mean=mean,
        std=std,
        crop_size=None,
        augment=False,
        static_path=root / "static.zarr" if (root / "static.zarr").exists() else None,
    )
    n = min(args.max_frames, len(ds))
    rows = []
    for i in tqdm(range(n), desc="bitstream"):
        batch = ds[i]
        x = batch["x"].unsqueeze(0).to(device)
        st = batch["static"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x, st)
            blob = model.compress(x, st)
            recon2 = model.decompress(blob, st)
            # exact index roundtrip
            from src.codec.bitstream import decode_indices

            _, ic, ifr = decode_indices(blob)
            ok_c = bool(torch.equal(ic.cpu(), out["indices_c"].detach().cpu()))
            ok_f = bool(torch.equal(ifr.cpu(), out["indices_f"].detach().cpu()))
            # recon from bitstream vs forward (SST land mask may match)
            recon_mse = float((recon2 - out["recon"]).pow(2).mean())
        h, w = x.shape[-2:]
        cr = official_cr(len(blob), h, w, 28, 1)
        rows.append(
            {
                "frame": i,
                "bytes": len(blob),
                "cr_official": cr,
                "cr_raw": float(out["cr_raw"]),
                "exact_indices": ok_c and ok_f,
                "recon_mse_vs_forward": recon_mse,
            }
        )
        print(
            f"  f{i}: B={len(blob)} CR×{cr:.2f} (raw×{float(out['cr_raw']):.1f}) "
            f"exact={ok_c and ok_f} mse={recon_mse:.2e}"
        )

    mean_cr = float(np.mean([r["cr_official"] for r in rows]))
    summary = {
        "ckpt": str(args.ckpt),
        "split": args.split,
        "n_frames": n,
        "mean_cr_official": mean_cr,
        "all_exact": all(r["exact_indices"] for r in rows),
        "frames": rows,
        "formula": "CR = 32*T*C*H*W / (8*B)",
    }
    out = args.out or (args.ckpt.parent / f"eval_bitstream_{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"mean official CR ×{mean_cr:.2f}  all_exact={summary['all_exact']} → {out}")


if __name__ == "__main__":
    main()
