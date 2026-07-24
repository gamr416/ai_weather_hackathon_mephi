"""Evaluate Spatial-FSQ checkpoint: metrics + plots + raw CR."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.era5 import CHANNEL_ORDER, Era5ZarrDataset, load_stats
from src.models.spatial_fsq import SpatialFSQAE, raw_cr_from_bits

PLOT_CHANNELS = ["t2m", "mslp", "tp6h", "tcwv", "T850", "Z850", "U850", "Q850"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test", "train"], default="val")
    p.add_argument("--stats", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-frames", type=int, default=32)
    p.add_argument("--plot-frames", type=int, default=3)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--crop", action="store_true")
    return p.parse_args()


def build_model(cfg: dict, state: dict, device: torch.device) -> SpatialFSQAE:
    mcfg = cfg["model"]
    model = SpatialFSQAE(
        in_channels=28,
        static_channels=int(mcfg.get("static_channels", 4)),
        patch_size=int(mcfg["patch_size"]),
        dim=int(mcfg["dim"]),
        depth=int(mcfg["depth"]),
        heads=int(mcfg["heads"]),
        latent_channels=int(mcfg["latent_channels"]),
        num_levels=int(mcfg["num_levels"]),
        dropout=float(mcfg.get("dropout", 0.0)),
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def collect_errors(
    model: SpatialFSQAE,
    ds: Era5ZarrDataset,
    device: torch.device,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, float, list[dict], dict]:
    n = min(len(ds), max_frames)
    sse_norm = np.zeros(28, dtype=np.float64)
    sse_phys = np.zeros(28, dtype=np.float64)
    wsum = 0.0
    examples: list[dict] = []
    mean = ds.mean.cpu().numpy().reshape(28, 1, 1)
    std = ds.std.cpu().numpy().reshape(28, 1, 1)
    rate_bits_total = 0.0
    cr_list: list[float] = []

    for i in tqdm(range(n), desc="eval"):
        batch = ds[i]
        x = batch["x"].unsqueeze(0).to(device)
        st = batch["static"].unsqueeze(0).to(device)
        w = batch["lat_weight"].to(device)
        out = model(x, st)
        recon = out["recon"][0].cpu().numpy()
        target = batch["x"].numpy()
        h, w_img = target.shape[-2:]

        rate_bits = float(out["rate_bits"].cpu())
        rate_bits_total += rate_bits
        cr_list.append(raw_cr_from_bits(rate_bits, h, w_img, 28))

        recon_phys = recon * std + mean
        target_phys = target * std + mean

        ww = w.cpu().numpy().astype(np.float64)
        w2 = ww[:, None]
        err_n = (recon - target) ** 2
        err_p = (recon_phys - target_phys) ** 2
        sse_norm += (err_n * w2[None, :, :]).sum(axis=(1, 2))
        sse_phys += (err_p * w2[None, :, :]).sum(axis=(1, 2))
        wsum += float(w2.sum() * target.shape[-1])

        if len(examples) < 8:
            examples.append(
                {
                    "index": int(batch["index"]),
                    "target": target_phys.astype(np.float32),
                    "recon": recon_phys.astype(np.float32),
                }
            )

    codec = {
        "mean_cr_raw": float(np.mean(cr_list)) if cr_list else None,
        "mean_rate_bits": rate_bits_total / max(n, 1),
        "expected_raw_cr": float(model.expected_raw_cr()),
        "note": "raw CR from uniform indices (no entropy coding yet)",
    }
    return sse_norm, sse_phys, wsum, examples, codec


def scores_from_sse(sse_norm: np.ndarray, sse_phys: np.ndarray, wsum: float) -> dict:
    rmse_norm = np.sqrt(sse_norm / max(wsum, 1e-12))
    nrmse = rmse_norm.copy()
    rmse_phys = np.sqrt(sse_phys / max(wsum, 1e-12))
    surface = float(nrmse[:8].mean())
    pressure = float(nrmse[8:].mean())
    per = {
        CHANNEL_ORDER[i]: {
            "rmse_norm": float(rmse_norm[i]),
            "nrmse": float(nrmse[i]),
            "rmse_physical": float(rmse_phys[i]),
        }
        for i in range(28)
    }
    return {
        "S_surface": surface,
        "S_pressure": pressure,
        "S_all": 0.5 * surface + 0.5 * pressure,
        "per_channel": per,
    }


def plot_frame_comparison(
    target: np.ndarray,
    recon: np.ndarray,
    out_path: Path,
    frame_id: int,
    channels: list[str],
) -> None:
    cols = len(channels)
    fig, axes = plt.subplots(3, cols, figsize=(3.2 * cols, 8), constrained_layout=True)
    if cols == 1:
        axes = np.asarray(axes).reshape(3, 1)
    for j, name in enumerate(channels):
        ci = CHANNEL_ORDER.index(name)
        t = target[ci]
        r = recon[ci]
        d = r - t
        is_wind = name.startswith(("U", "V", "u", "v"))
        if is_wind:
            vmax = np.nanpercentile(np.abs(t), 99)
            vmin, vmax_t = -vmax, vmax
        else:
            vmin = np.nanpercentile(t, 1)
            vmax_t = np.nanpercentile(t, 99)
        dlim = np.nanpercentile(np.abs(d), 99) or 1.0
        for row, img, title, cmap, v0, v1 in [
            (0, t, f"{name} truth", "viridis", vmin, vmax_t),
            (1, r, f"{name} recon", "viridis", vmin, vmax_t),
            (2, d, f"{name} diff", "RdBu_r", -dlim, dlim),
        ]:
            ax = axes[row, j]
            im = ax.imshow(img, origin="lower", cmap=cmap, vmin=v0, vmax=v1, aspect="auto")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Frame {frame_id}: truth vs reconstruction", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_nrmse_bars(metrics: dict, out_path: Path) -> None:
    names = CHANNEL_ORDER
    vals = [metrics["per_channel"][n]["nrmse"] for n in names]
    colors = ["#3b82f6"] * 8 + ["#10b981"] * 20
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.bar(range(28), vals, color=colors)
    ax.set_xticks(range(28))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("NRMSE (norm space)")
    ax.set_title(
        f"Per-channel NRMSE  |  S_all={metrics['S_all']:.3f}  "
        f"S_surface={metrics['S_surface']:.3f}  S_pressure={metrics['S_pressure']:.3f}"
    )
    ax.axvline(7.5, color="gray", ls="--", lw=0.8)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_history(history_path: Path, out_path: Path) -> None:
    if not history_path.exists():
        return
    hist = json.loads(history_path.read_text())
    steps, losses, recs = [], [], []
    vsteps, vrecs, vlosses = [], [], []
    for row in hist:
        if "loss" in row and "val_recon" not in row and "full_val_recon" not in row:
            steps.append(row["step"])
            losses.append(row["loss"])
            recs.append(row.get("recon"))
        if "val_recon" in row:
            vsteps.append(row["step"])
            vrecs.append(row["val_recon"])
            vlosses.append(row.get("val_loss"))
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    if steps:
        ax.plot(steps, losses, label="train loss", lw=1.5)
        if any(r is not None for r in recs):
            ax.plot(steps, recs, label="train recon", lw=1.2, alpha=0.8)
    if vsteps:
        ax.plot(vsteps, vrecs, label="val recon", marker="o", lw=1.5)
        if any(v is not None for v in vlosses):
            ax.plot(vsteps, vlosses, label="val loss", marker="x", lw=1.0, alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Spatial-FSQ training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg") or yaml.safe_load((ROOT / "configs/spatial_fsq_0p5.yaml").read_text())
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    stats_path = args.stats or (args.ckpt.parent / "norm_stats.npz")
    if not stats_path.exists():
        raise SystemExit(f"Missing stats: {stats_path}")
    mean, std = load_stats(stats_path)

    split_path = args.data / f"{args.split}.zarr"
    if not split_path.exists():
        raise SystemExit(f"Missing split: {split_path}")

    crop = cfg["data"].get("crop_size", 192) if args.crop else None
    static_path = args.data / "static.zarr"
    ds = Era5ZarrDataset(
        split_path,
        mean=mean,
        std=std,
        crop_size=crop,
        augment=False,
        seed=0,
        static_path=static_path if static_path.exists() else None,
    )
    model = build_model(cfg, ckpt["model"], device)

    out_dir = args.out or (args.ckpt.parent / f"eval_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sse_n, sse_p, wsum, examples, codec = collect_errors(model, ds, device, args.max_frames)
    metrics = scores_from_sse(sse_n, sse_p, wsum)
    metrics.update(
        {
            "split": args.split,
            "n_frames": min(len(ds), args.max_frames),
            "ckpt": str(args.ckpt),
            "crop_size": crop,
            "codec": codec,
        }
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_nrmse_bars(metrics, out_dir / "nrmse_bars.png")
    plot_history(args.ckpt.parent / "history.json", out_dir / "train_curves.png")
    for ex in examples[: args.plot_frames]:
        plot_frame_comparison(
            ex["target"],
            ex["recon"],
            out_dir / f"compare_frame{ex['index']:04d}.png",
            ex["index"],
            PLOT_CHANNELS,
        )

    print(
        json.dumps(
            {
                "S_all": metrics["S_all"],
                "S_surface": metrics["S_surface"],
                "S_pressure": metrics["S_pressure"],
                "mean_cr_raw": codec["mean_cr_raw"],
                "expected_raw_cr": codec["expected_raw_cr"],
                "n_frames": metrics["n_frames"],
            },
            indent=2,
        )
    )
    print(f"Wrote {out_dir}/metrics.json and plots")


if __name__ == "__main__":
    main()
