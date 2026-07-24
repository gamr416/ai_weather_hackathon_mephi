"""Train Perceiver-AE codec on local ERA5 0.5° zarr."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# allow `python scripts/train.py` from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.era5 import Era5ZarrDataset, load_stats, save_stats
from src.models.perceiver_ae import PerceiverAE, latitude_weighted_mse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Perceiver AE on ERA5 28ch")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "perceiver_0p5.yaml")
    p.add_argument("--data", type=Path, default=None, help="Override data.root")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model: PerceiverAE, loader: DataLoader, device: torch.device, vq_weight: float) -> dict:
    model.eval()
    tot_rec, tot_vq, n = 0.0, 0.0, 0
    for batch in loader:
        x = batch["x"].to(device)
        w = batch["lat_weight"].to(device)
        out = model(x)
        rec = latitude_weighted_mse(out["recon"], x, w)
        tot_rec += float(rec) * x.shape[0]
        tot_vq += float(out["vq_loss"]) * x.shape[0]
        n += x.shape[0]
    model.train()
    return {
        "val_recon": tot_rec / max(n, 1),
        "val_vq": tot_vq / max(n, 1),
        "val_loss": (tot_rec + vq_weight * tot_vq) / max(n, 1),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data is not None:
        cfg["data"]["root"] = str(args.data)
    if args.steps is not None:
        cfg["train"]["max_steps"] = args.steps

    device = torch.device(
        args.device
        or cfg["train"].get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(cfg["data"]["root"])
    train_path = root / "train.zarr"
    val_path = root / "val.zarr"
    if not train_path.exists():
        raise SystemExit(
            f"Missing {train_path}. Wait for download or pass --data path/to/era5_...zarr"
        )

    stats_path = out_dir / "norm_stats.npz"
    if stats_path.exists():
        mean, std = load_stats(stats_path)
        if not (np.isfinite(mean).all() and np.isfinite(std).all()):
            print(f"Corrupt stats in {stats_path} (NaNs) — recomputing")
            stats_path.unlink()
            mean = std = None
        else:
            print(f"Loaded stats from {stats_path}")
    else:
        mean = std = None

    if mean is None or std is None:
        print("Estimating train mean/std…")
        train_ds = Era5ZarrDataset(
            train_path,
            crop_size=cfg["data"]["crop_size"],
            augment=True,
            seed=cfg["train"]["seed"],
        )
        save_stats(train_ds.mean.numpy().reshape(28), train_ds.std.numpy().reshape(28), stats_path)
    else:
        train_ds = Era5ZarrDataset(
            train_path,
            mean=mean,
            std=std,
            crop_size=cfg["data"]["crop_size"],
            augment=True,
            seed=cfg["train"]["seed"],
        )

    val_ds = None
    if val_path.exists():
        val_ds = Era5ZarrDataset(
            val_path,
            mean=train_ds.mean.numpy().reshape(28),
            std=train_ds.std.numpy().reshape(28),
            crop_size=cfg["data"]["crop_size"],
            augment=False,
            seed=cfg["train"]["seed"] + 1,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=0,
        )

    mcfg = cfg["model"]
    model = PerceiverAE(
        in_channels=28,
        patch_size=mcfg["patch_size"],
        dim=mcfg["dim"],
        num_latents=mcfg["num_latents"],
        depth=mcfg["depth"],
        heads=mcfg["heads"],
        codebook_size=mcfg["codebook_size"],
        dropout=mcfg.get("dropout", 0.0),
    ).to(device)

    nparams = model.num_parameters()
    print(f"Device={device}  params={nparams/1e6:.2f}M  (limit 20M)")
    if nparams > 20_000_000:
        raise SystemExit(f"Model has {nparams} params > 20M hackathon limit")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.01),
    )
    max_steps = int(cfg["train"]["max_steps"])
    log_every = int(cfg["train"].get("log_every", 20))
    val_every = int(cfg["train"].get("val_every", 200))
    ckpt_every = int(cfg["train"].get("ckpt_every", 500))
    vq_weight = float(cfg["train"].get("vq_weight", 0.25))

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    model.train()
    step = 0
    best_val = math.inf
    history: list[dict] = []
    t0 = time.time()
    data_iter = iter(train_loader)

    pbar = tqdm(total=max_steps, desc="train")
    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x = batch["x"].to(device, non_blocking=True)
        w = batch["lat_weight"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            out = model(x)
            rec = latitude_weighted_mse(out["recon"], x, w)
            loss = rec + vq_weight * out["vq_loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 1.0))
        scaler.step(opt)
        scaler.update()

        step += 1
        pbar.update(1)
        if step % log_every == 0:
            pbar.set_postfix(
                loss=float(loss),
                rec=float(rec),
                vq=float(out["vq_loss"]),
                ppl=float(out["perplexity"]),
            )
            history.append(
                {
                    "step": step,
                    "loss": float(loss),
                    "recon": float(rec),
                    "vq": float(out["vq_loss"]),
                    "ppl": float(out["perplexity"]),
                    "sec": time.time() - t0,
                }
            )

        if val_loader is not None and step % val_every == 0:
            metrics = evaluate(model, val_loader, device, vq_weight)
            print(f"\n[val] step={step} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            history.append({"step": step, **metrics})
            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "step": step,
                        "best_val": best_val,
                        "nparams": nparams,
                    },
                    out_dir / "best.pt",
                )

        if step % ckpt_every == 0:
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "step": step, "nparams": nparams},
                out_dir / f"ckpt_{step:06d}.pt",
            )

    pbar.close()
    torch.save(
        {"model": model.state_dict(), "cfg": cfg, "step": step, "nparams": nparams},
        out_dir / "last.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. checkpoints → {out_dir}")
    print(
        "Evaluate + plots:\n"
        f"  python scripts/evaluate.py --ckpt {out_dir / 'best.pt'} "
        f"--data {root} --split val"
    )


if __name__ == "__main__":
    main()
