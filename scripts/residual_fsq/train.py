"""Train Residual Multi-Scale FSQ codec (scripts/residual_fsq/)."""
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.era5 import (
    DEFAULT_CHANNEL_WEIGHTS,
    Era5ZarrDataset,
    channel_weight_tensor,
    load_stats,
    save_stats,
)
from src.models.residual_fsq import (
    ResidualFSQAE,
    group_recon_stats,
    high_freq_grad_penalty,
    latitude_weighted_mse_sst_ocean,
    sst_land_zero_loss,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Residual-FSQ AE on ERA5 28ch")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "residual_fsq_0p5.yaml")
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--init-ckpt", type=Path, default=None, help="Warm-start student weights")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def make_run_dir(cfg: dict, args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        out = Path(args.out_dir)
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"Refusing to overwrite non-empty run dir: {out}")
        out.mkdir(parents=True, exist_ok=True)
        return out
    base = Path(cfg["train"].get("runs_root", "runs/residual_fsq_0p5"))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = args.run_name or cfg["train"].get("run_name") or "run"
    tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
    out = base / f"{stamp}_{tag}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def rate_weight_at(step: int, cfg: dict) -> float:
    target = float(cfg["train"].get("rate_weight", 0.02))
    warmup = int(cfg["train"].get("rate_warmup_steps", 1500))
    if warmup <= 0:
        return target
    return target * min(1.0, step / warmup)


def make_scheduler(opt: torch.optim.Optimizer, cfg: dict, max_steps: int):
    warmup = int(cfg["train"].get("warmup_steps", 500))

    def lr_lambda(step: int) -> float:
        s = step + 1
        if s <= warmup:
            return s / max(warmup, 1)
        progress = (s - warmup) / max(max_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def build_model(mcfg: dict) -> ResidualFSQAE:
    return ResidualFSQAE(
        in_channels=28,
        static_channels=int(mcfg.get("static_channels", 4)),
        stem_dim=int(mcfg.get("stem_dim", 64)),
        stem_blocks=int(mcfg.get("stem_blocks", 2)),
        coarse_patch=int(mcfg.get("coarse_patch", 8)),
        coarse_dim=int(mcfg.get("coarse_dim", 256)),
        coarse_depth=int(mcfg.get("coarse_depth", 4)),
        coarse_dec_depth=int(mcfg.get("coarse_dec_depth", 3)),
        coarse_heads=int(mcfg.get("coarse_heads", 8)),
        coarse_latent=int(mcfg.get("coarse_latent", 80)),
        coarse_levels=int(mcfg.get("coarse_levels", 256)),
        fine_patch=int(mcfg.get("fine_patch", 4)),
        fine_dim=int(mcfg.get("fine_dim", 128)),
        fine_depth=int(mcfg.get("fine_depth", 2)),
        fine_heads=int(mcfg.get("fine_heads", 4)),
        fine_latent=int(mcfg.get("fine_latent", 16)),
        fine_levels=int(mcfg.get("fine_levels", 64)),
        dropout=float(mcfg.get("dropout", 0.1)),
        drop_path=float(mcfg.get("drop_path", 0.1)),
    )


def compute_losses(
    out: dict,
    x: torch.Tensor,
    st: torch.Tensor,
    w: torch.Tensor,
    channel_w: torch.Tensor,
    rate_w: float,
    commit_w: float,
    hf_w: float,
    sst_land_w: float = 1.0,
    teacher: torch.Tensor | None = None,
    teacher_w: float = 0.0,
    teacher_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rec = latitude_weighted_mse_sst_ocean(out["recon"], x, w, channel_w, static=st)
    bsz, _, h, w_img = x.shape
    input_bits = 32.0 * 28 * h * w_img * bsz
    rate_norm = out["rate_bits"] / max(input_bits, 1.0)
    hf = high_freq_grad_penalty(out["recon"], x, lat_weight=w) if hf_w > 0 else x.new_zeros(())
    raw = out.get("recon_raw", out["recon"])
    land = sst_land_zero_loss(raw, st, w) if sst_land_w > 0 else x.new_zeros(())
    teach = x.new_zeros(())
    if teacher is not None and teacher_w > 0:
        # lat-weighted MSE on overlap channels only
        diff2 = (out["recon"] - teacher).pow(2)
        if teacher_mask is not None:
            m = teacher_mask.view(1, -1, 1, 1).to(diff2.device, diff2.dtype)
            diff2 = diff2 * m
            denom = m.sum().clamp_min(1.0)
        else:
            denom = float(diff2.shape[1])
        w_lat = w.view(w.shape[0], 1, -1, 1)
        teach = (diff2 * w_lat).sum() / (w_lat.sum() * denom * diff2.shape[-1] + 1e-8)
    loss = (
        rec
        + rate_w * rate_norm
        + commit_w * out["commit"]
        + hf_w * hf
        + sst_land_w * land
        + teacher_w * teach
    )
    return loss, rec, rate_norm, hf, land, teach


@torch.no_grad()
def evaluate(
    model: ResidualFSQAE,
    loader: DataLoader,
    device: torch.device,
    rate_w: float,
    commit_w: float,
    hf_w: float,
    channel_w: torch.Tensor,
    sst_land_w: float = 1.0,
    teacher_w: float = 0.0,
    teacher_mask: torch.Tensor | None = None,
) -> dict:
    model.eval()
    tot_rec = tot_rate = tot_commit = tot_cr = tot_hf = tot_land = tot_teach = n = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        st = batch["static"].to(device)
        w = batch["lat_weight"].to(device)
        teacher = batch["teacher"].to(device) if "teacher" in batch else None
        out = model(x, st)
        loss, rec, rate_norm, hf, land, teach = compute_losses(
            out,
            x,
            st,
            w,
            channel_w,
            rate_w,
            commit_w,
            hf_w,
            sst_land_w,
            teacher=teacher,
            teacher_w=teacher_w if teacher is not None else 0.0,
            teacher_mask=teacher_mask,
        )
        tot_rec += float(rec) * x.shape[0]
        tot_rate += float(rate_norm) * x.shape[0]
        tot_commit += float(out["commit"]) * x.shape[0]
        tot_cr += float(out["cr_raw"]) * x.shape[0]
        tot_hf += float(hf) * x.shape[0]
        tot_land += float(land) * x.shape[0]
        tot_teach += float(teach) * x.shape[0]
        n += x.shape[0]
    model.train()
    n = max(n, 1.0)
    return {
        "val_recon": tot_rec / n,
        "val_rate": tot_rate / n,
        "val_commit": tot_commit / n,
        "val_hf": tot_hf / n,
        "val_sst_land": tot_land / n,
        "val_teacher": tot_teach / n,
        "val_cr": tot_cr / n,
        "val_loss": (
            tot_rec
            + rate_w * tot_rate
            + commit_w * tot_commit
            + hf_w * tot_hf
            + sst_land_w * tot_land
            + teacher_w * tot_teach
        )
        / n,
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
    out_dir = make_run_dir(cfg, args)
    cfg["train"]["out_dir"] = str(out_dir)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    meta = {
        "out_dir": str(out_dir),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_root": cfg["data"]["root"],
        "config_src": str(args.config),
        "run_name": args.run_name,
        "max_steps": cfg["train"]["max_steps"],
        "pipeline": "residual_fsq",
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Run directory: {out_dir}")

    root = Path(cfg["data"]["root"])
    train_path = root / "train.zarr"
    val_path = root / "val.zarr"
    static_path = root / "static.zarr"
    if not train_path.exists():
        raise SystemExit(f"Missing {train_path}")

    stats_path = out_dir / "norm_stats.npz"
    mean = std = None
    if stats_path.exists():
        mean, std = load_stats(stats_path)
        if not (np.isfinite(mean).all() and np.isfinite(std).all()):
            stats_path.unlink()
            mean = std = None
        else:
            print(f"Loaded stats from {stats_path}")

    ds_kwargs = dict(
        crop_size=cfg["data"]["crop_size"],
        static_path=static_path if static_path.exists() else None,
    )

    # Optional CRA5 teacher cache
    distill_cfg = cfg.get("distill") or {}
    teacher_root = distill_cfg.get("teacher_cache")
    teacher_mask_np = None
    train_teacher = val_teacher = None
    if teacher_root:
        troot = Path(teacher_root)
        mask_path = troot / "overlap_mask.npz"
        if mask_path.exists():
            mz = np.load(mask_path, allow_pickle=True)
            teacher_mask_np = mz["mask"].astype(np.float32)
            distill_cfg["teacher_mask"] = teacher_mask_np.tolist()
            cfg["distill"] = distill_cfg
        train_npy = troot / "train" / "recon_norm.npy"
        val_npy = troot / "val" / "recon_norm.npy"
        if train_npy.exists():
            train_teacher = np.load(train_npy, mmap_mode="r")
            print(f"Teacher train cache: {train_npy} shape={train_teacher.shape}")
        if val_npy.exists():
            val_teacher = np.load(val_npy, mmap_mode="r")
            print(f"Teacher val cache: {val_npy} shape={val_teacher.shape}")
        cfg["train"]["lambda_teacher"] = float(
            cfg["train"].get("lambda_teacher", distill_cfg.get("lambda_teacher", 0.5))
        )

    if mean is None or std is None:
        print("Estimating train mean/std…")
        train_ds = Era5ZarrDataset(
            train_path,
            augment=True,
            seed=cfg["train"]["seed"],
            teacher_recon=train_teacher,
            teacher_mask=teacher_mask_np,
            teacher_index_offset=int(distill_cfg.get("train_offset", 0)),
            **ds_kwargs,
        )
        save_stats(train_ds.mean.numpy().reshape(28), train_ds.std.numpy().reshape(28), stats_path)
    else:
        train_ds = Era5ZarrDataset(
            train_path,
            mean=mean,
            std=std,
            augment=True,
            seed=cfg["train"]["seed"],
            teacher_recon=train_teacher,
            teacher_mask=teacher_mask_np,
            teacher_index_offset=int(distill_cfg.get("train_offset", 0)),
            **ds_kwargs,
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
            static_path=static_path if static_path.exists() else None,
            teacher_recon=val_teacher,
            teacher_mask=teacher_mask_np,
            teacher_index_offset=int(distill_cfg.get("val_offset", 0)),
        )
    val_full_ds = None
    if val_path.exists() and cfg["train"].get("full_frame_val_every"):
        val_full_ds = Era5ZarrDataset(
            val_path,
            mean=train_ds.mean.numpy().reshape(28),
            std=train_ds.std.numpy().reshape(28),
            crop_size=None,
            augment=False,
            seed=cfg["train"]["seed"] + 2,
            static_path=static_path if static_path.exists() else None,
            teacher_recon=val_teacher,
            teacher_mask=teacher_mask_np,
            teacher_index_offset=int(distill_cfg.get("val_offset", 0)),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)
        if val_ds is not None and len(val_ds) > 0
        else None
    )
    val_full_loader = (
        DataLoader(val_full_ds, batch_size=1, shuffle=False, num_workers=0)
        if val_full_ds is not None and len(val_full_ds) > 0
        else None
    )

    model = build_model(cfg["model"]).to(device)
    nparams = model.num_parameters()
    print(
        f"Device={device}  params={nparams/1e6:.2f}M  "
        f"expected_raw_CR≈×{model.expected_raw_cr():.1f}  (limit 20M)"
    )
    if nparams > 20_000_000:
        raise SystemExit(f"Model has {nparams} params > 20M hackathon limit")

    if args.init_ckpt is not None:
        init = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(init["model"], strict=False)
        print(f"Warm-start from {args.init_ckpt} missing={len(missing)} unexpected={len(unexpected)}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.05),
    )
    max_steps = int(cfg["train"]["max_steps"])
    sched = make_scheduler(opt, cfg, max_steps)
    accum = int(cfg["train"].get("grad_accum", 1))
    commit_w = float(cfg["train"].get("commit_weight", 0.05))
    hf_w = float(cfg["train"].get("hf_weight", 0.05))
    sst_land_w = float(cfg["train"].get("sst_land_weight", 1.0))
    teacher_w = float(cfg["train"].get("lambda_teacher", 0.0))
    select_full = bool(cfg["train"].get("select_full_frame", True))

    log_every = int(cfg["train"].get("log_every", 20))
    val_every = int(cfg["train"].get("val_every", 200))
    ckpt_every = int(cfg["train"].get("ckpt_every", 1000))
    full_every = int(cfg["train"].get("full_frame_val_every", 0) or 0)

    ch_w_cfg = cfg["train"].get("channel_weights") or DEFAULT_CHANNEL_WEIGHTS
    channel_w = channel_weight_tensor(ch_w_cfg).to(device)

    teacher_mask_t = None
    if teacher_w > 0 and "teacher_mask" in cfg.get("distill", {}):
        teacher_mask_t = torch.tensor(cfg["distill"]["teacher_mask"], dtype=torch.float32, device=device)
    elif teacher_w > 0:
        # default: all ones except sst/tcwv indices 5,6
        m = torch.ones(28, dtype=torch.float32, device=device)
        m[5] = 0.0
        m[6] = 0.0
        teacher_mask_t = m

    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    model.train()
    step = 0
    best_score = math.inf
    history: list[dict] = []
    t0 = time.time()
    data_iter = iter(train_loader)
    opt.zero_grad(set_to_none=True)

    pbar = tqdm(total=max_steps, desc="residual_fsq")
    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x = batch["x"].to(device, non_blocking=True)
        st = batch["static"].to(device, non_blocking=True)
        w = batch["lat_weight"].to(device, non_blocking=True)
        teacher = batch["teacher"].to(device, non_blocking=True) if "teacher" in batch else None
        has_t = batch.get("has_teacher")
        rw = rate_weight_at(step + 1, cfg)
        tw = teacher_w
        if teacher is None or teacher_w <= 0:
            tw = 0.0
        elif has_t is not None and float(has_t.float().mean()) < 0.5:
            tw = 0.0  # batch mostly without teacher cache

        with torch.amp.autocast("cuda", enabled=use_cuda):
            out = model(x, st)
            loss_full, rec, rate_norm, hf, land, teach = compute_losses(
                out,
                x,
                st,
                w,
                channel_w,
                rw,
                commit_w,
                hf_w,
                sst_land_w,
                teacher=teacher,
                teacher_w=tw,
                teacher_mask=teacher_mask_t,
            )
            loss = loss_full / accum

        scaler.scale(loss).backward()

        if (step + 1) % accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 1.0))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()

        step += 1
        pbar.update(1)

        if step % log_every == 0:
            groups = group_recon_stats(out["recon"], x, w)
            row = {
                "step": step,
                "loss": float(loss_full.detach()),
                "recon": float(rec.detach()),
                "rate": float(rate_norm.detach()),
                "commit": float(out["commit"].detach()),
                "hf": float(hf.detach()),
                "sst_land": float(land.detach()),
                "teacher": float(teach.detach()),
                "cr": float(out["cr_raw"].detach()),
                "usage": float(out["usage"]),
                "rate_w": rw,
                "teacher_w": tw,
                "lr": float(opt.param_groups[0]["lr"]),
                "sec": time.time() - t0,
                **{f"g_{k}": v for k, v in groups.items()},
            }
            pbar.set_postfix(
                loss=row["loss"],
                rec=row["recon"],
                cr=f"{row['cr']:.0f}",
                tw=f"{tw:.2f}",
            )
            history.append(row)

        if val_loader is not None and step % val_every == 0:
            metrics = evaluate(
                model,
                val_loader,
                device,
                rw,
                commit_w,
                hf_w,
                channel_w,
                sst_land_w,
                teacher_w=teacher_w,
                teacher_mask=teacher_mask_t,
            )
            print(
                f"\n[val] step={step} "
                + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )
            history.append({"step": step, **metrics})
            score = metrics["val_recon"]
            if (not select_full) and score < best_score:
                best_score = score
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "step": step,
                        "best_val_recon": best_score,
                        "nparams": nparams,
                    },
                    out_dir / "best.pt",
                )

        if full_every and val_full_loader is not None and step % full_every == 0:
            try:
                torch.cuda.empty_cache()
                mfull = evaluate(
                    model,
                    val_full_loader,
                    device,
                    rw,
                    commit_w,
                    hf_w,
                    channel_w,
                    sst_land_w,
                    teacher_w=teacher_w,
                    teacher_mask=teacher_mask_t,
                )
                print(
                    f"[val-full] step={step} "
                    + " ".join(f"{k}={v:.4f}" for k, v in mfull.items())
                )
                history.append({"step": step, **{f"full_{k}": v for k, v in mfull.items()}})
                if select_full and mfull["val_recon"] < best_score:
                    best_score = mfull["val_recon"]
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "cfg": cfg,
                            "step": step,
                            "best_full_val_recon": best_score,
                            "nparams": nparams,
                        },
                        out_dir / "best.pt",
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[val-full] step={step} skipped (CUDA OOM); using crop val for best")
                if select_full and val_loader is not None:
                    # fall back: keep crop-based best updates active for remainder
                    select_full = False

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
    if not (out_dir / "best.pt").exists():
        torch.save(
            {"model": model.state_dict(), "cfg": cfg, "step": step, "nparams": nparams},
            out_dir / "best.pt",
        )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. checkpoints → {out_dir}")
    print(
        "Evaluate:\n"
        f"  python scripts/residual_fsq/evaluate.py --ckpt {out_dir / 'best.pt'} "
        f"--data {root} --split val"
    )


if __name__ == "__main__":
    main()
