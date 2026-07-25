#!/usr/bin/env python3
"""Plot quality–N curve from Residual-FSQ eval metrics.json files."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "runs" / "residual_fsq_0p5",
        help="directory with run folders",
    )
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def infer_n(run_dir: Path, metrics: dict) -> int | None:
    # Prefer run name like ..._n128s9k or config data path
    m = re.search(r"_n(\d+)", run_dir.name)
    if m:
        return int(m.group(1))
    ckpt = metrics.get("ckpt", "")
    m = re.search(r"_n(\d+)", ckpt)
    if m:
        return int(m.group(1))
    cfg = run_dir / "config.yaml"
    if cfg.exists():
        text = cfg.read_text()
        m = re.search(r"n(\d+)\.zarr", text)
        if m:
            return int(m.group(1))
    return None


def collect_rows(runs_root: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / f"eval_{split}" / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        n = infer_n(run_dir, metrics)
        if n is None:
            continue
        rows.append(
            {
                "N": n,
                "run": run_dir.name,
                "S_all": metrics["S_all"],
                "S_surface": metrics["S_surface"],
                "S_pressure": metrics["S_pressure"],
                "n_frames": metrics.get("n_frames"),
                "sigma_path": metrics.get("sigma_path"),
                "metrics_path": str(metrics_path),
            }
        )
    # keep best (lowest S_all) per N
    best: dict[int, dict] = {}
    for r in rows:
        prev = best.get(r["N"])
        if prev is None or r["S_all"] < prev["S_all"]:
            best[r["N"]] = r
    return [best[k] for k in sorted(best)]


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.runs_root, args.split)
    if not rows:
        raise SystemExit(f"No eval_{args.split}/metrics.json under {args.runs_root}")

    out_dir = args.out or (args.runs_root / "quality_vs_N")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"quality_vs_N_{args.split}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["N", "S_all", "S_surface", "S_pressure", "n_frames", "run", "metrics_path"],
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    ns = [r["N"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(ns, [r["S_all"] for r in rows], "o-", label="S_all", lw=2)
    ax.plot(ns, [r["S_surface"] for r in rows], "s--", label="S_surface", lw=1.5)
    ax.plot(ns, [r["S_pressure"] for r in rows], "^--", label="S_pressure", lw=1.5)
    ax.set_xlabel("N train frames")
    ax.set_ylabel("NRMSE (RMSE_phys / σ_official)")
    ax.set_title(f"Residual-FSQ data efficiency ({args.split})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    png_path = out_dir / f"quality_vs_N_{args.split}.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    (out_dir / f"quality_vs_N_{args.split}.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"rows": rows, "csv": str(csv_path), "png": str(png_path)}, indent=2))


if __name__ == "__main__":
    main()
