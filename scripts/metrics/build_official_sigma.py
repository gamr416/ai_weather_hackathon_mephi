#!/usr/bin/env python3
"""Build data/ref_stats/sigma_official_28ch.npz from LadCast + train-pool gaps."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.era5 import CHANNEL_ORDER
from src.metrics.official_sigma import (
    DEFAULT_LADCAST_JSON,
    DEFAULT_SIGMA_PATH,
    build_official_sigma,
    save_official_sigma,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ladcast", type=Path, default=DEFAULT_LADCAST_JSON)
    p.add_argument(
        "--train-zarr",
        type=Path,
        default=ROOT / "data" / "era5_28ch_0p5_6h.zarr" / "train.zarr",
        help="train pool for tcc/tcwv (and any other gaps)",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_SIGMA_PATH)
    p.add_argument("--max-frames", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ladcast.exists():
        raise SystemExit(f"Missing LadCast JSON: {args.ladcast}")
    if not args.train_zarr.exists():
        raise SystemExit(f"Missing train zarr: {args.train_zarr}")

    payload = build_official_sigma(args.ladcast, args.train_zarr, max_frames=args.max_frames)
    save_official_sigma(payload, args.out)

    summary = {
        "out": str(args.out),
        "ladcast": str(args.ladcast),
        "train_zarr": str(args.train_zarr),
        "channels": {
            ch: {
                "mean": float(payload["mean"][i]),
                "std": float(payload["std"][i]),
                "source": str(payload["source"][i]),
            }
            for i, ch in enumerate(CHANNEL_ORDER)
        },
    }
    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"wrote": str(args.out), "meta": str(meta_path)}, indent=2))
    for i, ch in enumerate(CHANNEL_ORDER):
        print(f"  {ch:8s} σ={payload['std'][i]:.6g}  [{payload['source'][i]}]")


if __name__ == "__main__":
    main()
