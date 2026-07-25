#!/usr/bin/env python3
"""Plot quality–bitrate (official CR vs S_all) from Residual-FSQ evals."""
from __future__ import annotations

import argparse
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
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs" / "residual_fsq_0p5")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def infer_n(run_dir: Path) -> int | None:
    m = re.search(r"_n(\d+)", run_dir.name)
    return int(m.group(1)) if m else None


def main() -> None:
    args = parse_args()
    rows = []
    for run_dir in sorted(args.runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / f"eval_{args.split}" / "metrics.json"
        if not metrics_path.exists():
            continue
        # prefer padded CR≤63 eval if present
        bs = None
        for name in ("eval_bitstream_val_cr63", "eval_bitstream_val_cr64", "eval_bitstream_val"):
            p = run_dir / name / "metrics.json"
            if p.exists():
                bs = json.loads(p.read_text())
                break
        if bs is None:
            continue
        m = json.loads(metrics_path.read_text())
        n = infer_n(run_dir)
        rows.append(
            {
                "N": n,
                "run": run_dir.name,
                "S_all": m["S_all"],
                "S_surface": m["S_surface"],
                "S_pressure": m["S_pressure"],
                "CR": bs["mean_cr_official"],
                "exact": bs.get("all_exact"),
                "raw_cr": (bs["frames"][0].get("cr_raw") if bs.get("frames") else None),
            }
        )
    # best S_all per N
    best: dict[int, dict] = {}
    for r in rows:
        if r["N"] is None:
            continue
        prev = best.get(r["N"])
        if prev is None or r["S_all"] < prev["S_all"]:
            best[r["N"]] = r
    rows = [best[k] for k in sorted(best)]
    if not rows:
        raise SystemExit("no rows with both eval + bitstream metrics")

    out = args.out or (args.runs_root / "quality_vs_bitrate")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"quality_vs_bitrate_{args.split}.json").write_text(json.dumps(rows, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    crs = [r["CR"] for r in rows]
    salls = [r["S_all"] for r in rows]
    ax.plot(crs, salls, "o-", lw=2, color="#1f4e79")
    for r in rows:
        ax.annotate(f"N={r['N']}", (r["CR"], r["S_all"]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.axvspan(32, 64, color="green", alpha=0.08, label="allowed CR [32, 64]")
    ax.axvline(63, color="gray", ls="--", lw=1, label="pad target ×63")
    ax.set_xlabel("Official CR (bitstream)")
    ax.set_ylabel(r"$S_{\mathrm{all}}$ (NRMSE / $\sigma_{\mathrm{official}}$)")
    ax.set_title(f"Residual-FSQ quality–bitrate ({args.split})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    png = out / f"quality_vs_bitrate_{args.split}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(json.dumps({"rows": rows, "png": str(png)}, indent=2))


if __name__ == "__main__":
    main()
