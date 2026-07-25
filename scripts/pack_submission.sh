#!/usr/bin/env bash
# Pack the current best Residual-FSQ run into submission/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN="${1:-runs/residual_fsq_0p5/20260725_175622_n512s30k}"
SUB=submission

mkdir -p "$SUB/artifacts" "$SUB/metrics" "$SUB/plots" "$SUB/ref_stats"
cp -f "$RUN/best.pt" "$SUB/artifacts/best.pt"
cp -f "$RUN/config.yaml" "$SUB/artifacts/config.yaml"
cp -f "$RUN/norm_stats.npz" "$SUB/artifacts/norm_stats.npz"
cp -f "$RUN/eval_val/metrics.json" "$SUB/metrics/val_metrics.json"
cp -f "$RUN/eval_test/metrics.json" "$SUB/metrics/test_metrics.json"
if [[ -f "$RUN/eval_bitstream_val_cr63/metrics.json" ]]; then
  cp -f "$RUN/eval_bitstream_val_cr63/metrics.json" "$SUB/metrics/bitstream_val.json"
elif [[ -f "$RUN/eval_bitstream_val/metrics.json" ]]; then
  cp -f "$RUN/eval_bitstream_val/metrics.json" "$SUB/metrics/bitstream_val.json"
fi
cp -f "$RUN/eval_val/nrmse_bars.png" "$SUB/plots/" 2>/dev/null || true
cp -f "$RUN/eval_val/train_curves.png" "$SUB/plots/" 2>/dev/null || true
cp -f data/ref_stats/sigma_official_28ch.npz "$SUB/ref_stats/" 2>/dev/null || true
cp -f data/ref_stats/ERA5_normal_1979_2017.json "$SUB/ref_stats/" 2>/dev/null || true
echo "Packed $RUN → $SUB"
ls -lh "$SUB/artifacts/best.pt"
