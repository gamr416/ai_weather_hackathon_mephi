#!/usr/bin/env bash
# Data-efficiency N-sweep for Residual-FSQ (+ auto-eval with official σ).
# Uses only hackathon ERA5 frames; teacher cache must match the same N root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

CONFIG="${CONFIG:-configs/residual_fsq_0p5.yaml}"
STEPS="${STEPS:-5000}"
NS="${NS:-128 256}"
SPLIT="${SPLIT:-val}"
MAX_FRAMES="${MAX_FRAMES:-32}"
SIGMA="${SIGMA:-data/ref_stats/sigma_official_28ch.npz}"

if [[ ! -f "$SIGMA" ]]; then
  echo "Building official sigma → $SIGMA"
  python scripts/metrics/build_official_sigma.py --out "$SIGMA"
fi

CKPTS=()
for N in $NS; do
  DATA="data/era5_28ch_0p5_6h_n${N}.zarr"
  if [[ ! -d "$DATA/train.zarr" ]]; then
    echo "skip N=$N — missing $DATA"
    continue
  fi
  echo "=== train N=$N steps=$STEPS ==="
  python scripts/residual_fsq/train.py \
    --config "$CONFIG" \
    --data "$DATA" \
    --steps "$STEPS" \
    --run-name "n${N}s${STEPS}"

  # newest run matching this tag
  RUN_DIR="$(ls -dt runs/residual_fsq_0p5/*_n${N}s${STEPS} 2>/dev/null | head -1 || true)"
  if [[ -z "${RUN_DIR}" || ! -f "${RUN_DIR}/best.pt" ]]; then
    echo "warn: no best.pt for N=$N"
    continue
  fi
  echo "=== eval N=$N → $RUN_DIR ==="
  python scripts/residual_fsq/evaluate.py \
    --ckpt "${RUN_DIR}/best.pt" \
    --data "$DATA" \
    --split "$SPLIT" \
    --sigma "$SIGMA" \
    --max-frames "$MAX_FRAMES"
  CKPTS+=("${RUN_DIR}/best.pt")
done

echo "=== plot quality vs N ==="
python scripts/residual_fsq/plot_n_curve.py --split "$SPLIT"

echo "N-sweep finished. ckpts:"
printf '  %s\n' "${CKPTS[@]:-}"
