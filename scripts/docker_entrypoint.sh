#!/usr/bin/env bash
set -euo pipefail
cd /workspace

cat <<'EOF'
╔══════════════════════════════════════════════════════════╗
║  MEPhI hackathon — Residual-FSQ ERA5 codec (expert)     ║
╚══════════════════════════════════════════════════════════╝

Bundled checkpoint: submission/artifacts/best.pt
Metrics / plots / σ / time manifests: submission/

Quick checks (no data mount needed):
  python -c "import torch; from pathlib import Path; import yaml; \
ckpt=torch.load('submission/artifacts/best.pt',map_location='cpu',weights_only=False); \
print('params', ckpt.get('nparams')); print('keys', list(ckpt.keys()))"

With data mounted at /workspace/data/..._n512.zarr:

  python scripts/residual_fsq/evaluate.py \
    --ckpt submission/artifacts/best.pt \
    --data data/era5_28ch_0p5_6h_n512.zarr \
    --stats submission/artifacts/norm_stats.npz \
    --sigma submission/ref_stats/sigma_official_28ch.npz \
    --split test --max-frames 32

  python scripts/residual_fsq/eval_bitstream.py \
    --ckpt submission/artifacts/best.pt \
    --data data/era5_28ch_0p5_6h_n512.zarr \
    --split val --max-frames 8

Docs: README.md  |  docs/submission_notes.md  |  submission/MANIFEST.json
EOF

exec bash
