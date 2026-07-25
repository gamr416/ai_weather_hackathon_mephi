#!/usr/bin/env bash
set -euo pipefail
cd /workspace

cat <<'EOF'
╔══════════════════════════════════════════════════════════╗
║  MEPhI — Residual-FSQ A18+ ERA5 codec (expert)           ║
╚══════════════════════════════════════════════════════════╝

Checkpoint: submission/artifacts/best.pt  (19.11M, N=512)
Full-frame test S_all ≈ 0.0724 (CPU); crop160 ≈ 0.0605 (GPU)
Official CR ≈ ×38.7, exact roundtrip

Note: full-frame GPU may OOM on 12GB — use --device cpu for full grid.

  python scripts/residual_fsq/evaluate.py \
    --ckpt submission/artifacts/best.pt \
    --data data/era5_28ch_0p5_6h_n512.zarr \
    --stats submission/artifacts/norm_stats.npz \
    --sigma submission/ref_stats/sigma_official_28ch.npz \
    --split test --max-frames 8 --device cpu

Docs: README.md | submission/MANIFEST.json
EOF

exec bash
