# Submission notes (для эксперта)

Пакет: [`submission/`](../submission/) · инструкция: [`README.md`](../README.md) · Docker: [`Dockerfile`](../Dockerfile).

## Формат оценки

- Устной защиты нет.
- NRMSE: RMSE_phys / \(\sigma_f\); σ = LadCast (`submission/ref_stats/`).
- Shortcut pretrained VAEformer допустим; мы сдаём **свой ≤20 M** Residual-FSQ.

## Финальный артефакт (`n512s35k_ft`)

| Поле | Значение |
|------|----------|
| Model | `ResidualFSQAE` **8.46 M** |
| Ckpt | `submission/artifacts/best.pt` |
| Source run | `runs/residual_fsq_0p5/20260725_192609_n512s35k_ft` |
| val / test \(S_\mathrm{all}\) | **0.0790 / 0.0788** |
| Official CR | **×63.0**, exact roundtrip |
| Probe +6h | latent MSE 2.37e-5 < persist 3.91e-5 (`submission/probe_6h/`) |

### Data efficiency (test \(S_\mathrm{all}\))

| N | \(S_\mathrm{all}\) |
|---|-------------------|
| 128 | 0.131 |
| 256 | 0.083 |
| **512** | **0.079** |

Plots: `submission/plots/quality_vs_N_*.png`, `quality_vs_bitrate_*.png`.  
Time manifests: `submission/manifest_times_n*.json`.  
0.25°: `submission/SCOPE_0p25.md` (out of scope).

## Docker

```bash
docker build -t mephi-era5-residual-fsq:latest .
docker run --rm -it --gpus all -v "$PWD/data:/workspace/data:ro" mephi-era5-residual-fsq:latest
```

## Reproduce eval

```bash
source .venv/bin/activate
python scripts/residual_fsq/evaluate.py \
  --ckpt submission/artifacts/best.pt \
  --data data/era5_28ch_0p5_6h_n512.zarr \
  --stats submission/artifacts/norm_stats.npz \
  --sigma submission/ref_stats/sigma_official_28ch.npz \
  --split test --max-frames 128
```

## Не сдаём

- VAEformer as-is · 0.25° runs · undeclared weather-pretraining · CI non-inferiority без baseline организаторов  
