# Submission notes (для эксперта)

Короткий чеклист. Пакет: [`submission/`](../submission/) · инструкция: [`README.md`](../README.md).

## Формат оценки (уточнения организаторов)

- Устной защиты нет: эксперт смотрит решение в чате.
- Задача исследовательская: методология, воспроизводимость, оригинальность.
- Shortcut pretrained LadCast/VAEformer допустим; мы сдаём **свой ≤20 M** Residual-FSQ.
- NRMSE: RMSE в физ. единицах / \(\sigma_f\); \(\sigma\) = LadCast (или CDS). У нас: `submission/ref_stats/sigma_official_28ch.npz` из `ERA5_normal_1979_2017.json` (+ train-pool для `tcc`/`tcwv`).

## Позиция

| Решение | Почему |
|---------|--------|
| Residual-FSQ **8.46 M** | лимит ≤20 M; VAEformer ~405 M не сдаём as-is |
| Только **0.5°** | compute / 12 GB GPU |
| CRA5 — teacher only (не в финальном графе) | declared |
| Bitstream pad до CR ≤ **×63** | zlib иначе даёт ~×66; pad сохраняет exact roundtrip |
| Probe +6h | latent MSE vs persistence |

## Финальный артефакт

| Поле | Значение |
|------|----------|
| Model | `ResidualFSQAE` |
| Params | **8.46 M** |
| Ckpt | `submission/artifacts/best.pt` |
| Source run | `runs/residual_fsq_0p5/20260725_175622_n512s30k` |
| Data | N=512, 30k steps (warm-start chain n128→n256→n512) |
| val \(S_\mathrm{all}\) | **0.0812** |
| test \(S_\mathrm{all}\) | **0.0810** |
| Official CR | **×63.0**, exact roundtrip |
| σ | `submission/ref_stats/sigma_official_28ch.npz` |

### Data-efficiency (test \(S_\mathrm{all}\))

| N | run | \(S_\mathrm{all}\) |
|---|-----|-------------------|
| 128 | n128s9k | 0.131 |
| 256 | n256s25k | 0.091 |
| 256 | n256s50k_ft | 0.083 |
| **512** | **n512s30k** | **0.081** |

## Reproduce

```bash
source .venv/bin/activate

python scripts/residual_fsq/evaluate.py \
  --ckpt submission/artifacts/best.pt \
  --data data/era5_28ch_0p5_6h_n512.zarr \
  --stats submission/artifacts/norm_stats.npz \
  --sigma submission/ref_stats/sigma_official_28ch.npz \
  --split test --max-frames 128

python scripts/residual_fsq/eval_bitstream.py \
  --ckpt submission/artifacts/best.pt \
  --data data/era5_28ch_0p5_6h_n512.zarr \
  --split val --max-frames 32
```

## Не сдаём

- VAEformer / LadCast DC-AE как основную модель  
- 0.25° runs  
- Незадекларированное weather-pretraining  
- Формальный CI non-inferiority vs VAEformer (нет их baseline на 28ch/0.5°)  
