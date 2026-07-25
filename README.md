# Хакатон МИФИ — ERA5 codec (Residual-FSQ A18+)

| | |
|---|---|
| Модель | `ResidualFSQAE` **A18+** |
| Params | **19.11 M** (лимит ≤20 M) |
| Сетка | 0.5° · 28×360×720 |
| Ckpt | [`submission/artifacts/best.pt`](submission/artifacts/best.pt) |
| test \(S_\mathrm{all}\) full-frame | **0.0724** (CPU, 32 кадра) |
| test \(S_\mathrm{all}\) crop160 | **0.0605** (GPU, 128 кадров) |
| Official CR | **×38.7** · exact roundtrip |
| σ | LadCast 1979–2017 (+ train-pool `tcc`/`tcwv`) |

Предыдущий 8.46M (`n512s35k_ft`): full test \(S_\mathrm{all}\) **0.0788**. A18 лучше по качеству при raw CR≈×34.5.

Пакет: [`submission/`](submission/) · чеклист: [`docs/submission_notes.md`](docs/submission_notes.md) · ТЗ: [`docs/task.md`](docs/task.md).

**Методология (для эксперта):** [`docs/methodology.md`](docs/methodology.md) — архитектура, curriculum \(N\), loss/CR, результаты, ограничения.

---

## Важно для эксперта (VRAM)

На **12 GB** полный кадр 360×720 с `fine_patch=4` даёт OOM (attention).  
**Заявленные full-frame метрики** посчитаны на **CPU**. Crop-160 — быстрее на GPU и занижает ошибку относительно full-frame.

```bash
# full-frame (CPU)
python scripts/residual_fsq/evaluate.py \
  --ckpt submission/artifacts/best.pt \
  --data data/era5_28ch_0p5_6h_n512.zarr \
  --stats submission/artifacts/norm_stats.npz \
  --sigma submission/ref_stats/sigma_official_28ch.npz \
  --split test --max-frames 32 --device cpu

# crop GPU
python scripts/residual_fsq/evaluate.py \
  --ckpt submission/artifacts/best.pt \
  --data data/era5_28ch_0p5_6h_n512.zarr \
  --stats submission/artifacts/norm_stats.npz \
  --sigma submission/ref_stats/sigma_official_28ch.npz \
  --split test --max-frames 128 --crop
```

---

## Docker

```bash
docker build -t mephi-era5-residual-fsq:latest .
docker run --rm -it --gpus all \
  -v "$PWD/data:/workspace/data:ro" \
  mephi-era5-residual-fsq:latest
```

---

## Чеклист сдачи

| Пункт | Где |
|--------|-----|
| Код + ckpt | repo + `submission/artifacts/best.pt` |
| Bitstream | `src/codec/bitstream.py` · CR×38.7 exact |
| Манифест времён | `submission/manifest_times_n*.json` |
| Метрики JSON | `submission/metrics/` (+ crop в `metrics_crop/`) |
| N-curve / bitrate | `submission/plots/` |
| Container | `Dockerfile` |
| 0.25° | out of scope — `SCOPE_0p25.md` |

---

## Окружение без Docker

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Данные: `python scripts/download_era5_0p5.py --train-n 512 --val-n 128 --test-n 128 --out data/era5_28ch_0p5_6h_n512.zarr`

## Ресурсы

RTX 3060 12 GB · 19.11 M params · N=512 · 40k steps · 0.5° only  
