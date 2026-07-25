# Хакатон МИФИ — ERA5 codec (Residual-FSQ)

Решение для экспертов: компактный автоэнкодер **28 каналов нижней атмосферы** на сетке **0.5°**, сжатие bitstream **×32–×64**, обучение на **одной GPU** и малых \(N\).

| | |
|---|---|
| Модель | `ResidualFSQAE` — coarse FSQ + fine residual FSQ |
| Params | **8.46 M** (лимит ≤20 M) |
| Сетка | 0.5° · 28×360×720 |
| Лучший ckpt | [`submission/artifacts/best.pt`](submission/artifacts/best.pt) (`n512s35k_ft`) |
| test \(S_\mathrm{all}\) | **0.0788** |
| Official CR | **×63.0** (pad) · exact index roundtrip |
| σ (NRMSE) | LadCast 1979–2017 (+ train-pool `tcc`/`tcwv`) |
| Probe +6h | latent MSE **2.37e-5** < persistence **3.91e-5** |
| Container | [`Dockerfile`](Dockerfile) / [`docker-compose.yml`](docker-compose.yml) |

Пакет: [`submission/`](submission/) · чеклист: [`docs/submission_notes.md`](docs/submission_notes.md) · ТЗ: [`docs/task.md`](docs/task.md).

---

## Чеклист сдачи (§7 ТЗ)

| # | Пункт | Где |
|---|--------|-----|
| 1 | Код + checkpoint | репозиторий + `submission/artifacts/best.pt` |
| 2 | Encoder / decoder | `src/models/residual_fsq.py` |
| 3 | Bitstream | `src/codec/bitstream.py` (zlib + pad CR≤63) |
| 4 | Манифест времён | `submission/manifest_times_n{128,256,512}.json` |
| 5 | JSON метрик | `submission/metrics/` |
| 6 | Затраты ресурсов | этот README + `MANIFEST.json` |
| 7 | Качество–данные | `submission/plots/quality_vs_N_*.png` |
| 8 | Качество–битрейт | `submission/plots/quality_vs_bitrate_*.png` |
| 9 | Container | `Dockerfile`, `docker-compose.yml` |
| 10 | Probe +6h | `submission/probe_6h/` |
| 11 | 0.25° | out of scope — `submission/SCOPE_0p25.md` |

---

## Docker (рекомендуемый путь для эксперта)

```bash
# сборка
docker build -t mephi-era5-residual-fsq:latest .

# интерактивно (нужен nvidia-container-toolkit для GPU)
docker run --rm -it --gpus all \
  -v "$PWD/data:/workspace/data:ro" \
  mephi-era5-residual-fsq:latest

# или compose
docker compose run --rm residual-fsq
```

Без GPU (только проверка ckpt / CPU eval на малом `--max-frames`):

```bash
docker run --rm -it mephi-era5-residual-fsq:latest \
  python -c "import torch; c=torch.load('submission/artifacts/best.pt',map_location='cpu',weights_only=False); print(c['nparams'], c.get('step'))"
```

Данные **не** входят в образ — смонтируйте локальный zarr в `/workspace/data`.

---

## Для экспертов (без Docker)

### 1. Окружение

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 2. Содержимое `submission/`

```
submission/
  MANIFEST.json
  SCOPE_0p25.md
  manifest_times_n128.json
  manifest_times_n256.json
  manifest_times_n512.json
  artifacts/          # best.pt, config.yaml, norm_stats.npz
  ref_stats/          # σ official (LadCast)
  metrics/            # val / test / bitstream JSON
  plots/              # NRMSE bars, N-curve, bitrate curve, compares
  probe_6h/           # latent +6h probe on final ckpt
```

### 3. Метрики

- NRMSE\(_f\) = RMSE\(_f^{\mathrm{phys}}\) / \(\sigma_f\) (LadCast), не train-std.
- Official CR = \(32\,T\,C\,H\,W/(8\,B)\); pad до ×63 если zlib жмёт сильнее.
- Заявлено (финал `n512s35k_ft`):

| Split | \(S_\mathrm{all}\) | \(S_\mathrm{surface}\) | \(S_\mathrm{pressure}\) |
|-------|--------------------|------------------------|-------------------------|
| val   | 0.0790             | 0.0793                 | 0.0787                  |
| test  | **0.0788**         | 0.0791                 | 0.0784                  |
| bitstream | CR **×63.0**, exact | | |

Data efficiency (test): N=128 → 0.131 · N=256 → 0.083 · **N=512 → 0.079**.

### 4. Пересчёт

```bash
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

python scripts/probe/latent_probe.py \
  --ckpt submission/artifacts/best.pt \
  --data data/era5_28ch_0p5_6h_n512.zarr \
  --split train --pairs 256 --steps 5000 \
  --out /tmp/probe_rerun
```

Скачивание данных (GCS; из РФ — proxy):

```bash
python scripts/download_era5_0p5.py --train-n 512 --val-n 128 --test-n 128 \
  --out data/era5_28ch_0p5_6h_n512.zarr
```

### 5. Метод (кратко)

1. Residual-FSQ: coarse + fine residual quantizers.  
2. Data efficiency: warm-start \(N=128\to256\to512\).  
3. Не сдаём VAEformer as-is (≤20 M). CRA5 — optional teacher only.  
4. Только 0.5°; 0.25° — см. `SCOPE_0p25.md`.  
5. Non-inferiority CI vs VAEformer не заполнен: нет их baseline на 28ch/0.5°.

---

## Структура репозитория

| Путь | Назначение |
|------|------------|
| `src/models/residual_fsq.py` | модель |
| `src/codec/bitstream.py` | bitstream + CR pad |
| `src/metrics/` | official σ, scores |
| `scripts/residual_fsq/` | train / evaluate / bitstream / plots |
| `scripts/probe/` | latent +6h |
| `Dockerfile` | контейнер для эксперта |
| `submission/` | артефакты сдачи |

## Ресурсы

- GPU: RTX 3060 12 GB  
- Trainable: 8.46 M  
- Финальный прогон: N=512, 35k steps (fine-tune после 30k)  
