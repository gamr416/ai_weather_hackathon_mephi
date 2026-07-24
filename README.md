# Хакатон МИФИ — сжатие ERA5 (AE)

Компактное 28-канальное представление нижней атмосферы: сжатие **×32–×64**, обучение на **одной GPU** и малых выборках \(N\).

Полное ТЗ и база знаний: [`docs/`](docs/README.md) → [`docs/task.md`](docs/task.md).

## Окружение

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # если нужна CUDA
pip install -r requirements.txt
pip install gcsfs   # доступ к WeatherBench 2 / GCS
```

## Данные (0.5°)

Источник: WeatherBench 2 ERA5 Zarr (`gs://weatherbench2/...`).  
Скрипт: [`scripts/download_era5_0p5.py`](scripts/download_era5_0p5.py).

### Сплиты (по ТЗ)

| Сплит | Годы | По умолчанию |
|-------|------|--------------|
| train | 2014–2019 | 512 кадров |
| val | 2020 | 128 |
| test | 2021 | 128 |

Сезонно сбалансированная выборка, сетка **0.5°** (360×720), 28 каналов.

### GCS из РФ

Бакет Google часто недоступен напрямую. Нужен VPN/proxy, затем:

```bash
proxyon          # http://127.0.0.1:12334 (Hiddify и т.п.)
source .venv/bin/activate
python scripts/download_era5_0p5.py --train-n 512 --val-n 128 --test-n 128
```

Скрипт сам подставляет proxy `127.0.0.1:12334`, если переменные окружения пустые.  
Важно: `gcsfs` ходит через proxy только при `session_kwargs={"trust_env": True}` (уже в скрипте).

### Результат

```
data/era5_28ch_0p5_6h.zarr/
  train.zarr / val.zarr / test.zarr / static.zarr
  manifest.json
```

Форма кадра: `(time, channel=28, lat=360, lon=720)`.  
Ориентир по месту: **~17 GiB** для 512+128+128. Каталог `data/` в git не коммитится.

Проверка:

```python
import xarray as xr
ds = xr.open_zarr("data/era5_28ch_0p5_6h.zarr/train.zarr")
print(ds.fields.shape)  # (512, 28, 360, 720)
```

## Обучение (Perceiver-AE)

После того как zarr готов (`train.zarr` внутри `data/...`):

```bash
source .venv/bin/activate
# короткий прогон
python scripts/train.py --config configs/perceiver_0p5.yaml --data data/era5_28ch_0p5_6h_n64.zarr --steps 100 --run-name smoke

# обычный
python scripts/train.py --config configs/perceiver_0p5.yaml --data data/era5_28ch_0p5_6h_n64.zarr --run-name n64
```

Каждый прогон → новая папка `runs/perceiver_0p5/<YYYYMMDD_HHMMSS>_<tag>/` (старые не трогаются). Внутри: `config.yaml`, `run_meta.json`, `best.pt`, `norm_stats.npz`, …

Что внутри:
- `src/models/perceiver_ae.py` — Perceiver-IO encode/decode + VQ bottleneck  
- `src/data/era5.py` — ленивый loader zarr, lat-weighted loss, random crop  
- `scripts/train.py` — AdamW, AMP, checkpoint в `runs/`

Лимит: модель должна быть ≤20M params (проверяется при старте).  
Пока это **скелет кодека** (VQ); полноценный entropy bitstream под CR ×32/×64 — следующий шаг.

### Оценка и графики

После обучения (по умолчанию **full-frame** 360×720), подставь путь конкретного прогона:

```bash
python scripts/evaluate.py \
  --ckpt runs/perceiver_0p5/20260724_211500_n64/best.pt \
  --data data/era5_28ch_0p5_6h_n64.zarr \
  --split val
```

Для eval на training-crop: добавь `--crop`.

В каталог `runs/.../eval_val/` пишется:
- `metrics.json` — \(S_\mathrm{all}\), surface/pressure, NRMSE (= rmse в norm-space) + physical RMSE  
- `nrmse_bars.png`, `compare_frame*.png`, `train_curves.png`

Checkpoint `best.pt` выбирается по **val_recon** (не по VQ).  
Сравнение с VAEformer CI — когда появится их baseline/evaluator.




- 1 GPU ≤ 24 GB VRAM, ≤ 20M params, ≤ 50k шагов  
- CR по bitstream ×32/×64 + exact roundtrip  
- + latent probe +6h на замороженном encoder  
