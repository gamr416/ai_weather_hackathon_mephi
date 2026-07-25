# Хакатон МИФИ — ERA5 codec (Residual-FSQ)

Решение для экспертов: компактный автоэнкодер **28 каналов нижней атмосферы** на сетке **0.5°**, сжатие bitstream **×32–×64**, обучение на **одной GPU** и малых \(N\).

| | |
|---|---|
| Модель | `ResidualFSQAE` — coarse FSQ + fine residual FSQ |
| Params | **8.46 M** (лимит ≤20 M) |
| Сетка | 0.5° · 28×360×720 |
| Лучший ckpt | [`submission/artifacts/best.pt`](submission/artifacts/best.pt) |
| test \(S_\mathrm{all}\) | **0.081** |
| Official CR | **×63.0** (pad ≤×63) · exact index roundtrip |
| σ (NRMSE) | LadCast 1979–2017 (+ train-pool `tcc`/`tcwv`) |

Пакет сдачи: [`submission/`](submission/) · чеклист: [`docs/submission_notes.md`](docs/submission_notes.md) · ТЗ: [`docs/task.md`](docs/task.md).

---

## Для экспертов (быстрый путь)

### 1. Окружение

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 2. Что лежит в `submission/`

```
submission/
  MANIFEST.json              # сводка: метрики, пути, ресурсы
  artifacts/
    best.pt                  # финальный checkpoint (~33 MB)
    config.yaml              # архитектура / train cfg
    norm_stats.npz           # train mean/std (нормализация входа)
  ref_stats/
    sigma_official_28ch.npz  # знаменатель NRMSE (LadCast)
    ERA5_normal_1979_2017.json
  metrics/
    val_metrics.json
    test_metrics.json
    bitstream_val.json       # CR + exact roundtrip
  plots/
    nrmse_bars.png
    train_curves.png
    compare_frame0000.png
```

### 3. Метрики (как в ТЗ)

- **NRMSE** по каждому полю: \(\mathrm{RMSE}_f^{\mathrm{phys}} / \sigma_f\), где \(\sigma\) из LadCast  
  (`submission/ref_stats/sigma_official_28ch.npz`), **не** из мини-train.
- Агрегаты: \(S_\mathrm{surface}\), \(S_\mathrm{pressure}\), \(S_\mathrm{all}\).
- **Official CR** = \(32\,T\,C\,H\,W / (8\,B)\) по zlib-bitstream индексов FSQ.  
  Если zlib жмёт сильнее ×64, encoder **паддит** bitstream до CR ≤ **×63** (запас до лимита); decode читает `payload_len` из header → exact roundtrip сохраняется.

Заявленные цифры (N=512, run `n512s30k`):

| Split | \(S_\mathrm{all}\) | \(S_\mathrm{surface}\) | \(S_\mathrm{pressure}\) |
|-------|--------------------|------------------------|-------------------------|
| val   | 0.0812             | 0.0821                 | 0.0802                  |
| test  | **0.0810**         | 0.0820                 | 0.0799                  |
| bitstream val | official CR **×63.0**, `all_exact=true` | | |

### 4. Пересчёт метрик (нужен val/test zarr)

Данные WeatherBench 2 ERA5 0.5° (не в git; ~GB). Пример:

```bash
# скачать test/val (нужен доступ к GCS; из РФ — proxy)
python scripts/download_era5_0p5.py --train-n 512 --val-n 128 --test-n 128 \
  --out data/era5_28ch_0p5_6h_n512.zarr

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

Encode / decode одного кадра:

```python
import torch
from src.models.residual_fsq import ResidualFSQAE
# см. scripts/residual_fsq/evaluate.py — build_model + model.compress / decompress
```

### 5. Метод (кратко)

1. **Residual-FSQ**: coarse patch-FSQ (глобальная структура) + fine residual-FSQ (детали ветра/влаги/осадков).
2. **Data efficiency**: цепочка warm-start \(N=128\to256\to512\), без weather-pretraining на сторонних архивах.
3. **Не сдаём** VAEformer (~405 M) as-is: лимит ≤20 M trainable; CRA5 только как optional teacher (не в финальном графе).
4. Scope: **только 0.5°** (12 GB GPU / сутки хакатона). 0.25° — out of scope.
5. Latent +6h probe: `scripts/probe/latent_probe.py` (MSE в latent vs persistence).

Non-inferiority CI vs VAEformer (таблица 3 ТЗ) **не заполнена**: нет официального baseline VAEformer на тех же 28ch/0.5°/том же тесте. NRMSE нормируется на климатологическое σ (LadCast), как уточнили организаторы.

### 6. Воспроизведение обучения

```bash
python scripts/residual_fsq/train.py \
  --config configs/residual_fsq_0p5_n512_s30k.yaml \
  --data data/era5_28ch_0p5_6h_n512.zarr \
  --steps 30000 \
  --init-ckpt <prev_best.pt>
```

Перепаковка `submission/` после нового eval:

```bash
bash scripts/pack_submission.sh runs/residual_fsq_0p5/<run>
```

---

## Структура репозитория

| Путь | Назначение |
|------|------------|
| `src/models/residual_fsq.py` | модель |
| `src/codec/bitstream.py` | zlib bitstream + pad CR≤63 |
| `src/metrics/` | official σ, scores |
| `scripts/residual_fsq/` | train / evaluate / bitstream |
| `scripts/probe/` | latent +6h probe |
| `configs/` | yaml прогонов |
| `docs/` | ТЗ и методология |
| `submission/` | артефакты для эксперта |

---

## Ресурсы

- GPU: RTX 3060 12 GB  
- Trainable: 8.46 M  
- Бюджет: один GPU, порядок \(10^4\)–\(5\cdot10^4\) шагов на финальных прогонах  
