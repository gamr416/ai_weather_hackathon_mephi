# База знаний: Хакатон МИФИ — сжатие ERA5 AE

## С чего начать

1. [**Полное ТЗ**](task.md) ← главный документ
2. [**Скачать данные**](../README.md#данные-05) — скрипт 0.5°, proxy, N=512
3. [Обзор](00_overview.md) — зачем и что строим
4. [28 каналов хакатона](01_weather_variables.md#набор-хакатона-28-каналов) — surface + уровни
5. [Данные / WeatherBench 2](02_datasets.md) — откуда качать
6. [Задачи и метрики](05_tasks_and_metrics.md) — codec + probe + критерии
7. [Инструменты](07_tools.md) — xarray, zarr, cartopy…
8. [Aurora](03_aurora.md) — что реально на 12 GB
9. [Perceiver / VAE](04_perceiver_vae.md) — идеи кодека
10. [Глоссарий](06_glossary.md)
11. [**Чеклист сдачи / позиция команды**](submission_notes.md) ← эксперт без защиты
12. [**Методология**](methodology.md) ← архитектура, \(N\)-curriculum, результаты, gaps
13. [External models](external_models.md) — CRA5 / pretrained shortcut

## Окружение

```bash
source .venv/bin/activate   # Python 3.11 + torch cu128
```

`requirements.txt` / `requirements.lock.txt` в корне. `.venv/` не в git (~7–8 GB).

## Карта файлов

| Файл | Роль |
|------|------|
| [`task.md`](task.md) | **Полное ТЗ** (систематизировано) |
| [`real_task.md`](real_task.md) | Сырой дамп официальной концепции (PDF→текст) |
| [`more_info.md`](more_info.md) | Сырой FAQ + пост про инструменты |
| [`additional_info.md`](additional_info.md) | Ранний дамп (Aurora / параметры / датасеты) |
| `00`–`07_*.md` | Разложенная база знаний |

## В одном абзаце

На **одной GPU (≤24 GB)** и **малых выборках** \(N\in\{128\ldots8192\}\) уникальных 6h кадров обучить **AE с квантованным bottleneck** на **28 каналах** нижней атмосферы (8 surface + T/U/V/Z/Q на 1000/925/850/700 hPa), сетки **0.5° и 0.25°**, сжатие **×32–×64** по bitstream, качество **не хуже VAEformer** по формальным CI-критериям, плюс **latent +6h probe** на замороженном энкодере. Главный результат — карта **data efficiency**.

- [methodology.md](methodology.md) — **методология для эксперта**
- [submission_notes.md](submission_notes.md) — сдача эксперту, только 0.5°, почему не VAEformer as-is
- [external_models.md](external_models.md) — CRA5 teacher + pretrained shortcut
- Official NRMSE σ: [`data/ref_stats/SOURCE.md`](../data/ref_stats/SOURCE.md)

### Residual-FSQ: σ → eval → N-sweep

```bash
python scripts/metrics/build_official_sigma.py
python scripts/residual_fsq/evaluate.py \
  --ckpt runs/residual_fsq_0p5/<run>/best.pt \
  --data data/era5_28ch_0p5_6h_n128.zarr --split val
# NS="128 256" STEPS=5000 bash scripts/residual_fsq/n_sweep.sh
python scripts/residual_fsq/plot_n_curve.py --split val
```
