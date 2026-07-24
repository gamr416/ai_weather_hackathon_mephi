# База знаний: Хакатон МИФИ — сжатие ERA5 AE

## С чего начать

1. [**Полное ТЗ**](task.md) ← главный документ
2. [Обзор](00_overview.md) — зачем и что строим
3. [28 каналов хакатона](01_weather_variables.md#набор-хакатона-28-каналов) — surface + уровни
4. [Данные / WeatherBench 2](02_datasets.md) — откуда качать
5. [Задачи и метрики](05_tasks_and_metrics.md) — codec + probe + критерии
6. [Инструменты](07_tools.md) — xarray, zarr, cartopy…
7. [Aurora](03_aurora.md) — что реально на 12 GB
8. [Perceiver / VAE](04_perceiver_vae.md) — идеи кодека
9. [Глоссарий](06_glossary.md)

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
