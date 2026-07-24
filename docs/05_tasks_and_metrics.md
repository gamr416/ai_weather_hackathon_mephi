# Задачи и метрики (хакатон)

Полные формулы и таблица допуска — в [task.md](task.md) §4–§6. Здесь — рабочая шпаргалка.

---

## Задача 1: кодек

- Автоэнкодер, **квантованный** bottleneck.
- \(\mathrm{CR} = 32\,T\,C\,H\,W / (8B)\), \(C=28\), \(B\) — байты bitstream.
- Exact roundtrip квантованных символов обязателен.
- Без entropy coding → только «latent reduction», не codec-зачёт.

### Ось исследования (data efficiency)

Вложенные \(N \in \{128, 256, 512, 1024, 2048, 4096, 8192\}\) уникальных 6h кадров  
→ кривые **качество–данные** и **качество–битрейт** на 0.5° и 0.25°.

Лимиты: ≤20M params, ≤50k steps, ≤48 GPU-h, 1 GPU ≤24 GB.

---

## Задача 2: latent probe

| Параметр | Значение |
|----------|----------|
| Encoder/decoder | **заморожены** |
| Probe-модель | ≤ **2M** params |
| Train pairs | ровно **1024** |
| Steps | ≤ **5000** |
| Цель | latent\(_{t}\) → latent\(_{t+6h}\) → decode |

Сравнение: vs persistence; vs тот же probe на признаках VAEformer-референса.

---

## Scores

| Score | Что усредняет |
|-------|----------------|
| \(S_\mathrm{surface}\) | NRMSE по 8 surface (равные веса) |
| \(S_\mathrm{pressure}\) | NRMSE по 20 (var×level) |
| \(S_\mathrm{all}\) | \(0.5\,S_\mathrm{surface}+0.5\,S_\mathrm{pressure}\) |

База: **latitude-weighted RMSE**, затем NRMSE = RMSE / \(\sigma_{f,\mathrm{train}}\).

Ещё: физический RMSE, PSNR (0.5–99.5 pct), спектры, экстремумы осадков.

---

## Non-inferiority (допуск)

\(\Delta = 100(\mathrm{NRMSE}_\mathrm{model}/\mathrm{NRMSE}_\mathrm{ref}-1)\);  
95% CI — block bootstrap (7 дней, 2000×).

| | 32× | 64× |
|--|-----|-----|
| CI верх \(S_\mathrm{all}\) | ≤ +3% | ≤ +7% |
| CI верх surface/pressure | ≤ +5% | ≤ +8% |
| CI верх критическое поле | ≤ +7% | ≤ +10% |
| Δ PSNR низ | ≥ −0.25 dB | ≥ −0.5 dB |
| Spectral error | ≤ 5% | ≤ 10% |
| CR + roundtrip | да | да |

---

## Практичный план на 12 GB

1. Подготовка локального 28ch Zarr (срез N, начать с 0.5°).
2. Лёгкий AE ≤20M + quantization + (по возможности) entropy coding.
3. Sweep по \(N\) при целевых ×32 и ×64.
4. Probe на лучшем checkpoint.
5. Сравнение с референсом VAEformer по evaluator организаторов (когда будет).
