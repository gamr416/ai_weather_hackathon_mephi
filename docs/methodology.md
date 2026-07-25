# Методология

Документ для эксперта: **что сделали, почему так, что получилось, какие ограничения**.  
Артефакты: [`submission/`](../submission/) · воспроизведение: [`README.md`](../README.md).

---

## 1. Постановка и scope

Хакатон требует AE с квантованным bottleneck на **28 каналах** нижней атмосферы ERA5, CR **×32–×64** по bitstream, exact roundtrip индексов, плюс исследование **data efficiency** при малом \(N\) на **одной GPU ≤24 GB** и **≤20 M** trainable params.

| Выбор | Решение | Мотив |
|-------|---------|--------|
| Сетка | **только 0.5°** (360×720) | 12 GB VRAM + сутки; орги: «0.5 **или** 0.25» |
| Модель | свой **Residual-FSQ**, не VAEformer as-is | VAEformer ~405 M; лимит 20 M; другие каналы/CR |
| Pretraining | нет незадекларированного weather-pretrain | \(N\)-кривые считают только хакатонный ERA5 |
| CRA5 | optional teacher only | distill smoke не улучшил; primary без teacher |
| σ для NRMSE | LadCast 1979–2017 (+ train-pool `tcc`/`tcwv`) | как разрешили организаторы |

Исследовательский вопрос:

> Сколько уникальных 6h кадров нужно, чтобы на 1×3060 выучить полезный 28ch кодек с official CR ∈ [32, 64]?

---

## 2. Архитектура: Residual-FSQ

### 2.1 Идея

Один Spatial-FSQ либо слишком грубый, либо дорогой по битрейту. Residual-FSQ разделяет шкалы:

1. **Coarse branch** — patch 8, FSQ на низкочастотной структуре (геопотенциал, крупный T, MSLP).  
2. **Fine residual branch** — patch 4, FSQ на остатке (ветер, Q, tp6h, детали).  
3. **CNN stem / refine** + static conditioning (orography, land–sea).  
4. **Uniform FSQ** (finite scalar quantization) → целые индексы → lossless zlib bitstream.

Exact roundtrip: encode indices → bytes → decode indices → тот же quantised latent.

### 2.2 Две рабочие точки

| Вариант | Params | raw CR | роль |
|---------|--------|--------|------|
| Baseline Residual-FSQ | **8.46 M** | ×56 | data-efficiency sweep, стабильный full-frame GPU |
| **A18+** (сдача) | **19.11 M** | ×34.5 | больше ёмкости + бит; ближе к лимиту 20 M |

A18+: шире stem/transformer, `coarse_latent=112`, `fine_latent=32` → больше символов на кадр, official CR после zlib ≈ **×39** (в зоне [32, 64], pad до ×63 не нужен).

Код: `src/models/residual_fsq.py`, bitstream: `src/codec/bitstream.py`.

### 2.3 Почему не сдали pretrained wrap

Легальный shortcut (замороженный LadCast/VAEformer + маленький адаптер) возможен, но:

- ломает сравнимость \(N\)-кривых «обучение с хакатонных кадров»;
- VAEformer не drop-in под 28ch @ 0.5° и CR ×32–64;
- вклад команды сводится к обвязке.

Мы выбрали **обучаемый ≤20 M кодек** + явная карта data efficiency — ближе к заявленной оси исследования ТЗ.

---

## 3. Данные и сплиты

- Источник: WeatherBench 2 ERA5 6h Zarr.  
- Train **2014–2019**, val **2020**, test **2021**.  
- Сезонно сбалансированная подвыборка уникальных времён (seed фиксирован).  
- Манифесты: `submission/manifest_times_n{128,256,512}.json`.  
- Нормализация входа: train mean/std (`norm_stats.npz`).  
- **NRMSE** при оценке: \(\mathrm{RMSE}_f^{\mathrm{phys}} / \sigma_f\), \(\sigma\) из LadCast (не из мини-train).

---

## 4. Обучение

### 4.1 Curriculum data efficiency

Warm-start по \(N\), без смены семейства модели (для 8.46 M линии):

\[
N=128 \;\xrightarrow{\text{train}}\; N=256 \;\xrightarrow{\text{FT}}\; N=512
\]

Типичные шаги: ~9k → 25k → +50k FT → +30k/+35k на N=512.  
A18+ обучался **с нуля** на N=512 (40k steps) — другая ширина, веса 8.46 M не переносятся 1:1.

### 4.2 Loss

\[
\mathcal{L} =
\underbrace{w_c\cdot\mathrm{MSE}_{\varphi}}_{\text{recon}}
+ \lambda_{\mathrm{rate}}\cdot\frac{R}{R_{\mathrm{in}}}
+ \lambda_{\mathrm{commit}}\cdot\mathrm{commit}
+ \lambda_{\mathrm{hf}}\cdot\mathrm{HF}
+ \lambda_{\mathrm{sst}}\cdot\mathrm{SST\text{-}land}
\]

- latitude-weighted MSE, усиленные веса на **tp6h, sst, tcc, ветра, Q**;  
- rate warmup;  
- high-frequency gradient penalty;  
- штраф SST на суше;  
- static в decoder.

Оптимизатор: AdamW, cosine/warmup LR, grad accum, AMP, crop 160–192 при train.

### 4.3 Bitstream / CR

Official:

\[
\mathrm{CR} = \frac{32\,T\,C\,H\,W}{8\,B}
\]

Индексы FSQ → pack → **zlib**. Если zlib даёт CR > 64 (слишком хорошо жмёт), encoder **паддит** bitstream до CR ≤ **63** (decode читает `payload_len`). У A18+ raw/official ≈ ×35/×39 — pad обычно не срабатывает.

---

## 5. Оценка

| Протокол | Назначение |
|----------|------------|
| Full-frame 360×720 | основной score (\(S_\mathrm{surface}\), \(S_\mathrm{pressure}\), \(S_\mathrm{all}\)) |
| Crop-160 | диагностика / train-proxy (на A18 GPU full-frame OOM) |
| Bitstream eval | official CR + exact index roundtrip |
| Latent +6h probe | ≤2 M MLP на pooled latent vs persistence |

**A18+ VRAM:** fine patch-4 на полном кадре → OOM на 12 GB. Заявленный full-frame score для A18+ считался на **CPU** (32 кадра). Crop-160 на GPU даёт оптимистичнее (~0.06).

### 5.1 Результаты (кратко)

| Модель | N | test \(S_\mathrm{all}\) | CR | exact |
|--------|---|-------------------------|-----|-------|
| Residual-FSQ 8.46 M | 128 | 0.131 | ~×60 | ✓ |
| Residual-FSQ 8.46 M | 256 | 0.083 | ×64 (pad) | ✓ |
| Residual-FSQ 8.46 M | 512 | **0.079** | ×63 (pad) | ✓ |
| **A18+ 19.11 M** | 512 | **0.072** (CPU×32) | **×39** | ✓ |

Кривые: `submission/plots/quality_vs_N_*.png`, `quality_vs_bitrate_*.png`.

### 5.2 Отрицательный результат (distill)

Короткий distill от CRA5 на частичном teacher-cache **не улучшил** val vs control (0.151 vs 0.131). Primary без teacher; детали — [external_models.md](external_models.md).

---

## 6. Probe +6h

Замороженный encoder → pooled latent → TinyMLP (≪2 M, ≤5k steps) предсказывает latent через +6h.  
Метрика: MSE в latent vs **persistence**. На финальном 8.46 M прогоне probe был лучше persistence. На A18+ в пакете лежит предыдущий probe; полный re-run на A18+ — опционально тем же скриптом `scripts/probe/latent_probe.py`.

Оговорка: в subsampled zarr соседние индексы — не всегда соседние 6h времена; probe проверяет динамику представления, не operational forecast skill.

---

## 7. Воспроизводимость

- Код + Docker (`Dockerfile`) + ckpt в `submission/artifacts/best.pt`  
- Фиксированные seeds / манифесты времён  
- `configs/residual_fsq_0p5_n512_A18.yaml` — сдача  
- `bash scripts/pack_submission.sh <run>` — пересборка пакета  

Ресурсы: RTX 3060 12 GB, порядок \(10^4\)–\(4\cdot10^4\) шагов на финальных прогонах.

---

## 8. Ограничения и честные gaps

1. **Нет formal non-inferiority CI vs VAEformer** — нет их baseline на тех же 28ch/0.5°/σ.  
2. **Только 0.5°** — сознательно ([SCOPE_0p25.md](../submission/SCOPE_0p25.md)).  
3. **A18+ full-frame на GPU** — OOM; score через CPU subsample.  
4. Цель \(S_\mathrm{all}\le0.05\) **не достигнута**; улучшение 0.079→0.072 за счёт ёмкости/битрейта.  
5. N=1024 download не завершён — кривая efficiency обрывается на 512.

---

## 9. Вклад / originality (для оценки)

- Компактный **Residual-FSQ** под лимит 20 M и official CR, а не wrap foundation.  
- Явная **карта data efficiency** на фиксированном железе.  
- Engineering: official LadCast σ, zlib bitstream + CR-cap pad, Docker-пакет для эксперта.  
- Зафиксированный отрицательный distill — тоже результат.

---

## 10. Ссылки по репо

| Тема | Файл |
|------|------|
| ТЗ | [task.md](task.md) |
| Внешние модели | [external_models.md](external_models.md) |
| Метрики / σ | [05_tasks_and_metrics.md](05_tasks_and_metrics.md) |
| Сдача | [submission_notes.md](submission_notes.md), [`submission/MANIFEST.json`](../submission/MANIFEST.json) |
