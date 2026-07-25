# Submission notes

Пакет: [`submission/`](../submission/) · README: [`../README.md`](../README.md).  
**Методология:** [`methodology.md`](methodology.md).

## Финал: Residual-FSQ **A18+** (19.11 M)

| | |
|--|--|
| Run | `runs/residual_fsq_0p5/20260725_212101_n512s40k_A18` |
| Full-frame test \(S_\mathrm{all}\) | **0.0724** (CPU, 32 frames) |
| Crop-160 test \(S_\mathrm{all}\) | **0.0605** (GPU, 128 frames) |
| Official CR | **×38.7**, exact |
| Prev 8.46M full test | 0.0788 (`n512s35k_ft`) |

Full-frame GPU OOM на 12GB — см. README. Организаторы: достаточно **0.5° или 0.25°** (мы — 0.5°).

## Docker

```bash
docker build -t mephi-era5-residual-fsq:latest .
docker run --rm -it --gpus all -v "$PWD/data:/workspace/data:ro" mephi-era5-residual-fsq:latest
```
