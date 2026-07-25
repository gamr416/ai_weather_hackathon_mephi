# External / third-party models

## Organizer note (pretrained shortcut + оценка)

Организаторы уточнили:

- задачу можно решить **без обучения с нуля** или на малых \(N\): взять pretrained **LadCast** / **VAEformer**, настроить квантование/сжатие или короткий fine-tune;
- за сутки foundation «с нуля» не ожидается — это ок;
- балл сильно зависит от **методологии, воспроизводимости и оригинальности**;
- **защиты не будет** — эксперт проверяет решение в чате (см. [submission_notes.md](submission_notes.md)).

### Почему не сдаём VAEformer / LadCast AE as-is

| Барьер | Детали |
|--------|--------|
| Лимит **≤20 M trainable** | VAEformer ~405 M; как обучаемая модель — вне лимита |
| Каналы / сетка | CRA5: 268 vars @ 0.25° (721×1440); хакатон: **28** @ **0.5°**; нет teacher для `sst`, `tcwv` |
| Bitstream ×32–64 | У VAEformer свой latent/rate под ~300× CRA5 — не drop-in под наш official CR |
| Compute | CRA5 forward ~40 s/frame CPU на полном 268ch кадре |

Легальный shortcut = **замороженный** AE + маленький quant/bitstream-адаптер ≤20 M (Plan B).  
Primary сдача — **Residual-FSQ ≤20 M** на хакатонном ERA5. CRA5 — только объявленный teacher.  
σ для NRMSE: LadCast JSON → `data/ref_stats/` (см. [SOURCE.md](../data/ref_stats/SOURCE.md)).

## CRA5 VAEformer (teacher only)

| Field | Value |
|-------|--------|
| Source | [taohan10200/CRA5](https://github.com/taohan10200/CRA5) |
| Paper | Han et al., arXiv:2405.03376 |
| Checkpoint | `https://cra5.s3.ap-southeast-2.amazonaws.com/cra5_268v_300k.pth` (~1.6 GB) |
| Local path | `third_party/checkpoints/cra5_268v_300k.pth` |
| Code | `third_party/CRA5` (vendored, circular-import patch in `cra5/models/compressai/__init__.py`) |
| Params | ~405 M (frozen; **not** submitted) |
| Role | Soft reconstruction teacher for overlapping channels → distill into Residual-FSQ ≤20 M |

### Hackathon compliance

- Submitted student: **ResidualFSQAE** ≤20 M trainable params.
- External weather pretraining is **declared** here and in the run manifest.
- Teacher does **not** count toward data-efficiency \(N\) curves — those use only hackathon ERA5 frames (2014–2019 / 2020 / 2021).
- CRA5 forward is run offline on CPU (~40 s/frame @ 268×721×1440); results cached under `data/teacher_cra5_*`.

### Channel overlap

Mapped: `t2m, mslp(msl), u10, v10, tp6h(tp mm), tcc, T/U/V/Z/Q @ 1000/925/850/700`.

No teacher: `sst`, `tcwv` (student recon loss only).

### Commands

```bash
# smoke (CPU)
PYTHONPATH=third_party/CRA5:$PYTHONPATH \
  python scripts/teacher/smoke_cra5.py --data data/era5_28ch_0p5_6h_n128.zarr --n 2 --device cpu

# offline cache (partial OK)
PYTHONPATH=third_party/CRA5:$PYTHONPATH \
  python scripts/teacher/build_cache.py \
    --data data/era5_28ch_0p5_6h_n128.zarr \
    --out data/teacher_cra5_n128 \
    --stats runs/residual_fsq_0p5/20260725_010045_n128s9k/norm_stats.npz \
    --max-frames 32 --device cpu

# distill
python scripts/residual_fsq/train.py --config configs/residual_fsq_distill_0p5.yaml --run-name distill_n128
```

### Distill smoke (2026-07-25)

Partial cache (16 train / 16 val frames) + 2k warm-start steps: val `S_all` **0.151** vs control **0.131** — no gain. Primary submission remains control Residual-FSQ; re-try needs full teacher cache over all \(N\) frames.
