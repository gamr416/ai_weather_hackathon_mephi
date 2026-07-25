# Official NRMSE denominator (σ)

## LadCast

- File: `ERA5_normal_1979_2017.json`
- Upstream: https://github.com/tonyzyl/ladcast (`ladcast/static/ERA5_normal_1979_2017.json`)
- Period: ERA5 mean/std over **1979–2017**
- License: see LadCast repository `LICENSE`

Organizers allow taking the NRMSE normalization denominator from LadCast or CDS climatology.

## Gaps filled from local train pool

LadCast does **not** ship `total_cloud_cover` (`tcc`) or `total_column_water_vapour` (`tcwv`).
Those channels use latitude-weighted std over the local hackathon train zarr (2014–2019 pool),
written into `sigma_official_28ch.npz` with `source=train_pool`.

```bash
python scripts/metrics/build_official_sigma.py \
  --ladcast data/ref_stats/ERA5_normal_1979_2017.json \
  --train-zarr data/era5_28ch_0p5_6h.zarr/train.zarr \
  --out data/ref_stats/sigma_official_28ch.npz
```
