# 0.25° comparison — out of scope for this submission

## Decision

We deliver **only 0.5°** (360×720, 28 channels).

## Why

| Constraint | Impact at 0.25° |
|------------|-----------------|
| GPU | RTX 3060 **12 GB** — full-frame 0.25° AE training is memory-tight |
| Time | hackathon ≤ ~1 day wall-clock; 0.5° N-sweep + long fine-tunes consumed budget |
| Data | 0.25° zarr download attempted but blocked by proxy/SSL issues mid-run; 0.5° N=512 completed |

## What we prepared

- Download script exists: `scripts/download_era5_0p25.py`
- Same Residual-FSQ architecture can be pointed at 0.25° zarr with tile/crop training
- Scientific N-curve and bitrate plots in this package are for **0.5°**

## Claim

Primary research result: **data efficiency at 0.5°** for a ≤20 M Residual-FSQ codec with official CR ∈ [32, 64] and LadCast σ-NRMSE. A matched 0.25° curve is left as future work, not a silent omission.
