# Spatial-FSQ training / eval scripts (separate from legacy Perceiver-VQ).

Train:
  python scripts/spatial_fsq/train.py --config configs/spatial_fsq_0p5.yaml \
    --data data/era5_28ch_0p5_6h_n128.zarr --run-name n128

Eval:
  python scripts/spatial_fsq/evaluate.py \
    --ckpt runs/spatial_fsq_0p5/<stamp>_n128/best.pt \
    --data data/era5_28ch_0p5_6h_n128.zarr --split val
