# Residual Multi-Scale FSQ (coarse patch8 + fine residual patch4).

Train:
  python scripts/residual_fsq/train.py --config configs/residual_fsq_0p5.yaml \
    --data data/era5_28ch_0p5_6h_n128.zarr --run-name n128

Smoke (short):
  python scripts/residual_fsq/train.py --config configs/residual_fsq_0p5.yaml \
    --data data/era5_28ch_0p5_6h_n128.zarr --run-name smoke --steps 40

Eval:
  python scripts/residual_fsq/evaluate.py \
    --ckpt runs/residual_fsq_0p5/<stamp>_n128/best.pt \
    --data data/era5_28ch_0p5_6h_n128.zarr --split val
