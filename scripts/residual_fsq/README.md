# Residual Multi-Scale FSQ (coarse patch8 + fine residual patch4).

Train:
  python scripts/residual_fsq/train.py --config configs/residual_fsq_0p5.yaml \
    --data data/era5_28ch_0p5_6h_n128.zarr --run-name n128

Smoke (short):
  python scripts/residual_fsq/train.py --config configs/residual_fsq_0p5.yaml \
    --data data/era5_28ch_0p5_6h_n128.zarr --run-name smoke --steps 40

Official σ (once):
  python scripts/metrics/build_official_sigma.py

Eval (NRMSE = RMSE_phys / σ_official):
  python scripts/residual_fsq/evaluate.py \
    --ckpt runs/residual_fsq_0p5/<stamp>_n128/best.pt \
    --data data/era5_28ch_0p5_6h_n128.zarr --split val

Bitstream CR + exact roundtrip (all val frames):
  python scripts/residual_fsq/eval_bitstream.py \
    --ckpt runs/residual_fsq_0p5/<stamp>_n128/best.pt \
    --data data/era5_28ch_0p5_6h_n128.zarr --split val --max-frames 64

N-sweep + quality–N plot:
  NS="128 256" STEPS=5000 bash scripts/residual_fsq/n_sweep.sh
  python scripts/residual_fsq/plot_n_curve.py --split val
