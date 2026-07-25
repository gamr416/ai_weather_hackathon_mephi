"""Aggregate reconstruction scores with official NRMSE."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.era5 import CHANNEL_ORDER
from src.metrics.official_sigma import load_official_sigma


def scores_from_sse(
    sse_norm: np.ndarray,
    sse_phys: np.ndarray,
    wsum: np.ndarray | float,
    sigma: np.ndarray | None = None,
    sigma_source: np.ndarray | None = None,
    sigma_path: Path | str | None = None,
) -> dict:
    """Build metrics dict from per-channel weighted SSE.

    NRMSE_f = RMSE_physical_f / σ_f_official
    S_surface / S_pressure / S_all use official NRMSE only.
    Legacy rmse_norm is kept for training diagnostics.
    """
    w = np.asarray(wsum, dtype=np.float64)
    if w.ndim == 0:
        w = np.full(28, float(w), dtype=np.float64)
    rmse_norm = np.sqrt(sse_norm / np.maximum(w, 1e-12))
    rmse_phys = np.sqrt(sse_phys / np.maximum(w, 1e-12))

    if sigma is None:
        _mean, sigma, sigma_source = load_official_sigma(sigma_path)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(28)
    sigma = np.maximum(sigma, 1e-8)
    nrmse = rmse_phys / sigma

    surface = float(nrmse[:8].mean())
    pressure = float(nrmse[8:].mean())
    per = {}
    for i, name in enumerate(CHANNEL_ORDER):
        entry = {
            "rmse_norm": float(rmse_norm[i]),
            "rmse_physical": float(rmse_phys[i]),
            "nrmse": float(nrmse[i]),
            "sigma_official": float(sigma[i]),
        }
        if sigma_source is not None:
            entry["sigma_source"] = str(sigma_source[i])
        per[name] = entry

    src_note = None
    if sigma_source is not None:
        src_note = {ch: str(sigma_source[i]) for i, ch in enumerate(CHANNEL_ORDER)}

    return {
        "S_surface": surface,
        "S_pressure": pressure,
        "S_all": 0.5 * surface + 0.5 * pressure,
        "per_channel": per,
        "sigma_path": str(sigma_path) if sigma_path else None,
        "sigma_source_map": src_note,
        "note": (
            "nrmse = rmse_physical / sigma_official (LadCast 1979–2017 + train_pool gaps). "
            "rmse_norm is train-normalization diagnostic only; scores use official NRMSE."
        ),
    }


def plot_nrmse_ylabel() -> str:
    return "NRMSE (RMSE_phys / σ_official)"
