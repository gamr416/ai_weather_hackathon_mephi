"""Frozen CRA5 VAEformer teacher wrapper (reconstruction soft targets)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.teacher.cra5_channels import (
    build_overlap_maps,
    load_cra5_mean_std,
    pack_to_cra5_physical,
    unpack_from_cra5_physical,
)

ROOT = Path(__file__).resolve().parents[2]
CRA5_ROOT = ROOT / "third_party" / "CRA5"
DEFAULT_CKPT = ROOT / "third_party" / "checkpoints" / "cra5_268v_300k.pth"


def _ensure_cra5_path() -> None:
    p = str(CRA5_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_vaeformer(ckpt: Path | str = DEFAULT_CKPT, device: str | torch.device = "cpu") -> nn.Module:
    """Load pretrained VAEformer-268 from local checkpoint (no torch.hub)."""
    _ensure_cra5_path()
    from cra5.models.vaeformer.vaeformer import VAEformer

    ckpt = Path(ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"CRA5 checkpoint missing: {ckpt}\n"
            "Download: https://cra5.s3.ap-southeast-2.amazonaws.com/cra5_268v_300k.pth"
        )
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # from_state_dict expects backbone.* keys
    if not any(str(k).startswith("backbone.") for k in sd):
        sd = {f"backbone.{k}": v for k, v in sd.items()}
    net = VAEformer.from_state_dict(sd)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net.to(device)


class CRA5Teacher(nn.Module):
    """
    Maps our physical/normed 28ch frames → CRA5 soft reconstructions on overlap channels.

    Forward expects *physical* 28ch tensors (B,28,H,W) on our 0.5° grid.
    Returns physical recon (B,28,H,W) with non-overlap channels left as zeros
    (caller should blend with GT for those).
    """

    def __init__(
        self,
        ckpt: Path | str = DEFAULT_CKPT,
        device: str | torch.device = "cpu",
        use_quantized: bool = False,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.use_quantized = bool(use_quantized)
        self.net = load_vaeformer(ckpt, device=self.device)
        mean, std = load_cra5_mean_std()
        self.register_buffer("cra5_mean", torch.from_numpy(mean).view(1, 268, 1, 1))
        self.register_buffer("cra5_std", torch.from_numpy(std).view(1, 268, 1, 1).clamp_min(1e-6))
        our_idx, cra5_idx, names = build_overlap_maps()
        self.our_idx = our_idx
        self.cra5_idx = cra5_idx
        self.overlap_names = names
        self.overlap_mask = torch.zeros(28, dtype=torch.bool)
        for i in our_idx:
            self.overlap_mask[i] = True

    @torch.no_grad()
    def reconstruct_physical(self, x_phys: torch.Tensor) -> torch.Tensor:
        """x_phys (B,28,H,W) physical → soft recon physical (B,28,H,W)."""
        x_phys = x_phys.to(self.device)
        # sanitize non-finite pixels (e.g. SST over land) per-channel median
        x_clean = x_phys.clone()
        for c in range(x_clean.shape[1]):
            ch = x_clean[:, c]
            if not torch.isfinite(ch).all():
                finite = ch[torch.isfinite(ch)]
                fill = finite.median() if finite.numel() else ch.new_zeros(())
                x_clean[:, c] = torch.where(torch.isfinite(ch), ch, fill)

        packed = pack_to_cra5_physical(
            x_clean, self.our_idx, self.cra5_idx, cra5_mean=self.cra5_mean.view(268)
        )
        mean = self.cra5_mean.to(dtype=packed.dtype, device=packed.device)
        std = self.cra5_std.to(dtype=packed.dtype, device=packed.device)
        x_n = (packed - mean) / std

        if self.use_quantized:
            y, y_hat, _ = self.net.encode_latent(x_n, type="quantized")
            x_hat_n = self.net.decode_latent(y_hat)
        else:
            y, _, _ = self.net.encode_latent(x_n, type="float")
            x_hat_n = self.net.decode_latent(y)

        x_hat = x_hat_n * std + mean
        return unpack_from_cra5_physical(x_hat, self.our_idx, self.cra5_idx)

    def reconstruct_normalized(
        self,
        x_norm: torch.Tensor,
        our_mean: torch.Tensor,
        our_std: torch.Tensor,
    ) -> torch.Tensor:
        """x_norm in our train stats → soft recon in same normalized space."""
        mean = our_mean.view(1, 28, 1, 1).to(x_norm.device, x_norm.dtype)
        std = our_std.view(1, 28, 1, 1).to(x_norm.device, x_norm.dtype).clamp_min(1e-4)
        x_phys = x_norm * std + mean
        rec_phys = self.reconstruct_physical(x_phys)
        return (rec_phys - mean) / std
