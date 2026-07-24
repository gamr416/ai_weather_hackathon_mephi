"""Spatial patch AE with uniform/FSQ bottleneck sized for CR ×32–×64."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.perceiver_ae import Attention, FeedForward, group_recon_stats, latitude_weighted_mse

__all__ = [
    "SpatialFSQAE",
    "latitude_weighted_mse",
    "group_recon_stats",
    "raw_cr_from_bits",
]


def raw_cr_from_bits(bits: float, h: int, w: int, channels: int = 28) -> float:
    """CR = (32 * C * H * W) / bits  (float32 numerator, bitstream bits in denom)."""
    num = 32.0 * channels * h * w
    return num / max(bits, 1.0)


class TransformerBlock(nn.Module):
    """Pre-LN self-attention + FFN."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, patch: int, dim: int) -> None:
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        tokens = self.proj(x)
        gh, gw = tokens.shape[-2:]
        tokens = rearrange(tokens, "b d gh gw -> b (gh gw) d")
        return tokens, (gh, gw)


class UniformQuantizer(nn.Module):
    """
    Per-channel uniform quantization in [-1, 1] with STE.
    Raw bitstream bits = N * C_lat * log2(num_levels) (no entropy yet).
    """

    def __init__(self, num_levels: int = 256) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError("num_levels must be >= 2")
        self.num_levels = int(num_levels)
        self.register_buffer("log2_levels", torch.tensor(math.log2(num_levels)), persistent=False)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        # z: B, N, C
        z_b = torch.tanh(z)
        span = float(self.num_levels - 1)
        scaled = (z_b + 1.0) * 0.5 * span
        idx = scaled.round().clamp(0, span)
        q = idx / span * 2.0 - 1.0
        z_q = z_b + (q - z_b).detach()  # STE
        bits_per = float(self.log2_levels)
        n_elem = int(z.shape[0] * z.shape[1] * z.shape[2])
        # keep rate in fp32 — batch*tokens*C_lat*bits overflows fp16 under AMP
        rate_bits = torch.tensor(n_elem * bits_per, device=z.device, dtype=torch.float32)
        commit = F.mse_loss(z_b.float(), q.detach().float())
        return {
            "z_q": z_q,
            "indices": idx.to(torch.int32),
            "rate_bits": rate_bits,
            "bits_per_elem": torch.tensor(bits_per, device=z.device, dtype=torch.float32),
            "commit": commit,
            "usage": int(idx.detach().float().unique().numel()),
        }


class SpatialFSQAE(nn.Module):
    """
    Patch tokens → transformer encoder → uniform quant (C_lat) → transformer decoder → image.

    Raw CR ≈ 32 * 28 * patch^2 / (C_lat * log2(L)).
    Default patch=8, C_lat=96, L=256 → ~×75 (near ×64 target).
    """

    def __init__(
        self,
        in_channels: int = 28,
        static_channels: int = 4,
        patch_size: int = 8,
        dim: int = 160,
        depth: int = 4,
        heads: int = 4,
        latent_channels: int = 96,
        num_levels: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.in_channels = in_channels
        self.static_channels = static_channels
        self.latent_channels = latent_channels
        self.num_levels = num_levels

        self.patch_embed = PatchEmbed(in_channels + static_channels, patch_size, dim)
        self.static_patch = PatchEmbed(static_channels, patch_size, dim)
        self.encoder_blocks = nn.ModuleList(
            [TransformerBlock(dim, heads=heads, dropout=dropout) for _ in range(depth)]
        )
        self.to_latent = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, latent_channels))
        self.quant = UniformQuantizer(num_levels)
        self.from_latent = nn.Linear(latent_channels, dim)
        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(dim, heads=heads, dropout=dropout) for _ in range(depth)]
        )
        self.to_patch = nn.Linear(dim, in_channels * patch_size * patch_size)
        self.pos_mlp = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def expected_raw_cr(self) -> float:
        bits_per = math.log2(self.num_levels)
        return (32.0 * self.in_channels * self.patch_size**2) / (self.latent_channels * bits_per)

    def _grid_pos(self, gh: int, gw: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1, 1, gh, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, gw, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        feat = torch.stack([yy, xx, torch.sin(math.pi * yy), torch.sin(math.pi * xx)], dim=-1)
        feat = feat.reshape(1, gh * gw, 4)
        return self.pos_mlp(feat)

    def encode(self, x: torch.Tensor, static: torch.Tensor) -> dict[str, torch.Tensor]:
        x_in = torch.cat([x, static], dim=1)
        tokens, (gh, gw) = self.patch_embed(x_in)
        tokens = tokens + self._grid_pos(gh, gw, x.device, tokens.dtype)
        for blk in self.encoder_blocks:
            tokens = blk(tokens)
        z = self.to_latent(tokens)
        qout = self.quant(z)
        return {
            "z_q": qout["z_q"],
            "indices": qout["indices"],
            "rate_bits": qout["rate_bits"],
            "bits_per_elem": qout["bits_per_elem"],
            "commit": qout["commit"],
            "usage": qout["usage"],
            "gh": gh,
            "gw": gw,
        }

    def decode(self, z_q: torch.Tensor, static: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        tokens = self.from_latent(z_q)
        tokens = tokens + self._grid_pos(gh, gw, z_q.device, tokens.dtype)
        st_tok, _ = self.static_patch(static)
        tokens = tokens + st_tok
        for blk in self.decoder_blocks:
            tokens = blk(tokens)
        patches = self.to_patch(tokens)
        return rearrange(
            patches,
            "b (gh gw) (c ph pw) -> b c (gh ph) (gw pw)",
            gh=gh,
            gw=gw,
            c=self.in_channels,
            ph=self.patch_size,
            pw=self.patch_size,
        )

    def forward(self, x: torch.Tensor, static: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        b, _, h, w = x.shape
        if static is None:
            static = x.new_zeros(b, self.static_channels, h, w)
        ph = self.patch_size
        pad_h = (ph - h % ph) % ph
        pad_w = (ph - w % ph) % ph
        if pad_h or pad_w:
            x_in = F.pad(x, (0, pad_w, 0, pad_h))
            st_in = F.pad(static, (0, pad_w, 0, pad_h))
        else:
            x_in, st_in = x, static

        enc = self.encode(x_in, st_in)
        recon = self.decode(enc["z_q"], st_in, enc["gh"], enc["gw"])
        recon = recon[:, :, :h, :w]

        # rate for this (possibly padded) spatial grid; CR reported vs original HxW
        rate_bits = enc["rate_bits"]
        cr = raw_cr_from_bits(float(rate_bits.detach()) / max(b, 1), h, w, self.in_channels)

        return {
            "recon": recon,
            "z_q": enc["z_q"],
            "indices": enc["indices"],
            "rate_bits": rate_bits,
            "commit": enc["commit"],
            "usage": enc["usage"],
            "cr_raw": torch.tensor(cr, device=x.device, dtype=torch.float32),
            "bits_per_elem": enc["bits_per_elem"],
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
