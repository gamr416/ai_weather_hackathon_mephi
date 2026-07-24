"""Residual Multi-Scale Spatial FSQ: coarse patch8 + fine residual patch4."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.perceiver_ae import FeedForward, group_recon_stats
from src.models.spatial_fsq import PatchEmbed, UniformQuantizer, raw_cr_from_bits

__all__ = [
    "ResidualFSQAE",
    "latitude_weighted_mse_sst_ocean",
    "high_freq_grad_penalty",
    "group_recon_stats",
    "raw_cr_from_bits",
]

# CHANNEL_ORDER index for sst / high-freq channels
SST_IDX = 5
TP6H_IDX = 4
TCC_IDX = 7


class DropPath(nn.Module):
    """Stochastic depth (per-sample)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class SDPAAttention(nn.Module):
    """Self-attention via scaled_dot_product_attention (flash/mem-efficient when available)."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dim_head = dim // heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self.norm(x)
        qkv = self.to_qkv(x_n)
        q, k, v = rearrange(qkv, "b n (three h d) -> three b h n d", three=3, h=self.heads)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Pre-LN self-attention + FFN with optional DropPath."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.attn = SDPAAttention(dim, heads=heads, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.ff(x))
        return x


class ResConvBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CNNStem(nn.Module):
    def __init__(self, in_ch: int, dim: int = 64, n_blocks: int = 2) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ResConvBlock(dim) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))


class CNNRefine(nn.Module):
    def __init__(self, channels: int = 28, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def latitude_weighted_mse_sst_ocean(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_weight: torch.Tensor,
    channel_weight: torch.Tensor | None = None,
    static: torch.Tensor | None = None,
    sst_idx: int = SST_IDX,
) -> torch.Tensor:
    """
    Lat-weighted MSE; SST channel (idx 5) only over ocean (1 - land_sea_mask).
    static: B,4,H,W with channel 0 = land_sea_mask (1=land).
    """
    if lat_weight.dim() == 1:
        w = lat_weight.view(1, 1, -1, 1)
    else:
        w = lat_weight.view(lat_weight.shape[0], 1, -1, 1)
    err = (pred - target).pow(2) * w
    if static is not None and static.shape[1] >= 1:
        # ocean = 1 where land_sea_mask < 0.5
        ocean = (1.0 - static[:, 0:1]).clamp(0.0, 1.0)
        err_sst = err[:, sst_idx : sst_idx + 1] * ocean
        err = torch.cat([err[:, :sst_idx], err_sst, err[:, sst_idx + 1 :]], dim=1)
    if channel_weight is not None:
        cw = channel_weight.view(1, -1, 1, 1).to(device=err.device, dtype=err.dtype)
        err = err * cw
    return err.mean()


def high_freq_grad_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_indices: tuple[int, ...] = (TP6H_IDX, TCC_IDX),
    lat_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Light spatial-gradient MSE on selected channels (tp6h, tcc by default)."""
    if not channel_indices:
        return pred.new_zeros(())
    idx = list(channel_indices)
    p = pred[:, idx]
    t = target[:, idx]
    if lat_weight is not None:
        if lat_weight.dim() == 1:
            w = lat_weight.view(1, 1, -1, 1)
        else:
            w = lat_weight.view(lat_weight.shape[0], 1, -1, 1)
        w_x = w  # B,1,H,1 → broadcast over W-1
        w_y = w[..., :-1, :]  # B,1,H-1,1
    else:
        w_x = w_y = 1.0

    gx_p = p[..., :, 1:] - p[..., :, :-1]
    gx_t = t[..., :, 1:] - t[..., :, :-1]
    gy_p = p[..., 1:, :] - p[..., :-1, :]
    gy_t = t[..., 1:, :] - t[..., :-1, :]
    loss_x = ((gx_p - gx_t).pow(2) * w_x).mean()
    loss_y = ((gy_p - gy_t).pow(2) * w_y).mean()
    return loss_x + loss_y


class ResidualFSQAE(nn.Module):
    """
    CNN stem → coarse spatial quant (patch 8) + fine residual quant (patch 4) → refine.

    Raw CR ≈ 32 * 28 / (bits_per_pixel), where
      bits_pp = C_c * log2(L_c) / P_c^2 + C_f * log2(L_f) / P_f^2
    Default: 80*8/64 + 16*6/16 = 10 + 6 = 16 → CR ≈ ×56.
    """

    def __init__(
        self,
        in_channels: int = 28,
        static_channels: int = 4,
        stem_dim: int = 64,
        stem_blocks: int = 2,
        # coarse
        coarse_patch: int = 8,
        coarse_dim: int = 256,
        coarse_depth: int = 4,
        coarse_dec_depth: int = 3,
        coarse_heads: int = 8,
        coarse_latent: int = 80,
        coarse_levels: int = 256,
        # fine
        fine_patch: int = 4,
        fine_dim: int = 128,
        fine_depth: int = 2,
        fine_heads: int = 4,
        fine_latent: int = 16,
        fine_levels: int = 64,
        dropout: float = 0.1,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.static_channels = static_channels
        self.coarse_patch = coarse_patch
        self.fine_patch = fine_patch
        self.coarse_latent = coarse_latent
        self.fine_latent = fine_latent
        self.coarse_levels = coarse_levels
        self.fine_levels = fine_levels
        self.coarse_dim = coarse_dim
        self.fine_dim = fine_dim

        if coarse_patch % fine_patch != 0:
            raise ValueError("coarse_patch must be divisible by fine_patch")
        self.scale = coarse_patch // fine_patch  # typically 2

        self.stem = CNNStem(in_channels + static_channels, stem_dim, stem_blocks)
        self.refine = CNNRefine(in_channels, hidden=stem_dim)

        # --- coarse branch ---
        self.coarse_embed = PatchEmbed(stem_dim, coarse_patch, coarse_dim)
        self.coarse_static = PatchEmbed(static_channels, coarse_patch, coarse_dim)
        dpr_c = torch.linspace(0, drop_path, coarse_depth).tolist()
        self.coarse_enc = nn.ModuleList(
            [
                TransformerBlock(coarse_dim, coarse_heads, dropout, dpr_c[i])
                for i in range(coarse_depth)
            ]
        )
        self.coarse_to_lat = nn.Sequential(nn.LayerNorm(coarse_dim), nn.Linear(coarse_dim, coarse_latent))
        self.coarse_quant = UniformQuantizer(coarse_levels)
        self.coarse_from_lat = nn.Linear(coarse_latent, coarse_dim)
        dpr_cd = torch.linspace(0, drop_path, coarse_dec_depth).tolist()
        self.coarse_dec = nn.ModuleList(
            [
                TransformerBlock(coarse_dim, coarse_heads, dropout, dpr_cd[i])
                for i in range(coarse_dec_depth)
            ]
        )
        self.coarse_to_patch = nn.Linear(coarse_dim, in_channels * coarse_patch * coarse_patch)
        self.coarse_pos = nn.Sequential(nn.Linear(4, coarse_dim), nn.GELU(), nn.Linear(coarse_dim, coarse_dim))

        # --- fine branch ---
        self.fine_embed = PatchEmbed(stem_dim, fine_patch, fine_dim)
        self.fine_static = PatchEmbed(static_channels, fine_patch, fine_dim)
        self.coarse_ctx_proj = nn.Linear(coarse_latent, fine_dim)
        dpr_f = torch.linspace(0, drop_path, fine_depth).tolist()
        self.fine_enc = nn.ModuleList(
            [
                TransformerBlock(fine_dim, fine_heads, dropout, dpr_f[i])
                for i in range(fine_depth)
            ]
        )
        self.fine_to_lat = nn.Sequential(nn.LayerNorm(fine_dim), nn.Linear(fine_dim, fine_latent))
        self.fine_quant = UniformQuantizer(fine_levels)
        self.fine_from_lat = nn.Linear(fine_latent, fine_dim)
        self.fine_dec = nn.ModuleList(
            [
                TransformerBlock(fine_dim, fine_heads, dropout, dpr_f[i])
                for i in range(fine_depth)
            ]
        )
        self.fine_to_patch = nn.Linear(fine_dim, in_channels * fine_patch * fine_patch)
        self.fine_pos = nn.Sequential(nn.Linear(4, fine_dim), nn.GELU(), nn.Linear(fine_dim, fine_dim))

    def expected_raw_cr(self) -> float:
        bits_pp = (
            self.coarse_latent * math.log2(self.coarse_levels) / (self.coarse_patch**2)
            + self.fine_latent * math.log2(self.fine_levels) / (self.fine_patch**2)
        )
        return (32.0 * self.in_channels) / bits_pp

    def _grid_pos(self, gh: int, gw: int, mlp: nn.Module, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1, 1, gh, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, gw, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        feat = torch.stack([yy, xx, torch.sin(math.pi * yy), torch.sin(math.pi * xx)], dim=-1)
        feat = feat.reshape(1, gh * gw, 4)
        return mlp(feat)

    def _upsample_coarse_to_fine(
        self, z_c: torch.Tensor, gh_c: int, gw_c: int, gh_f: int, gw_f: int
    ) -> torch.Tensor:
        """B, Nc, Cc → B, Nf, fine_dim via nearest upsample of spatial map."""
        x = rearrange(z_c, "b (gh gw) c -> b c gh gw", gh=gh_c, gw=gw_c)
        x = F.interpolate(x.float(), size=(gh_f, gw_f), mode="nearest").to(dtype=z_c.dtype)
        x = rearrange(x, "b c gh gw -> b (gh gw) c")
        return self.coarse_ctx_proj(x)

    def encode(self, x: torch.Tensor, static: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.stem(torch.cat([x, static], dim=1))

        tok_c, (gh_c, gw_c) = self.coarse_embed(feat)
        tok_c = tok_c + self._grid_pos(gh_c, gw_c, self.coarse_pos, x.device, tok_c.dtype)
        for blk in self.coarse_enc:
            tok_c = blk(tok_c)
        z_c = self.coarse_to_lat(tok_c)
        qc = self.coarse_quant(z_c)

        tok_f, (gh_f, gw_f) = self.fine_embed(feat)
        ctx = self._upsample_coarse_to_fine(qc["z_q"], gh_c, gw_c, gh_f, gw_f)
        tok_f = tok_f + self._grid_pos(gh_f, gw_f, self.fine_pos, x.device, tok_f.dtype) + ctx
        for blk in self.fine_enc:
            tok_f = blk(tok_f)
        z_f = self.fine_to_lat(tok_f)
        qf = self.fine_quant(z_f)

        rate_bits = qc["rate_bits"] + qf["rate_bits"]
        commit = qc["commit"] + qf["commit"]
        return {
            "z_c": qc["z_q"],
            "z_f": qf["z_q"],
            "indices_c": qc["indices"],
            "indices_f": qf["indices"],
            "rate_bits": rate_bits,
            "rate_bits_c": qc["rate_bits"],
            "rate_bits_f": qf["rate_bits"],
            "commit": commit,
            "usage_c": qc["usage"],
            "usage_f": qf["usage"],
            "gh_c": gh_c,
            "gw_c": gw_c,
            "gh_f": gh_f,
            "gw_f": gw_f,
        }

    def decode(
        self,
        z_c: torch.Tensor,
        z_f: torch.Tensor,
        static: torch.Tensor,
        gh_c: int,
        gw_c: int,
        gh_f: int,
        gw_f: int,
    ) -> torch.Tensor:
        # coarse image
        tok = self.coarse_from_lat(z_c)
        tok = tok + self._grid_pos(gh_c, gw_c, self.coarse_pos, z_c.device, tok.dtype)
        st_c, _ = self.coarse_static(static)
        tok = tok + st_c
        for blk in self.coarse_dec:
            tok = blk(tok)
        coarse = rearrange(
            self.coarse_to_patch(tok),
            "b (gh gw) (c ph pw) -> b c (gh ph) (gw pw)",
            gh=gh_c,
            gw=gw_c,
            c=self.in_channels,
            ph=self.coarse_patch,
            pw=self.coarse_patch,
        )

        # fine residual
        tok = self.fine_from_lat(z_f)
        tok = tok + self._grid_pos(gh_f, gw_f, self.fine_pos, z_f.device, tok.dtype)
        st_f, _ = self.fine_static(static)
        ctx = self._upsample_coarse_to_fine(z_c, gh_c, gw_c, gh_f, gw_f)
        tok = tok + st_f + ctx
        for blk in self.fine_dec:
            tok = blk(tok)
        residual = rearrange(
            self.fine_to_patch(tok),
            "b (gh gw) (c ph pw) -> b c (gh ph) (gw pw)",
            gh=gh_f,
            gw=gw_f,
            c=self.in_channels,
            ph=self.fine_patch,
            pw=self.fine_patch,
        )

        return self.refine(coarse + residual)

    def forward(self, x: torch.Tensor, static: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        b, _, h, w = x.shape
        if static is None:
            static = x.new_zeros(b, self.static_channels, h, w)

        # pad to lcm of patches (coarse_patch since fine divides it)
        ph = self.coarse_patch
        pad_h = (ph - h % ph) % ph
        pad_w = (ph - w % ph) % ph
        if pad_h or pad_w:
            x_in = F.pad(x, (0, pad_w, 0, pad_h))
            st_in = F.pad(static, (0, pad_w, 0, pad_h))
        else:
            x_in, st_in = x, static

        enc = self.encode(x_in, st_in)
        recon = self.decode(
            enc["z_c"],
            enc["z_f"],
            st_in,
            enc["gh_c"],
            enc["gw_c"],
            enc["gh_f"],
            enc["gw_f"],
        )
        recon = recon[:, :, :h, :w]

        rate_bits = enc["rate_bits"]
        cr = raw_cr_from_bits(float(rate_bits.detach()) / max(b, 1), h, w, self.in_channels)
        usage = int(enc["usage_c"]) + int(enc["usage_f"])

        return {
            "recon": recon,
            "z_c": enc["z_c"],
            "z_f": enc["z_f"],
            "indices_c": enc["indices_c"],
            "indices_f": enc["indices_f"],
            "rate_bits": rate_bits,
            "rate_bits_c": enc["rate_bits_c"],
            "rate_bits_f": enc["rate_bits_f"],
            "commit": enc["commit"],
            "usage": usage,
            "usage_c": enc["usage_c"],
            "usage_f": enc["usage_f"],
            "cr_raw": torch.tensor(cr, device=x.device, dtype=torch.float32),
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
