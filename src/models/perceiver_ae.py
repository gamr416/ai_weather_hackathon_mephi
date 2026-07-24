"""Compact Perceiver-IO autoencoder with VQ bottleneck (hackathon codec skeleton)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        x_n = self.norm(x)
        ctx = x_n if context is None else context
        q = self.to_q(x_n)
        k, v = self.to_kv(ctx).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class PerceiverBlock(nn.Module):
    """Cross-attend latents←context, then self-attend + FFN on latents."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.cross = Attention(dim, heads=heads, dropout=dropout)
        self.self_attn = Attention(dim, heads=heads, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        latents = latents + self.cross(latents, context)
        latents = latents + self.self_attn(latents)
        latents = latents + self.ff(latents)
        return latents


class VectorQuantizer(nn.Module):
    """VQ-VAE bottleneck with straight-through estimator."""

    def __init__(self, codebook_size: int, dim: int, beta: float = 0.25) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.beta = beta
        self.embed = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.embed.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, N, D)
        flat = z.reshape(-1, z.shape[-1])
        # distances to codebook
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1)
        )
        idx = dist.argmin(dim=1)
        z_q = self.embed(idx).view(z.shape)

        # losses
        commit = F.mse_loss(z_q.detach(), z)
        code = F.mse_loss(z_q, z.detach())
        vq_loss = code + self.beta * commit

        # STE
        z_q_st = z + (z_q - z).detach()
        # perplexity diagnostic
        ah = torch.histc(idx.float(), bins=self.codebook_size, min=0, max=self.codebook_size - 1)
        probs = ah / ah.sum().clamp_min(1)
        perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
        return z_q_st, vq_loss, perplexity


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, patch: int, dim: int) -> None:
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        # x: B,C,H,W
        tokens = self.proj(x)  # B,D,Gh,Gw
        gh, gw = tokens.shape[-2:]
        tokens = rearrange(tokens, "b d gh gw -> b (gh gw) d")
        return tokens, (gh, gw)


class PerceiverAE(nn.Module):
    """
    Encode patches → Perceiver latents → VQ → decode queries → image.

    Designed for ERA5 28ch @ 0.5° crops; keep params ≤ ~20M.
    """

    def __init__(
        self,
        in_channels: int = 28,
        patch_size: int = 16,
        dim: int = 192,
        num_latents: int = 128,
        depth: int = 4,
        heads: int = 4,
        codebook_size: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.num_latents = num_latents

        self.patch_embed = PatchEmbed(in_channels, patch_size, dim)
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.encoder_blocks = nn.ModuleList(
            [PerceiverBlock(dim, heads=heads, dropout=dropout) for _ in range(depth)]
        )
        self.pre_vq = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.vq = VectorQuantizer(codebook_size, dim)
        self.decoder_blocks = nn.ModuleList(
            [PerceiverBlock(dim, heads=heads, dropout=dropout) for _ in range(depth)]
        )
        self.query_pos = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # broadcast add via grid below
        self.to_patch = nn.Linear(dim, in_channels * patch_size * patch_size)
        self.in_channels = in_channels

        # fixed 2d sin-cos style learnable grid bias built on the fly via Linear of coords
        self.pos_mlp = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def _grid_pos(self, gh: int, gw: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1, 1, gh, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, gw, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        feat = torch.stack([yy, xx, torch.sin(math.pi * yy), torch.sin(math.pi * xx)], dim=-1)
        feat = feat.reshape(1, gh * gw, 4)
        return self.pos_mlp(feat)  # 1,N,D

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        tokens, (gh, gw) = self.patch_embed(x)
        tokens = tokens + self._grid_pos(gh, gw, x.device, tokens.dtype)
        latents = repeat(self.latents, "n d -> b n d", b=x.shape[0])
        for blk in self.encoder_blocks:
            latents = blk(latents, tokens)
        z = self.pre_vq(latents)
        z_q, vq_loss, ppl = self.vq(z)
        return z_q, vq_loss, ppl, (gh, gw)

    def decode(self, z_q: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        queries = self._grid_pos(gh, gw, z_q.device, z_q.dtype).expand(z_q.shape[0], -1, -1)
        queries = queries + self.query_pos
        for blk in self.decoder_blocks:
            queries = blk(queries, z_q)
        patches = self.to_patch(queries)  # B, Gh*Gw, C*P*P
        x = rearrange(
            patches,
            "b (gh gw) (c ph pw) -> b c (gh ph) (gw pw)",
            gh=gh,
            gw=gw,
            c=self.in_channels,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # pad to multiple of patch
        b, c, h, w = x.shape
        ph = self.patch_size
        pad_h = (ph - h % ph) % ph
        pad_w = (ph - w % ph) % ph
        if pad_h or pad_w:
            x_in = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_in = x
        z_q, vq_loss, ppl, (gh, gw) = self.encode(x_in)
        recon = self.decode(z_q, gh, gw)
        recon = recon[:, :, :h, :w]
        return {"recon": recon, "vq_loss": vq_loss, "perplexity": ppl, "z_q": z_q}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def latitude_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_weight: torch.Tensor,
) -> torch.Tensor:
    """pred/target: B,C,H,W; lat_weight: B,H or H."""
    if lat_weight.dim() == 1:
        w = lat_weight.view(1, 1, -1, 1)
    else:
        w = lat_weight.view(lat_weight.shape[0], 1, -1, 1)
    err = (pred - target).pow(2) * w
    return err.mean()
