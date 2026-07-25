"""Lossless bitstream for Residual-FSQ indices (exact roundtrip) + official CR."""
from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass

import numpy as np
import torch

MAGIC = b"RFSQ1"
# header: magic(5) + H(u16) + W(u16) + gh_c(u16) + gw_c(u16) + gh_f(u16) + gw_f(u16)
#         + Lc(u16) + Lf(u16) + Cc(u16) + Cf(u16) + payload_len(u32)
HEADER_FMT = "!5sHHHHHHHHHHI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def official_cr(num_bytes: int, h: int, w: int, channels: int = 28, t: int = 1) -> float:
    """CR = 32 * T * C * H * W / (8 * B)."""
    return (32.0 * t * channels * h * w) / (8.0 * max(num_bytes, 1))


def min_bytes_for_max_cr(
    h: int, w: int, *, max_cr: float = 64.0, channels: int = 28, t: int = 1
) -> int:
    """Smallest B such that official_cr(B) ≤ max_cr (ceil)."""
    numer = 32.0 * t * channels * h * w
    return int(np.ceil(numer / (8.0 * max_cr)))


def _pack_indices(idx_c: np.ndarray, idx_f: np.ndarray, levels_c: int, levels_f: int) -> bytes:
    """Pack coarse/fine index arrays to compact bytes then zlib."""
    # idx shapes: (N_c,) and (N_f,) flattened over batch=1, tokens, channels
    if levels_c <= 256:
        c_bytes = idx_c.astype(np.uint8).tobytes()
    else:
        c_bytes = idx_c.astype(np.uint16).tobytes()
    if levels_f <= 256:
        f_bytes = idx_f.astype(np.uint8).tobytes()
    else:
        f_bytes = idx_f.astype(np.uint16).tobytes()
    raw = struct.pack("!II", len(c_bytes), len(f_bytes)) + c_bytes + f_bytes
    return zlib.compress(raw, level=9)


def _unpack_indices(
    payload: bytes, levels_c: int, levels_f: int, n_c: int, n_f: int
) -> tuple[np.ndarray, np.ndarray]:
    raw = zlib.decompress(payload)
    buf = io.BytesIO(raw)
    lc, lf = struct.unpack("!II", buf.read(8))
    c_raw = buf.read(lc)
    f_raw = buf.read(lf)
    dtype_c = np.uint8 if levels_c <= 256 else np.uint16
    dtype_f = np.uint8 if levels_f <= 256 else np.uint16
    idx_c = np.frombuffer(c_raw, dtype=dtype_c).astype(np.int64)
    idx_f = np.frombuffer(f_raw, dtype=dtype_f).astype(np.int64)
    if idx_c.size != n_c or idx_f.size != n_f:
        raise ValueError(f"index size mismatch: got {idx_c.size}/{idx_f.size}, expected {n_c}/{n_f}")
    return idx_c, idx_f


def indices_to_zq(indices: torch.Tensor, num_levels: int) -> torch.Tensor:
    """Map integer indices → quantized values in [-1,1] (same as UniformQuantizer)."""
    span = float(num_levels - 1)
    return indices.float() / span * 2.0 - 1.0


@dataclass
class BitstreamMeta:
    h: int
    w: int
    gh_c: int
    gw_c: int
    gh_f: int
    gw_f: int
    levels_c: int
    levels_f: int
    c_latent: int
    f_latent: int


def encode_indices(
    indices_c: torch.Tensor,
    indices_f: torch.Tensor,
    *,
    h: int,
    w: int,
    gh_c: int,
    gw_c: int,
    gh_f: int,
    gw_f: int,
    levels_c: int,
    levels_f: int,
    max_cr: float = 63.0,
) -> bytes:
    """
    Encode a single-frame pair of index tensors.
    indices_*: (1, N, C) or (N, C) int tensors.

    If zlib payload is smaller than the byte budget for official CR ≤ max_cr
    (default 63 — margin under the ×64 limit), append zero padding after the
    payload (decode uses payload_len from header).
    """
    ic = indices_c.detach().cpu().reshape(-1).numpy().astype(np.int64)
    ifr = indices_f.detach().cpu().reshape(-1).numpy().astype(np.int64)
    cc = int(indices_c.shape[-1])
    cf = int(indices_f.shape[-1])
    payload = _pack_indices(ic, ifr, levels_c, levels_f)
    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        int(h),
        int(w),
        int(gh_c),
        int(gw_c),
        int(gh_f),
        int(gw_f),
        int(levels_c),
        int(levels_f),
        cc,
        cf,
        len(payload),
    )
    blob = header + payload
    if max_cr is not None and max_cr > 0:
        target = min_bytes_for_max_cr(int(h), int(w), max_cr=float(max_cr))
        if len(blob) < target:
            blob = blob + bytes(target - len(blob))
    return blob


def decode_indices(blob: bytes) -> tuple[BitstreamMeta, torch.Tensor, torch.Tensor]:
    if len(blob) < HEADER_SIZE:
        raise ValueError("bitstream too short")
    magic, h, w, gh_c, gw_c, gh_f, gw_f, lc, lf, cc, cf, plen = struct.unpack(
        HEADER_FMT, blob[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    # Trailing CR-cap padding (if any) is ignored via payload_len.
    payload = blob[HEADER_SIZE : HEADER_SIZE + plen]
    n_c = int(gh_c) * int(gw_c) * int(cc)
    n_f = int(gh_f) * int(gw_f) * int(cf)
    ic, ifr = _unpack_indices(payload, lc, lf, n_c, n_f)
    meta = BitstreamMeta(h, w, gh_c, gw_c, gh_f, gw_f, lc, lf, cc, cf)
    idx_c = torch.from_numpy(ic.reshape(1, gh_c * gw_c, cc))
    idx_f = torch.from_numpy(ifr.reshape(1, gh_f * gw_f, cf))
    return meta, idx_c, idx_f
