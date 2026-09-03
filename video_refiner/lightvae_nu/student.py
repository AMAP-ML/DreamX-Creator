"""Causal non-uniform Wan2.2 decoder student (scheme1).

Same graph as ``wan.modules.vae2_2.Decoder3d`` (causal 3D conv, Attention,
DupUp3D shortcuts, per-frame feat_cache). The only change is per-stage channel
counts so high-res stages stay wide.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from lightvae_nu.vae_blocks import wan_vae22

_v = wan_vae22()
CACHE_T = _v.CACHE_T
AttentionBlock = _v.AttentionBlock
CausalConv3d = _v.CausalConv3d
RMS_norm = _v.RMS_norm
ResidualBlock = _v.ResidualBlock
Up_ResidualBlock = _v.Up_ResidualBlock
count_conv3d = _v.count_conv3d
unpatchify = _v.unpatchify

from lightvae_nu.channels import (
    SCHEME1_DIMS,
    TEMPORAL_UPSAMPLE,
    Z_DIM,
    validate_dims,
)

# Copied from WanVAEWrapper22 so this package does not import sr_dit_wrapper
# (that module pulls in the DiT).
MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]
STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]


class Decoder3dNU(nn.Module):
    """Decoder3d with an explicit 5-tuple of stage channels."""

    def __init__(
        self,
        z_dim: int = Z_DIM,
        dims: Sequence[int] = SCHEME1_DIMS,
        num_res_blocks: int = 2,
        temperal_upsample: Sequence[bool] = TEMPORAL_UPSAMPLE,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = validate_dims(dims)
        self.z_dim = z_dim
        self.dims = dims
        self.num_res_blocks = num_res_blocks
        self.temperal_upsample = tuple(temperal_upsample)

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        dim_mult_len = 4  # Wan2.2: 4 up stages, last has up_flag=False
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = (
                self.temperal_upsample[i]
                if i < len(self.temperal_upsample)
                else False
            )
            upsamples.append(
                Up_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temperal_upsample=t_up_flag,
                    up_flag=i != dim_mult_len - 1,
                )
            )
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(
            RMS_norm(dims[-1], images=False),
            nn.SiLU(),
            CausalConv3d(dims[-1], 12, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache=None,
        feat_idx=None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :]
                        .unsqueeze(2)
                        .to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class StudentVAE(nn.Module):
    """Decoder-only causal student. ``conv2`` is the teacher 1×1 and stays frozen."""

    def __init__(
        self,
        dims: Sequence[int] = SCHEME1_DIMS,
        z_dim: int = Z_DIM,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.dims = validate_dims(dims)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3dNU(
            z_dim=z_dim, dims=self.dims, dropout=dropout
        )
        self.mean = torch.tensor(MEAN, dtype=torch.float32)
        self.std = torch.tensor(STD, dtype=torch.float32)
        self.conv2.requires_grad_(False)
        self.clear_cache()

    @property
    def scale(self) -> List[torch.Tensor]:
        device = next(self.decoder.parameters()).device
        dtype = next(self.decoder.parameters()).dtype
        mean = self.mean.to(device=device, dtype=dtype)
        inv_std = (1.0 / self.std.to(device=device, dtype=dtype))
        return [mean, inv_std]

    def clear_cache(self) -> None:
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num

    def freeze_conv2(self) -> None:
        self.conv2.requires_grad_(False)

    def trainable_parameters(self):
        return (p for p in self.decoder.parameters() if p.requires_grad)

    def decode_causal(
        self,
        z: torch.Tensor,
        already_denorm: bool = False,
    ) -> torch.Tensor:
        """``z`` is ``[B, 48, T, H, W]``. Returns ``[B, 3, T_pix, H*16, W*16]``.

        ``z`` is the *normalized* teacher encode (same as ``WanVAE_.encode``)
        unless ``already_denorm=True``.
        """
        if z.dim() != 5:
            raise ValueError(f"expected [B,C,T,H,W], got {tuple(z.shape)}")
        scale = self.scale
        z_in = z
        if not already_denorm:
            z_in = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        self.clear_cache()
        x = self.conv2(z_in)
        outs = []
        for i in range(x.shape[2]):
            self._conv_idx = [0]
            out_ = self.decoder(
                x[:, :, i : i + 1],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                first_chunk=(i == 0),
            )
            outs.append(out_)
        out = torch.cat(outs, dim=2)
        out = unpatchify(out, patch_size=2)
        return out

    def forward(self, z: torch.Tensor, already_denorm: bool = False) -> torch.Tensor:
        """DDP-compatible alias of ``decode_causal``."""
        return self.decode_causal(z, already_denorm=already_denorm)
