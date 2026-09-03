"""Lightweight 2D latent upsampler for Wan2.2 (48-ch latents).

Structure follows the Minimax-H3 2D latent upscaler (Comfyui_Minimax_h3_
latent_Upscaler, ``VideoLatentResizer`` = 2D ``LatentResizer`` backbone +
shared TemporalConv modules):

    conv_in (Conv2d, per-frame)
    in_blocks  : ResBlockEmb2D (GroupNorm+SiLU+Conv2d, FiLM scale embedding)
                 with temporal_in (shared depthwise TemporalConv) applied
                 every ``temporal_every`` blocks
    bilinear interpolate to target HW                 <- mid-network upsample
    out_blocks : same, with the shared temporal_out module
    norm_out + SiLU + conv_out (Conv2d)

vs the 3D variant (wan22_3d_latent_up): spatial compute is pure 2D (per-frame
conv kernels, 1/3 the taps of a 3x3x3 Conv3d) and temporal modeling is done by
only TWO shared depthwise TemporalConv modules (one per phase) instead of one
after every ``temporal_every`` ResBlocks — the reference's parameter-efficient
2D+temporal factorization.

Lightweight adaptations for Wan2.2 video SR (same as the 3D variant):
  - mid-network interpolation: in_blocks run at LR resolution, only out_blocks
    at HR -> roughly half the network runs at 1/4 the spatial compute.
  - global residual: ``out = bilinear(x) + conv_out(blocks(...))`` with
    zero-init ``conv_out`` -> the UNTRAINED model is an exact bilinear upsample,
    so training starts at the bilinear baseline. (The H3 reference has no
    residual; set ``residual=False`` for the faithful structure.)
  - no input latent normalization: the Wan2.2 VAE wrapper already applies the
    per-channel mean/std scaling at encode/decode time.

forward: [B, C, T, H, W] -> [B, C, T, H*scale, W*scale] (spatial-only upsample,
temporal length is preserved). ``scale`` / ``target_size=(T,H,W)`` /
``target_hw=(H,W)`` override the built-in ``upsample_scale``; ``parallel`` is
accepted (and ignored) for API compatibility with the trainer call sites.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _normalization(channels: int, num_groups: int) -> nn.GroupNorm:
    if channels % num_groups != 0:
        raise ValueError(
            f"channels ({channels}) must be divisible by num_groups ({num_groups}).")
    return nn.GroupNorm(num_groups, channels)


def _zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResBlockEmb2D(nn.Module):
    """Per-frame GroupNorm+SiLU+Conv2d residual block with FiLM scale
    conditioning. The output conv is zero-initialized so each block starts as
    an identity (same as the H3 reference's ``zero_module`` usage)."""

    def __init__(self, channels: int, emb_channels: int,
                 num_groups: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(
            _normalization(channels, num_groups),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * channels),
        )
        self.out_norm = _normalization(channels, num_groups)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            _zero_module(nn.Conv2d(channels, channels, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return x + h


class TemporalConv(nn.Module):
    """Depthwise temporal conv with zero-init pointwise projection (H3 style).

    GroupNorm+SiLU run per frame (2D, on the flattened batch-time dim) exactly
    like the reference; the depthwise (k_t,1,1) conv + pointwise projection mix
    temporal context. The zero-init pointwise makes it an identity at init.
    With ``causal=True`` the depthwise kernel is left-padded by k-1 zero frames
    (after norm+SiLU, so SiLU(0)=0 keeps the pad clean) -> output frame t
    depends only on frames <= t."""

    def __init__(self, channels: int, kernel_size: int = 3,
                 num_groups: int = 32, causal: bool = False) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.causal = bool(causal)
        self.norm = _normalization(channels, num_groups)
        self.dwconv = nn.Conv3d(
            channels, channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(0, 0, 0) if causal else (kernel_size // 2, 0, 0),
            groups=channels,
        )
        self.pwconv = _zero_module(nn.Conv3d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        y = rearrange(x, "b c t h w -> (b t) c h w")
        y = self.norm(y)  # per-frame 2D stats (reference behavior; causal-safe)
        y = rearrange(y, "(b t) c h w -> b c t h w", b=b, t=t)
        if self.causal:
            y = F.pad(y, (0, 0, 0, 0, self.kernel_size - 1, 0))
        return x + self.pwconv(self.dwconv(F.silu(y)))


class LatentUpsampler(nn.Module):
    """Lightweight 2D (+shared temporal) latent upsampler.

    Input:  [B, in_channels, T, H, W]
    Output: [B, out_channels, T, H*scale, W*scale]

    Defaults (~3.9M params for 48->48 channels, channels=128, 6+6 blocks):
    cheaper per-parameter than the 3D variant (2D kernels, only 2 shared
    temporal modules), hence the wider default `channels`.

    Causal mode (``causal=True``): the two shared TemporalConv modules
    left-pad their depthwise temporal kernels by k-1 zero frames, so output
    frame t depends only on input frames <= t (streaming-safe). Spatial convs
    and GroupNorm are already per-frame in both modes.
    """

    def __init__(
        self,
        in_channels: int = 48,
        out_channels: int | None = None,
        channels: int = 128,
        in_blocks: int = 6,
        out_blocks: int = 6,
        embed_dim: int = 64,
        num_groups: int = 32,
        dropout: float = 0.0,
        temporal_every: int = 2,
        temporal_kernel: int = 3,
        upsample_scale: int = 2,
        residual: bool = True,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if in_blocks <= 0 or out_blocks <= 0:
            raise ValueError("in_blocks/out_blocks must be positive.")
        if upsample_scale < 1:
            raise ValueError(f"upsample_scale must be >= 1, got {upsample_scale}.")

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.upsample_scale = int(upsample_scale)
        self.residual = bool(residual)
        self.temporal_every = int(temporal_every)
        self.causal = bool(causal)

        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.in_blocks = nn.ModuleList(
            [ResBlockEmb2D(channels, embed_dim,
                           num_groups=num_groups, dropout=dropout)
             for _ in range(in_blocks)])
        self.out_blocks = nn.ModuleList(
            [ResBlockEmb2D(channels, embed_dim,
                           num_groups=num_groups, dropout=dropout)
             for _ in range(out_blocks)])
        # One SHARED temporal module per phase (reference design): applied after
        # every `temporal_every` ResBlocks within the phase, weights shared.
        # With causal=True the depthwise temporal kernels are left-padded, so
        # output frame t only depends on input frames <= t (spatial convs are
        # already per-frame; GroupNorm here is per-frame 2D stats in both modes).
        self.temporal_in = (
            TemporalConv(channels, temporal_kernel, num_groups=num_groups,
                         causal=self.causal)
            if temporal_every > 0 else None)
        self.temporal_out = (
            TemporalConv(channels, temporal_kernel, num_groups=num_groups,
                         causal=self.causal)
            if temporal_every > 0 else None)

        self.norm_out = _normalization(channels, num_groups)
        self.conv_out = nn.Conv2d(channels, self.out_channels, 3, padding=1)

        if self.residual:
            # Zero-init so the untrained model is an exact bilinear upsample.
            nn.init.zeros_(self.conv_out.weight)
            if self.conv_out.bias is not None:
                nn.init.zeros_(self.conv_out.bias)

    def _resolve_target(self, x: torch.Tensor, scale, target_size, target_hw):
        """Return ((TH, TW), scale_for_embedding) or None if passthrough."""
        _, t, h, w = x.shape[1:]
        if target_hw is not None:
            th, tw = int(target_hw[0]), int(target_hw[1])
        elif target_size is not None:
            _, th, tw = (int(s) for s in target_size)
        elif scale is not None and int(round(scale)) != 1:
            s = float(scale)
            th, tw = int(round(h * s)), int(round(w * s))
        elif self.upsample_scale != 1:
            s = float(self.upsample_scale)
            th, tw = int(round(h * s)), int(round(w * s))
        else:
            return None
        if (th, tw) == (h, w):
            return None
        return (th, tw), 0.5 * (th / h + tw / w)

    def _run_phase(self, blocks: nn.ModuleList, temporal: nn.Module | None,
                   h: torch.Tensor, b: int, t: int, emb: torch.Tensor) -> torch.Tensor:
        """Run ResBlocks on [BT, C, h, w]; insert the shared temporal module
        (on the 5D view) after every `temporal_every` blocks."""
        for i, blk in enumerate(blocks):
            h = blk(h, emb)
            if temporal is not None and i % self.temporal_every == 0:
                h5 = rearrange(h, "(b t) c h w -> b c t h w", b=b, t=t)
                h5 = temporal(h5)
                h = rearrange(h5, "b c t h w -> (b t) c h w")
        return h

    def forward(
        self,
        latent: torch.Tensor,
        scale: float | None = None,
        target_size=None,       # (T, H, W); T is preserved
        target_hw=None,         # (H, W)
        parallel: bool | None = None,  # accepted for API compat; ignored
        **kwargs,
    ) -> torch.Tensor:
        if latent.ndim != 5:
            raise ValueError(f"Expected latent [B, C, T, H, W], got {latent.shape}.")
        if latent.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {latent.shape[1]}.")

        resolved = self._resolve_target(latent, scale, target_size, target_hw)
        if resolved is None:
            return latent
        (th, tw), s_eff = resolved

        b, _, t, _, _ = latent.shape
        scale_emb = torch.tensor([s_eff - 1.0], dtype=latent.dtype,
                                 device=latent.device).unsqueeze(0)
        emb = self.embed(scale_emb).expand(b * t, -1)

        h = rearrange(latent, "b c t h w -> (b t) c h w")
        h = self.conv_in(h)
        h = self._run_phase(self.in_blocks, self.temporal_in, h, b, t, emb)
        h = F.interpolate(h, size=(th, tw), mode="bilinear", align_corners=False)
        h = self._run_phase(self.out_blocks, self.temporal_out, h, b, t, emb)
        h = self.conv_out(F.silu(self.norm_out(h)))

        if self.residual:
            base = F.interpolate(
                rearrange(latent, "b c t h w -> (b t) c h w"),
                size=(th, tw), mode="bilinear", align_corners=False)
            if base.shape[1] != h.shape[1]:
                base = base[:, : h.shape[1]]
            h = base + h
        return rearrange(h, "(b t) c h w -> b c t h w", b=b, t=t)


__all__ = ["LatentUpsampler", "ResBlockEmb2D", "TemporalConv"]
