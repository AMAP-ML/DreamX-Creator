from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _conv3x3(in_channels: int, out_channels: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs)


def _icnr_(weight: torch.Tensor, upscale_factor: int,
           initializer=nn.init.kaiming_normal_) -> None:
    """ICNR initialization for a sub-pixel conv (Conv -> PixelShuffle).

    Initializes so the r^2 output channels that feed each PixelShuffle output
    position share the same kernel -> at init PixelShuffle == nearest-neighbor
    upsampling (no checkerboard). In-place on `weight`. Pure weight init: does
    NOT change the module structure or forward pass.
    """
    r2 = int(upscale_factor) ** 2
    out_c = weight.shape[0]
    if r2 <= 1 or out_c % r2 != 0:
        return
    sub = torch.zeros(out_c // r2, *weight.shape[1:],
                      dtype=weight.dtype, device=weight.device)
    initializer(sub)
    sub = sub.repeat_interleave(r2, dim=0)  # [c0,c0,..(r2)..,c1,c1,..] matches PixelShuffle
    with torch.no_grad():
        weight.copy_(sub)


def _apply_icnr_to_head(head: nn.Sequential) -> None:
    """Apply ICNR to every Conv2d immediately followed by a PixelShuffle in `head`."""
    layers = list(head)
    for i, layer in enumerate(layers):
        if (isinstance(layer, nn.PixelShuffle) and i > 0
                and isinstance(layers[i - 1], nn.Conv2d)):
            conv = layers[i - 1]
            _icnr_(conv.weight, layer.upscale_factor)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)


def _set_bilinear_taps_(w: torch.Tensor, out_ch: int, in_ch: int,
                        si: int, sj: int, r: int, kh: int, kw: int,
                        sign: float = 1.0) -> None:
    """Set 3x3 kernel weights for one sub-pixel position (si, sj) to produce
    bilinear interpolation (align_corners=False, scale factor r)."""
    fy_off = (si + 0.5) / r - 0.5
    fx_off = (sj + 0.5) / r - 0.5
    y0 = -1 if fy_off < 0 else 0
    fy = fy_off + 1.0 if fy_off < 0 else fy_off
    x0 = -1 if fx_off < 0 else 0
    fx = fx_off + 1.0 if fx_off < 0 else fx_off
    cy, cx = kh // 2, kw // 2
    for dy, dx, wt in [(y0, x0, (1 - fy) * (1 - fx)),
                        (y0, x0 + 1, (1 - fy) * fx),
                        (y0 + 1, x0, fy * (1 - fx)),
                        (y0 + 1, x0 + 1, fy * fx)]:
        ky, kx = dy + cy, dx + cx
        if 0 <= ky < kh and 0 <= kx < kw and wt > 0:
            w[out_ch, in_ch, ky, kx] = sign * wt


def _nearest_identity_head_(head: nn.Sequential) -> None:
    """Init a (linear, no-activation) Conv->PixelShuffle head to EXACT
    nearest-neighbor upsampling of the input (channel identity).

    For each Conv2d(in=C, out=C*r*r) followed by PixelShuffle(r): set the center
    tap so output channel group c copies input channel c to all r*r positions.
    Result: head(x) == nearest-neighbor spatial upsample of x. In-place.
    Requires out_channels == in_channels * r*r (i.e. same #channels in/out).
    """
    layers = list(head)
    for i, layer in enumerate(layers):
        if not (isinstance(layer, nn.PixelShuffle) and i > 0
                and isinstance(layers[i - 1], nn.Conv2d)):
            continue
        conv = layers[i - 1]
        r = layer.upscale_factor
        r2 = r * r
        w = conv.weight  # [out, in, kh, kw]
        out_c, in_c, kh, kw = w.shape
        sub = out_c // r2
        ci, cj = kh // 2, kw // 2
        with torch.no_grad():
            w.zero_()
            for c in range(min(sub, in_c)):
                for j in range(r2):
                    w[c * r2 + j, c, ci, cj] = 1.0
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)


def _bilinear_identity_head_(head: nn.Sequential) -> None:
    """Init a (linear, no-activation) Conv->PixelShuffle head to EXACT
    bilinear upsampling (align_corners=False) of the input (channel identity).

    Like _nearest_identity_head_ but each output pixel is a bilinear blend of
    4 input neighbours instead of a copy of the nearest one. In-place.
    Requires out_channels == in_channels * r*r.
    """
    layers = list(head)
    for i, layer in enumerate(layers):
        if not (isinstance(layer, nn.PixelShuffle) and i > 0
                and isinstance(layers[i - 1], nn.Conv2d)):
            continue
        conv = layers[i - 1]
        r = layer.upscale_factor
        r2 = r * r
        w = conv.weight
        out_c, in_c, kh, kw = w.shape
        sub = out_c // r2
        with torch.no_grad():
            w.zero_()
            if conv.bias is not None:
                conv.bias.zero_()
            for c in range(min(sub, in_c)):
                for si in range(r):
                    for sj in range(r):
                        ch = c * r2 + si * r + sj
                        _set_bilinear_taps_(w, ch, c, si, sj, r, kh, kw)


def _make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "silu":
        return nn.SiLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'. Choose from relu/silu/gelu.")


def _build_upsample_head(
    in_channels: int,
    out_channels: int,
    upsample_scale: int,
    activation: str | None = None,
) -> nn.Sequential:
    """
    Build a multi-stage PixelShuffle upsampling head for any integer scale.

    An integer scale s is decomposed into n 2x upsamplings (the factors of 2
    in s) plus one PixelShuffle(r) for the remaining factor r.
    E.g. s=2 -> PS(2); s=4 -> PS(2)+PS(2); s=3 -> PS(3); s=6 -> PS(2)+PS(3).
    """
    if upsample_scale <= 0:
        raise ValueError(f"upsample_scale must be positive, got {upsample_scale}.")
    if upsample_scale == 1:
        layers: list[nn.Module] = [_conv3x3(in_channels, out_channels)]
        if activation is not None:
            layers.append(_make_activation(activation))
        return nn.Sequential(*layers)

    factors: list[int] = []
    remaining = upsample_scale
    while remaining % 2 == 0:
        factors.append(2)
        remaining //= 2
    if remaining > 1:
        factors.append(remaining)

    layers = []
    current_ch = in_channels
    for i, factor in enumerate(factors):
        is_last = (i == len(factors) - 1)
        next_ch = out_channels if is_last else out_channels
        layers.append(_conv3x3(current_ch, next_ch * factor * factor))
        layers.append(nn.PixelShuffle(factor))
        current_ch = next_ch
    if activation is not None:
        layers.append(_make_activation(activation))
    return nn.Sequential(*layers)


def _init_residual_free_identity_(pre_shuffle: nn.Sequential,
                                  blocks: nn.ModuleList,
                                  final: nn.Conv2d,
                                  in_ch: int, mid_ch: int, out_ch: int,
                                  scale: int) -> bool:
    """Initialize the WHOLE residual-free path to an EXACT nearest-neighbor
    upsample (pure weight init; structure/forward unchanged).

    Trick to survive the ReLUs on signed latents (pos/neg split):
      pre_shuffle conv -> shuffle-out ch 2c = +up(x_c), 2c+1 = -up(x_c)
      ReLU            -> h[2c]=relu(up(x_c)), h[2c+1]=relu(-up(x_c))
      each MemBlock   -> last conv zeroed => block(h)=relu(h)=h (h>=0) => identity
      final           -> out_c = h[2c] - h[2c+1] = up(x_c)   (exact nearest up)

    Requires mid_ch >= 2*in_ch, out_ch == in_ch, and pre_shuffle == a single
    [Conv2d -> PixelShuffle(scale) -> activation]. Returns True on success.
    """
    if out_ch != in_ch or mid_ch < 2 * in_ch:
        return False
    convs = [l for l in pre_shuffle if isinstance(l, nn.Conv2d)]
    shuffles = [l for l in pre_shuffle if isinstance(l, nn.PixelShuffle)]
    if len(convs) != 1 or len(shuffles) != 1:
        return False  # multi-stage head not supported
    conv0 = convs[0]
    r = shuffles[0].upscale_factor
    r2 = r * r
    if conv0.weight.shape[0] != mid_ch * r2:
        return False
    kh, kw = conv0.weight.shape[-2:]
    cc = (kh // 2, kw // 2)
    with torch.no_grad():
        conv0.weight.zero_()
        if conv0.bias is not None:
            conv0.bias.zero_()
        # shuffle-output channel m <- conv output channels [m*r2 : m*r2+r2]
        for c in range(in_ch):
            for j in range(r2):
                conv0.weight[(2 * c) * r2 + j, c, cc[0], cc[1]] = 1.0       # +up(x_c)
                conv0.weight[(2 * c + 1) * r2 + j, c, cc[0], cc[1]] = -1.0  # -up(x_c)
        # blocks -> identity: zero each block's last conv (conv branch = 0)
        for blk in blocks:
            last_conv = [m for m in blk.conv if isinstance(m, nn.Conv2d)][-1]
            last_conv.weight.zero_()
            if last_conv.bias is not None:
                last_conv.bias.zero_()
        # final: out_c = h[2c] - h[2c+1]
        final.weight.zero_()
        if final.bias is not None:
            final.bias.zero_()
        fkh, fkw = final.weight.shape[-2:]
        fcc = (fkh // 2, fkw // 2)
        for c in range(out_ch):
            final.weight[c, 2 * c, fcc[0], fcc[1]] = 1.0
            final.weight[c, 2 * c + 1, fcc[0], fcc[1]] = -1.0
    return True


def _init_residual_free_bilinear_(pre_shuffle: nn.Sequential,
                                  blocks: nn.ModuleList,
                                  final: nn.Conv2d,
                                  in_ch: int, mid_ch: int, out_ch: int,
                                  scale: int) -> bool:
    """Like _init_residual_free_identity_ but produces EXACT bilinear upsample
    (align_corners=False) instead of nearest-neighbor. Same pos/neg ReLU trick;
    only the pre_shuffle conv weights differ (4-tap bilinear kernel per sub-pixel
    instead of center-tap-only). Same constraints: mid>=2*in, out==in, single-stage head."""
    if out_ch != in_ch or mid_ch < 2 * in_ch:
        return False
    convs = [l for l in pre_shuffle if isinstance(l, nn.Conv2d)]
    shuffles = [l for l in pre_shuffle if isinstance(l, nn.PixelShuffle)]
    if len(convs) != 1 or len(shuffles) != 1:
        return False
    conv0 = convs[0]
    r = shuffles[0].upscale_factor
    r2 = r * r
    if conv0.weight.shape[0] != mid_ch * r2:
        return False
    kh, kw = conv0.weight.shape[-2:]
    with torch.no_grad():
        conv0.weight.zero_()
        if conv0.bias is not None:
            conv0.bias.zero_()
        for c in range(in_ch):
            for si in range(r):
                for sj in range(r):
                    pos_ch = (2 * c) * r2 + si * r + sj
                    neg_ch = (2 * c + 1) * r2 + si * r + sj
                    _set_bilinear_taps_(conv0.weight, pos_ch, c, si, sj, r, kh, kw, sign=1.0)
                    _set_bilinear_taps_(conv0.weight, neg_ch, c, si, sj, r, kh, kw, sign=-1.0)
        for blk in blocks:
            last_conv = [m for m in blk.conv if isinstance(m, nn.Conv2d)][-1]
            last_conv.weight.zero_()
            if last_conv.bias is not None:
                last_conv.bias.zero_()
        final.weight.zero_()
        if final.bias is not None:
            final.bias.zero_()
        fkh, fkw = final.weight.shape[-2:]
        fcc = (fkh // 2, fkw // 2)
        for c in range(out_ch):
            final.weight[c, 2 * c, fcc[0], fcc[1]] = 1.0
            final.weight[c, 2 * c + 1, fcc[0], fcc[1]] = -1.0
    return True


class FlashLatentMemBlock(nn.Module):
    """
    Lightweight TAEHV-style causal memory block.

    - Spatial modeling uses only Conv2d, so the 3x3 convs act on single frames;
    - Temporal fusion is a conv over [current feature | previous feature memory];
    - No attention masks / 3D convs, so no future-frame leakage.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            _conv3x3(in_channels * 2, out_channels),
            _make_activation(activation),
            _conv3x3(out_channels, out_channels),
            _make_activation(activation),
            _conv3x3(out_channels, out_channels),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = _make_activation(activation)

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], dim=1)) + self.skip(x))


class FlashLatentUpsampler(nn.Module):
    """
    FlashLatent upsampler with a configurable spatial upsampling scale.

    Structure:
    - Multi-stage PixelShuffle upsampling head (any integer scale 2/3/4/...)
    - TAEHV-style MemBlocks x num_blocks
    - Conv2d projection back to the output channels

    Input:
        [B, in_channels, T, H, W]
    Output:
        [B, out_channels, T, H*scale, W*scale]
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int | None = None,
        mid_channels: int = 128,
        num_blocks: int = 9,
        upsample_scale: int = 2,
        activation: str = "relu",
        residual: bool = False,
        residual_scale: float = 1.0,
        memory_init: Literal["zero", "replicate"] = "zero",
        zero_init_final: bool = True,
        default_parallel: bool = True,
        ema_alpha: float | None = None,
        init_mode: str = "nearest",
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")
        if memory_init not in ("zero", "replicate"):
            raise ValueError("memory_init must be 'zero' or 'replicate'.")
        if not isinstance(upsample_scale, int) or upsample_scale < 1:
            raise ValueError(f"upsample_scale must be a positive integer, got {upsample_scale}.")

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.mid_channels = mid_channels
        self.num_blocks = num_blocks
        self.upsample_scale = upsample_scale
        self.activation_name = activation
        self.residual = residual
        self.residual_scale = float(residual_scale)
        self.memory_init = memory_init
        self.default_parallel = bool(default_parallel)
        self.ema_alpha = float(ema_alpha) if ema_alpha is not None else None

        self.pre_shuffle = _build_upsample_head(
            in_channels, mid_channels, upsample_scale, activation=activation,
        )
        self.blocks = nn.ModuleList(
            [FlashLatentMemBlock(mid_channels, mid_channels, activation=activation) for _ in range(num_blocks)]
        )
        self.final = _conv3x3(mid_channels, self.out_channels)

        if self.residual:
            self.skip_up = _build_upsample_head(
                in_channels, self.out_channels, upsample_scale, activation=None,
            )
        else:
            self.skip_up = None

        self.init_mode = str(init_mode).lower()
        self._init_weights(self.init_mode, zero_init_final)

    def _init_weights(self, init_mode: str, zero_init_final: bool) -> None:
        """Unified weight initialization (pure init; structure/forward unchanged).

        init_mode:
          "default"  — PyTorch default conv init; honor `zero_init_final`.
          "icnr"     — ICNR on the sub-pixel heads (checkerboard-free nearest at
                       init); honor `zero_init_final`.
          "nearest"  — the UNTRAINED model IS an exact nearest-neighbor upsample
                       (best starting point). Mechanism auto-selected by `residual`:
                         residual=True : skip_up = identity-nearest, final = 0.
                         residual=False: pos/neg-through-ReLU identity over the whole
                                         path (needs mid>=2*in, out==in). Falls back
                                         to "icnr" if those conditions aren't met.
          "bilinear" — like "nearest" but the UNTRAINED model IS an exact bilinear
                       upsample (align_corners=False). Same constraints as "nearest".
        """
        import warnings
        if init_mode not in ("default", "icnr", "nearest", "bilinear"):
            raise ValueError(f"init_mode must be default/icnr/nearest/bilinear, got {init_mode!r}.")

        # ICNR on every sub-pixel head (base for icnr/nearest/bilinear modes).
        if init_mode in ("icnr", "nearest", "bilinear"):
            _apply_icnr_to_head(self.pre_shuffle)
            if self.skip_up is not None:
                _apply_icnr_to_head(self.skip_up)

        if init_mode in ("nearest", "bilinear"):
            if init_mode == "nearest":
                head_fn = _nearest_identity_head_
                free_fn = _init_residual_free_identity_
            else:
                head_fn = _bilinear_identity_head_
                free_fn = _init_residual_free_bilinear_

            if self.residual and self.skip_up is not None:
                head_fn(self.skip_up)
                nn.init.zeros_(self.final.weight)
                if self.final.bias is not None:
                    nn.init.zeros_(self.final.bias)
            elif not self.residual:
                ok = free_fn(
                    self.pre_shuffle, self.blocks, self.final,
                    self.in_channels, self.mid_channels, self.out_channels,
                    self.upsample_scale,
                )
                if not ok:
                    warnings.warn(
                        f"init_mode='{init_mode}' unavailable for residual=False "
                        "(needs mid_channels>=2*in_channels, out_channels==in_channels, "
                        "single-stage head) -> fell back to 'icnr'.")
            return

        # default / icnr: optionally zero the final projection.
        if zero_init_final:
            nn.init.zeros_(self.final.weight)
            if self.final.bias is not None:
                nn.init.zeros_(self.final.bias)

    def _initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        if self.memory_init == "replicate":
            return x
        return torch.zeros_like(x)

    def _forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, _, _ = x.shape
        h = rearrange(x, "b c t h w -> (b t) c h w")
        h = self.pre_shuffle(h)

        alpha = self.ema_alpha
        for block in self.blocks:
            _, c, hh, ww = h.shape
            h_bt = h.reshape(b, t, c, hh, ww)
            if self.memory_init == "replicate":
                first = h_bt[:, :1]
            else:
                first = h_bt[:, :1] * 0
            if alpha is not None:
                # EMA: past_t = α * h_{t-1} + (1-α) * past_{t-1}
                # Compute via cumulative weighted sum along time dimension
                weights = torch.tensor(
                    [(1.0 - alpha) ** i for i in range(t)],
                    device=h.device, dtype=h.dtype,
                ).flip(0).reshape(1, t, 1, 1, 1)
                # past_t = sum_{k=0}^{t-1} α*(1-α)^{t-1-k} * h_k
                # Use a sequential scan for exact EMA
                ema_states = torch.zeros_like(h_bt[:, :1])  # [B, 1, C, H, W]
                past_list = [first[:, 0]]
                for ti in range(1, t):
                    ema_states = alpha * h_bt[:, ti - 1:ti] + (1.0 - alpha) * ema_states
                    past_list.append(ema_states[:, 0])
                past = torch.stack(past_list, dim=1).reshape(h.shape)
            else:
                past = torch.cat([first, h_bt[:, :-1]], dim=1).reshape(h.shape)
            h = block(h, past)

        h = self.final(h)
        return rearrange(h, "(b t) c h w -> b c t h w", b=b, t=t)

    def _forward_sequential(
        self,
        x: torch.Tensor,
        caches: list[torch.Tensor | None] | None = None,
        detach_caches: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        b, _, t, _, _ = x.shape
        if caches is None:
            caches = [None] * len(self.blocks)
        if len(caches) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} caches, got {len(caches)}.")

        alpha = self.ema_alpha
        outputs = []
        block_caches = list(caches)
        for frame_idx in range(t):
            h = self.pre_shuffle(x[:, :, frame_idx])
            next_block_caches: list[torch.Tensor] = []
            for block_idx, block in enumerate(self.blocks):
                past = block_caches[block_idx]
                if past is None:
                    past = self._initial_memory(h)
                else:
                    past = past.to(device=h.device, dtype=h.dtype)
                h_next = block(h, past)
                h_cur = h.detach() if detach_caches else h
                if alpha is not None and past is not None:
                    next_cache = alpha * h_cur + (1.0 - alpha) * past
                else:
                    next_cache = h_cur
                next_block_caches.append(next_cache)
                h = h_next
            block_caches = next_block_caches
            outputs.append(self.final(h))
        out = torch.stack(outputs, dim=2)
        return out, block_caches

    def forward(
        self,
        latent: torch.Tensor,
        parallel: bool | None = None,
        caches: list[torch.Tensor | None] | None = None,
        return_caches: bool = False,
        detach_caches: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if latent.ndim != 5:
            raise ValueError(f"Expected latent shape [B, C, T, H, W], got {latent.shape}.")
        if latent.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {latent.shape[1]}.")

        if parallel is None:
            parallel = self.default_parallel
        if caches is not None and parallel:
            raise ValueError("caches are only supported when parallel=False.")

        if parallel:
            correction = self._forward_parallel(latent)
            new_caches = [None] * len(self.blocks)
        else:
            correction, new_caches = self._forward_sequential(
                latent,
                caches=caches,
                detach_caches=detach_caches,
            )

        if self.residual:
            b, _, t, _, _ = latent.shape
            skip = rearrange(latent, "b c t h w -> (b t) c h w")
            skip = self.skip_up(skip)
            skip = rearrange(skip, "(b t) c h w -> b c t h w", b=b, t=t)
            out = skip + self.residual_scale * correction
        else:
            out = correction

        if return_caches:
            return out, new_caches
        return out

# [B, t, c, h, w]: spatially upsample an LQ latent.