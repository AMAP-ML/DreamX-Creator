# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from __future__ import annotations
import math
import os
import warnings

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor import DTensor
try:
    from torch.distributed.tensor import Replicate, Shard
except Exception:  # pragma: no cover - older torch uses _tensor symbols
    from torch.distributed._tensor import Replicate, Shard


def _is_dtensor(x: object) -> bool:
    return hasattr(x, "to_local") and hasattr(x, "placements") and hasattr(x, "device_mesh")


def _is_replicate_placement(p: object) -> bool:
    return isinstance(p, Replicate) or p.__class__.__name__ == "Replicate"
try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    _HAS_DIFFUSERS = True
except ImportError:
    ConfigMixin = object
    ModelMixin = nn.Module
    _HAS_DIFFUSERS = False
    def register_to_config(fn):
        return fn

from .attention import (
    flash_attention,
    mask_free_window_attention,
    FLASH_ATTN_3_AVAILABLE,
    FLASH_ATTN_2_AVAILABLE,
)

if FLASH_ATTN_3_AVAILABLE:
    import flash_attn_interface
if FLASH_ATTN_2_AVAILABLE:
    import flash_attn

# FlexAttention for parallel block-wise causal attention (PyTorch 2.6+)
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    if os.environ.get("DISABLE_FLEX_COMPILE", "0") != "1":
        flex_attention = torch.compile(flex_attention, dynamic=False, mode="default")
    FLEX_ATTN_AVAILABLE = True
except ImportError:
    FLEX_ATTN_AVAILABLE = False

__all__ = ['WanModel', 'Transformer3DModel', 'WanAttentionBlock', 'sinusoidal_embedding_1d']

_FALLBACK_TEXT_CONTEXT_TOKEN_NUMBER = 512
_FALLBACK_FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2
_FALLBACK_STREAM_CHUNK_SIZE = 8

# `enable_window_attention` warns once per process when the two temporal rules'
# knobs are both set (see the warning text there). Tests reset it.
_WINDOW_KNOB_WARNED = False


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000, device=None):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len, device=device),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2, device=device).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def _rope_axis_dims(head_dim: int) -> tuple[int, int, int]:
    """Real (i.e. non-complex) RoPE dims along the T/H/W axes.

    head_dim=128 yields (44, 42, 42), i.e. complex freq dims (22, 21, 21).
    This split matches the Wan/FlashVSR `precompute_freqs_cis_3d()` convention.
    """
    return (head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6))


def _normalize_hw_stride(stride) -> tuple[int, int]:
    """Normalize `None` / `int` / `(s_h, s_w)` to `(s_h, s_w)`; non-positive raises."""
    if stride is None:
        return (1, 1)
    if isinstance(stride, int):
        s_h = s_w = stride
    else:
        s_h, s_w = (int(v) for v in stride)
    if s_h < 1 or s_w < 1:
        raise ValueError(f"rope_hw_stride must be >= 1, got {stride!r}")
    return (int(s_h), int(s_w))


def spatial_strided_freqs(freqs: torch.Tensor, stride) -> torch.Tensor:
    """Thin out the *spatial* columns of a RoPE freq table: row i takes source row `i*s`.

    Used by the native LQ-anchor: the anchor has only `h/s * w/s` tokens on the LQ
    grid, but their RoPE positions must land on HR coordinates `0, s, 2s, ...` --
    same range as the HR queries, s-times coarser stride. Because ``rope_apply``
    only does integer table lookups (``freqs[1][:h]``), "positions with a stride"
    is exactly equivalent to "the table's spatial columns pre-thinned", so
    ``rope_apply`` itself needs no change.

    Args:
        freqs: `[L, c]` complex with column layout `t | h | w`, split as in
            ``rope_apply`` (``c - 2*(c//3), c//3, c//3``).
        stride: int or `(s_h, s_w)`; h and w occupy separate column segments,
            so anisotropy comes for free.

    Returns:
        `[L, c]` complex. Temporal columns are kept as-is (anchor frame
        positions are not scaled); spatial row i comes from source row `i*s`.
        Any trailing rows beyond the source length are NaN -- a read faults
        immediately instead of silently using a plausible-looking wrong frequency.
    """
    s_h, s_w = _normalize_hw_stride(stride)
    if s_h == 1 and s_w == 1:
        return freqs
    c = freqs.size(1)
    t_cols = c - 2 * (c // 3)
    ch = slice(t_cols, t_cols + c // 3)
    cw = slice(t_cols + c // 3, c)
    rows = freqs.size(0)
    out = freqs.clone()
    for cols, s in ((ch, s_h), (cw, s_w)):
        if s == 1:
            continue
        n = (rows + s - 1) // s          # row i needs source row i*s
        n = min(n, rows)
        idx = torch.arange(n, device=freqs.device) * s
        valid = idx < rows
        idx = idx[valid]
        out[:idx.numel(), cols] = freqs[idx][:, cols]
        # NaN in BOTH parts: a scalar float('nan') would produce nan+0j, which
        # looks like half a valid value.
        out[idx.numel():, cols] = complex(float('nan'), float('nan'))
    return out


def _normalize_temporal_offsets(t_offset, batch_size: int) -> list[int]:
    if t_offset is None:
        return [0] * batch_size
    if isinstance(t_offset, int):
        return [t_offset] * batch_size
    if isinstance(t_offset, torch.Tensor):
        offsets = t_offset.detach().cpu().tolist()
    else:
        offsets = list(t_offset)
    if len(offsets) != batch_size:
        raise ValueError('temporal_offset length must match batch size')
    return [int(v) for v in offsets]


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs, t_offset=None):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    t_offsets = _normalize_temporal_offsets(t_offset, grid_sizes.size(0))

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        t_off = int(t_offsets[i])
        if t_off < 0 or t_off + f > freqs[0].size(0):
            raise ValueError(
                f'temporal_offset out of range for rope freqs: offset={t_off}, f={f}, cache={freqs[0].size(0)}. '
                'Call WanModel._ensure_rope_freqs() before rope_apply.'
            )
        if h > freqs[1].size(0) or w > freqs[2].size(0):
            raise ValueError(
                f'spatial size out of range for rope freqs: h={h}, w={w}, cache={freqs[1].size(0)}. '
                'Call WanModel._ensure_rope_freqs() before rope_apply.'
            )

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][t_off:t_off + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(dtype=x.dtype)


def _build_blockwise_causal_mask(num_frames, frame_seqlen, chunk_frames, device, total_tokens):
    """Build block-wise causal BlockMask for FlexAttention.

    Semantics identical to the serial chunk loop:
      - Within a chunk: bidirectional (all tokens see each other)
      - Across chunks: causal (chunk i sees chunks 0..i)

    Args:
        num_frames: number of latent frames
        frame_seqlen: tokens per frame (H * W in latent space)
        chunk_frames: frames per chunk (stream_chunk_size)
        device: torch device
        total_tokens: total number of tokens (num_frames * frame_seqlen)

    Returns:
        block_mask: FlexAttention BlockMask
        padded_len: padded sequence length (multiple of 128)
    """
    chunk_tokens = chunk_frames * frame_seqlen
    # FlexAttention requires sequence length to be a multiple of 128
    padded_len = math.ceil(total_tokens / 128) * 128
    # Compute the upper bound of visible kv positions for each query position
    # Pad to padded_len: padding positions get total_tokens (never matched)
    ends = torch.full((padded_len,), total_tokens, device=device, dtype=torch.long)
    frame_indices = torch.arange(0, total_tokens, step=chunk_tokens, device=device)
    for start in frame_indices:
        end = min(start + chunk_tokens, total_tokens)
        ends[start:end] = end

    def mask_fn(b, h, q_idx, kv_idx):
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

    block_mask = create_block_mask(
        mask_fn, B=None, H=None,
        Q_LEN=padded_len, KV_LEN=padded_len,
        device=device, _compile=False
    )

    DEBUG = False
    if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            # # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
            # #                     padded_length, KV_LEN=total_length + padded_length, device=device)
            # import cv2
            # mask = cv2.resize(block_mask[0, 0].cpu().float().numpy(), (1024, 1024))
            # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

    return block_mask, padded_len


def _build_teacher_forcing_mask(
    num_frames: int,
    frame_seqlen: int,
    chunk_frames: int,
    device: torch.device,
) -> "BlockMask":
    """Build teacher-forcing BlockMask: [clean tokens | noisy tokens].

    Clean tokens use block-wise causal attention among themselves.
    Noisy tokens attend to previous clean blocks plus the current noisy block.
    """
    if not FLEX_ATTN_AVAILABLE:
        raise RuntimeError("Teacher-forcing training requires FlexAttention (PyTorch 2.6+)")

    total_length = num_frames * frame_seqlen * 2
    padded_length = math.ceil(total_length / 128) * 128 - total_length
    clean_ends = num_frames * frame_seqlen

    context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

    attention_block_size = frame_seqlen * chunk_frames
    frame_indices = torch.arange(
        0, num_frames * frame_seqlen, step=attention_block_size,
        device=device, dtype=torch.long,
    )
    for start in frame_indices:
        context_ends[start:start + attention_block_size] = start + attention_block_size

    noisy_image_start_list = torch.arange(
        num_frames * frame_seqlen, total_length,
        step=attention_block_size, device=device, dtype=torch.long,
    )
    noisy_image_end_list = noisy_image_start_list + attention_block_size
    for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
        noise_noise_starts[start:end] = start
        noise_noise_ends[start:end] = end
        noise_context_ends[start:end] = block_index * attention_block_size

    def attention_mask(b, h, q_idx, kv_idx):
        clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
        c1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
        c2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
        noise_mask = (q_idx >= clean_ends) & (c1 | c2)
        eye_mask = q_idx == kv_idx
        return eye_mask | clean_mask | noise_mask

    return create_block_mask(
        attention_mask, B=None, H=None,
        Q_LEN=total_length + padded_length,
        KV_LEN=total_length + padded_length,
        _compile=False, device=device,
    )


def _build_teacher_forcing_anchor_mask(
    num_frames: int,
    frame_seqlen: int,
    chunk_frames: int,
    device: torch.device,
) -> "BlockMask":
    """Teacher-forcing BlockMask extended with a per-frame LR-anchor KV region.

    Query layout : [ clean (L) | noisy (L) ]              -> Q_LEN  = pad128(2L)
    Key   layout : [ clean (L) | noisy (L) | anchor (L) ] -> KV_LEN = pad128(3L)
    with L = num_frames * frame_seqlen.

    The main region (kv_idx < 2L) reproduces ``_build_teacher_forcing_mask``
    exactly (clean block-causal; noisy attends previous clean blocks + own block).
    The anchor region (2L <= kv_idx < 3L) is visible ONLY to noisy queries, and
    each noisy query at frame i attends ONLY its own frame's anchor column
    (frame-aligned 1:1, HR frame i <-> LR frame i) — clean queries never see it.
    Per-sample cond_drop is applied on top at attention time via a score_mod
    (see ``_make_anchor_drop_score_mod``), so this mask is drop-agnostic and can
    be cached across steps.
    """
    if not FLEX_ATTN_AVAILABLE:
        raise RuntimeError("Teacher-forcing training requires FlexAttention (PyTorch 2.6+)")

    L = num_frames * frame_seqlen
    main_len = 2 * L
    kv_real = 3 * L
    q_len = math.ceil(main_len / 128) * 128
    kv_len = math.ceil(kv_real / 128) * 128
    clean_ends = L

    # Per-query bounds (indexed by q_idx), identical to the plain TF mask.
    context_ends = torch.zeros(q_len, device=device, dtype=torch.long)
    noise_context_starts = torch.zeros(q_len, device=device, dtype=torch.long)
    noise_context_ends = torch.zeros(q_len, device=device, dtype=torch.long)
    noise_noise_starts = torch.zeros(q_len, device=device, dtype=torch.long)
    noise_noise_ends = torch.zeros(q_len, device=device, dtype=torch.long)

    attention_block_size = frame_seqlen * chunk_frames
    frame_indices = torch.arange(
        0, L, step=attention_block_size, device=device, dtype=torch.long)
    for start in frame_indices:
        context_ends[start:start + attention_block_size] = start + attention_block_size

    noisy_start_list = torch.arange(
        L, main_len, step=attention_block_size, device=device, dtype=torch.long)
    noisy_end_list = noisy_start_list + attention_block_size
    for block_index, (start, end) in enumerate(zip(noisy_start_list, noisy_end_list)):
        noise_noise_starts[start:end] = start
        noise_noise_ends[start:end] = end
        noise_context_ends[start:end] = block_index * attention_block_size

    fs = frame_seqlen
    two_l = main_len
    anchor_end = main_len + L

    def attention_mask(b, h, q_idx, kv_idx):
        clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
        c1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
        c2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
        noise_mask = (q_idx >= clean_ends) & (q_idx < two_l) & (c1 | c2)
        eye_mask = q_idx == kv_idx
        # Anchor region: noisy frame i <-> anchor frame i (frame-aligned 1:1).
        is_noisy = (q_idx >= clean_ends) & (q_idx < two_l)
        in_anchor = (kv_idx >= two_l) & (kv_idx < anchor_end)
        frame_q = (q_idx - clean_ends) // fs
        frame_a = (kv_idx - two_l) // fs
        anchor_ok = is_noisy & in_anchor & (frame_q == frame_a)
        return eye_mask | clean_mask | noise_mask | anchor_ok

    return create_block_mask(
        attention_mask, B=None, H=None,
        Q_LEN=q_len, KV_LEN=kv_len,
        _compile=False, device=device,
    )


def _make_anchor_drop_score_mod(keep_mask, anchor_start: int, anchor_end: int):
    """score_mod implementing per-sample cond_drop for the TF+anchor path.

    ``keep_mask`` is a bool tensor [B]; a False sample must attend to NO anchor.
    The anchor columns occupy kv_idx in [anchor_start, anchor_end); for dropped
    samples those columns get -inf (removed from softmax) while the clean/noisy
    columns are untouched — i.e. a genuine "no anchor" for that sample. Shapes are
    fixed (full batch, full KV), so only tensor *values* change across steps and
    the compiled flex graph is not re-traced.
    """
    def score_mod(score, b, h, q_idx, kv_idx):
        in_anchor = (kv_idx >= anchor_start) & (kv_idx < anchor_end)
        drop = ~keep_mask[b]
        return torch.where(
            in_anchor & drop, torch.full_like(score, float("-inf")), score)
    return score_mod


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        if _is_dtensor(x):
            orig_dtype = x.dtype
            orig_placements = x.placements
            mesh = x.device_mesh
            if not all(_is_replicate_placement(p) for p in orig_placements):
                x = x.redistribute(placements=[Replicate()] * mesh.ndim)
            x_local = x.to_local()
            x_full = x_local
            expected_dim = self.normalized_shape[0]
            if x_local.shape[-1] != expected_dim and dist.is_initialized():
                mesh_dim = None
                for placement in orig_placements:
                    if placement.__class__.__name__ == "Shard" and hasattr(placement, "dim"):
                        mesh_dim = placement.dim
                        break
                if hasattr(mesh, "get_group"):
                    group = mesh.get_group(mesh_dim=mesh_dim) if mesh_dim is not None else None
                elif hasattr(mesh, "get_all_groups"):
                    all_groups = mesh.get_all_groups()
                    group = all_groups[0] if all_groups else None
                else:
                    group = None
                world_size = dist.get_world_size(group)
                if world_size > 1 and expected_dim % x_local.shape[-1] == 0:
                    gathered = [torch.empty_like(x_local) for _ in range(world_size)]
                    dist.all_gather(gathered, x_local, group=group)
                    x_full = torch.cat(gathered, dim=-1)
            y_local = nn.functional.layer_norm(
                x_full.float(),
                self.normalized_shape,
                weight,
                bias,
                self.eps,
            ).to(dtype=orig_dtype)
            y = DTensor.from_local(
                y_local,
                device_mesh=mesh,
                placements=[Replicate()] * mesh.ndim,
            )
            if y.placements != orig_placements:
                y = y.redistribute(placements=orig_placements)
            return y
        return nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            weight,
            bias,
            self.eps,
        ).type_as(x)


class WanSelfAttention(nn.Module):
    """MMDiT self-attention layer: dense FA, dense block-causal, or window attention."""

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 sparse_causal=False,  # block-causal switch (legacy name; see the dense block-causal branch in forward)
                 stream_chunk_size=None,
                 text_context_token_number=None,
                 use_flex_causal=False,
                 use_window_attn=False,
                 window_causal_kv_len=None,
                 window_block_hw=(4, 4),
                 window_block_radius_hw=(2, 2),
                 window_block_t=None,
                 window_radius_t=None):
        assert dim % num_heads == 0
        super().__init__()
        # Mask-free causal block-grid window self-attention (optional, additive).
        # Disabled by default so existing dense/causal configs are unaffected.
        self.use_window_attn = use_window_attn
        self.window_chunk = None       # None = one-shot; set to bound peak memory
        # execution backend forwarded to mask_free_window_attention: "auto" =
        # gather (trains correctly, but keeps the gathered per-window KV live for
        # the backward); "flex" = flex_attention BlockMask (no KV gather
        # -> far lower training-backward memory, see
        # attention._blockgrid_flex_attend); "magi" = MagiAttention FFA over a
        # rect range table (needs window_radius_t); "triton" = fused Triton
        # kernels that gather and pack nothing (fastest, but FORWARD-ONLY — it
        # raises under autograd, so it is an inference backend only).
        self.window_attn_impl = "auto"
        # Chunk-causal window attention on the PARALLEL (no-KV-cache) forward, in
        # latent frames. None = off = the original mask-free path, whose time axis
        # is global and therefore BIDIRECTIONAL when no cache bounds it — fine for
        # a bidirectional model, wrong for teacher_forcing / diffusion_forcing.
        # Set it to the streaming `stream_kv_len` to align the two receptive
        # fields.
        self.window_causal_kv_len = window_causal_kv_len
        # Spatial geometry: contiguous block grid of window_block_hw query blocks,
        # each attending its clamped window_block_radius_hw neighborhood.
        # Temporal geometry: by default block_t / radius are derived from
        # stream_chunk_size / window_causal_kv_len at forward time. When
        # window_block_t / window_radius_t are given explicitly, the temporal rule
        # is chunk-anchored instead: every query attends its WHOLE own chunk
        # (stream_chunk_size frames) plus the last (2*window_radius_t+1)*
        # window_block_t history frames behind the chunk — the streaming and
        # parallel receptive fields are then frame-identical by construction.
        self.window_block_hw = tuple(window_block_hw)
        self.window_block_radius_hw = tuple(window_block_radius_hw)
        self.window_block_t = (None if window_block_t is None
                               else int(window_block_t))
        self.window_radius_t = (None if window_radius_t is None
                                else int(window_radius_t))
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.sparse_causal = sparse_causal  # block-causal attention
        self.stream_chunk_size = int(_FALLBACK_STREAM_CHUNK_SIZE if stream_chunk_size is None else stream_chunk_size)
        self.use_flex_causal = use_flex_causal and FLEX_ATTN_AVAILABLE  # parallel causal via FlexAttention
        self._flex_block_mask = None
        self._flex_mask_key = None
        self._flex_padded_len = None

        # projection layers for Q/K/V/O
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, temporal_offset=None, train_img=False, is_stream=False, pre_cache_k=None, pre_cache_v=None, kv_len=None, block_mask=None,
                anchor_k=None, anchor_v=None, anchor_scale=1.0, anchor_mode="v",
                anchor_keep=None, anchor_align="frame",
                anchor_window_scope="window", anchor_hw=None):
        """Run attention on variable-resolution videos.

        LQ-anchor (optional, dense streaming path only): when ``anchor_k``/``anchor_v``
        are provided, they are prepended to the attention K/V so the current chunk also
        attends to the low-res (LQ) version of the same chunk. The anchor is used for the
        attention computation ONLY and is never written back into the rolling KV cache.

        ``anchor_hw=(h_a, w_a)`` (the NATIVE LQ-anchor) says the anchor carries its
        own, coarser token grid. Only the mask-free window path can express that
        second geometry; the dense paths slice the anchor by the HR frame stride,
        so they raise instead of reading the wrong tokens.
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function (kept as nested fn to share logic)
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # ---- mask-free causal block-grid window path (optional, additive) ----
        # Intercepts before the dense / block-causal branches; only active when
        # explicitly enabled via WanModel.enable_window_attention(). Leaves all
        # existing code paths untouched when self.use_window_attn is False.
        if self.use_window_attn and not train_img:
            if block_mask is not None:
                # Would otherwise die deep inside the gather with a confusing
                # "grid inconsistent with s_cur" assert (s_cur is doubled).
                raise NotImplementedError(
                    "window attention does not support teacher forcing: the TF path "
                    "concatenates clean_x with the noisy latents (2x sequence) and "
                    "expresses visibility through block_mask, which the mask-free "
                    "window path cannot represent. Use generator_mode=diffusion_forcing "
                    "with window_causal_kv_len set, or disable use_window_attn."
                )
            roped_q = rope_apply(q, grid_sizes, freqs, t_offset=temporal_offset).type_as(v)
            roped_k = rope_apply(k, grid_sizes, freqs, t_offset=temporal_offset).type_as(v)
            # ── LQ-anchor (Scheme B, window path) ──
            # anchor K/V are already RoPE'd at each frame's absolute temporal
            # offset during the per-frame harvest; mask_free_window_attention
            # prepends each query frame's own anchor block (frame-aligned 1:1,
            # matching the dense streaming path) or the whole chunk when
            # anchor_align == "chunk". `anchor_window_scope` decides whether the
            # anchor is restricted to the query's own spatial window ("window",
            # default) or spans the whole frame/chunk ("global", the legacy
            # behaviour). Never enters the KV cache. `anchor_keep` (per-sample
            # cond_drop) is honoured inside the op: a dropped sample's anchor
            # columns are simply not gathered.
            win_anchor_k = win_anchor_v = None
            if anchor_k is not None and anchor_v is not None:
                win_anchor_k = (anchor_k * anchor_scale if anchor_mode == "k"
                                else anchor_k).type_as(v)
                win_anchor_v = (anchor_v * anchor_scale if anchor_mode == "v"
                                else anchor_v).type_as(v)
            # Parallel forward: no cache bounds the time axis, so opt in to the
            # chunk-causal gather. `+ stream_chunk_size` because the span counts
            # the query's own chunk, matching streaming's cache(kv_len) + current.
            causal_cf = causal_span = None
            if not is_stream and self.window_causal_kv_len is not None:
                causal_cf = int(self.stream_chunk_size)
                causal_span = int(self.window_causal_kv_len) + causal_cf
            win_out, win_ck, win_cv = mask_free_window_attention(
                roped_q, roped_k, v,
                grid_sizes=grid_sizes,
                pre_cache_k=pre_cache_k if is_stream else None,
                pre_cache_v=pre_cache_v if is_stream else None,
                kv_len=kv_len if is_stream else None,
                anchor_k=win_anchor_k, anchor_v=win_anchor_v,
                anchor_align=anchor_align,
                anchor_window_scope=anchor_window_scope,
                anchor_hw=anchor_hw,
                anchor_keep=anchor_keep,
                win_chunk=self.window_chunk,
                attn_impl=self.window_attn_impl,
                causal_chunk_frames=causal_cf,
                causal_kv_span=causal_span,
                block_t=(self.window_block_t
                         if self.window_block_t is not None
                         else int(self.stream_chunk_size)),
                block_hw=self.window_block_hw,
                block_radius_hw=self.window_block_radius_hw,
                block_radius_t=self.window_radius_t,
            )
            win_out = self.o(win_out.to(dtype=self.o.weight.dtype).flatten(2))
            if is_stream:
                return win_out, win_ck, win_cv
            return win_out

        cache_k = None
        cache_v = None

        if anchor_hw is not None and anchor_k is not None:
            # Every dense branch below addresses the anchor with the HR
            # frame_seqlen (h*w) — `ak_full[:, cs:ce]`, the TF concat, the varlen
            # k_lens. A coarser anchor would be silently mis-sliced into the wrong
            # frames, which is exactly the kind of plausible-looking wrong number
            # that costs a benchmark.
            raise NotImplementedError(
                f"the native LQ-anchor (anchor_hw={tuple(anchor_hw)}) is only "
                f"supported by the mask-free window path; this forward took a "
                f"dense branch (use_window_attn="
                f"{getattr(self, 'use_window_attn', False)}, train_img={train_img}).")

        # ---- dense path: chunk-level causal + streaming KV cache / teacher forcing ----
        if block_mask is not None:
            is_tf = s == int(seq_lens[0].item()) * 2
            if not is_tf:
                raise ValueError(
                    f"Teacher-forcing block_mask requires doubled sequence length, "
                    f"got s={s}, expected {int(seq_lens[0].item()) * 2}"
                )
            q_chunk = torch.chunk(q, 2, dim=1)
            k_chunk = torch.chunk(k, 2, dim=1)
            roped_query = torch.cat([
                rope_apply(q_chunk[0], grid_sizes, freqs, t_offset=temporal_offset).type_as(v),
                rope_apply(q_chunk[1], grid_sizes, freqs, t_offset=temporal_offset).type_as(v),
            ], dim=1)
            roped_key = torch.cat([
                rope_apply(k_chunk[0], grid_sizes, freqs, t_offset=temporal_offset).type_as(v),
                rope_apply(k_chunk[1], grid_sizes, freqs, t_offset=temporal_offset).type_as(v),
            ], dim=1)

            half_dtype = torch.bfloat16
            if anchor_k is not None and anchor_v is not None:
                # ── TF + LR-anchor (Scheme B, teacher-forcing) ──
                # Append the per-frame LR anchor K/V (already RoPE'd at
                # extraction, frame-aligned to HR) as extra KV columns, so the
                # sequence becomes [clean(L) | noisy(L) | anchor(L)] on the KV
                # side while queries stay [clean(L) | noisy(L)]. `block_mask`
                # here is the anchor-extended TF mask (Q=pad128(2L),
                # KV=pad128(3L)); it lets each noisy frame i attend its own
                # anchor frame i only. Per-sample cond_drop is applied via
                # score_mod (dropped samples get -inf on the anchor columns).
                anchor_len = anchor_k.shape[1]
                ak = anchor_k * anchor_scale if anchor_mode == "k" else anchor_k
                av = anchor_v * anchor_scale if anchor_mode == "v" else anchor_v
                key_cat = torch.cat([roped_key, ak.type_as(roped_key)], dim=1)
                val_cat = torch.cat([v, av.type_as(v)], dim=1)

                q_pad = math.ceil(s / 128) * 128 - s
                kv_real = s + anchor_len
                kv_pad = math.ceil(kv_real / 128) * 128 - kv_real
                padded_roped_query = torch.cat([
                    roped_query, roped_query.new_zeros(b, q_pad, n, d)], dim=1)
                padded_roped_key = torch.cat([
                    key_cat, key_cat.new_zeros(b, kv_pad, n, d)], dim=1)
                padded_v = torch.cat([
                    val_cat, val_cat.new_zeros(b, kv_pad, n, d)], dim=1)

                score_mod = None
                if anchor_keep is not None:
                    keep = anchor_keep.to(device=roped_query.device).view(-1).bool()
                    if not bool(keep.all()):
                        score_mod = _make_anchor_drop_score_mod(keep, s, kv_real)

                out = flex_attention(
                    query=padded_roped_query.transpose(1, 2).to(half_dtype),
                    key=padded_roped_key.transpose(1, 2).to(half_dtype),
                    value=padded_v.transpose(1, 2).to(half_dtype),
                    block_mask=block_mask,
                    score_mod=score_mod,
                )
                x = out.transpose(1, 2)[:, :s].to(roped_query.dtype)
            else:
                padded_length = math.ceil(s / 128) * 128 - s
                padded_roped_query = torch.cat([
                    roped_query,
                    roped_query.new_zeros(b, padded_length, n, d),
                ], dim=1)
                padded_roped_key = torch.cat([
                    roped_key,
                    roped_key.new_zeros(b, padded_length, n, d),
                ], dim=1)
                padded_v = torch.cat([
                    v,
                    v.new_zeros(b, padded_length, n, d),
                ], dim=1)

                out = flex_attention(
                    query=padded_roped_query.transpose(1, 2).to(half_dtype),
                    key=padded_roped_key.transpose(1, 2).to(half_dtype),
                    value=padded_v.transpose(1, 2).to(half_dtype),
                    block_mask=block_mask,
                )
                x = out.transpose(1, 2)[:, :s].to(roped_query.dtype)

        else:
            roped_q = rope_apply(q, grid_sizes, freqs, t_offset=temporal_offset).type_as(v)
            roped_k = rope_apply(k, grid_sizes, freqs, t_offset=temporal_offset).type_as(v)

            if is_stream:
                # Streaming inference: concat history KV -> flash_attention
                # (the KV cache is causal by construction).
                if pre_cache_k is not None and pre_cache_v is not None:
                    full_k = torch.cat([pre_cache_k, roped_k], dim=1)
                    full_v = torch.cat([pre_cache_v, v], dim=1)
                else:
                    full_k = roped_k
                    full_v = v
                # ── LQ-anchor (Scheme B, streaming): STRICT frame alignment ──
                # Match the training TF anchor mask (frame_q == frame_a): the
                # current chunk's frame i attends ONLY to its own LR anchor frame i
                # (plus all history + current-chunk KV carried by full_k/full_v).
                # anchor_k/anchor_v cover exactly this chunk's frames, frame-aligned
                # and already RoPE'd at each frame's absolute temporal offset at
                # extraction. Done as a per-frame loop (cf. the bidirectional branch
                # above) so the anchor stays 1:1 rather than chunk-level. The anchor
                # is never written into cache_k/cache_v below, so cross-chunk
                # streaming is unaffected.
                if anchor_k is not None and anchor_v is not None:
                    ak = (anchor_k * anchor_scale if anchor_mode == "k" else anchor_k).type_as(full_k)
                    av = (anchor_v * anchor_scale if anchor_mode == "v" else anchor_v).type_as(full_v)
                    if anchor_align == "chunk":
                        # Per-chunk anchor: the ENTIRE anchor chunk is
                        # prepended and every query attends all of it
                        # (matches the per-chunk training/inference
                        # semantics, e.g. Causal-Forcing-VSR).
                        k_f = torch.cat([ak, full_k], dim=1)
                        v_f = torch.cat([av, full_v], dim=1)
                        x = flash_attention(roped_q, k_f, v_f)
                    else:
                        f0 = int(grid_sizes[0, 0].item())
                        frame_seqlen = int(grid_sizes[0, 1].item()) * int(grid_sizes[0, 2].item())
                        outs = []
                        for fi in range(f0):
                            cs = fi * frame_seqlen
                            ce = cs + frame_seqlen
                            k_f = torch.cat([ak[:, cs:ce], full_k], dim=1)
                            v_f = torch.cat([av[:, cs:ce], full_v], dim=1)
                            outs.append(flash_attention(roped_q[:, cs:ce], k_f, v_f))
                        x = torch.cat(outs, dim=1)
                        if x.shape[1] < roped_q.shape[1]:
                            pad = roped_q.shape[1] - x.shape[1]
                            x = torch.cat([x, x.new_zeros(b, pad, n, d)], dim=1)
                else:
                    x = flash_attention(roped_q, full_k, full_v)
                cache_k = full_k
                cache_v = full_v
                if kv_len is not None and kv_len > 0:
                    chunk_tokens = roped_k.shape[1]
                    max_tokens = kv_len * chunk_tokens
                    if cache_k.shape[1] > max_tokens:
                        cache_k = cache_k[:, -max_tokens:]
                        cache_v = cache_v[:, -max_tokens:]

            elif self.sparse_causal:
                # Training-time causal: chunk-level causal attention. Frames
                # within a chunk see each other; chunks only see past chunks --
                # the same semantics as KV-cache inference.
                f0 = int(grid_sizes[0, 0].item()) # total frames
                frame_seqlen = int(grid_sizes[0, 1].item()) * int(grid_sizes[0, 2].item())
                chunk_frames = self.stream_chunk_size
                chunk_tokens = chunk_frames * frame_seqlen
                total_tokens = f0 * frame_seqlen
                num_chunks = (total_tokens + chunk_tokens - 1) // chunk_tokens

                if self.use_flex_causal and FLEX_ATTN_AVAILABLE:
                    # ── FlexAttention parallel path (single kernel launch) ──
                    mask_key = (total_tokens, chunk_tokens, roped_q.device)
                    if self._flex_mask_key != mask_key:
                        self._flex_block_mask, self._flex_padded_len = _build_blockwise_causal_mask(
                            f0, frame_seqlen, chunk_frames, roped_q.device, total_tokens
                        )
                        self._flex_mask_key = mask_key

                    half_dtype = torch.bfloat16
                    q_3d = roped_q[0, :total_tokens].to(half_dtype)  # [L, H, D]
                    k_3d = roped_k[0, :total_tokens].to(half_dtype)
                    v_3d = v[0, :total_tokens].to(half_dtype)

                    padded_len = self._flex_padded_len
                    pad_size = padded_len - total_tokens
                    if pad_size > 0:
                        q_3d = torch.cat([q_3d, q_3d.new_zeros(pad_size, n, d)], dim=0)
                        k_3d = torch.cat([k_3d, k_3d.new_zeros(pad_size, n, d)], dim=0)
                        v_3d = torch.cat([v_3d, v_3d.new_zeros(pad_size, n, d)], dim=0)

                    # FlexAttention expects [B, H, L, D]
                    out = flex_attention(
                        query=q_3d.transpose(0, 1).unsqueeze(0),   # [1, H, padded_len, D]
                        key=k_3d.transpose(0, 1).unsqueeze(0),
                        value=v_3d.transpose(0, 1).unsqueeze(0),
                        block_mask=self._flex_block_mask,
                    )  # [1, H, padded_len, D]

                    x_cat = out[0, :, :total_tokens, :].transpose(0, 1)  # [total_tokens, H, D]
                    if total_tokens < roped_q.shape[1]:
                        x = roped_q.new_zeros(1, roped_q.shape[1], n, d)
                        x[0, :total_tokens] = x_cat.to(roped_q.dtype)
                    else:
                        x = x_cat.unsqueeze(0).to(roped_q.dtype)

                else:
                    # ── Serial chunk-loop path ──
                    half_dtype = torch.bfloat16
                    q_3d = roped_q[0, :total_tokens].to(half_dtype)  # [L, H, D] contiguous
                    k_3d = roped_k[0, :total_tokens].to(half_dtype)
                    v_3d = v[0, :total_tokens].to(half_dtype)

                    chunks_out = []
                    for ci in range(num_chunks):
                        qs = ci * chunk_tokens
                        qe = min(qs + chunk_tokens, total_tokens)
                        kve = qe  # chunk i attends [0, qe)

                        q_chunk = q_3d[qs:qe]       # contiguous slice
                        k_chunk = k_3d[:kve]         # contiguous slice
                        v_chunk = v_3d[:kve]         # contiguous slice

                        q_len = qe - qs
                        kv_len_i = kve
                        cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=q_chunk.device)
                        cu_k = torch.tensor([0, kv_len_i], dtype=torch.int32, device=k_chunk.device)

                        if FLASH_ATTN_3_AVAILABLE:
                            o_chunk = flash_attn_interface.flash_attn_varlen_func(
                                q=q_chunk, k=k_chunk, v=v_chunk,
                                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                seqused_q=None, seqused_k=None,
                                max_seqlen_q=q_len, max_seqlen_k=kv_len_i,
                                softmax_scale=None, causal=False,
                                deterministic=False,
                            )
                        else:
                            o_chunk = flash_attn.flash_attn_varlen_func(
                                q=q_chunk, k=k_chunk, v=v_chunk,
                                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                max_seqlen_q=q_len, max_seqlen_k=kv_len_i,
                                dropout_p=0.0, softmax_scale=None, causal=False,
                                deterministic=False,
                            )
                        chunks_out.append(o_chunk)

                    x_cat = torch.cat(chunks_out, dim=0)  # [total_tokens, H, D]
                    if total_tokens < roped_q.shape[1]:
                        x = roped_q.new_zeros(1, roped_q.shape[1], n, d)
                        x[0, :total_tokens] = x_cat.to(roped_q.dtype)
                    else:
                        x = x_cat.unsqueeze(0).to(roped_q.dtype)

            elif anchor_k is not None and anchor_v is not None:
                # ── Scheme B: bidirectional HR + per-frame LR anchor (frame-aligned) ──
                # The HR self-attention stays fully bidirectional over the whole
                # sequence; additionally every HR query frame attends the LR
                # anchor K/V of ITS OWN frame (1:1: HR frame i <-> LR frame i).
                # anchor_k/anchor_v were RoPE'd at each frame's temporal offset
                # during extraction and laid out frame-aligned with the HR
                # sequence. The block is split into a per-frame query loop where
                # each flash_attention call uses keys = [own frame's anchor | all
                # HR], so no mask is needed. (Decoupled from stream_chunk_size:
                # the anchor is frame-aligned by default.) With
                # anchor_align == "chunk" it becomes a single forward with
                # keys = [whole anchor | all HR], matching per-chunk
                # training/inference semantics.
                f0 = int(grid_sizes[0, 0].item())
                frame_seqlen = int(grid_sizes[0, 1].item()) * int(grid_sizes[0, 2].item())
                total_tokens = f0 * frame_seqlen

                ak_full = anchor_k.type_as(roped_k)
                av_full = anchor_v.type_as(v)
                if anchor_mode == "k":
                    ak_full = ak_full * anchor_scale
                else:  # "v"
                    av_full = av_full * anchor_scale

                k_hr = roped_k[:, :total_tokens]
                v_hr = v[:, :total_tokens]

                # Per-sample anchor drop (unified cond_drop entry). anchor_keep is a
                # bool [B] mask; a False sample must attend to NO anchor. To express
                # that in one batched flash call we put the anchor AFTER the HR keys
                # and use a per-sample k_lens: a kept sample sees [HR | anchor], a
                # dropped sample's k_len stops at total_tokens so its trailing anchor
                # is varlen-masked (−inf) — no value, no softmax mass, i.e. genuinely
                # "no anchor". anchor_keep=None (inference / no drop) keeps the
                # original anchor-as-prefix concat, numerically unchanged.
                # anchor_align: "frame" walks the query frames and shows each
                # one only its OWN anchor frame (1:1); "chunk" is a single
                # pass where every query sees the ENTIRE anchor (matches the
                # per-chunk streaming/rollout semantics). The frame case is
                # byte-for-byte the original loop (q_step == frame_seqlen).
                q_step = total_tokens if anchor_align == "chunk" else frame_seqlen
                outs = []
                for cs in range(0, total_tokens, q_step):
                    ce = min(cs + q_step, total_tokens)
                    q_c = roped_q[:, cs:ce]
                    if anchor_align == "chunk":
                        a_k, a_v = ak_full[:, :total_tokens], av_full[:, :total_tokens]
                    else:
                        a_k, a_v = ak_full[:, cs:ce], av_full[:, cs:ce]
                    if anchor_keep is None:
                        k_all = torch.cat([a_k, k_hr], dim=1)
                        v_all = torch.cat([a_v, v_hr], dim=1)
                        outs.append(flash_attention(
                            q=q_c, k=k_all, v=v_all,
                            window_size=self.window_size))
                    else:
                        anchor_len = a_k.shape[1]
                        k_all = torch.cat([k_hr, a_k], dim=1)
                        v_all = torch.cat([v_hr, a_v], dim=1)
                        keep = anchor_keep.to(device=q_c.device).view(-1).bool()
                        k_lens = torch.where(
                            keep,
                            torch.full((b,), total_tokens + anchor_len,
                                       dtype=torch.int32, device=q_c.device),
                            torch.full((b,), total_tokens,
                                       dtype=torch.int32, device=q_c.device),
                        )
                        outs.append(flash_attention(
                            q=q_c, k=k_all, v=v_all, k_lens=k_lens,
                            window_size=self.window_size))
                x = torch.cat(outs, dim=1)
                if total_tokens < roped_q.shape[1]:
                    pad = roped_q.shape[1] - total_tokens
                    x = torch.cat([x, x.new_zeros(b, pad, n, d)], dim=1)

            else:
                # Non-causal: standard bidirectional flash attention.
                x = flash_attention(
                    q=roped_q, k=roped_k, v=v,
                    k_lens=seq_lens,
                    window_size=self.window_size)

        x = x.to(dtype=self.o.weight.dtype)

        # output projection expects shape [B, L, C]
        x = x.flatten(2)
        x = self.o(x)
        if is_stream:
            return x, cache_k, cache_v
        return x


class WanT2VCrossAttention(WanSelfAttention):
    """Cross-attention block that reuses encoder context across diffusion steps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_k = None
        self.cache_v = None

    def init_cache(self, context):
        b = context.size(0)
        n = self.num_heads
        d = self.head_dim
        self.cache_k = self.norm_k(self.k(context)).view(b, -1, n, d)
        self.cache_v = self.v(context).view(b, -1, n, d)

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, context, context_lens):
        """Attend from video tokens (`x`) to cached text/image context."""
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        if self.cache_k is not None and self.cache_v is not None:
            # re-use cached context from previous call (saves recomputation)
            k = self.cache_k
            v = self.cache_v
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 sparse_causal=False,  # unused: kept for signature parity with WanSelfAttention
                 text_context_token_number=None):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.text_context_token_number = int(
            _FALLBACK_TEXT_CONTEXT_TOKEN_NUMBER if text_context_token_number is None else text_context_token_number
        )
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.cache_k = None
        self.cache_v = None
        self.cache_k_img = None
        self.cache_v_img = None

    def init_cache(self, context):
        image_context_length = context.shape[1] - self.text_context_token_number
        context_img = context[:, :image_context_length]
        context_txt = context[:, image_context_length:]
        b = context.size(0)
        n = self.num_heads
        d = self.head_dim
        self.cache_k = self.norm_k(self.k(context_txt)).view(b, -1, n, d)
        self.cache_v = self.v(context_txt).view(b, -1, n, d)
        self.cache_k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        self.cache_v_img = self.v_img(context_img).view(b, -1, n, d)

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None
        self.cache_k_img = None
        self.cache_v_img = None

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - self.text_context_token_number
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        if self.cache_k is not None and self.cache_v is not None:
            k = self.cache_k
            v = self.cache_v
            k_img = self.cache_k_img
            v_img = self.cache_v_img
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
            v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 sparse_causal=False,  # block-causal switch at the block level
                 text_context_token_number=None,
                 stream_chunk_size=None,
                 use_flex_causal=False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.sparse_causal = sparse_causal  # block-causal attention
        self.stream_chunk_size = int(_FALLBACK_STREAM_CHUNK_SIZE if stream_chunk_size is None else stream_chunk_size)
        self.text_context_token_number = int(
            _FALLBACK_TEXT_CONTEXT_TOKEN_NUMBER if text_context_token_number is None else text_context_token_number
        )

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            sparse_causal=sparse_causal,  # pass the block-causal switch to self-attention
            stream_chunk_size=self.stream_chunk_size,
            use_flex_causal=use_flex_causal,
        )
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps,
                                                                      text_context_token_number=self.text_context_token_number)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        temporal_offset=None,
        train_img=False,
        is_stream=False,
        pre_cache_k=None,
        pre_cache_v=None,
        kv_len=None,
        block_mask=None,
        teacher_forcing=False,
        anchor_k=None,
        anchor_v=None,
        anchor_scale=1.0,
        anchor_mode="v",
        anchor_keep=None,
        anchor_align="frame",
        anchor_window_scope="window",
        anchor_hw=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        modulation = self.modulation
        if isinstance(e, DTensor) and not isinstance(modulation, DTensor):
            if modulation.device != e.device:
                modulation = modulation.to(e.device)
            modulation = DTensor.from_local(
                modulation,
                device_mesh=e.device_mesh,
                placements=e.placements,
            )
        elif isinstance(e, torch.Tensor) and isinstance(modulation, DTensor):
            modulation = modulation.to_local()
        orig_dtype = x.dtype
        cache_k = cache_v = None

        def cross_attn_ffn(x_in, e_mod_parts):
            x_out = x_in + self.cross_attn(self.norm3(x_in), context, context_lens)
            ffn_in = self.norm2(x_out).float() * (1 + e_mod_parts[4]) + e_mod_parts[3]
            y_ffn = self.ffn(ffn_in.to(dtype=orig_dtype))
            with amp.autocast(dtype=torch.float32):
                x_out = x_out + y_ffn * e_mod_parts[5]
            return x_out.to(dtype=orig_dtype)

        if teacher_forcing and e.dim() == 4 and e.shape[1] == 2:
            half = x.shape[1] // 2
            x_clean, x_noisy = x[:, :half], x[:, half:]
            with amp.autocast(dtype=torch.float32):
                e_clean = (modulation + e[:, 0]).chunk(6, dim=1)
                e_noisy = (modulation + e[:, 1]).chunk(6, dim=1)
            attn_in = torch.cat([
                self.norm1(x_clean).float() * (1 + e_clean[1]) + e_clean[0],
                self.norm1(x_noisy).float() * (1 + e_noisy[1]) + e_noisy[0],
            ], dim=1).to(dtype=orig_dtype)
            if is_stream:
                y, cache_k, cache_v = self.self_attn(
                    attn_in, seq_lens, grid_sizes, freqs,
                    temporal_offset=temporal_offset, train_img=train_img,
                    is_stream=True, pre_cache_k=pre_cache_k, pre_cache_v=pre_cache_v,
                    kv_len=kv_len, block_mask=block_mask,
                )
            else:
                y = self.self_attn(
                    attn_in, seq_lens, grid_sizes, freqs,
                    temporal_offset=temporal_offset, train_img=train_img,
                    block_mask=block_mask,
                    anchor_k=anchor_k,
                    anchor_v=anchor_v,
                    anchor_scale=anchor_scale,
                    anchor_mode=anchor_mode,
                    anchor_keep=anchor_keep,
                )
            y_clean, y_noisy = y[:, :half], y[:, half:]
            with amp.autocast(dtype=torch.float32):
                x = torch.cat([
                    x_clean + y_clean * e_clean[2],
                    x_noisy + y_noisy * e_noisy[2],
                ], dim=1)
            x = x.to(dtype=orig_dtype)
            x_clean, x_noisy = x[:, :half], x[:, half:]
            x = torch.cat([
                cross_attn_ffn(x_clean, e_clean),
                cross_attn_ffn(x_noisy, e_noisy),
            ], dim=1)
        else:
            with amp.autocast(dtype=torch.float32):
                e_parts = (modulation + e).chunk(6, dim=1)
            assert e_parts[0].dtype == torch.float32
            attn_in = self.norm1(x).float() * (1 + e_parts[1]) + e_parts[0]
            if is_stream:
                y, cache_k, cache_v = self.self_attn(
                    attn_in.to(dtype=orig_dtype),
                    seq_lens,
                    grid_sizes,
                    freqs,
                    temporal_offset=temporal_offset,
                    train_img=train_img,
                    is_stream=True,
                    pre_cache_k=pre_cache_k,
                    pre_cache_v=pre_cache_v,
                    kv_len=kv_len,
                    block_mask=block_mask,
                    anchor_k=anchor_k,
                    anchor_v=anchor_v,
                    anchor_scale=anchor_scale,
                    anchor_mode=anchor_mode,
                    anchor_keep=anchor_keep,
                    anchor_align=anchor_align,
                    anchor_window_scope=anchor_window_scope,
                    anchor_hw=anchor_hw,
                )
            else:
                y = self.self_attn(
                    attn_in.to(dtype=orig_dtype),
                    seq_lens,
                    grid_sizes,
                    freqs,
                    temporal_offset=temporal_offset,
                    train_img=train_img,
                    block_mask=block_mask,
                    anchor_k=anchor_k,
                    anchor_v=anchor_v,
                    anchor_scale=anchor_scale,
                    anchor_mode=anchor_mode,
                    anchor_keep=anchor_keep,
                    anchor_align=anchor_align,
                    anchor_window_scope=anchor_window_scope,
                    anchor_hw=anchor_hw,
                )
            with amp.autocast(dtype=torch.float32):
                x = x + y * e_parts[2]
            x = cross_attn_ffn(x.to(dtype=orig_dtype), e_parts)
        if is_stream:
            return x, cache_k, cache_v
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False, first_last_frame_context_token_number=None):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            first_last_frame_context_token_number = int(
                _FALLBACK_FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER
                if first_last_frame_context_token_number is None
                else first_last_frame_context_token_number
            )
            self.emb_pos = nn.Parameter(
                torch.zeros(1, first_last_frame_context_token_number, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 text_context_token_number=None,
                 first_last_frame_context_token_number=None,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 sparse_causal=False,  # global block-causal switch (legacy name; see WanSelfAttention.forward)
                 stream_chunk_size=None,
                 rope_max_seq_len=1024,
                 rope_theta=10000.0,
                 rope_cache_multiple=1024,
                 # parallel causal via FlexAttention
                 use_flex_causal=False,
                 # Mask-free causal block-grid window self-attention (config-driven,
                 # optional). Additive: when False the model is identical to before.
                 use_window_attn=False,
                 window_chunk=None,
                 # Chunk-causal window attention on the parallel (no-KV-cache)
                 # forward, in latent frames. None = off = original behavior.
                 window_causal_kv_len=None,
                 # Spatial query-block size and neighborhood radius in block units.
                 window_block_hw=(4, 4),
                 window_block_radius_hw=(2, 2),
                 window_block_t=None,
                 window_radius_t=None,
                 # execution backend: "auto" = gather path,
                 # "flex" = flex_attention BlockMask (no KV gather materialised;
                 # training-backward memory lever, numerically equivalent),
                 # "magi" = MagiAttention flex_flash_attn_func over a
                 # block-packed range table (fastest; needs the magi_attention
                 # package, single-sample calls).
                 window_attn_impl="auto"):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.text_context_token_number = int(
            text_len if text_context_token_number is None else text_context_token_number
        )
        self.first_last_frame_context_token_number = int(
            _FALLBACK_FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER
            if first_last_frame_context_token_number is None
            else first_last_frame_context_token_number
        )
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.sparse_causal = sparse_causal  # block-causal attention across the model
        self.stream_chunk_size = int(_FALLBACK_STREAM_CHUNK_SIZE if stream_chunk_size is None else stream_chunk_size)
        self.use_flex_causal = use_flex_causal
        self._tf_block_mask = None
        self._tf_mask_key = None
        # TF + LR-anchor: separate BlockMask whose KV is extended by an anchor
        # region (see _build_teacher_forcing_anchor_mask). Cached under the same
        # key as _tf_block_mask; only built when a TF step actually carries anchor_kv.
        self._tf_anchor_block_mask = None
        self._tf_anchor_mask_key = None
        self.rope_theta = float(rope_theta)
        # Spatially thinned freqs tables for the native LQ-anchor, keyed by
        # (stride, source-table identity).
        self._strided_freqs_cache = {}
        self.rope_cache_multiple = max(1, int(rope_cache_multiple))
        self.rope_max_seq_len = max(1, int(rope_max_seq_len))

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(
                cross_attn_type,
                dim,
                ffn_dim,
                num_heads,
                window_size,
                qk_norm,
                cross_attn_norm,
                eps,
                sparse_causal=sparse_causal,  # pass the global block-causal switch to every block
                text_context_token_number=self.text_context_token_number,
                stream_chunk_size=self.stream_chunk_size,
                use_flex_causal=use_flex_causal,
            )
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self.head_dim = dim // num_heads
        self.freqs = self._build_rope_freqs(self.rope_max_seq_len, device=torch.device("cpu"))

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = MLPProj(
                1280,
                dim,
                flf_pos_emb=model_type == 'flf2v',
                first_last_frame_context_token_number=self.first_last_frame_context_token_number,
            )

        # initialize weights
        self.init_weights()

        # Mask-free causal block-grid window self-attention (config-driven, optional).
        # Flips the self-attn path on every block after construction; leaves the
        # dense / block-causal path untouched when use_window_attn is False.
        #
        # These are kept as REAL attributes (not only registered into the diffusers
        # config): the distillation trainers read them back off the model
        # (`gen_is_causal` in model/sr_{dmd,vsd,gan}.py). Going through
        # ConfigMixin.__getattr__ would (a) rely on a deprecated fallback and
        # (b) return the CONSTRUCTION-time value, i.e. a stale False/None after a
        # runtime enable_window_attention() / inference-script override.
        self.use_window_attn = False
        self.window_chunk = window_chunk
        self.window_causal_kv_len = None
        self.window_block_hw = tuple(window_block_hw)
        self.window_block_radius_hw = tuple(window_block_radius_hw)
        self.window_block_t = (None if window_block_t is None
                               else int(window_block_t))
        self.window_radius_t = (None if window_radius_t is None
                                else int(window_radius_t))
        self.window_attn_impl = str(window_attn_impl)
        if use_window_attn:
            self.enable_window_attention(win_chunk=window_chunk,
                                         causal_kv_len=window_causal_kv_len,
                                         window_block_hw=window_block_hw,
                                         window_block_radius_hw=window_block_radius_hw,
                                         window_block_t=window_block_t,
                                         window_radius_t=window_radius_t,
                                         window_attn_impl=window_attn_impl)

        self.gradient_checkpointing = False
        self.block_mask = None
        self.num_frame_per_block = 1
        self.independent_first_frame = False
        self._cross_kv_initialized = False

    def enable_window_attention(self, win_chunk=None,
                                causal_kv_len=None,
                                window_block_hw=(4, 4), window_block_radius_hw=(2, 2),
                                window_block_t=None, window_radius_t=None,
                                window_attn_impl="auto"):
        r"""Switch every self-attention block to the mask-free causal block-grid
        window path (additive, runtime toggle).

        Query blocks of ``window_block_hw`` hard-partition each frame; every block
        attends its clamped ``window_block_radius_hw`` neighborhood. Existing
        dense / block-causal config is untouched until this is called.

        ``causal_kv_len`` (latent frames, None = off) makes the PARALLEL forward
        chunk-causal. Leave it off for a bidirectional model; set it to the
        streaming ``stream_kv_len`` for teacher_forcing / diffusion_forcing,
        otherwise the unconstrained time axis attends to future frames.

        Temporal geometry: by default the block size and radius are derived from
        ``stream_chunk_size`` / ``causal_kv_len`` at forward time. Passing
        ``window_block_t`` / ``window_radius_t`` explicitly switches to the
        chunk-anchored rule (own chunk + ``(2*window_radius_t+1)*window_block_t``
        history frames), which is what ``window_attn_impl="magi"`` requires.
        Under that rule ``causal_kv_len``'s VALUE no longer sets the span — only
        whether it is set at all still matters (it gates the chunk-causal parallel
        forward) — so passing both warns once.
        """
        global _WINDOW_KNOB_WARNED
        if (causal_kv_len is not None and window_radius_t is not None
                and not _WINDOW_KNOB_WARNED):
            _WINDOW_KNOB_WARNED = True
            bt = int(window_block_t if window_block_t is not None
                     else getattr(self, "stream_chunk_size",
                                  _FALLBACK_STREAM_CHUNK_SIZE))
            warnings.warn(
                f"window_causal_kv_len={causal_kv_len} no longer sets the temporal "
                f"span: window_radius_t={window_radius_t} selects the chunk-anchored "
                f"rule, whose history is (2*{window_radius_t}+1)*{bt} = "
                f"{(2 * int(window_radius_t) + 1) * bt} latent frames. It is still "
                "read for whether it is set (that gates the chunk-causal PARALLEL "
                "forward) and by gen_is_causal, so leave it set — just do not expect "
                "its number to change visibility. Drop window_radius_t to go back to "
                "the legacy rule, where the span IS window_causal_kv_len.",
                UserWarning, stacklevel=2)
        n = 0
        for block in self.blocks:
            attn = getattr(block, "self_attn", None)
            if attn is None:
                continue
            attn.use_window_attn = True
            attn.window_chunk = win_chunk
            attn.window_causal_kv_len = causal_kv_len
            attn.window_block_hw = tuple(window_block_hw)
            attn.window_block_radius_hw = tuple(window_block_radius_hw)
            attn.window_block_t = (None if window_block_t is None
                                   else int(window_block_t))
            attn.window_radius_t = (None if window_radius_t is None
                                    else int(window_radius_t))
            attn.window_attn_impl = str(window_attn_impl)
            n += 1
        # Mirror onto the model so `gen_is_causal` and friends see the CURRENT
        # setting rather than the construction-time config value.
        self.use_window_attn = True
        self.window_chunk = win_chunk
        self.window_causal_kv_len = causal_kv_len
        self.window_block_hw = tuple(window_block_hw)
        self.window_block_radius_hw = tuple(window_block_radius_hw)
        self.window_block_t = (None if window_block_t is None
                               else int(window_block_t))
        self.window_radius_t = (None if window_radius_t is None
                                else int(window_radius_t))
        return n

    def disable_window_attention(self):
        r"""Revert every self-attention block to its original (dense / block-causal) path."""
        for block in self.blocks:
            attn = getattr(block, "self_attn", None)
            if attn is not None:
                attn.use_window_attn = False
        self.use_window_attn = False

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None, **kwargs):
        self.gradient_checkpointing = enable

    def _round_rope_cache_len(self, required_len: int) -> int:
        """Round the RoPE cache length up, avoiding frequent rebuilds in long-video streaming."""
        required_len = max(1, int(required_len))
        m = self.rope_cache_multiple
        return ((required_len + m - 1) // m) * m

    def _build_rope_freqs(self, max_seq_len: int, device: torch.device | None = None) -> torch.Tensor:
        """Build the 3D RoPE frequency cache.

        Returns a `[max_seq_len, head_dim/2]` complex tensor with the temporal /
        height / width segments concatenated. Each axis is cached to the same
        length and dynamically sliced at forward time per the current
        `(f, h, w)`.
        """
        t_dim, h_dim, w_dim = _rope_axis_dims(self.head_dim)
        return torch.cat([
            rope_params(max_seq_len, t_dim, theta=self.rope_theta, device=device),
            rope_params(max_seq_len, h_dim, theta=self.rope_theta, device=device),
            rope_params(max_seq_len, w_dim, theta=self.rope_theta, device=device),
        ], dim=1)

    def _required_rope_seq_len(self, grid_sizes: torch.Tensor, temporal_offset=None,
                               rope_hw_stride=None) -> int:
        """Compute the max RoPE length for the current batch's f/h/w and temporal_offset.

        With a non-trivial `rope_hw_stride` (the native LQ-anchor harvest), grid
        row i reads source row `i*s`, so the source table must cover `s*h` /
        `s*w` rather than `h` / `w`.
        """
        if isinstance(grid_sizes, DTensor):
            grid_sizes = grid_sizes.to_local()
        offsets = _normalize_temporal_offsets(temporal_offset, grid_sizes.size(0))
        s_h, s_w = _normalize_hw_stride(rope_hw_stride)
        required = 1
        for (f, h, w), t_off in zip(grid_sizes.detach().cpu().tolist(), offsets):
            if t_off < 0:
                raise ValueError(f"temporal_offset must be non-negative, got {t_off}")
            # Temporal axis uses offset-absolute positions; spatial axes are
            # sliced by the current grid length directly.
            required = max(required, int(t_off) + int(f),
                           s_h * int(h), s_w * int(w))
        return required

    def _freqs_for_stride(self, rope_hw_stride, grid_sizes, temporal_offset=None):
        """The freqs table handed to the blocks: with no stride, `self.freqs` itself.

        The native LQ-anchor harvest swaps in a spatially thinned table here so
        that the anchor's RoPE positions land on HR coordinates `0, s, 2s, ...`.
        Thinned tables are cached by (stride, source-table identity) -- the
        geometry is constant within one rollout, otherwise every layer and step
        would rebuild a `[L, c]` complex table.
        """
        s_h, s_w = _normalize_hw_stride(rope_hw_stride)
        if s_h == 1 and s_w == 1:
            return self.freqs
        required_len = self._required_rope_seq_len(
            grid_sizes, temporal_offset, rope_hw_stride=(s_h, s_w))
        if self.freqs.size(0) < required_len:
            # A too-short source table would leave NaN tails after thinning;
            # grow it first.
            self.freqs = self._build_rope_freqs(
                self._round_rope_cache_len(required_len), device=self.freqs.device)
        key = ((s_h, s_w), self.freqs.size(0), self.freqs.data_ptr(),
               str(self.freqs.device))
        cache = self._strided_freqs_cache
        out = cache.get(key)
        if out is None:
            cache.clear()          # keep only the current geometry; don't let long videos pile up tables
            out = cache[key] = spatial_strided_freqs(self.freqs, (s_h, s_w))
        return out

    def _ensure_rope_freqs(
        self,
        grid_sizes: torch.Tensor,
        temporal_offset=None,
        device: torch.device | None = None,
    ) -> None:
        """Dynamically extend the RoPE cache, supporting temporal/spatial extrapolation.

        The reference implementation assembles freqs from the current f/h/w on
        every forward; this keeps the same semantics but caches the three-axis
        frequencies and rebuilds a longer cache whenever the current resolution
        or streaming offset outgrows it.
        """
        device = device or self.patch_embedding.weight.device
        required_len = self._required_rope_seq_len(grid_sizes, temporal_offset)
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if self.freqs.size(0) >= required_len:
            return
        new_len = self._round_rope_cache_len(required_len)
        self.freqs = self._build_rope_freqs(new_len, device=device)

    def clear_cross_kv(self):
        for blk in self.blocks:
            if hasattr(blk.cross_attn, "clear_cache"):
                blk.cross_attn.clear_cache()
        self._cross_kv_initialized = False

    def reinit_cross_kv(self, context):
        if context is None:
            return
        if context.dim() == 3 and context.size(-1) != self.dim:
            context = self.text_embedding(context)
        for blk in self.blocks:
            if hasattr(blk.cross_attn, "init_cache"):
                blk.cross_attn.init_cache(context)
        self._cross_kv_initialized = True

    def _prepare_embeddings(self, x, t, context, seq_len, clip_fea, y, temporal_offset):
        r"""Shared embedding logic for both train and inference paths."""
        device = self.patch_embedding.weight.device
        dtype = self.patch_embedding.weight.dtype

        x = [u.to(device=device, dtype=dtype) for u in x]
        if y is not None:
            y = [v.to(device=device, dtype=dtype) for v in y]
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        self._ensure_rope_freqs(grid_sizes, temporal_offset=temporal_offset, device=device)
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len

        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        context_lens = None
        if context is None:
            raise ValueError("context must be provided for WanModel forward.")
        if isinstance(context, (list, tuple)):
            context = torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]).to(device=device, dtype=dtype)
            if context.size(-1) != self.dim:
                context = self.text_embedding(context)
        elif torch.is_tensor(context):
            if context.dim() == 2:
                context = context.unsqueeze(0)
            if context.dim() == 3 and context.size(-1) != self.dim:
                context = self.text_embedding(context.to(device=device, dtype=dtype))
        else:
            raise TypeError("context must be a tensor or a list of tensors.")

        if clip_fea is not None:
            if not hasattr(self, "img_emb"):
                raise ValueError("clip_fea is provided but model has no image embedding module.")
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        return x, grid_sizes, seq_lens, e, e0, context, context_lens, device, dtype

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        temporal_offset=None,
        kv_len=None,
        train_img=False,
        kv_caches=None,
        anchor_kv=None,
        anchor_cfg=None,
        rope_hw_stride=None,
    ):
        r"""
        Streaming inference path with KV caches.

        Args:
            x (List[Tensor]): List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor): Diffusion timesteps tensor of shape [B]
            context (List[Tensor]): List of text embeddings each with shape [L, C]
            seq_len (int): Maximum sequence length for positional encoding
            clip_fea (Tensor, optional): CLIP image features for I2V mode
            y (List[Tensor], optional): Conditional video inputs for I2V mode
            kv_caches (List[Tuple], optional): Per-layer KV caches from previous chunks

        Returns:
            Tuple[List[Tensor], List[Tuple]]: (output tensors, new KV caches)
        """
        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            assert clip_fea is not None and y is not None

        (x, grid_sizes, seq_lens, e, e0, context, context_lens,
         device, dtype) = self._prepare_embeddings(
            x, t, context, seq_len, clip_fea, y, temporal_offset)

        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            # The native LQ-anchor harvest uses a spatially thinned table;
            # with rope_hw_stride=None this is self.freqs itself (zero behavior
            # change). RoPE is baked into K at harvest time and the consumer
            # never touches it again, so the table MUST be swapped here.
            freqs=self._freqs_for_stride(rope_hw_stride, grid_sizes,
                                        temporal_offset),
            context=context,
            context_lens=context_lens,
            temporal_offset=temporal_offset,
            train_img=train_img,
            kv_len=kv_len,
            block_mask=self.block_mask,
            teacher_forcing=False,
        )

        anchor_layers = anchor_cfg.get('layers', None) if anchor_cfg is not None else None
        # Per-sample cond_drop gate. The streaming rollout at batch_size=1 usually
        # expresses the drop as anchor_layers=[] instead, but keep_sample must also
        # work here so B>1 drops (and the window path) are not silently ignored.
        anchor_keep = anchor_cfg.get('keep_sample', None) if anchor_cfg is not None else None
        anchor_window_scope = (anchor_cfg.get('window_scope', 'window')
                               if anchor_cfg is not None else 'window')
        if anchor_cfg is not None and anchor_cfg.get('align', 'frame') not in ('frame', 'chunk'):
            raise ValueError(
                f"anchor_cfg['align'] must be 'frame' or 'chunk', got "
                f"{anchor_cfg.get('align')!r}.")
        if anchor_window_scope not in ('window', 'global'):
            raise ValueError(
                f"anchor_cfg['window_scope'] must be 'window' or 'global', got "
                f"{anchor_window_scope!r}.")
        # ── native LQ-anchor: the anchor keeps its own, coarser token grid ──
        # anchor_cfg['hw'] = (h_a, w_a). Only the window path can express a second
        # spatial geometry; None (every other caller, including the "repeat"
        # scheme that expands the harvested K/V back to the HR grid) leaves the
        # consumers on the single-grid path untouched.
        anchor_hw = anchor_cfg.get('hw', None) if anchor_cfg is not None else None
        if anchor_hw is not None:
            anchor_hw = (int(anchor_hw[0]), int(anchor_hw[1]))
        new_kv_caches = []
        for block_id, block in enumerate(self.blocks):
            block_kwargs = dict(kwargs, is_stream=True)
            if kv_caches is not None and block_id < len(kv_caches):
                block_kwargs['pre_cache_k'] = kv_caches[block_id][0]
                block_kwargs['pre_cache_v'] = kv_caches[block_id][1]
            # ── LQ-anchor: per-layer anchor K/V for attention-only injection ──
            # anchor_layers (from anchor_cfg) optionally restricts injection to a
            # subset of blocks; None = all blocks (backward-compatible).
            if (anchor_kv is not None and block_id < len(anchor_kv)
                    and (anchor_layers is None or block_id in anchor_layers)):
                block_kwargs['anchor_k'] = anchor_kv[block_id][0]
                block_kwargs['anchor_v'] = anchor_kv[block_id][1]
                if anchor_cfg is not None:
                    block_kwargs['anchor_scale'] = anchor_cfg.get('scale', 1.0)
                    block_kwargs['anchor_mode'] = anchor_cfg.get('mode', 'v')
                    block_kwargs['anchor_align'] = anchor_cfg.get('align', 'frame')
                    block_kwargs['anchor_window_scope'] = anchor_window_scope
                    block_kwargs['anchor_hw'] = anchor_hw
                    block_kwargs['anchor_keep'] = anchor_keep
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs, **kw):
                        return module(*inputs, **kw)
                    return custom_forward
                x, cache_k, cache_v = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **block_kwargs,
                    use_reentrant=False,
                )
            else:
                x, cache_k, cache_v = block(x, **block_kwargs)
            new_kv_caches.append((cache_k, cache_v))

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        outputs = [u.to(dtype=self.patch_embedding.weight.dtype) for u in x]
        return outputs, new_kv_caches

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        temporal_offset=None,
        kv_len=None,
        train_img=False,
        lq_features=None,
        lq_proj=None,
        sigma=None,
        clean_x=None,
        aug_t=None,
        anchor_kv=None,
        anchor_cfg=None,
    ):
        r"""
        Training forward pass (full-attn / block-causal / teacher-forcing).

        LQ-anchor (Scheme B, dense bidirectional path only): when ``anchor_kv`` is
        provided (per-layer list of ``(k, v)`` extracted from the upsampled-LR latent),
        each HR query chunk additionally attends to its own chunk's LR anchor K/V.

        Args:
            x (List[Tensor]): List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor): Diffusion timesteps tensor of shape [B]
            context (List[Tensor]): List of text embeddings each with shape [L, C]
            seq_len (int): Maximum sequence length for positional encoding
            clip_fea (Tensor, optional): CLIP image features for I2V mode
            y (List[Tensor], optional): Conditional video inputs for I2V mode
            clean_x (List[Tensor], optional): Clean latents for teacher-forcing
            aug_t (Tensor, optional): Timestep for clean GT in teacher-forcing

        Returns:
            List[Tensor]: Denoised video tensors with shape [C_out, F, H/8, W/8]
        """
        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            assert clip_fea is not None and y is not None

        (x, grid_sizes, seq_lens, e, e0, context, context_lens,
         device, dtype) = self._prepare_embeddings(
            x, t, context, seq_len, clip_fea, y, temporal_offset)

        teacher_forcing = clean_x is not None

        # ── LQ-anchor visibility (anchor_cfg["align"]) ──
        # Parsed BEFORE the (expensive, compiled) TF BlockMask is built so an
        # unsupported combination fails fast. Under teacher forcing the anchor
        # visibility is expressed by self._tf_anchor_block_mask, NOT by
        # anchor_align, so a chunk-align request there would be silently ignored
        # — raise instead of lying about the semantics.
        anchor_align = "frame"
        anchor_window_scope = "window"
        if anchor_cfg is not None:
            anchor_align = anchor_cfg.get("align", "frame")
            if anchor_align not in ("frame", "chunk"):
                raise ValueError(
                    f"anchor_cfg['align'] must be 'frame' or 'chunk', got "
                    f"{anchor_align!r}.")
            # Spatial extent of the anchor inside the window path. Only the
            # window path reads it (the dense/TF paths are spatially global on
            # the HR side too), but validate here so a typo fails fast.
            anchor_window_scope = anchor_cfg.get("window_scope", "window")
            if anchor_window_scope not in ("window", "global"):
                raise ValueError(
                    f"anchor_cfg['window_scope'] must be 'window' or 'global', "
                    f"got {anchor_window_scope!r}.")
            if teacher_forcing and anchor_align == "chunk":
                raise NotImplementedError(
                    "anchor_align='chunk' is not supported on the teacher-forcing "
                    "path: TF expresses anchor visibility through the anchor-extended "
                    "BlockMask (KV = [clean|noisy|anchor]), which is built frame-aligned. "
                    "Use align='frame' for TF, or run the streaming (self_forcing) "
                    "rollout for per-chunk anchors.")

        pad_len = int(seq_lens.max().item()) if teacher_forcing else seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, pad_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        block_mask = None
        if teacher_forcing:
            if not FLEX_ATTN_AVAILABLE:
                raise RuntimeError("Teacher-forcing training requires FlexAttention (PyTorch 2.6+)")
            clean_x = [u.to(device=device, dtype=dtype) for u in clean_x]
            if y is not None:
                clean_x = [torch.cat([u, v], dim=0) for u, v in zip(clean_x, y)]
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, pad_len - u.size(1), u.size(2))], dim=1)
                for u in clean_x
            ])
            x = torch.cat([clean_x, x], dim=1)

            num_frames = int(grid_sizes[0, 0].item())
            frame_seqlen = int(grid_sizes[0, 1].item()) * int(grid_sizes[0, 2].item())
            tf_mask_key = (num_frames, frame_seqlen, self.stream_chunk_size, device)
            if self._tf_mask_key != tf_mask_key:
                self._tf_block_mask = _build_teacher_forcing_mask(
                    num_frames, frame_seqlen, self.stream_chunk_size, device,
                )
                self._tf_mask_key = tf_mask_key
            block_mask = self._tf_block_mask
            # TF + LR-anchor: build the anchor-extended mask (KV = [clean|noisy|
            # anchor]) once per shape. Injected only into the blocks selected by
            # anchor_layers below; non-anchor blocks keep the plain TF mask.
            if anchor_kv is not None and self._tf_anchor_mask_key != tf_mask_key:
                self._tf_anchor_block_mask = _build_teacher_forcing_anchor_mask(
                    num_frames, frame_seqlen, self.stream_chunk_size, device,
                )
                self._tf_anchor_mask_key = tf_mask_key

            if aug_t is None:
                aug_t = torch.zeros_like(t)
            with amp.autocast(dtype=torch.float32):
                e_clean = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, aug_t).float())
                e0_clean = self.time_projection(e_clean).unflatten(1, (6, self.dim))
                e0 = torch.stack([e0_clean, e0], dim=1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            temporal_offset=temporal_offset,
            train_img=train_img,
            kv_len=kv_len,
            block_mask=block_mask,
            teacher_forcing=teacher_forcing,
        )

        anchor_scale = 1.0
        anchor_mode = "v"
        anchor_layers = None
        anchor_keep = None
        if anchor_cfg is not None:
            anchor_scale = anchor_cfg.get("scale", 1.0)
            anchor_mode = anchor_cfg.get("mode", "v")
            # Optional layer selection: only these block ids receive the LR anchor
            # K/V. None = all blocks (backward-compatible). Set both at training
            # (prefix_keep drop) and inference (explicit anchor_layers) via anchor_cfg.
            anchor_layers = anchor_cfg.get("layers", None)
            # Optional per-sample gate (training cond_drop): bool [B]; False => that
            # sample attends to no anchor in any block. None = all samples keep it.
            anchor_keep = anchor_cfg.get("keep_sample", None)

        for block_id, block in enumerate(self.blocks):
            block_kwargs = kwargs
            # ── LQ-anchor (Scheme B): per-layer anchor K/V ──
            # Works for both the dense bidirectional path and the teacher-forcing
            # causal path; under TF the block additionally switches to the
            # anchor-extended BlockMask (KV = [clean|noisy|anchor]).
            if (anchor_kv is not None
                    and block_id < len(anchor_kv)
                    and (anchor_layers is None or block_id in anchor_layers)):
                block_kwargs = dict(
                    kwargs,
                    anchor_k=anchor_kv[block_id][0],
                    anchor_v=anchor_kv[block_id][1],
                    anchor_scale=anchor_scale,
                    anchor_mode=anchor_mode,
                    anchor_keep=anchor_keep,
                    anchor_align=anchor_align,
                    anchor_window_scope=anchor_window_scope,
                )
                if teacher_forcing:
                    block_kwargs["block_mask"] = self._tf_anchor_block_mask
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs, **kw):
                        return module(*inputs, **kw)
                    return custom_forward
                ckpt_kwargs = {"use_reentrant": False}
                if anchor_kv is not None:
                    # The per-chunk anchor path issues extra flash_attn calls whose
                    # int32 cu_seqlens bookkeeping tensors trip checkpoint's (overly
                    # strict) saved-tensor metadata check. The anchor forward is verified
                    # deterministic, so skip that check for this path.
                    ckpt_kwargs["determinism_check"] = "none"
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **block_kwargs,
                    **ckpt_kwargs,
                )
            else:
                x = block(x, **block_kwargs)
            if teacher_forcing:
                continue
            if lq_features is not None and lq_proj is not None and lq_proj.is_inject_active(block_id):
                out_idx = lq_proj.get_inject_index(block_id)
                if out_idx < len(lq_features):
                    lq_feat = lq_features[out_idx].to(device=x.device, dtype=x.dtype)
                    x = lq_proj.gate(x, lq_feat, sigma=sigma, out_idx=out_idx)
            elif lq_features is not None and block_id < len(lq_features):
                lq_feat = lq_features[block_id].to(device=x.device, dtype=x.dtype)
                x = x + lq_feat

        if teacher_forcing:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.to(dtype=self.patch_embedding.weight.dtype) for u in x]

    def forward(self, *args, **kwargs):
        r"""Dispatch to training or inference path based on kv_caches / is_stream."""
        if kwargs.get('kv_caches', None) is not None or kwargs.get('is_stream', False):
            # Remove train-only params before forwarding to inference
            for k in ('lq_features', 'lq_proj', 'sigma', 'clean_x', 'aug_t', 'is_stream'):
                kwargs.pop(k, None)
            return self._forward_inference(*args, **kwargs)
        else:
            # Remove inference-only params before forwarding to train.
            # anchor_kv / anchor_cfg are kept: the dense training path (Scheme B)
            # consumes them to inject the per-chunk LR anchor K/V.
            for k in ('kv_caches', 'is_stream'):
                kwargs.pop(k, None)
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)


# Keep backward compatibility with previous naming
Transformer3DModel = WanModel
