"""Configurable gated cross-attention joint audio-video model.

This experimental variant follows the AV cross-attention design but allows
audio-to-video (A2V) and video-to-audio (V2A) cross attention to be enabled
independently for each layer. Cross-modal attention can optionally apply a
fixed per-layer alpha times a per-head sigmoid gate to the attention context
before the output projection.
"""

import logging
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from .attention_utils import attention
from .creator_audio import CreatorAudioModel
from .wan_transformer3d_prope import (
    Wan2_2Transformer3DModel,
    WanRMSNorm,
    WanTransformer3DModel,
    rope_apply_qk,
)
from .creator.creator_video_dit import sinusoidal_embedding_1d
from .creator.creator_video_dit import rope_apply_head_dim
from ..dist.sequence_parallel import all_gather_sequence, ulysses_attention


LayerSelection = Optional[Union[bool, str, Sequence[bool], Sequence[int], torch.Tensor]]
LayerAlphas = Optional[
    Union[float, Sequence[float], Mapping[Union[int, str], float], torch.Tensor]
]


@torch.amp.autocast("cuda", enabled=False)
def temporal_rope_1d(
    x: torch.Tensor,
    temporal_positions: torch.Tensor,
    inv_freqs_1d: torch.Tensor,
) -> torch.Tensor:
    """Apply 1D temporal RoPE to [B, L, num_heads, head_dim] tensors."""
    dtype = x.dtype
    batch_size, seq_len, num_heads, head_dim = x.shape
    half_dim = head_dim // 2

    freqs = torch.einsum(
        "bl,d->bld",
        temporal_positions.to(torch.float64),
        inv_freqs_1d.to(x.device, torch.float64),
    )
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(batch_size, seq_len, num_heads, half_dim, 2)
    )
    x_out = torch.view_as_real(x_complex * freqs_cis.unsqueeze(2)).flatten(3)
    return x_out.to(dtype)


def compute_video_temporal_positions(
    grid_sizes: torch.Tensor,
    seq_len: int,
    device: torch.device,
    audio_fps: float = 48000.0 / 960.0,
    video_fps: float = 16.0,
    vae_temporal_stride: int = 4,
) -> torch.Tensor:
    """Compute video token temporal positions in audio-token time units."""
    batch_size = grid_sizes.size(0)
    positions = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float64)
    video_latent_fps = video_fps / vae_temporal_stride
    scale = audio_fps / video_latent_fps

    for sample_idx, (num_frames, height, width) in enumerate(grid_sizes.tolist()):
        spatial_size = int(height * width)
        num_tokens = int(num_frames * spatial_size)
        frame_indices = torch.arange(num_tokens, device=device, dtype=torch.float64) // spatial_size
        positions[sample_idx, :num_tokens] = frame_indices * scale
    return positions


def compute_audio_temporal_positions(
    seq_lens: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute sequential temporal positions for audio tokens."""
    batch_size = seq_lens.size(0)
    positions = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float64)
    base_positions = torch.arange(seq_len, device=device, dtype=torch.float64)
    for sample_idx in range(batch_size):
        valid_len = int(seq_lens[sample_idx].item())
        positions[sample_idx, :valid_len] = base_positions[:valid_len]
    return positions


def _apply_video_rope_local(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    sp_rank: int,
    sp_world_size: int,
) -> torch.Tensor:
    """Apply the Wan 3D RoPE slice belonging to this sequence-parallel rank."""
    if sp_world_size <= 1:
        return rope_apply_qk(x, x, grid_sizes, freqs)[0]

    local_len, num_heads, complex_dim = x.size(1), x.size(2), x.size(3) // 2
    freq_parts = freqs.split(
        [complex_dim - 2 * (complex_dim // 3), complex_dim // 3, complex_dim // 3],
        dim=1,
    )
    output = []
    for sample_idx, (frames, height, width) in enumerate(grid_sizes.tolist()):
        full_len = int(frames * height * width)
        sample = x[sample_idx, :local_len].to(torch.float64)
        sample_complex = torch.view_as_complex(
            sample.reshape(local_len, num_heads, -1, 2)
        )
        full_freqs = torch.cat(
            [
                freq_parts[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                freq_parts[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                freq_parts[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(full_len, 1, -1)
        if full_freqs.size(0) < local_len * sp_world_size:
            full_freqs = torch.cat(
                [
                    full_freqs,
                    torch.ones(
                        local_len * sp_world_size - full_freqs.size(0),
                        full_freqs.size(1),
                        full_freqs.size(2),
                        dtype=full_freqs.dtype,
                        device=full_freqs.device,
                    ),
                ],
                dim=0,
            )
        start = sp_rank * local_len
        local_freqs = full_freqs[start : start + local_len]
        rotated = torch.view_as_real(sample_complex * local_freqs).flatten(2)
        if x.size(1) > local_len:
            rotated = torch.cat([rotated, x[sample_idx, local_len:]], dim=0)
        output.append(rotated)
    return torch.stack(output).to(x.dtype)


def _to_list(value) -> List[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [sample for sample in value]
    return list(value)


def _build_time_embeddings(
    time_embedding: nn.Module,
    time_projection: nn.Module,
    freq_dim: int,
    dim: int,
    timesteps: torch.Tensor,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build time embeddings and projected modulations."""

    with torch.amp.autocast("cuda", dtype=torch.float32):
        if timesteps.dim() != 1:
            if timesteps.size(1) < seq_len:
                pad_size = seq_len - timesteps.size(1)
                timesteps = torch.cat(
                    [timesteps, timesteps[:, -1:].repeat(1, pad_size)], dim=1
                )
            batch_size = timesteps.size(0)
            embedding = time_embedding(
                sinusoidal_embedding_1d(freq_dim, timesteps.flatten())
                .unflatten(0, (batch_size, seq_len))
                .float()
            )
            modulation = time_projection(embedding).unflatten(2, (6, dim))
        else:
            embedding = time_embedding(sinusoidal_embedding_1d(freq_dim, timesteps).float())
            modulation = time_projection(embedding).unflatten(1, (6, dim))
    return embedding, modulation


def _embed_context(text_embedding: nn.Module, context, text_len: int) -> torch.Tensor:
    """Embed and right-pad text context to the model fixed text length."""
    if isinstance(context, torch.Tensor):
        samples = [sample for sample in context]
    else:
        samples = list(context)
    return text_embedding(
        torch.stack([
            torch.cat([sample, sample.new_zeros(text_len - sample.size(0), sample.size(1))])
            for sample in samples
        ])
    )


def _logit_from_gate_value(gate_init_value: Optional[float]) -> float:
    """Convert an initial gate value in [0, 1] to a sigmoid bias."""
    if gate_init_value is None:
        return 0.0
    gate_value = min(max(float(gate_init_value), 1e-6), 1.0 - 1e-6)
    return math.log(gate_value / (1.0 - gate_value))


def _expand_layer_selection(
    selection: LayerSelection,
    num_layers: int,
    name: str,
) -> List[bool]:
    """Expand a layer-selection config to a bool mask of length ``num_layers``.

    Accepted forms:
      - ``None`` or ``True``: enable every layer.
      - ``False``: disable every layer.
      - bool mask with length ``num_layers``.
      - 0/1 mask with length ``num_layers``.
      - list/tuple/tensor of layer indices to enable.
      - strings: ``"all"``, ``"none"``, ``"0,2,5"``.
    """
    if selection is None:
        return [False] * num_layers
    if isinstance(selection, bool):
        return [selection] * num_layers
    if isinstance(selection, torch.Tensor):
        selection = selection.cpu().tolist()
    if isinstance(selection, str):
        normalized = selection.strip().lower()
        if normalized in {"", "none", "false", "off", "0"}:
            return [False] * num_layers
        if normalized in {"all", "true", "on", "1"}:
            return [True] * num_layers
        indices = [int(part.strip()) for part in selection.split(",") if part.strip()]
        mask = [False] * num_layers
        for layer_idx in indices:
            if layer_idx < 0 or layer_idx >= num_layers:
                raise ValueError(f"{name} layer index {layer_idx} out of range [0, {num_layers})")
            mask[layer_idx] = True
        return mask

    values = list(selection)
    if not values:
        return [False] * num_layers

    if all(isinstance(value, bool) for value in values):
        if len(values) != num_layers:
            raise ValueError(f"{name} bool mask must have length {num_layers}, got {len(values)}")
        return [bool(value) for value in values]

    if all(isinstance(value, int) for value in values):
        if len(values) == num_layers and all(int(value) in {0, 1} for value in values):
            return [bool(value) for value in values]
        mask = [False] * num_layers
        for layer_idx in values:
            if layer_idx < 0 or layer_idx >= num_layers:
                raise ValueError(f"{name} layer index {layer_idx} out of range [0, {num_layers})")
            mask[int(layer_idx)] = True
        return mask

    raise TypeError(
        f"{name} must be None, bool, string, bool mask, 0/1 mask, or layer-index sequence"
    )


def _expand_layer_alphas(
    values: LayerAlphas,
    num_layers: int,
    name: str,
) -> List[float]:
    """Expand fixed per-layer gate multipliers, defaulting each layer to 1.0.

    Accepted forms are a scalar shared by all layers, a full sequence with
    ``num_layers`` entries, or a mapping of layer index to alpha. Unspecified
    mapping entries retain the default value 1.0.
    """
    if values is None:
        return [1.0] * num_layers
    if isinstance(values, torch.Tensor):
        values = values.cpu().tolist()

    def validate(value, layer_label: str) -> float:
        alpha = float(value)
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError(f"{name} {layer_label} must be finite and non-negative, got {value}")
        return alpha

    if isinstance(values, (int, float)) and not isinstance(values, bool):
        return [validate(values, "scalar")] * num_layers

    if isinstance(values, Mapping):
        alphas = [1.0] * num_layers
        for raw_layer_idx, value in values.items():
            try:
                layer_idx = int(raw_layer_idx)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} mapping key must be a layer index, got {raw_layer_idx!r}"
                ) from exc
            if layer_idx < 0 or layer_idx >= num_layers:
                raise ValueError(f"{name} layer index {layer_idx} out of range [0, {num_layers})")
            alphas[layer_idx] = validate(value, f"layer {layer_idx}")
        return alphas

    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if len(values) != num_layers:
            raise ValueError(f"{name} sequence must have length {num_layers}, got {len(values)}")
        return [validate(value, f"layer {layer_idx}") for layer_idx, value in enumerate(values)]

    raise TypeError(f"{name} must be None, a scalar, a mapping, or a full layer sequence")


class GatedCrossModalAttention(nn.Module):
    """Cross-modal attention with optional alpha-scaled sigmoid context gating.

    For A2V, ``x`` is video hidden states and ``y`` is audio hidden states.
    For V2A, ``x`` is audio hidden states and ``y`` is video hidden states.
    When ``use_gating=False``, this module follows the reference cross-attn
    behavior: query from raw ``x`` and key/value from normalized ``y``.
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        zero_init_output: bool = False,
        use_gating: bool = True,
        zero_init_gating: bool = False,
        gate_init_value: Optional[float] = None,
        gate_alpha: float = 1.0,
    ):
        super().__init__()
        assert q_dim % num_heads == 0
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.use_gating = bool(use_gating)
        self.gate_alpha = _expand_layer_alphas(gate_alpha, 1, "gate_alpha")[0]

        self.norm = nn.LayerNorm(kv_dim, eps=eps)
        # if self.use_gating:
        self.norm_x = nn.LayerNorm(q_dim, eps=eps)

        self.q = nn.Linear(q_dim, q_dim)
        self.k = nn.Linear(kv_dim, q_dim)
        self.v = nn.Linear(kv_dim, q_dim)
        self.o = nn.Linear(q_dim, q_dim)

        self.norm_q = WanRMSNorm(q_dim, eps=eps)
        self.norm_k = WanRMSNorm(q_dim, eps=eps)

        if self.use_gating:
            self.gate_hidden = nn.Linear(q_dim, num_heads, bias=False)
            self.gate_context_norm = nn.LayerNorm(self.head_dim, eps=eps)
            self.gate_context = nn.Linear(self.head_dim, 1, bias=False)
            self.gate_bias = nn.Parameter(torch.empty(num_heads))

        self._init_weights(
            zero_init_output=zero_init_output,
            zero_init_gating=zero_init_gating,
            gate_init_value=gate_init_value,
        )

    def _init_weights(
        self,
        zero_init_output: bool,
        zero_init_gating: bool,
        gate_init_value: Optional[float],
    ):
        for module in [self.q, self.k, self.v, self.o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if zero_init_output:
            nn.init.zeros_(self.o.weight)
            nn.init.zeros_(self.o.bias)

        if self.use_gating:
            if zero_init_gating:
                nn.init.zeros_(self.gate_hidden.weight)
                nn.init.zeros_(self.gate_context.weight)
            else:
                nn.init.xavier_uniform_(self.gate_hidden.weight)
                nn.init.xavier_uniform_(self.gate_context.weight)
            nn.init.constant_(self.gate_bias, _logit_from_gate_value(gate_init_value))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_lens: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.bfloat16,
        q_temporal_pos: Optional[torch.Tensor] = None,
        k_temporal_pos: Optional[torch.Tensor] = None,
        temporal_rope_inv_freq: Optional[torch.Tensor] = None,
        sequence_parallel: bool = False,
        sp_group=None,
    ) -> torch.Tensor:
        """
        Args:
            x: primary hidden states [B, Lq, q_dim]
            y: conditioning hidden states [B, Lk, kv_dim]
            y_lens: valid lengths of y per sample [B]
            q_temporal_pos: [B, Lq] temporal positions for query
            k_temporal_pos: [B, Lk] temporal positions for key
            temporal_rope_inv_freq: [head_dim // 2] inv frequencies for temporal RoPE
        Returns:
            Cross-attention output [B, Lq, q_dim].
        """
        batch_size = x.size(0)
        num_heads = self.num_heads
        head_dim = self.head_dim

        # In SP mode x is a local query chunk while y is a local conditioning
        # chunk. Cross-modal attention needs the complete conditioning sequence.
        if sequence_parallel:
            y = all_gather_sequence(y, group=sp_group)
            if k_temporal_pos is not None:
                k_temporal_pos = all_gather_sequence(
                    k_temporal_pos.unsqueeze(-1), group=sp_group
                ).squeeze(-1)

        x_for_q = self.norm_x(x)
        y_norm = self.norm(y)

        query = self.norm_q(self.q(x_for_q.to(dtype))).view(batch_size, -1, num_heads, head_dim)
        key = self.norm_k(self.k(y_norm.to(dtype))).view(batch_size, -1, num_heads, head_dim)
        value = self.v(y_norm.to(dtype)).view(batch_size, -1, num_heads, head_dim)

        if q_temporal_pos is not None and k_temporal_pos is not None and temporal_rope_inv_freq is not None:
            query = temporal_rope_1d(query, q_temporal_pos, temporal_rope_inv_freq)
            key = temporal_rope_1d(key, k_temporal_pos, temporal_rope_inv_freq)

        context = attention(query.to(dtype), key.to(dtype), value.to(dtype), k_lens=y_lens)
        context = context.to(dtype)

        if self.use_gating:
            hidden_gate = self.gate_hidden(x_for_q.to(dtype)).view(batch_size, -1, num_heads, 1)
            context_gate = self.gate_context(
                self.gate_context_norm(context)
            )
            gate = self.gate_alpha * torch.sigmoid(
                hidden_gate + context_gate + self.gate_bias.view(1, 1, num_heads, 1)
            )
            context = gate * context
        return self.o(context.flatten(2))


class GatedJointBlock(nn.Module):
    """One joint block with independently configurable A2V and V2A attention."""

    def __init__(
        self,
        video_block: nn.Module,
        audio_block: nn.Module,
        video_dim: int,
        audio_dim: int,
        video_num_heads: int,
        audio_num_heads: int,
        enable_a2v_cross_attn: bool = True,
        enable_v2a_cross_attn: bool = True,
        zero_init_output: bool = False,
        zero_init_video_cross_attn: bool | None = None,
        zero_init_audio_cross_attn: bool | None = None,
        use_a2v_gating: bool = True,
        use_v2a_gating: bool = True,
        zero_init_a2v_gating: bool = False,
        zero_init_v2a_gating: bool = False,
        a2v_gate_init_value: Optional[float] = None,
        v2a_gate_init_value: Optional[float] = None,
        a2v_gate_alpha: float = 1.0,
        v2a_gate_alpha: float = 1.0,
    ):
        super().__init__()
        self.video_block = video_block
        self.audio_block = audio_block
        self.enable_a2v_cross_attn = bool(enable_a2v_cross_attn)
        self.enable_v2a_cross_attn = bool(enable_v2a_cross_attn)

        zero_init_video = zero_init_video_cross_attn if zero_init_video_cross_attn is not None else zero_init_output
        zero_init_audio = zero_init_audio_cross_attn if zero_init_audio_cross_attn is not None else zero_init_output

        if self.enable_a2v_cross_attn:
            self.video_cross_attn_audio = GatedCrossModalAttention(
                q_dim=video_dim,
                kv_dim=audio_dim,
                num_heads=video_num_heads,
                zero_init_output=zero_init_video,
                use_gating=use_a2v_gating,
                zero_init_gating=zero_init_a2v_gating,
                gate_init_value=a2v_gate_init_value,
                gate_alpha=a2v_gate_alpha,
            )
        else:
            self.video_cross_attn_audio = None

        if self.enable_v2a_cross_attn:
            self.audio_cross_attn_video = GatedCrossModalAttention(
                q_dim=audio_dim,
                kv_dim=video_dim,
                num_heads=audio_num_heads,
                zero_init_output=zero_init_audio,
                use_gating=use_v2a_gating,
                zero_init_gating=zero_init_v2a_gating,
                gate_init_value=v2a_gate_init_value,
                gate_alpha=v2a_gate_alpha,
            )
        else:
            self.audio_cross_attn_video = None

    def forward(
        self,
        video_x: torch.Tensor,
        audio_x: torch.Tensor,
        video_kwargs: Dict[str, Any],
        audio_kwargs: Dict[str, Any],
        dtype: torch.dtype = torch.bfloat16,
        enable_a2v: Optional[Union[bool, torch.Tensor]] = None,
        enable_v2a: Optional[Union[bool, torch.Tensor]] = None,
        sequence_parallel: bool = False,
        sp_group=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video_block = self.video_block
        audio_block = self.audio_block

        video_x = self._video_selfattn_and_text(
            video_x, video_block, video_kwargs, dtype,
            sequence_parallel=sequence_parallel,
            sp_group=sp_group,
        )
        audio_x = self._audio_selfattn_and_text(
            audio_x, audio_block, audio_kwargs, dtype,
            sequence_parallel=sequence_parallel,
            sp_group=sp_group,
        )

        temporal_rope_inv_freq = video_kwargs.get("temporal_rope_inv_freq")
        video_temporal_pos = video_kwargs.get("temporal_positions")
        audio_temporal_pos = audio_kwargs.get("temporal_positions")

        # Runtime switches can be batch-wide booleans or per-branch CFG masks.
        a2v_is_tensor = isinstance(enable_a2v, torch.Tensor)
        v2a_is_tensor = isinstance(enable_v2a, torch.Tensor)
        run_a2v = self.video_cross_attn_audio is not None and (
            a2v_is_tensor or enable_a2v is None or enable_a2v
        )
        run_v2a = self.audio_cross_attn_video is not None and (
            v2a_is_tensor or enable_v2a is None or enable_v2a
        )

        # Cache pre-cross-attention states so A2V and V2A are updated jointly:
        # both directions must attend to the *same* pre-update snapshot, otherwise
        # V2A would condition on the already-A2V-updated video (serial dependency).
        video_x_pre = video_x
        audio_x_pre = audio_x

        if run_a2v:
            a2v_result = self.video_cross_attn_audio(
                x=video_x_pre,
                y=audio_x_pre,
                y_lens=audio_kwargs.get("seq_lens"),
                dtype=dtype,
                q_temporal_pos=video_temporal_pos,
                k_temporal_pos=audio_temporal_pos,
                temporal_rope_inv_freq=temporal_rope_inv_freq,
                sequence_parallel=sequence_parallel,
                sp_group=sp_group,
            )
            a2v_out = a2v_result
            if a2v_is_tensor:
                a2v_mask = enable_a2v.view(-1, *([1] * (a2v_out.dim() - 1))).to(
                    device=a2v_out.device, dtype=a2v_out.dtype
                )
                a2v_out = a2v_out * a2v_mask
            video_x = video_x + a2v_out

        if run_v2a:
            v2a_result = self.audio_cross_attn_video(
                x=audio_x_pre,
                y=video_x_pre,
                y_lens=video_kwargs.get("seq_lens"),
                dtype=dtype,
                q_temporal_pos=audio_temporal_pos,
                k_temporal_pos=video_temporal_pos,
                temporal_rope_inv_freq=temporal_rope_inv_freq,
                sequence_parallel=sequence_parallel,
                sp_group=sp_group,
            )
            v2a_out = v2a_result
            if v2a_is_tensor:
                v2a_mask = enable_v2a.view(-1, *([1] * (v2a_out.dim() - 1))).to(
                    device=v2a_out.device, dtype=v2a_out.dtype
                )
                v2a_out = v2a_out * v2a_mask
            audio_x = audio_x + v2a_out

        video_x = self._video_ffn(video_x, video_block, video_kwargs, dtype)
        audio_x = self._audio_ffn(audio_x, audio_block, audio_kwargs, dtype)
        return video_x, audio_x

    def _video_selfattn_and_text(
        self,
        x: torch.Tensor,
        block,
        kwargs: Dict[str, Any],
        dtype: torch.dtype,
        sequence_parallel: bool = False,
        sp_group=None,
    ) -> torch.Tensor:
        e0 = kwargs["e0"]
        seq_lens = kwargs["seq_lens"]
        grid_sizes = kwargs["grid_sizes"]
        freqs = kwargs["freqs"]
        context = kwargs["context"]
        context_lens = kwargs.get("context_lens")

        if e0.dim() > 3:
            modulation = (block.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
            modulation = [part.squeeze(2) for part in modulation]
        else:
            modulation = (block.modulation + e0).chunk(6, dim=1)

        kwargs["_video_e"] = modulation

        temp_x = block.norm1(x) * (1 + modulation[1]) + modulation[0]
        temp_x = temp_x.to(dtype)

        self_attn = block.self_attn
        batch_size, seq_len = temp_x.shape[:2]
        num_heads, head_dim = self_attn.num_heads, self_attn.head_dim
        query = self_attn.norm_q(self_attn.q(temp_x)).view(batch_size, seq_len, num_heads, head_dim)
        key = self_attn.norm_k(self_attn.k(temp_x)).view(batch_size, seq_len, num_heads, head_dim)
        value = self_attn.v(temp_x).view(batch_size, seq_len, num_heads, head_dim)
        if sequence_parallel:
            sp_rank = int(kwargs["sp_rank"])
            sp_world_size = int(kwargs["sp_world_size"])
            query = _apply_video_rope_local(query, grid_sizes, freqs, sp_rank, sp_world_size)
            key = _apply_video_rope_local(key, grid_sizes, freqs, sp_rank, sp_world_size)
            attn_output = ulysses_attention(
                query.to(dtype),
                key.to(dtype),
                value.to(dtype),
                attention,
                k_lens=seq_lens,
                window_size=getattr(self_attn, "window_size", (-1, -1)),
                group=sp_group,
            )
        else:
            query, key = rope_apply_qk(query, key, grid_sizes, freqs)
            attn_output = attention(
                query.to(dtype),
                key.to(dtype),
                v=value.to(dtype),
                k_lens=seq_lens,
                window_size=getattr(self_attn, "window_size", (-1, -1)),
            )
        attn_output = attn_output.to(dtype).flatten(2)
        attn_output = self_attn.o(attn_output)

        x = x + attn_output * modulation[2]
        x = x + block.cross_attn(block.norm3(x), context, context_lens, dtype)
        return x

    def _audio_selfattn_and_text(
        self,
        x: torch.Tensor,
        block,
        kwargs: Dict[str, Any],
        dtype: torch.dtype,
        sequence_parallel: bool = False,
        sp_group=None,
    ) -> torch.Tensor:
        time_mod = kwargs["e0"]
        freqs = kwargs["freqs"]
        context = kwargs["context"]

        has_seq_mod = len(time_mod.shape) == 4
        chunk_dim = 2 if has_seq_mod else 1
        modulation = (
            block.modulation.to(dtype=time_mod.dtype, device=time_mod.device) + time_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
                part.squeeze(2) for part in modulation
            ]
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation

        kwargs["_audio_mod"] = (shift_mlp, scale_mlp, gate_mlp)

        # norm1 (LayerNorm) upcasts to fp32 and the modulation terms are fp32,
        # so input_x is fp32 here. block.self_attn calls flash-attn, which only
        # supports fp16/bf16 -> cast down first (mirrors the video self-attn path).
        input_x = (block.norm1(x) * (1 + scale_msa) + shift_msa).to(dtype)
        if sequence_parallel:
            self_attn = block.self_attn
            batch_size, seq_len = input_x.shape[:2]
            num_heads, head_dim = self_attn.num_heads, self_attn.head_dim
            query = self_attn.norm_q(self_attn.q(input_x)).view(batch_size, seq_len, num_heads, head_dim)
            key = self_attn.norm_k(self_attn.k(input_x)).view(batch_size, seq_len, num_heads, head_dim)
            value = self_attn.v(input_x).view(batch_size, seq_len, num_heads, head_dim)
            query = rope_apply_head_dim(
                query.flatten(2), freqs, head_dim
            ).view(batch_size, seq_len, num_heads, head_dim)
            key = rope_apply_head_dim(
                key.flatten(2), freqs, head_dim
            ).view(batch_size, seq_len, num_heads, head_dim)
            self_attn_output = ulysses_attention(
                query,
                key,
                value,
                attention,
                k_lens=kwargs.get("seq_lens"),
                window_size=getattr(self_attn, "window_size", (-1, -1)),
                group=sp_group,
            ).flatten(2)
            self_attn_output = self_attn.o(self_attn_output)
        else:
            self_attn_output = block.self_attn(input_x, freqs, seq_lens=kwargs.get("seq_lens"))
        x = block.gate(x, gate_msa, self_attn_output)
        # norm3 (LayerNorm) upcasts to fp32 and context (text embeds) may be fp32;
        # audio cross_attn calls flash-attn (fp16/bf16 only) with no internal cast,
        # so cast both q-source and kv-source down first.
        x = x + block.cross_attn(block.norm3(x).to(dtype), context.to(dtype))
        return x

    def _video_ffn(
        self, x: torch.Tensor, block, kwargs: Dict[str, Any], dtype: torch.dtype
    ) -> torch.Tensor:
        modulation = kwargs["_video_e"]
        temp_x = block.norm2(x) * (1 + modulation[4]) + modulation[3]
        temp_x = temp_x.to(dtype)
        return x + block.ffn(temp_x) * modulation[5]

    def _audio_ffn(
        self, x: torch.Tensor, block, kwargs: Dict[str, Any], dtype: torch.dtype
    ) -> torch.Tensor:
        shift_mlp, scale_mlp, gate_mlp = kwargs["_audio_mod"]
        input_x = block.norm2(x) * (1 + scale_mlp) + shift_mlp
        return block.gate(x, gate_mlp, block.ffn(input_x))


class WanCreatorGatingAVModel(nn.Module):
    """Wan+Creator audio-video model with configurable gated cross attention.

    A2V means the video branch attends to audio and updates video tokens.
    V2A means the audio branch attends to video and updates audio tokens.
    """

    def __init__(
        self,
        video_model: WanTransformer3DModel,
        audio_model: CreatorAudioModel,
        use_temporal_rope: bool = True,
        audio_fps: float = 48000 / 960,
        vae_temporal_stride: int = 4,
        zero_init_cross_attn: bool = False,
        zero_init_video_cross_attn: bool | None = None,
        zero_init_audio_cross_attn: bool | None = None,
        a2v_cross_attn_layers: LayerSelection = None,
        v2a_cross_attn_layers: LayerSelection = None,
        use_gating: bool = True,
        use_a2v_gating: bool | None = None,
        use_v2a_gating: bool | None = None,
        zero_init_gating: bool = False,
        zero_init_a2v_gating: bool | None = None,
        zero_init_v2a_gating: bool | None = None,
        gate_init_value: Optional[float] = None,
        a2v_gate_init_value: Optional[float] = None,
        v2a_gate_init_value: Optional[float] = None,
        a2v_gate_alphas: LayerAlphas = None,
        v2a_gate_alphas: LayerAlphas = None,
    ):
        nn.Module.__init__(self)
        self.video_model = video_model
        self.audio_model = audio_model

        self.video_dim = int(video_model.dim)
        self.audio_dim = int(audio_model.dim)
        self.video_num_heads = int(video_model.num_heads)
        self.audio_num_heads = int(audio_model.num_heads)
        self.num_layers = int(video_model.num_layers)
        self.video_patch_size = tuple(int(value) for value in video_model.patch_size)
        self.audio_patch_size = tuple(int(value) for value in audio_model.patch_size)

        assert int(audio_model.num_layers) == self.num_layers, (
            f"Video ({self.num_layers}) and audio ({audio_model.num_layers}) must have same number of layers"
        )

        a2v_enabled = _expand_layer_selection(
            a2v_cross_attn_layers, self.num_layers, "a2v_cross_attn_layers"
        )
        v2a_enabled = _expand_layer_selection(
            v2a_cross_attn_layers, self.num_layers, "v2a_cross_attn_layers"
        )
        self.a2v_cross_attn_layers = a2v_enabled
        self.v2a_cross_attn_layers = v2a_enabled
        resolved_a2v_gate_alphas = _expand_layer_alphas(
            a2v_gate_alphas, self.num_layers, "a2v_gate_alphas"
        )
        resolved_v2a_gate_alphas = _expand_layer_alphas(
            v2a_gate_alphas, self.num_layers, "v2a_gate_alphas"
        )
        self.a2v_gate_alphas = resolved_a2v_gate_alphas
        self.v2a_gate_alphas = resolved_v2a_gate_alphas

        resolved_use_a2v_gating = use_gating if use_a2v_gating is None else use_a2v_gating
        resolved_use_v2a_gating = use_gating if use_v2a_gating is None else use_v2a_gating
        resolved_zero_init_a2v_gating = (
            zero_init_gating if zero_init_a2v_gating is None else zero_init_a2v_gating
        )
        resolved_zero_init_v2a_gating = (
            zero_init_gating if zero_init_v2a_gating is None else zero_init_v2a_gating
        )
        resolved_a2v_gate_init_value = (
            gate_init_value if a2v_gate_init_value is None else a2v_gate_init_value
        )
        resolved_v2a_gate_init_value = (
            gate_init_value if v2a_gate_init_value is None else v2a_gate_init_value
        )
        video_blocks = list(video_model.blocks)
        audio_blocks = list(audio_model.blocks)
        self.joint_blocks = nn.ModuleList([
            GatedJointBlock(
                video_block=video_block,
                audio_block=audio_block,
                video_dim=self.video_dim,
                audio_dim=self.audio_dim,
                video_num_heads=self.video_num_heads,
                audio_num_heads=self.audio_num_heads,
                enable_a2v_cross_attn=a2v_enabled[layer_idx],
                enable_v2a_cross_attn=v2a_enabled[layer_idx],
                zero_init_output=zero_init_cross_attn,
                zero_init_video_cross_attn=zero_init_video_cross_attn,
                zero_init_audio_cross_attn=zero_init_audio_cross_attn,
                use_a2v_gating=resolved_use_a2v_gating,
                use_v2a_gating=resolved_use_v2a_gating,
                zero_init_a2v_gating=resolved_zero_init_a2v_gating,
                zero_init_v2a_gating=resolved_zero_init_v2a_gating,
                a2v_gate_init_value=resolved_a2v_gate_init_value,
                v2a_gate_init_value=resolved_v2a_gate_init_value,
                a2v_gate_alpha=resolved_a2v_gate_alphas[layer_idx],
                v2a_gate_alpha=resolved_v2a_gate_alphas[layer_idx],
            )
            for layer_idx, (video_block, audio_block) in enumerate(zip(video_blocks, audio_blocks))
        ])

        video_model.blocks = nn.ModuleList()
        audio_model.blocks = nn.ModuleList()

        self.use_temporal_rope = use_temporal_rope
        self.audio_fps = audio_fps
        self.vae_temporal_stride = vae_temporal_stride
        if use_temporal_rope:
            head_dim = self.video_dim // self.video_num_heads
            self.temporal_rope_inv_freq = 1.0 / (
                10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
            )
        else:
            self.temporal_rope_inv_freq = None

        # Ulysses-style sequence-parallel inference state. All ranks keep a
        # complete copy of the weights and exchange token/head dimensions in
        # attention, matching the Wan2.2 inference design.
        self.sp_world_size = 1
        self.sp_world_rank = 0
        self.sp_group = None

    def enable_multi_gpus_inference(self, group=None) -> None:
        """Enable raw-process-group sequence parallelism for inference."""
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError("Sequence-parallel inference requires an initialized process group")
        self.sp_world_size = dist.get_world_size(group)
        self.sp_world_rank = dist.get_rank(group)
        self.sp_group = group
        if self.video_num_heads % self.sp_world_size != 0:
            raise ValueError(
                f"Video attention heads ({self.video_num_heads}) must be divisible by "
                f"SP size ({self.sp_world_size})"
            )
        if self.audio_num_heads % self.sp_world_size != 0:
            raise ValueError(
                f"Audio attention heads ({self.audio_num_heads}) must be divisible by "
                f"SP size ({self.sp_world_size})"
            )

    def _apply_sequence_parallel(self, video_state: Dict[str, Any], audio_state: Dict[str, Any]):
        """Shard prepared video/audio token states along their sequence axes."""
        if self.sp_world_size <= 1:
            video_state["sp_rank"] = 0
            video_state["sp_world_size"] = 1
            audio_state["sp_rank"] = 0
            audio_state["sp_world_size"] = 1
            return video_state, audio_state

        def shard(state: Dict[str, Any], *, audio: bool):
            local_len = state["x"].size(1) // self.sp_world_size
            rank = self.sp_world_rank
            state["x"] = torch.chunk(state["x"], self.sp_world_size, dim=1)[rank]
            if not audio:
                if state["e"].dim() >= 3:
                    state["e"] = torch.chunk(state["e"], self.sp_world_size, dim=1)[rank]
                if state["e0"].dim() >= 4:
                    state["e0"] = torch.chunk(state["e0"], self.sp_world_size, dim=1)[rank]
            state["freqs"] = (
                torch.chunk(state["freqs"], self.sp_world_size, dim=0)[rank]
                if audio else state["freqs"]
            )
            if state.get("temporal_positions") is not None:
                state["temporal_positions"] = torch.chunk(
                    state["temporal_positions"], self.sp_world_size, dim=1
                )[rank]
            state["local_seq_lens"] = (
                state["seq_lens"] - rank * local_len
            ).clamp(min=0, max=local_len)
            state["sp_rank"] = rank
            state["sp_world_size"] = self.sp_world_size
            return state

        return shard(video_state, audio=False), shard(audio_state, audio=True)

    def _prepare_video(self, video_inputs: Dict[str, Any], dtype: torch.dtype) -> Dict[str, Any]:
        video_model = self.video_model
        device = video_model.patch_embedding.weight.device

        if video_model.freqs.device != device:
            video_model.freqs = video_model.freqs.to(device)

        x_list = _to_list(video_inputs["x"])
        y = video_inputs.get("y")
        if y is not None:
            y_list = _to_list(y)
            x_list = [torch.cat([sample, condition], dim=0) for sample, condition in zip(x_list, y_list)]

        x_list = [video_model.patch_embedding(sample.unsqueeze(0)) for sample in x_list]
        grid_sizes = torch.stack([
            torch.tensor(sample.shape[2:], dtype=torch.long, device=device) for sample in x_list
        ])
        x_list = [sample.flatten(2).transpose(1, 2) for sample in x_list]
        seq_lens = torch.tensor([sample.size(1) for sample in x_list], dtype=torch.long, device=device)

        seq_len = self._round_seq_len(int(video_inputs["seq_len"]))
        assert int(seq_lens.max().item()) <= seq_len
        x = torch.cat([
            torch.cat([sample, sample.new_zeros(1, seq_len - sample.size(1), sample.size(2))], dim=1)
            for sample in x_list
        ])

        timesteps = video_inputs["t"].to(device)
        embedding, modulation = _build_time_embeddings(
            video_model.time_embedding,
            video_model.time_projection,
            int(video_model.freq_dim),
            int(video_model.dim),
            timesteps,
            seq_len,
        )

        context = _embed_context(video_model.text_embedding, video_inputs["context"], int(video_model.text_len))

        return {
            "x": x,
            "e": embedding,
            "e0": modulation,
            "seq_lens": seq_lens,
            "grid_sizes": grid_sizes,
            "freqs": video_model.freqs,
            "context": context,
            "context_lens": None,
            "seq_len": seq_len,
        }

    def _prepare_audio(self, audio_inputs: Dict[str, Any], dtype: torch.dtype) -> Dict[str, Any]:
        audio_model = self.audio_model
        device = audio_model.patch_embedding.weight.device

        x_list = _to_list(audio_inputs["x"])
        y = audio_inputs.get("y")
        if y is not None:
            y_list = _to_list(y)
            x_list = [torch.cat([sample, condition], dim=0) for sample, condition in zip(x_list, y_list)]

        original_audio_shapes = [tuple(sample.shape) for sample in x_list]

        patchified = []
        grid_sizes_list = []
        for sample in x_list:
            tokens = audio_model.patch_embedding(sample.unsqueeze(0).to(device))
            tokens = rearrange(tokens, "1 c f -> f c").contiguous()
            patchified.append(tokens)
            grid_sizes_list.append(tokens.shape[0])

        grid_sizes = torch.tensor([[grid_size] for grid_size in grid_sizes_list], dtype=torch.long, device=device)
        seq_lens = grid_sizes[:, 0]
        seq_len = self._round_seq_len(int(audio_inputs["seq_len"]))
        assert int(seq_lens.max().item()) <= seq_len

        x = torch.stack([
            torch.cat([sample, sample.new_zeros(seq_len - sample.size(0), sample.size(1))], dim=0)
            for sample in patchified
        ])

        timesteps = audio_inputs["t"].to(device)
        embedding, modulation = _build_time_embeddings(
            audio_model.time_embedding,
            audio_model.time_projection,
            int(audio_model.freq_dim),
            int(audio_model.dim),
            timesteps,
            seq_len,
        )

        context = _embed_context(audio_model.text_embedding, audio_inputs["context"], int(audio_model.text_len))
        freqs = audio_model._build_freqs(seq_len, device)

        clip_fea = audio_inputs.get("clip_fea")
        if audio_model.has_image_input and clip_fea is not None:
            clip_embedding = audio_model.img_emb(clip_fea)
            context = torch.cat([clip_embedding, context], dim=1)

        return {
            "x": x,
            "e": embedding,
            "e0": modulation,
            "seq_lens": seq_lens,
            "grid_sizes": grid_sizes,
            "freqs": freqs,
            "context": context,
            "context_lens": None,
            "seq_len": seq_len,
            "original_audio_shapes": original_audio_shapes,
        }

    def forward(
        self,
        video: Dict[str, Any],
        audio: Dict[str, Any],
        dtype: torch.dtype = torch.bfloat16,
        return_dict: bool = True,
        enable_a2v: Optional[Union[bool, torch.Tensor]] = None,
        enable_v2a: Optional[Union[bool, torch.Tensor]] = None,
    ):
        """Forward pass of the joint audio-video model.

        Args:
            video: video input dict with keys 'x', 't', 'context', 'seq_len', etc.
            audio: audio input dict with keys 'x', 't', 'context', 'seq_len', etc.
            dtype: computation dtype for attention ops (default bfloat16).
            return_dict: if True, return dict with 'video'/'audio' keys; else tuple.
            enable_a2v: gate A2V cross-attention (video attending to audio).
                A bool tensor can select the enabled CFG branches per sample.
            enable_v2a: gate V2A cross-attention (audio attending to video), same
                semantics as enable_a2v.

        Returns:
            dict or tuple of (video_output, audio_output) tensors.
        """
        video_state = self._prepare_video(video, dtype)
        audio_state = self._prepare_audio(audio, dtype)

        device = video_state["x"].device

        video_temporal_pos = None
        audio_temporal_pos = None
        temporal_rope_inv_freq = None
        if self.use_temporal_rope and self.temporal_rope_inv_freq is not None:
            video_fps = float(video.get("video_fps", 16.0))
            temporal_rope_inv_freq = self.temporal_rope_inv_freq
            video_temporal_pos = compute_video_temporal_positions(
                video_state["grid_sizes"],
                video_state["x"].size(1),
                device,
                audio_fps=self.audio_fps,
                video_fps=video_fps,
                vae_temporal_stride=self.vae_temporal_stride,
            )
            audio_temporal_pos = compute_audio_temporal_positions(
                audio_state["seq_lens"],
                audio_state["x"].size(1),
                device,
            )

        video_state["temporal_positions"] = video_temporal_pos
        audio_state["temporal_positions"] = audio_temporal_pos
        video_state, audio_state = self._apply_sequence_parallel(video_state, audio_state)
        video_x = video_state["x"]
        audio_x = audio_state["x"]

        video_kwargs = {
            "e0": video_state["e0"],
            "seq_lens": video_state["seq_lens"],
            "grid_sizes": video_state["grid_sizes"],
            "freqs": video_state["freqs"],
            "context": video_state["context"],
            "context_lens": video_state["context_lens"],
            "temporal_positions": video_state["temporal_positions"],
            "temporal_rope_inv_freq": temporal_rope_inv_freq,
            "sp_rank": video_state["sp_rank"],
            "sp_world_size": video_state["sp_world_size"],
        }
        audio_kwargs = {
            "e0": audio_state["e0"],
            "seq_lens": audio_state["seq_lens"],
            "freqs": audio_state["freqs"],
            "context": audio_state["context"],
            "temporal_positions": audio_state["temporal_positions"],
            "sp_rank": audio_state["sp_rank"],
            "sp_world_size": audio_state["sp_world_size"],
        }

        runtime_cross_attn = {
            "enable_a2v": enable_a2v,
            "enable_v2a": enable_v2a,
            "sequence_parallel": self.sp_world_size > 1,
            "sp_group": self.sp_group,
        }

        for joint_block in self.joint_blocks:
            video_x, audio_x = joint_block(
                video_x, audio_x, video_kwargs, audio_kwargs, dtype,
                **runtime_cross_attn,
            )

        video_output = self.video_model.head(video_x, video_state["e"])
        audio_output = self.audio_model.head(audio_x, audio_state["e"])

        if self.sp_world_size > 1:
            video_output = all_gather_sequence(video_output, group=self.sp_group)
            audio_output = all_gather_sequence(audio_output, group=self.sp_group)

        video_output = torch.stack(
            self.video_model.unpatchify(video_output, video_state["grid_sizes"])
        )
        audio_output = torch.stack(
            self.audio_model.unpatchify(
                audio_output,
                audio_state["grid_sizes"],
                audio_state["original_audio_shapes"],
            )
        )
        result = {"video": video_output, "audio": audio_output}
        return result if return_dict else (video_output, audio_output)

    def _round_seq_len(self, seq_len: int) -> int:
        if self.sp_world_size > 1:
            return int(math.ceil(seq_len / self.sp_world_size) * self.sp_world_size)
        return int(seq_len)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        video_pretrained_model_path: Optional[str] = None,
        audio_pretrained_model_path: Optional[str] = None,
        video_subfolder: Optional[str] = None,
        audio_subfolder: Optional[str] = None,
        video_kwargs: Optional[Dict] = None,
        audio_kwargs: Optional[Dict] = None,
        video_model_cls=Wan2_2Transformer3DModel,
        audio_model_cls=CreatorAudioModel,
        low_cpu_mem_usage: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        use_temporal_rope: bool = True,
        audio_fps: float = 48000.0 / 960.0,
        vae_temporal_stride: int = 4,
        zero_init_cross_attn: bool = False,
        zero_init_video_cross_attn: bool | None = None,
        zero_init_audio_cross_attn: bool | None = None,
        a2v_cross_attn_layers: LayerSelection = None,
        v2a_cross_attn_layers: LayerSelection = None,
        use_gating: bool = True,
        use_a2v_gating: bool | None = None,
        use_v2a_gating: bool | None = None,
        zero_init_gating: bool = False,
        zero_init_a2v_gating: bool | None = None,
        zero_init_v2a_gating: bool | None = None,
        gate_init_value: Optional[float] = None,
        a2v_gate_init_value: Optional[float] = None,
        v2a_gate_init_value: Optional[float] = None,
        a2v_gate_alphas: LayerAlphas = None,
        v2a_gate_alphas: LayerAlphas = None,
    ):
        video_kwargs = dict(video_kwargs or {})
        audio_kwargs = dict(audio_kwargs or {})

        if video_pretrained_model_path is not None and audio_pretrained_model_path is not None:
            video_path = video_pretrained_model_path
            audio_path = audio_pretrained_model_path
        elif pretrained_model_path is not None:
            video_path = os.path.join(pretrained_model_path, "video_model")
            audio_path = os.path.join(pretrained_model_path, "audio_model")
        else:
            raise ValueError(
                "Must provide either pretrained_model_path or both video_pretrained_model_path "
                "and audio_pretrained_model_path"
            )

        logging.info("Loading video model from: %s", video_path)
        video_model = video_model_cls.from_pretrained(
            video_path,
            subfolder=video_subfolder,
            transformer_additional_kwargs=video_kwargs,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype,
        )

        logging.info("Loading audio model from: %s", audio_path)
        audio_model = audio_model_cls.from_pretrained(
            audio_path,
            subfolder=audio_subfolder,
            transformer_additional_kwargs=audio_kwargs,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype,
        )

        model = cls(
            video_model=video_model,
            audio_model=audio_model,
            use_temporal_rope=use_temporal_rope,
            audio_fps=audio_fps,
            vae_temporal_stride=vae_temporal_stride,
            zero_init_cross_attn=zero_init_cross_attn,
            zero_init_video_cross_attn=zero_init_video_cross_attn,
            zero_init_audio_cross_attn=zero_init_audio_cross_attn,
            a2v_cross_attn_layers=a2v_cross_attn_layers,
            v2a_cross_attn_layers=v2a_cross_attn_layers,
            use_gating=use_gating,
            use_a2v_gating=use_a2v_gating,
            use_v2a_gating=use_v2a_gating,
            zero_init_gating=zero_init_gating,
            zero_init_a2v_gating=zero_init_a2v_gating,
            zero_init_v2a_gating=zero_init_v2a_gating,
            gate_init_value=gate_init_value,
            a2v_gate_init_value=a2v_gate_init_value,
            v2a_gate_init_value=v2a_gate_init_value,
            a2v_gate_alphas=a2v_gate_alphas,
            v2a_gate_alphas=v2a_gate_alphas,
        ).to(torch_dtype)
        if pretrained_model_path is not None:
            cross_attn_file = os.path.join(pretrained_model_path, "cross_attn_weights.safetensors")
            cross_attn_file_bin = os.path.join(pretrained_model_path, "cross_attn_weights.bin")

            if os.path.exists(cross_attn_file):
                from safetensors.torch import load_file

                cross_attn_state = load_file(cross_attn_file)
                logging.info("Loading cross-attn weights from: %s (%d keys)", cross_attn_file, len(cross_attn_state))
            elif os.path.exists(cross_attn_file_bin):
                cross_attn_state = torch.load(cross_attn_file_bin, map_location="cpu")
                logging.info("Loading cross-attn weights from: %s (%d keys)", cross_attn_file_bin, len(cross_attn_state))
            else:
                cross_attn_state = None
                logging.warning("No cross_attn_weights found in %s, skipping.", pretrained_model_path)

            if cross_attn_state is not None:
                missing, unexpected = model.load_state_dict(cross_attn_state, strict=False)
                logging.info("Cross-attn load: %d missing, %d unexpected keys", len(missing), len(unexpected))
                if unexpected:
                    logging.warning("Unexpected keys in cross_attn_weights: %s", unexpected[:10])

        return model


WanCreatorCrossAttnGatingAVModel = WanCreatorGatingAVModel
WanCreatorGatedCrossAttnAVModel = WanCreatorGatingAVModel

__all__ = [
    "GatedCrossModalAttention",
    "GatedJointBlock",
    "WanCreatorGatingAVModel",
    "WanCreatorCrossAttnGatingAVModel",
    "WanCreatorGatedCrossAttnAVModel",
    "compute_audio_temporal_positions",
    "compute_video_temporal_positions",
]
