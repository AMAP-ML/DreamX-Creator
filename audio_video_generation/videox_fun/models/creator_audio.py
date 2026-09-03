"""Creator audio diffusion transformer used by the release inference path."""

import glob
import json
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

from .creator.creator_video_dit import DiTBlock
from .creator.creator_audio_dit import (
    Head,
    MLP,
    sinusoidal_embedding_1d,
    precompute_freqs_cis_1d,
)


class CreatorAudioModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    """Creator Audio DiT with wan_audio2-compatible forward interface."""

    @register_to_config
    def __init__(
        self,
        patch_size=(1,),
        text_len=512,
        in_dim=128,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=128,
        num_heads=16,
        num_layers=32,
        eps=1e-6,
        has_image_input=False,
        has_image_pos_emb=False,
        has_ref_conv=False,
        vae_type="dac",
        **kwargs
    ):
        super().__init__()

        self.patch_size = tuple(patch_size) if isinstance(patch_size, (list, tuple)) else (patch_size,)
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.eps = eps
        self.has_image_input = has_image_input
        self.vae_type = vae_type

        # Patch embedding (1D conv)
        self.patch_embedding = nn.Conv1d(
            in_dim, dim, kernel_size=self.patch_size[0], stride=self.patch_size[0],
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6),
        )

        # DiTBlock stack
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])

        self.head = Head(dim, out_dim, self.patch_size, eps)

        # RoPE precomputation
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_1d(head_dim)

        # Optional image / reference embeddings
        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_ref_conv = has_ref_conv

    # -----------------------------------------------------------------
    # Unpatchify
    # -----------------------------------------------------------------

    def unpatchify(self, x, grid_sizes, output_shapes):
        """Unpatchify tokens back to audio latent shape.

        Args:
            x: [B, seq_len, out_dim * patch_size]
            grid_sizes: [B, 1] token lengths
            output_shapes: list of original audio shapes [(C, T), ...]

        Returns:
            list of restored tensors matching output_shapes.
        """
        output = []
        for i, (grid_size, original_shape) in enumerate(zip(grid_sizes, output_shapes)):
            f = int(grid_size[0].item())
            restored = rearrange(
                x[i, :f].unsqueeze(0),
                "b f (p c) -> b c (f p)",
                f=f, p=self.patch_size[0],
            )
            # Trim to original time length
            orig_T = original_shape[-1]
            restored = restored[:, :, :orig_T].squeeze(0)
            output.append(restored)
        return output

    # -----------------------------------------------------------------
    # RoPE frequencies
    # -----------------------------------------------------------------
    def _build_freqs(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build RoPE complex frequencies [seq_len, 1, rope_dim]."""
        freqs = torch.cat([
            self.freqs[0][:seq_len].view(seq_len, -1),
            self.freqs[1][:seq_len].view(seq_len, -1),
            self.freqs[2][:seq_len].view(seq_len, -1),
        ], dim=-1).reshape(seq_len, 1, -1).to(device)
        return freqs

    # -----------------------------------------------------------------
    # Weight initialisation
    # -----------------------------------------------------------------
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)
        for module in self.text_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        for module in self.time_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

        nn.init.zeros_(self.head.head.weight)
        if self.head.head.bias is not None:
            nn.init.zeros_(self.head.head.bias)

    # -----------------------------------------------------------------
    # forward (matches wan_audio2.WanAudioModel interface)
    # -----------------------------------------------------------------
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        dtype=torch.bfloat16,
        **kwargs,
    ):
        """
        Args:
            x: list of [C, T] audio latent samples, or [B, C, T] tensor.
            t: [B] or [B, T] timesteps.
            context: list of [S, text_dim] or [B, S, text_dim] tensor.
            seq_len: max sequence length after patching.
            clip_fea: optional CLIP features for image conditioning.
            y: optional conditioning latent (e.g. for i2a), same format as x.
        """
        # --- Normalise inputs to lists ---
        if isinstance(x, torch.Tensor):
            x = [sample for sample in x]
        if isinstance(context, torch.Tensor):
            context = [sample for sample in context]
        if y is not None and isinstance(y, torch.Tensor):
            y = [sample for sample in y]

        device = self.patch_embedding.weight.device
        dtype = x[0].dtype if len(x) > 0 else self.patch_embedding.weight.dtype
        batch_size = len(x)

        # Concatenate conditioning y (e.g. for i2v/i2a)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Remember original shapes for unpatchify
        original_audio_shapes = [tuple(sample.shape) for sample in x]

        # --- Patchify per sample, then pad tokens ---
        patchified_samples = []
        grid_sizes_list = []
        for sample in x:
            tokens_i = self.patch_embedding(sample.unsqueeze(0).to(device))  # [1, dim, T']
            tokens_i = rearrange(tokens_i, "1 c f -> f c").contiguous()
            patchified_samples.append(tokens_i)
            grid_sizes_list.append(tokens_i.shape[0])

        grid_sizes = torch.tensor(
            [[g] for g in grid_sizes_list], dtype=torch.long, device=device,
        )
        seq_lens = grid_sizes[:, 0]
        assert seq_lens.max() <= seq_len, (
            f"Max token length {seq_lens.max().item()} exceeds seq_len {seq_len}"
        )

        # Pad each sample's tokens to seq_len and stack
        x = torch.stack([
            torch.cat([u, u.new_zeros(seq_len - u.size(0), u.size(1))], dim=0)
            for u in patchified_samples
        ])

        # --- Time embedding ---
        with torch.amp.autocast("cuda", dtype=torch.float32):
            if t.dim() != 1:
                # Per-token timesteps [B, T]
                if t.size(1) < seq_len:
                    pad_size = seq_len - t.size(1)
                    last_elements = t[:, -1].unsqueeze(1)
                    t = torch.cat([t, last_elements.repeat(1, pad_size)], dim=1)
                bt = t.size(0)
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t.flatten())
                    .unflatten(0, (bt, seq_len)).float()
                )
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float()
                )
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # --- Context embedding ---
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )

        # Image embedding
        if self.has_image_input and clip_fea is not None:
            clip_embedding = self.img_emb(clip_fea)
            context = torch.cat([clip_embedding, context], dim=1)

        # --- RoPE frequencies ---
        freqs = self._build_freqs(seq_len, device)

        for block in self.blocks:
            x = block(x, context, e0, freqs, seq_lens=seq_lens)

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes, original_audio_shapes)
        return x

    # -----------------------------------------------------------------
    # from_pretrained
    # -----------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        subfolder=None,
        transformer_additional_kwargs=None,
        low_cpu_mem_usage=False,
        in_dim=None,
        out_dim=None,
        patch_size=None,
        torch_dtype=torch.bfloat16,
    ):
        transformer_additional_kwargs = dict(transformer_additional_kwargs or {})
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        config_file = os.path.join(pretrained_model_path, "config.json")
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")

        with open(config_file, "r") as fp:
            config = json.load(fp)

        from diffusers.utils import WEIGHTS_NAME

        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        # Explicit architecture overrides
        if in_dim is not None:
            transformer_additional_kwargs["in_dim"] = in_dim
        if out_dim is not None:
            transformer_additional_kwargs["out_dim"] = out_dim
        if patch_size is not None:
            transformer_additional_kwargs["patch_size"] = patch_size

        if "dict_mapping" in transformer_additional_kwargs:
            for key, value in transformer_additional_kwargs["dict_mapping"].items():
                if value not in transformer_additional_kwargs:
                    transformer_additional_kwargs[value] = config[key]

        # Merge overrides into config
        model_config = dict(config)
        model_config.update(transformer_additional_kwargs)

        model = cls.from_config(model_config, **transformer_additional_kwargs)

        # --- Load state dict ---
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file
            state_dict = {}
            for shard in glob.glob(os.path.join(pretrained_model_path, "*.safetensors")):
                state_dict.update(load_file(shard))

        # Filter by shape match
        model_sd = model.state_dict()
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key in model_sd and model_sd[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                logging.info("Skipping key %s due to size mismatch or absence.", key)

        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        logging.info(
            "CreatorAudioModel missing keys: %d, unexpected keys: %d",
            len(missing), len(unexpected),
        )
        return model.to(torch_dtype)

# Convenience aliases
CreatorAudioTransformerModel = CreatorAudioModel
