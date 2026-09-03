import os
from typing import List
import torch

from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.t5 import umt5_xxl
from wan.modules.vae import _video_vae


class WanTextEncoder(torch.nn.Module):
    def __init__(self, wan_model_dir="wan_models/Wan2.1-T2V-1.3B") -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(os.path.join(wan_model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(wan_model_dir, "google/umt5-xxl/"), seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    """Wan2.1 VAE wrapper (z_dim=16, spatial /8)."""

    def __init__(self, vae_pth="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device = pixel.device
        model_dtype = next(self.model.parameters()).dtype
        scale = [self.mean.to(device=device, dtype=model_dtype),
                 1.0 / self.std.to(device=device, dtype=model_dtype)]

        output = [
            self.model.encode(u.to(dtype=model_dtype).unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False,
                        offload_output: bool = False) -> torch.Tensor:
        # `offload_output` is accepted for interface parity with the 2.2 VAE
        # wrapper (which supports CPU accumulation to cut peak memory). This 2.1
        # decode path does not implement it and ignores the flag.
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device = latent.device
        model_dtype = next(self.model.parameters()).dtype
        scale = [self.mean.to(device=device, dtype=model_dtype),
                 1.0 / self.std.to(device=device, dtype=model_dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.to(dtype=model_dtype).unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output
