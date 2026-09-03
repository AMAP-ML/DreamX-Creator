"""Single-image Creator audio-video inference.

The supplied image is encoded as the first video frame. The prompt conditions
the joint Creator video/audio denoiser, which writes a video, a WAV, and a merged
MP4. This entrypoint intentionally has no dataset, Verse-Bench, training, or
model-download workflow.
"""

import argparse
import inspect
import math
import os
import sys
import time
import subprocess

import numpy as np
import torch
import torch.distributed as dist
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer as HFAutoTokenizer
from torchvision.io import write_video
from torch import nn
from tqdm.auto import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

current_file_path = os.path.abspath(__file__)
release_root = os.path.dirname(current_file_path)
if release_root not in sys.path:
    sys.path.insert(0, release_root)

from videox_fun.models import AutoencoderKLWan3_8, WanT5EncoderModel
from videox_fun.models.creator_gating import WanCreatorGatingAVModel
from videox_fun.models.creator_dac_vae import CreatorDACVAE


def filter_kwargs(cls, kwargs):
    """Keep scheduler options compatible across diffusers versions."""
    valid = set(inspect.signature(cls.__init__).parameters)
    return {key: value for key, value in kwargs.items() if key in valid}


class DirectionalMultimodalCFGAdapter(nn.Module):
    """Preserve the experiment's three-branch multimodal CFG in one call."""

    def __init__(self, model, video_scale, audio_scale, enable_a2v, enable_v2a):
        super().__init__()
        self.model = model
        self.video_scale = float(video_scale)
        self.audio_scale = float(audio_scale)
        self.enable_a2v = bool(enable_a2v)
        self.enable_v2a = bool(enable_v2a)

    @property
    def video_patch_size(self):
        return self.model.video_patch_size

    @property
    def audio_patch_size(self):
        return self.model.audio_patch_size

    @staticmethod
    def _expand(value, name):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0 or value.shape[0] != 2:
                raise ValueError(f"Multimodal CFG expects {name} batch size 2")
            return torch.cat((value[0:1], value[0:1], value[1:2]), dim=0)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return [value[0], value[0], value[1]]
        raise TypeError(f"Unsupported {name} batch value: {type(value)!r}")

    @classmethod
    def _expand_inputs(cls, inputs):
        expanded = dict(inputs)
        for key in ("x", "t", "context", "y", "clip_fea"):
            if expanded.get(key) is not None:
                expanded[key] = cls._expand(expanded[key], key)
        return expanded

    @staticmethod
    def _collapse(prediction, bridge_scale):
        if prediction.shape[0] != 3:
            raise ValueError("Creator multimodal CFG expects three model predictions")
        d00, d0b, dtb = prediction[0:1], prediction[1:2], prediction[2:3]
        bridge = d00 + bridge_scale * (d0b - d00)
        return torch.cat((bridge, bridge + (dtb - d0b)), dim=0)

    def forward(self, video, audio, dtype=torch.bfloat16, return_dict=True, **kwargs):
        if self.model.training:
            raise RuntimeError("Inference adapter received a training model")
        model_video = self._expand_inputs(video)
        model_audio = self._expand_inputs(audio)
        device = model_video["t"].device
        bridge_mask = torch.tensor([False, True, True], device=device, dtype=torch.bool)
        output = self.model(
            video=model_video,
            audio=model_audio,
            dtype=dtype,
            return_dict=True,
            enable_a2v=bridge_mask & self.enable_a2v,
            enable_v2a=bridge_mask & self.enable_v2a,
        )
        result = {
            "video": self._collapse(output["video"], self.video_scale),
            "audio": self._collapse(output["audio"], self.audio_scale),
        }
        return result if return_dict else (result["video"], result["audio"])

try:
    import soundfile as sf
except ImportError:
    sf = None


# ============================================================================
    # Single-process helpers
# ============================================================================

def print_info(*args, **kwargs):
    print(*args, **kwargs)


def synchronize_tensor(tensor: torch.Tensor, args) -> torch.Tensor:
    """Broadcast a rank-0 tensor when the SP launcher requests exact inputs."""
    if getattr(args, "synchronize_noise", False) and dist.is_initialized():
        dist.broadcast(tensor, src=0)
    return tensor


# ============================================================================
# Argument parsing
# ============================================================================

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Single-image Creator audio-video inference")

    # Model paths
    parser.add_argument("--config_path", type=str, default=os.path.join(os.path.dirname(__file__), "config/config.yaml"))
    parser.add_argument("--model_name", type=str, required=True,
                        help="Wan2.2-TI2V-5B model directory")
    parser.add_argument("--transformer_path", type=str, required=True,
                        help="Creator checkpoint directory with video_model/ and audio_model/")
    parser.add_argument("--audio_vae_path", type=str, required=True,
                        help="Path to CreatorDACVAE audio VAE")

    # Input / Output
    parser.add_argument("--image", type=str, required=True,
                        help="Input image used as the first video frame")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output", type=str, default="./outputs/output.mp4",
                        help="Output MP4; a WAV with the same stem is also written")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    # Generation parameters
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Target duration in seconds")
    parser.add_argument("--target_spatial_tokens", type=int, default=880,
                        help="Maximum spatial tokens per frame")
    parser.add_argument("--min_token_ratio", type=float, default=0.95,
                        help="Minimum fraction of target spatial tokens")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--cfg_mode", type=str, default="multimodal",
                        choices=["text", "multimodal"],
                        help="CFG mode. 'multimodal' uses Creator dual CFG with separate "
                             "bridge and text guidance (3 model evaluations per step).")
    parser.add_argument("--video_bridge_guidance_scale", type=float, default=3.5)
    parser.add_argument("--audio_bridge_guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video_shift", type=float, default=5.0,
                        help="Noise schedule shift for video denoising")
    parser.add_argument("--audio_shift", type=float, default=5.0,
                        help="Noise schedule shift for audio denoising")

    # Sampler
    parser.add_argument("--sampler_name", type=str, default="Flow", choices=["Flow"])

    # Memory & compute
    parser.add_argument("--weight_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--GPU_memory_mode", type=str, default="model_full_load",
                        choices=["model_full_load", "model_cpu_offload"])
    offload_action = argparse.BooleanOptionalAction
    parser.add_argument(
        "--text_encoder_cpu_offload",
        action=offload_action,
        default=None,
        help="Keep the text encoder on CPU except while encoding prompts. "
             "Defaults to the GPU_memory_mode setting.",
    )
    parser.add_argument(
        "--video_vae_cpu_offload",
        action=offload_action,
        default=None,
        help="Keep the video VAE on CPU except while encoding/decoding. "
             "Defaults to the GPU_memory_mode setting.",
    )
    parser.add_argument(
        "--audio_vae_cpu_offload",
        action=offload_action,
        default=None,
        help="Keep the audio VAE on CPU except while decoding. "
             "Defaults to the GPU_memory_mode setting.",
    )
    parser.add_argument(
        "--vae_cpu_offload",
        action=offload_action,
        default=None,
        help="Enable or disable CPU offload for both video and audio VAEs. "
             "Individual VAE flags take precedence.",
    )

    # Temporal RoPE
    parser.add_argument("--use_temporal_rope", type=bool, default=True)
    parser.add_argument("--audio_fps", type=float, default=48000.0 / 960.0,
                        help="Audio latent FPS (DAC: 48000/960=50)")
    parser.add_argument("--vae_temporal_stride", type=int, default=4)

    # Runtime cross-attention controls
    parser.add_argument("--disable_a2v_cross_attn", "--disable-a2v-cross-attn",
                        "--disable_a2v", "--disable-a2v", action="store_true",
                        help="Disable audio-to-video cross attention at inference time.")
    parser.add_argument("--disable_v2a_cross_attn", "--disable-v2a-cross-attn",
                        "--disable_v2a", "--disable-v2a", action="store_true",
                        help="Disable video-to-audio cross attention at inference time.")

    args = parser.parse_args()
    return args


# ============================================================================
# Device / dtype helpers
# ============================================================================

def init_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    return device


def resolve_weight_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    return torch.float32


def resolve_cpu_offload_flags(args):
    """Resolve dedicated offload flags, preserving the legacy memory mode."""
    legacy_offload = args.GPU_memory_mode == "model_cpu_offload"

    def resolve(value):
        return legacy_offload if value is None else bool(value)

    vae_override = getattr(args, "vae_cpu_offload", None)

    def resolve_vae(value):
        return resolve(vae_override if value is None else value)

    return {
        "transformer": legacy_offload,
        "text_encoder": resolve(getattr(args, "text_encoder_cpu_offload", None)),
        "video_vae": resolve_vae(getattr(args, "video_vae_cpu_offload", None)),
        "audio_vae": resolve_vae(getattr(args, "audio_vae_cpu_offload", None)),
    }


def clear_cuda_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================================
# Text encoding
# ============================================================================

def _get_t5_prompt_embeds(tokenizer, text_encoder, prompt, max_sequence_length, device, dtype):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_attention_mask = text_inputs.attention_mask
    seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(
        text_input_ids.to(device),
        attention_mask=prompt_attention_mask.to(device),
    )[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    return [embed[:seq_len] for embed, seq_len in zip(prompt_embeds, seq_lens.tolist())]


def encode_prompt(tokenizer, text_encoder, prompt, negative_prompt, guidance_scale,
                  max_sequence_length, device, dtype):
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer, text_encoder, prompt, max_sequence_length, device, dtype)

    if guidance_scale <= 1.0:
        return prompt_embeds

    negative_prompt = negative_prompt or ""
    negative_prompt_embeds = _get_t5_prompt_embeds(
        tokenizer, text_encoder, negative_prompt, max_sequence_length, device, dtype)
    return negative_prompt_embeds + prompt_embeds


# ============================================================================
# Latent shape helpers
# ============================================================================

def compute_audio_latent_length(duration: float, audio_vae: CreatorDACVAE) -> int:
    """Compute audio latent time length from duration.

    For DAC VAE: latent_T = ceil(duration * sample_rate / hop_length)
    """
    sample_rate = audio_vae.sample_rate
    hop_length = audio_vae.hop_length
    num_samples = int(duration * sample_rate)
    latent_T = math.ceil(num_samples / hop_length)
    return latent_T


def get_audio_num_tokens(latent_T: int, patch_size: tuple) -> int:
    """Compute number of audio tokens after patching."""
    return latent_T // patch_size[0]


def compute_video_latent_shape(duration, fps, vae_temporal_ratio=4, vae_spatial_ratio=8,
                               height=480, width=832):
    """Compute video latent dimensions from duration and resolution."""
    num_frames = int(duration * fps)
    num_frames = int((num_frames - 1) // vae_temporal_ratio * vae_temporal_ratio) + 1
    latent_frames = (num_frames - 1) // vae_temporal_ratio + 1
    latent_height = height // vae_spatial_ratio
    latent_width = width // vae_spatial_ratio
    return num_frames, latent_frames, latent_height, latent_width


# ============================================================================
# Audio saving
# ============================================================================

def save_audio_wav(waveform: torch.Tensor, sample_rate: int, output_path: str):
    """Save waveform tensor to wav file.

    Args:
        waveform: [B, 1, T] or [1, T] or [T] tensor
        sample_rate: audio sample rate
        output_path: path to save wav file
    """
    waveform = waveform.cpu().float()
    if waveform.ndim == 3:
        waveform = waveform.squeeze(0)  # [1, T]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # [1, T]

    if sf is not None:
        sf.write(output_path, waveform.transpose(0, 1).numpy(), sample_rate)
        return

    try:
        import torchaudio
        torchaudio.save(output_path, waveform, sample_rate)
        return
    except ImportError:
        pass

    import wave
    waveform = waveform.clamp(-1.0, 1.0)
    waveform_int16 = (waveform * 32767.0).to(torch.int16).transpose(0, 1).contiguous().numpy()
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(int(waveform_int16.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(waveform_int16.tobytes())


# ============================================================================
# Scheduler helpers
# ============================================================================

def retrieve_timesteps(scheduler, num_inference_steps=None, device=None,
                       timesteps=None, sigmas=None, **kwargs):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        return scheduler.timesteps, num_inference_steps


def prepare_extra_step_kwargs(scheduler, eta=0.0):
    extra_step_kwargs = {}
    if "eta" in set(inspect.signature(scheduler.step).parameters.keys()):
        extra_step_kwargs["eta"] = eta
    return extra_step_kwargs


def resolve_scheduler_shifts(args):
    """Resolve separate AV shifts while supporting callers that only define --shift."""
    video_shift = getattr(args, "video_shift", 5.0)
    audio_shift = getattr(args, "audio_shift", 5.0)
    return float(video_shift), float(audio_shift)


# ============================================================================
# Image to latent encoding
# ============================================================================

def encode_first_frame(image: Image.Image, vae, device, dtype):
    """Encode first frame image into VAE latent space.

    Returns latent of shape [1, C, 1, H//8, W//8].
    """
    image = image.convert("RGB")
    img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor * 2.0 - 1.0  # normalize to [-1, 1]
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
    img_tensor = img_tensor.to(device=device, dtype=dtype)
    vae = vae.to(dtype=dtype, device=device)
    with torch.no_grad():
        posterior = vae.encode(img_tensor)[0]
        latent = posterior.sample()
    return latent  # [1, C, 1, H//8, W//8]


def _budget_spatial_size(
    target_spatial_tokens: int,
    source_height: int,
    source_width: int,
    spatial_divisor_h: int,
    spatial_divisor_w: int,
    min_token_ratio: float = 0.95,
):
    """Pick a near-aspect-ratio size within the spatial-token budget."""
    if target_spatial_tokens <= 0:
        raise ValueError("target_spatial_tokens must be positive")
    if source_height <= 0 or source_width <= 0:
        raise ValueError(f"Invalid source size: {(source_height, source_width)}")
    if not 0.0 < min_token_ratio <= 1.0:
        raise ValueError("min_token_ratio must be in (0, 1]")

    min_spatial_tokens = max(1, int(math.ceil(target_spatial_tokens * min_token_ratio)))
    source_ratio = float(source_height) / float(source_width)
    best = None
    for token_height in range(1, target_spatial_tokens + 1):
        max_token_width = target_spatial_tokens // token_height
        if max_token_width < 1:
            continue
        ideal_token_width = (
            source_width * token_height * spatial_divisor_h
            / (source_height * spatial_divisor_w)
        )
        for token_width in {
            1,
            max_token_width,
            int(math.floor(ideal_token_width)),
            int(math.ceil(ideal_token_width)),
        }:
            if not 1 <= token_width <= max_token_width:
                continue
            spatial_tokens = token_height * token_width
            height = token_height * spatial_divisor_h
            width = token_width * spatial_divisor_w
            aspect_error = abs(math.log((height / width) / source_ratio))
            score = (
                max(0, min_spatial_tokens - spatial_tokens),
                aspect_error,
                target_spatial_tokens - spatial_tokens,
                height,
                width,
            )
            if best is None or score < best[0]:
                best = (score, height, width, spatial_tokens)

    if best is None:
        raise ValueError(f"Unable to resolve target_spatial_tokens={target_spatial_tokens}")
    return best[1], best[2], best[3]


def compute_dynamic_resolution(
    image: Image.Image,
    target_spatial_tokens: int = 880,
    min_token_ratio: float = 0.95,
    spatial_divisor_h: int = 32,
    spatial_divisor_w: int = 32,
):
    """Preserve input aspect ratio while using 0.95-1.0 of the token budget."""
    source_width, source_height = image.size
    height, width, actual_tokens = _budget_spatial_size(
        int(target_spatial_tokens),
        source_height,
        source_width,
        spatial_divisor_h,
        spatial_divisor_w,
        min_token_ratio,
    )
    return height, width, int(target_spatial_tokens), actual_tokens


# ============================================================================
# Model setup
# ============================================================================

def setup_models(args, device, weight_dtype):
    config = OmegaConf.load(args.config_path)

    # Video transformer kwargs
    video_transformer_kwargs = OmegaConf.to_container(
        config.get("video_transformer_additional_kwargs",
                   config.get("transformer_additional_kwargs", {})),
        resolve=True,
    )
    # Audio transformer kwargs
    audio_transformer_kwargs = OmegaConf.to_container(
        config.get("audio_transformer_additional_kwargs",
                   config.get("transformer_additional_kwargs", {})),
        resolve=True,
    )

    # Resolve transformer paths
    video_path = args.transformer_path
    audio_path = args.transformer_path

    # Check if joint checkpoint (has video_model/ and audio_model/ subdirs)
    if args.transformer_path is not None:
        video_sub = os.path.join(args.transformer_path, "video_model")
        audio_sub = os.path.join(args.transformer_path, "audio_model")
        if os.path.isdir(video_sub) and os.path.isdir(audio_sub):
            video_path = video_sub
            audio_path = audio_sub

    # Creator gating kwargs
    creator_gating_kwargs = OmegaConf.to_container(
        config.get("creator_gating_kwargs", {}),
        resolve=True,
    )

    print_info(f"Loading Creator gating AV transformer: video={video_path}, audio={audio_path}")
    transformer = WanCreatorGatingAVModel.from_pretrained(
        pretrained_model_path=args.transformer_path,
        video_pretrained_model_path=video_path,
        audio_pretrained_model_path=audio_path,
        video_subfolder=video_transformer_kwargs.get("transformer_low_noise_model_subpath", None),
        audio_subfolder=audio_transformer_kwargs.get("transformer_low_noise_model_subpath", None),
        video_kwargs=video_transformer_kwargs,
        audio_kwargs=audio_transformer_kwargs,
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
        # Pass gating-specific kwargs from config
        use_temporal_rope=creator_gating_kwargs.get("use_temporal_rope", args.use_temporal_rope),
        audio_fps=creator_gating_kwargs.get("audio_fps", args.audio_fps),
        vae_temporal_stride=creator_gating_kwargs.get("vae_temporal_stride", args.vae_temporal_stride),
        a2v_cross_attn_layers=creator_gating_kwargs.get("a2v_cross_attn_layers", None),
        v2a_cross_attn_layers=creator_gating_kwargs.get("v2a_cross_attn_layers", None),
        use_gating=creator_gating_kwargs.get("use_gating", True),
        zero_init_cross_attn=creator_gating_kwargs.get("zero_init_cross_attn", False),
        zero_init_gating=creator_gating_kwargs.get("zero_init_gating", True),
        gate_init_value=creator_gating_kwargs.get("gate_init_value", 0.0),
        a2v_gate_alphas=creator_gating_kwargs.get("a2v_gate_alphas", None),
        v2a_gate_alphas=creator_gating_kwargs.get("v2a_gate_alphas", None),
    )

    # Load video VAE
    video_vae_path = os.path.join(
        args.model_name, config.get("video_vae_kwargs", {}).get("vae_subpath", "vae"))
    print_info(f"Loading video VAE from: {video_vae_path}")
    vae_kwargs = OmegaConf.to_container(config.get("video_vae_kwargs", {}), resolve=True)
    video_vae = AutoencoderKLWan3_8.from_pretrained(video_vae_path, additional_kwargs=vae_kwargs)

    # Load audio VAE (CreatorDACVAE)
    print_info(f"Loading audio VAE (CreatorDACVAE) from: {args.audio_vae_path}")
    audio_vae = CreatorDACVAE.from_pretrained(args.audio_vae_path, strict=False)

    # Load tokenizer and text encoder
    text_encoder_kwargs = OmegaConf.to_container(config.get("text_encoder_kwargs", {}), resolve=True)
    tokenizer_path = os.path.join(args.model_name, text_encoder_kwargs.get("tokenizer_subpath", "tokenizer"))
    text_encoder_path = os.path.join(
        args.model_name, text_encoder_kwargs.get("text_encoder_subpath", "text_encoder"))

    print_info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = HFAutoTokenizer.from_pretrained(tokenizer_path)

    print_info(f"Loading text encoder from: {text_encoder_path}")
    text_encoder = WanT5EncoderModel.from_pretrained(
        text_encoder_path,
        additional_kwargs=text_encoder_kwargs,
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    # Setup schedulers
    scheduler_dict = {"Flow": FlowMatchEulerDiscreteScheduler}
    video_shift, audio_shift = resolve_scheduler_shifts(args)
    scheduler_kwargs = OmegaConf.to_container(config.get("scheduler_kwargs", {}), resolve=True)
    video_scheduler_kwargs = dict(scheduler_kwargs)
    audio_scheduler_kwargs = dict(scheduler_kwargs)
    video_scheduler_kwargs["shift"] = video_shift
    audio_scheduler_kwargs["shift"] = audio_shift
    Chosen_Scheduler = scheduler_dict[args.sampler_name]
    video_scheduler = Chosen_Scheduler(**filter_kwargs(Chosen_Scheduler, video_scheduler_kwargs))
    audio_scheduler = Chosen_Scheduler(**filter_kwargs(Chosen_Scheduler, audio_scheduler_kwargs))

    # Set models to eval
    transformer.eval()
    if args.cfg_mode == "multimodal":
        transformer = DirectionalMultimodalCFGAdapter(
            transformer,
            video_scale=args.video_bridge_guidance_scale,
            audio_scale=args.audio_bridge_guidance_scale,
            enable_a2v=not args.disable_a2v_cross_attn,
            enable_v2a=not args.disable_v2a_cross_attn,
        ).eval()
    text_encoder.eval()
    video_vae.eval()
    audio_vae.eval()

    offload_flags = resolve_cpu_offload_flags(args)

    # Keep explicitly offloaded modules on CPU until their short GPU phase.
    if not offload_flags["transformer"]:
        transformer.to(device)
    if not offload_flags["text_encoder"]:
        text_encoder.to(device)
    if not offload_flags["video_vae"]:
        video_vae.to(device)
    if not offload_flags["audio_vae"]:
        audio_vae.to(device)

    return {
        "config": config,
        "transformer": transformer,
        "video_vae": video_vae,
        "audio_vae": audio_vae,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "video_scheduler": video_scheduler,
        "audio_scheduler": audio_scheduler,
        "max_sequence_length": int(text_encoder_kwargs.get("text_length", 512)),
    }


# ============================================================================
# Joint denoising loop
# ============================================================================

@torch.no_grad()
def generate_joint_audio_video(args, models, device, weight_dtype, item):
    """Generate audio and video jointly for a single item."""
    transformer = models["transformer"]
    video_vae = models["video_vae"]
    audio_vae = models["audio_vae"]
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    video_scheduler = models["video_scheduler"]
    audio_scheduler = models["audio_scheduler"]
    max_sequence_length = models["max_sequence_length"]
    offload_flags = resolve_cpu_offload_flags(args)

    prompt = item["prompt"]
    video_prompt = item.get("video_prompt", prompt)
    audio_prompt = item.get("audio_prompt", prompt)
    negative_prompt = item.get("negative_prompt", args.negative_prompt)
    audio_negative_prompt = item.get("audio_negative_prompt", args.negative_prompt)
    duration = float(item.get("duration", args.duration))
    guidance_scale = float(item.get("guidance_scale", args.guidance_scale))
    num_inference_steps = int(item.get("num_inference_steps", args.num_inference_steps))
    seed = int(item.get("seed", args.seed))
    # ---- Step 1: Load the supplied first frame ----
    first_frame_path = item.get("first_frame_path", args.image)
    if not os.path.isfile(first_frame_path):
        raise FileNotFoundError(f"Input image not found: {first_frame_path}")
    print_info(f"Loading first frame from: {first_frame_path}")
    with Image.open(first_frame_path) as source_image:
        source_image = source_image.convert("RGB")
    video_patch_size = tuple(int(v) for v in transformer.video_patch_size)
    spatial_compression = int(getattr(video_vae.config, "spatial_compression_ratio", 16))
    height, width, target_tokens, actual_tokens = compute_dynamic_resolution(
        source_image,
        target_spatial_tokens=args.target_spatial_tokens,
        min_token_ratio=args.min_token_ratio,
        spatial_divisor_h=spatial_compression * video_patch_size[1],
        spatial_divisor_w=spatial_compression * video_patch_size[2],
    )
    print_info(
        f"Dynamic resolution: {height}x{width}, "
        f"spatial_tokens={actual_tokens}/{target_tokens}"
    )
    first_frame_image = source_image.resize((width, height), Image.Resampling.BICUBIC)

    do_classifier_free_guidance = guidance_scale > 1.0

    # Save the resized frame once in distributed inference.
    if not getattr(args, "suppress_aux_writes", False):
        first_frame_save_path = os.path.splitext(args.output)[0] + "_first_frame.png"
        first_frame_image.save(first_frame_save_path)
        print_info(f"First frame saved to: {first_frame_save_path}")

    # ---- Step 2: Encode first frame to video latent ----
    if offload_flags["video_vae"]:
        video_vae.to(device)

    first_frame_latent = encode_first_frame(first_frame_image, video_vae, device, weight_dtype)
    first_frame_latent = synchronize_tensor(first_frame_latent, args)
    latent_channels = first_frame_latent.shape[1]
    latent_height = first_frame_latent.shape[3]
    latent_width = first_frame_latent.shape[4]

    if offload_flags["video_vae"]:
        video_vae.to("cpu")
        clear_cuda_cache(device)

    # ---- Step 3: Compute latent shapes ----
    vae_temporal_ratio = int(getattr(video_vae.config, "temporal_compression_ratio", 4))
    num_frames = int(duration * args.fps)
    num_frames = int((num_frames - 1) // vae_temporal_ratio * vae_temporal_ratio) + 1
    latent_frames = (num_frames - 1) // vae_temporal_ratio + 1

    # Video sequence length
    video_patch_size = tuple(int(v) for v in transformer.video_patch_size)
    video_seq_len = (latent_frames * latent_height * latent_width) // math.prod(video_patch_size)

    # Audio latent shape (DAC VAE: latent is [D, T])
    audio_duration = num_frames / args.fps
    audio_latent_T = compute_audio_latent_length(audio_duration, audio_vae)
    audio_latent_dim = audio_vae.latent_dim
    audio_patch_size = tuple(int(v) for v in transformer.audio_patch_size)
    # Align latent_T to patch_size boundary
    if audio_latent_T % audio_patch_size[0] != 0:
        audio_latent_T = math.ceil(audio_latent_T / audio_patch_size[0]) * audio_patch_size[0]
    audio_seq_len = get_audio_num_tokens(audio_latent_T, audio_patch_size)

    print_info(f"Video: {num_frames} frames, latent [{latent_frames}, {latent_height}, {latent_width}], "
               f"seq_len={video_seq_len}")
    print_info(f"Audio: duration={audio_duration}s, latent [{audio_latent_dim}, {audio_latent_T}], "
               f"seq_len={audio_seq_len}")

    # ---- Step 4: Encode text prompts ----
    if offload_flags["text_encoder"]:
        text_encoder.to(device)

    video_prompt_embeds = encode_prompt(
        tokenizer, text_encoder, video_prompt, negative_prompt,
        guidance_scale, max_sequence_length, device, weight_dtype)

    if audio_prompt != video_prompt or audio_negative_prompt != negative_prompt:
        audio_prompt_embeds = encode_prompt(
            tokenizer, text_encoder, audio_prompt, audio_negative_prompt,
            guidance_scale, max_sequence_length, device, weight_dtype)
    else:
        audio_prompt_embeds = video_prompt_embeds

    if offload_flags["text_encoder"]:
        text_encoder.to("cpu")
        clear_cuda_cache(device)

    # ---- Step 5: Initialize noise latents ----
    generator = torch.Generator(device=device).manual_seed(seed)

    # Video noise: [1, C, latent_frames, H_lat, W_lat]
    video_latents = torch.randn(
        (1, latent_channels, latent_frames, latent_height, latent_width),
        generator=generator, device=device, dtype=weight_dtype,
    )
    # Apply first-frame conditioning
    video_latents[:, :, 0:1, :, :] = first_frame_latent.to(weight_dtype)

    # Audio noise: [1, D, T] (DAC latent format)
    audio_latents = torch.randn(
        (1, audio_latent_dim, audio_latent_T),
        generator=generator, device=device, dtype=weight_dtype,
    )
    video_latents = synchronize_tensor(video_latents, args)
    audio_latents = synchronize_tensor(audio_latents, args)

    # ---- Step 6: Setup schedulers ----
    video_shift, audio_shift = resolve_scheduler_shifts(args)
    timestep_kwargs = {}
    if getattr(args, "flow_match_mu", None) is not None:
        timestep_kwargs["mu"] = float(args.flow_match_mu)
    video_timesteps, num_inference_steps = retrieve_timesteps(
        video_scheduler, num_inference_steps, device, **timestep_kwargs)
    audio_timesteps, _ = retrieve_timesteps(
        audio_scheduler, num_inference_steps, device, **timestep_kwargs)

    timesteps = video_timesteps
    if len(audio_timesteps) != len(video_timesteps):
        raise ValueError(
            f"Video/audio timestep count mismatch: {len(video_timesteps)} vs {len(audio_timesteps)}"
        )
    if hasattr(video_scheduler, "init_noise_sigma"):
        video_latents[:, :, 1:, :, :] = video_latents[:, :, 1:, :, :] * video_scheduler.init_noise_sigma
    if hasattr(audio_scheduler, "init_noise_sigma"):
        audio_latents = audio_latents * audio_scheduler.init_noise_sigma

    video_extra_step_kwargs = prepare_extra_step_kwargs(video_scheduler)
    audio_extra_step_kwargs = prepare_extra_step_kwargs(audio_scheduler)

    # ---- Step 7: Per-timestep video token count for first frame ----
    patch_t, patch_h, patch_w = video_patch_size
    first_frame_tokens = (latent_height * latent_width) // (patch_h * patch_w)

    # ---- Step 8: Denoising loop ----
    if offload_flags["transformer"]:
        transformer.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    enable_a2v = not getattr(args, "disable_a2v_cross_attn", False)
    enable_v2a = not getattr(args, "disable_v2a_cross_attn", False)

    for step_idx, t in enumerate(tqdm(
        timesteps,
        desc="Denoising",
        unit="step",
        disable=bool(getattr(args, "disable_progress", False)),
    )):
        audio_t = audio_timesteps[step_idx]
        video_noisy_input = video_latents.clone()

        # Per-token timestep for video (first frame = 0, rest = t)
        t_scalar = t.item() if t.dim() == 0 else t
        video_timestep_tokens = torch.ones(1, video_seq_len, device=device) * t_scalar
        video_timestep_tokens[:, :first_frame_tokens] = 0

        audio_timestep = audio_t.unsqueeze(0) if audio_t.dim() == 0 else audio_t
        # video_timestep_tokens = (video_timestep_tokens / 1000) ** 0.25 * 1000
        # Build model inputs
        if do_classifier_free_guidance:
            video_x_input = [video_noisy_input[0], video_noisy_input[0]]
            video_t_input = torch.cat([video_timestep_tokens, video_timestep_tokens], dim=0)
            video_context_input = video_prompt_embeds

            audio_x_input = [audio_latents[0], audio_latents[0]]
            audio_t_input = audio_timestep.expand(2)
            audio_context_input = audio_prompt_embeds
        else:
            video_x_input = [video_noisy_input[0]]
            video_t_input = video_timestep_tokens
            video_context_input = video_prompt_embeds

            audio_x_input = [audio_latents[0]]
            audio_t_input = audio_timestep
            audio_context_input = audio_prompt_embeds

        video_inputs = {
            "x": video_x_input,
            "t": video_t_input,
            "context": video_context_input,
            "y": None,
            "seq_len": video_seq_len,
            "video_fps": float(args.fps),
        }
        audio_inputs = {
            "x": audio_x_input,
            "t": audio_t_input,
            "context": audio_context_input,
            "y": None,
            "seq_len": audio_seq_len,
        }

        with torch.autocast("cuda", dtype=weight_dtype):
            model_output = transformer(
                video=video_inputs,
                audio=audio_inputs,
                dtype=weight_dtype,
                enable_a2v=enable_a2v,
                enable_v2a=enable_v2a,
            )

        video_pred = model_output["video"]
        audio_pred = model_output["audio"]

        # Apply CFG
        if do_classifier_free_guidance:
            video_pred_uncond = video_pred[0:1]
            video_pred_text = video_pred[1:2]
            video_noise_pred = video_pred_uncond + guidance_scale * (video_pred_text - video_pred_uncond)
            audio_pred_uncond = audio_pred[0:1]
            audio_pred_text = audio_pred[1:2]
            audio_noise_pred = audio_pred_uncond + guidance_scale * (audio_pred_text - audio_pred_uncond)
        else:
            video_noise_pred = video_pred
            audio_noise_pred = audio_pred

        # Scheduler step
        video_latents_denoised = video_scheduler.step(
            video_noise_pred, t, video_latents, **video_extra_step_kwargs, return_dict=False)[0]
        video_latents_denoised[:, :, 0:1, :, :] = first_frame_latent.to(weight_dtype)
        video_latents = video_latents_denoised

        audio_latents = audio_scheduler.step(
            audio_noise_pred, audio_t, audio_latents, **audio_extra_step_kwargs, return_dict=False)[0]

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    print_info(f"Denoising completed in {elapsed:.2f}s")

    if offload_flags["transformer"]:
        transformer.to("cpu")
        clear_cuda_cache(device)

    if getattr(args, "skip_output_decode", False):
        return None, None, num_frames

    # ---- Step 9: Decode video ----
    if offload_flags["video_vae"]:
        video_vae.to(device)

    with torch.no_grad():
        video_decoded = video_vae.decode(video_latents.to(video_vae.dtype))[0]
    video_decoded = (video_decoded / 2.0 + 0.5).clamp(0, 1).float().cpu()

    if offload_flags["video_vae"]:
        video_vae.to("cpu")
        clear_cuda_cache(device)

    # ---- Step 10: Decode audio (CreatorDACVAE) ----
    if offload_flags["audio_vae"]:
        audio_vae.to(device)

    with torch.no_grad():
        # CreatorDACVAE.decode expects [B, D, T'] and returns [B, 1, T]
        audio_decoded = audio_vae.decode(audio_latents.float().to(audio_vae.dac.device if hasattr(audio_vae, 'dac') else device))

    if offload_flags["audio_vae"]:
        audio_vae.to("cpu")
        clear_cuda_cache(device)

    return video_decoded, audio_decoded, num_frames


def main():
    args = parse_args()
    device = init_device()
    weight_dtype = resolve_weight_dtype(args.weight_dtype, device)

    output_path = os.path.abspath(args.output)
    if not os.path.splitext(output_path)[1]:
        output_path += ".mp4"
    args.output = output_path
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    print_info("=" * 80)
    print_info("Creator single-image audio-video inference")
    print_info("=" * 80)
    print_info(f"Device: {device} | dtype: {weight_dtype}")
    offload_flags = resolve_cpu_offload_flags(args)
    print_info(
        "CPU offload: "
        f"text_encoder={'on' if offload_flags['text_encoder'] else 'off'}, "
        f"video_vae={'on' if offload_flags['video_vae'] else 'off'}, "
        f"audio_vae={'on' if offload_flags['audio_vae'] else 'off'}"
    )
    print_info(
        "Cross attention: "
        f"A2V={'disabled' if args.disable_a2v_cross_attn else 'enabled'}, "
        f"V2A={'disabled' if args.disable_v2a_cross_attn else 'enabled'}"
    )

    # Load models
    models = setup_models(args, device, weight_dtype)

    item = {
        "prompt": args.prompt,
        "video_prompt": args.prompt,
        "audio_prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "audio_negative_prompt": args.negative_prompt,
        "duration": args.duration,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "name": "sample",
    }
    video_decoded, audio_decoded, _ = generate_joint_audio_video(
        args, models, device, weight_dtype, item)
    video_path = os.path.splitext(output_path)[0] + ".video.mp4"
    audio_path = os.path.splitext(output_path)[0] + ".wav"
    frames = (video_decoded[0].permute(1, 2, 3, 0).clamp(0, 1).numpy() * 255).astype("uint8")
    write_video(video_path, torch.from_numpy(frames), fps=args.fps, video_codec="h264")
    save_audio_wav(audio_decoded, int(models["audio_vae"].sample_rate), audio_path)
    mux_result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
         "-c:v", "copy", "-c:a", "aac", "-shortest", output_path],
        check=False,
        capture_output=True,
        text=True,
    )
    if mux_result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed while muxing {output_path}:\n{mux_result.stderr.strip()}"
        )
    print_info(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
