"""SR-DiT causal few-step video super-resolution inference.

Usage:
    python inference_sr.py \
        --config_path configs/sr_dit_5b.yaml \
        --checkpoint_path /path/to/model.pt \
        --input_path /path/to/lr_videos_or_folder_or_eval.json \
        --output_folder outputs/sr_results \
        --causal

The input may be a single .mp4 file, a directory of .mp4 files, or a JSON list
of {"video_path": ..., "prompt": ...} entries. Clips whose output file already
exists under --output_folder are skipped, so every knob that changes pixels
must also change the output path (see run_inference.sh).
"""
import argparse
import glob
import json
import logging
import os
import subprocess
import time

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--use_flex_attention", action="store_true", default=False)
_pre_args, _ = _pre_parser.parse_known_args()
if not _pre_args.use_flex_attention:
    os.environ["DISABLE_FLEX_COMPILE"] = "1"

import peft
import torch

# cudnn plan-selection switches (both default OFF, preserving original
# behaviour). Some cudnn builds pick a slow plan for the VAE decoder's cached
# bf16 Conv3d shapes; the two environment variables bypass that:
#   CUDNN_BENCHMARK=1  -> torch.backends.cudnn.benchmark=True; autotunes every
#                         new shape (per-clip autotune cost, faster steady state)
#   CUDNN_V8_OFF=1     -> TORCH_CUDNN_V8_API_DISABLED=1 (must be exported
#                         before torch is imported); falls back to the old API
if os.environ.get("CUDNN_BENCHMARK", "0") == "1":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import read_video, write_video
from tqdm import tqdm

from utils.lora_utils import configure_lora_for_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def _prompt_slug(text, maxlen=120):
    """Filesystem-safe, truncated slug of the prompt for embedding in filenames."""
    import re
    s = re.sub(r"\s+", "_", str(text).strip())
    s = re.sub(r"[^0-9A-Za-z_\-]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen] or "noprompt"

def _decode_timed(pipeline, latent):
    """VAE-decode, billed to the pipeline's `decode` stage.

    The few-step branches decode outside the pipeline (they pass
    ``return_video=False`` so the latent can be saved and truncated first), so
    without this the df/sf paths would report ``decode=0.0s``.
    """
    with pipeline.stage_timer.stage("decode"):
        # offload_output: decoded video accumulates on CPU per temporal chunk.
        # At long-clip / 2K+ resolutions the full fp32 video on GPU OOMs the
        # normalize line right after this call.
        return pipeline.vae.decode_to_pixel(latent, offload_output=True)


def _normalize_decode(video_out, chunk=32):
    """[-1,1] -> [0,1] in-place, chunked over time.

    decode_to_pixel(offload_output=True) hands back the full clip on CPU; the
    one-liner `(v*0.5+0.5).clamp(0,1)` would materialize a second full-length
    fp32 copy next to the first.
    """
    with torch.no_grad():
        for i in range(0, video_out.shape[1], chunk):
            v = video_out[:, i:i + chunk]
            v.mul_(0.5).add_(0.5).clamp_(0, 1)
    return video_out


def _maybe_save_latent(latent, basename):
    if not args.save_latent_dir or not isinstance(latent, torch.Tensor):
        return
    os.makedirs(args.save_latent_dir, exist_ok=True)
    path = os.path.join(args.save_latent_dir, f"{basename}.pt")
    torch.save(latent.detach().float().cpu(), path)
    logging.info("Saved latent: %s %s", path, list(latent.shape))


def _has_audio_stream(path):
    """True when the container has an audio stream (ffmpeg-probed)."""
    r = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", path, "-map", "0:a", "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0


def _mux_audio(refined_path, source_path, basename):
    """Copy the source video's audio track into the refined output in place.

    The refiner only synthesizes video, so the audio of the input clip is
    preserved by stream-copying it back after the video is written.
    """
    try:
        if not _has_audio_stream(source_path):
            logging.info("[%s] source has no audio stream; video-only output", basename)
            return
        tmp_path = refined_path + ".mux.mp4"
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", refined_path, "-i", source_path,
             "-map", "0:v:0", "-map", "1:a:0",
             "-c", "copy", "-shortest", tmp_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            logging.warning("[%s] audio mux failed, keeping video-only output: %s",
                            basename, r.stderr.strip()[-300:])
            _silent_remove(tmp_path)
            return
        os.replace(tmp_path, refined_path)
        logging.info("[%s] Audio track copied from source", basename)
    except FileNotFoundError:
        logging.warning("[%s] ffmpeg not found; video-only output", basename)


def _silent_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


parser = argparse.ArgumentParser(description="SR-DiT inference")
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--input_path", type=str, required=True,
                    help="LR video file (.mp4), folder of .mp4 files, or JSON file")
parser.add_argument("--output_folder", type=str, default="outputs/sr_results")
parser.add_argument("--save_latent_dir", type=str, default=None,
                    help="Save the final SR latent ([1,T,C,H,W], normalized, fp32, cpu) "
                         "to <dir>/<basename>.pt before the VAE decode. For offline "
                         "decoder benchmarks; off unless set.")
parser.add_argument("--prompt", type=str,
                    default="Cinematic, High Contrast, highly detailed, taken using a Canon EOS R "
                            "camera, hyper detailed photo-realistic maximum detail, Color Grading, "
                            "ultra HD, extreme meticulous detailing, skin pore detailing, hyper "
                            "sharpness, perfect without deformations")
parser.add_argument("--keep_audio", action=argparse.BooleanOptionalAction, default=True,
                    help="Copy the source video's audio track into the refined output "
                         "(stream copy, no re-encode; off when the source has no audio).")
parser.add_argument("--guidance_scale", type=float, default=7.5)
parser.add_argument("--sigma_start", type=float, default=None,
                    help="Override refiner_sigma_start from config")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_frames", type=int, default=81,
                    help="Frames to process; -1 = use all frames in each video "
                         "(snapped down to the nearest 4n+1 for VAE temporal alignment)")
parser.add_argument("--target_size", type=int, nargs=2, default=[960, 1664])
parser.add_argument("--auto_target_size", action="store_true",
                    help="Derive target_size per clip from its native resolution * --sr_scale "
                         "(snapped to /32), preserving each clip's aspect ratio. Use for datasets "
                         "with mixed resolutions/aspects. Overrides --target_size.")
parser.add_argument("--sr_scale", type=float, default=2.0,
                    help="SR upscale factor used by --auto_target_size (native size * sr_scale).")
parser.add_argument("--causal", action="store_true", default=None,
                    help="Force causal pipeline (auto-detected from config if not set)")
parser.add_argument("--kv_len", type=int, default=None,
                    help="Max KV cache chunks to keep (None=unlimited, 6 recommended for 960x1664)")
parser.add_argument("--cond_noise_sigma", type=float, default=0.0,
                    help="Noise sigma for concat cond_y (0=clean, training used U[0.3,0.5])")
parser.add_argument("--lq_cfg", action="store_true", default=False,
                    help="Use LQ CFG (guide on cond_y presence) instead of the default text CFG "
                         "(guide on positive vs negative prompt). Only affects concat mode.")
parser.add_argument("--use_flex_attention", action="store_true", default=False,
                    help="Use FlexAttention full-sequence inference instead of streaming KV cache "
                         "(denoising-forcing only). Eliminates KV cache, fewer forward passes per step.")
parser.add_argument("--use_window_attn", action="store_true", default=False,
                    help="Swap the generator self-attention to mask-free causal block-grid window "
                         "attention (kv_len still applies). Block params come from the model config "
                         "unless explicitly overridden below.")
parser.add_argument("--window_chunk", type=int, default=None,
                    help="Process N windows per batch to bound peak memory (None = one-shot).")
parser.add_argument("--window_block_hw", type=int, nargs=2, default=None,
                    help="Override the spatial query-block size (default: config value).")
parser.add_argument("--window_block_radius_hw", type=int, nargs=2, default=None,
                    help="Override spatial neighborhood radius (per axis).")
parser.add_argument("--window_block_t", type=int, default=None,
                    help="Override the temporal block size (default: config value, else "
                         "stream_chunk_size).")
parser.add_argument("--window_radius_t", type=int, default=None,
                    help="Explicit temporal radius (in block_t units). Activates the "
                         "chunk-anchored rule: own chunk + (2r+1)*block_t history frames. "
                         "Required for --window_attn_impl magi.")
parser.add_argument("--window_attn_impl", type=str, default=None,
                    choices=["auto", "flash", "sdpa", "flex", "magi", "triton"],
                    help="Override the window execution backend (default: config value). "
                         "'triton' is inference-only (forward-only kernels).")
# ── fp8 DiT GEMMs ──
# OFF by default because it is a quality trade, not a free win: per-GEMM
# relative error is ~22x bf16's, compounding over 30 blocks.
parser.add_argument("--fp8_linear", action="store_true", default=False,
                    help="Compute the DiT block GEMMs in fp8 e4m3 with row-wise dynamic "
                         "scaling (needs sm89+, and --merge_lora if a LoRA is loaded). "
                         "~1.23x per block; check PSNR before trusting it.")
parser.add_argument("--fp8_targets", type=str, default="all", choices=["all", "ffn"],
                    help="Which GEMMs go to fp8: 'all' (10 per block) or 'ffn' (the 2 FFN "
                         "Linears = 60%% of the block's GEMM FLOPs, attention stays bf16).")
parser.add_argument("--fp8_skip_first", type=int, default=0,
                    help="Leave the first N blocks in bf16 (they set the feature scale).")
parser.add_argument("--fp8_skip_last", type=int, default=0,
                    help="Leave the last N blocks in bf16 (they feed the output head).")
parser.add_argument("--pixel_upsample", action="store_true", default=False,
                    help="Upsample in pixel space before VAE encoding instead of upsampling in latent space")
parser.add_argument("--pixel_upsample_mode", type=str, default="bicubic",
                    choices=["bilinear", "bicubic", "nearest"],
                    help="Interpolation mode for pixel-space upsampling (default: bicubic)")
parser.add_argument("--latent_upsampler_config", type=str, default=None,
                    help="Path to a YAML holding latent_upsample_mode / latent_upsampler_arch_config "
                         "/ latent_upsampler_precision (e.g. the upsampler training config). Enables "
                         "the trained latent upsampler instead of latent-bilinear/pixel upsampling.")
parser.add_argument("--latent_upsampler_ckpt", type=str, default=None,
                    help="Checkpoint (.pt) for the latent upsampler; overrides the ckpt in config.")
parser.add_argument("--enable_nu_lightvae", action="store_true",
                    help="Decode with a LightVAE-NU student (non-uniform channel widths, "
                         "lightvae_nu/) instead of the full Wan2.2 VAE. Decoder-only, same "
                         "latent space. OFF by default; enabling trades some fidelity for "
                         "decode speed.")
parser.add_argument("--nu_lightvae_type", type=str, default="scheme3",
                    choices=["scheme2", "scheme2b", "scheme3", "scheme2_f33"],
                    help="Distilled width. 'scheme3' is the released default; the other "
                         "schemes require an explicit --nu_lightvae_ckpt.")
parser.add_argument("--nu_lightvae_ckpt", type=str, default=None,
                    help="Path to the LightVAE-NU checkpoint (default: auto from "
                         "--nu_lightvae_type). The 'ema' key is loaded, not 'student'.")
parser.add_argument("--nu_lightvae_module_path", type=str, default=None,
                    help="Tree providing the lightvae_nu package (default: this repo). The "
                         "package binds its Wan VAE blocks relative to its own location, so this "
                         "also picks which tree's wan/modules/vae2_2.py builds the decoder -- "
                         "keep it on the repo the rest of the pipeline runs from.")
parser.add_argument("--lora_checkpoint_path", type=str, default=None,
                    help="Path to a LoRA adapter checkpoint (containing 'generator_lora'). "
                         "Loaded on top of the base --checkpoint_path. When set, an adapter "
                         "config is built automatically (rank inferred from the checkpoint, "
                         "alpha=rank). Optional: the released checkpoints are pre-merged.")
parser.add_argument("--lora_rank", type=int, default=0,
                    help="LoRA rank for the auto-built adapter (0 = infer from checkpoint).")
parser.add_argument("--lora_alpha", type=int, default=0,
                    help="LoRA alpha for the auto-built adapter (0 = equal to rank).")
parser.add_argument("--merge_lora", action="store_true",
                    help="Merge LoRA weights into base model after loading (eliminates "
                         "adapter overhead during inference).")
# ── LQ-anchor: inject the LQ (upsampled-LR) chunk's K/V as extra attention context ──
# during each chunk's denoising. Only supported for the causal few-step sf path
# (teacher_forcing: true). Default OFF -> behaviour unchanged.
parser.add_argument("--use_lq_anchor", action="store_true",
                    help="Enable per-chunk LQ-anchor KV injection (sf causal few-step only).")
parser.add_argument("--lq_guidance_scale", type=float, nargs="+", default=[1.0],
                    help="Per-denoise-step scale applied to the anchor K or V "
                         "(e.g. '1.0 0.5'; last value repeats). Default 1.0.")
parser.add_argument("--lq_guidance_mode", type=str, default="v", choices=["k", "v"],
                    help="Scale the anchor's K (attention strength) or V (content). Default v.")
parser.add_argument("--lq_anchor_align", type=str, default="frame",
                    choices=["frame", "chunk"],
                    help="LQ-anchor harvest/injection alignment: 'frame' (default) "
                         "harvests each LR frame independently (cf=1) and lets HR "
                         "frame i attend ONLY its own anchor frame i; 'chunk' "
                         "harvests the whole chunk in one forward and lets every HR "
                         "query attend the ENTIRE anchor chunk.")
parser.add_argument("--lq_anchor_window_scope", type=str, default="window",
                    choices=["window", "global"],
                    help="Spatial scope of the LQ anchor UNDER WINDOW ATTENTION "
                         "(no effect otherwise): 'window' (default) lets a window "
                         "block attend only the anchor tokens inside its own "
                         "spatial extent; 'global' restores the "
                         "old whole-frame / whole-chunk anchor. Must match the "
                         "trainer's anchor_window_scope.")
parser.add_argument("--lq_anchor_native", action="store_true",
                    help="Harvest the LQ-anchor from the NON-upsampled LR latent "
                    "(scheme B): the anchor keeps the LQ grid but is RoPE'd at "
                    "HR positions with a coarser stride (same h/w range, step "
                    "s), then nearest-neighbour repeated back to HR token "
                    "counts so the consumers are unchanged. INFERENCE ONLY -- "
                    "the deployed LoRA was distilled on an upsampled-LQ anchor, "
                    "so this is a deliberate train/test mismatch. Default OFF.")
parser.add_argument("--lq_anchor_native_scheme", type=str, default="repeat",
                    choices=["repeat", "true"],
                    help="How the native anchor reaches attention (needs "
                         "--lq_anchor_native): 'repeat' (default, scheme B) "
                         "nearest-neighbour repeats the LQ-grid K/V back to HR "
                         "token counts, so every consumer sees HR shapes and the "
                         "anchor's softmax mass is the mass the LoRA was distilled "
                         "with; 'true' (scheme A) keeps the anchor on its own "
                         "coarser grid all the way into the attention op "
                         "(anchor_hw), which is s^2 fewer anchor tokens and "
                         "therefore a weaker anchor.")
# ── Scheme-B anchor knobs ──
# Override the config's anchor_scale / anchor_context_noise at inference. These MUST
# match the semantics used in training: anchor_context_noise = number of flow-matching
# steps of noise added to the LR anchor before its K/V is extracted (0 = clean anchor,
# larger = noisier).
parser.add_argument("--anchor_context_noise", type=int, default=None,
                    help="Override anchor_context_noise (0=clean LR anchor, larger=noisier). "
                         "Matches training's actual-noise semantics.")
parser.add_argument("--anchor_scale", type=float, default=None,
                    help="Override anchor injection scale (config default 1.0).")
# ── Anchor layer selection (which blocks receive the LR anchor K/V) ──
# Restrict the LR-anchor injection to a subset of DiT blocks. This MUST match how
# the LoRA was trained (prefix_keep drop) — layers the model never learned to run
# without an anchor will otherwise be off-distribution. None => all layers.
parser.add_argument("--anchor_layers", type=int, nargs="+", default=None,
                    help="Explicit list of block ids that receive the LR anchor K/V "
                         "(e.g. '0 1 2 3 4 5'); all other blocks get NO LR-KV. "
                         "Overrides --anchor_keep_prefix. Default: all layers.")
parser.add_argument("--anchor_keep_prefix", type=int, default=None,
                    help="Shortcut for a prefix: keep the LR anchor only in blocks "
                         "[0, K), matching training's prefix_keep. Ignored when "
                         "--anchor_layers is given.")
parser.add_argument("--anchor_influence_probe", type=str, default=None,
                    help="Diagnostic: dump per-block LQ-anchor influence to this JSON "
                         "path. Measures, for each block that receives an anchor, the "
                         "relative L2 change of its self-attention output when that "
                         "block alone loses the anchor (input held fixed). Bit-exact "
                         "for the run itself, so it does not change the videos; point "
                         "--output_folder at a FRESH dir or every clip is skipped and "
                         "nothing is recorded. Default OFF.")
args = parser.parse_args()

torch.set_grad_enabled(False)
torch.manual_seed(args.seed)
device = torch.device("cuda")

# ── Load config ──
config = OmegaConf.load(args.config_path)
config.guidance_scale = args.guidance_scale
config.num_train_timestep = getattr(config, "num_train_timestep", 1000)
# Anchor knob overrides (Scheme-B): take precedence over the config values so the
# same base config can be reused across curriculum stages without editing the yaml.
if args.anchor_context_noise is not None:
    config.anchor_context_noise = args.anchor_context_noise
if args.anchor_scale is not None:
    config.anchor_scale = args.anchor_scale
# Anchor layer selection: explicit list wins over the prefix shortcut.
if args.anchor_layers is not None:
    config.anchor_layers = list(args.anchor_layers)
elif args.anchor_keep_prefix is not None:
    config.anchor_layers = list(range(int(args.anchor_keep_prefix)))
sigma_start = args.sigma_start if args.sigma_start is not None else getattr(config, "refiner_sigma_start", 0.9)

# Optional student decoder (read by the causal pipeline when building the VAE).
# It replaces pipeline.vae outright.
config.enable_nu_lightvae = args.enable_nu_lightvae
config.nu_lightvae_type = args.nu_lightvae_type
config.nu_lightvae_ckpt = args.nu_lightvae_ckpt
config.nu_lightvae_module_path = args.nu_lightvae_module_path

# ── Optional: enable a trained latent upsampler ──
# Loaded by the pipeline's _load_latent_upsampler() from these config fields, so we
# inject them before the pipeline is constructed. Mutually exclusive with pixel upsampling.
if args.latent_upsampler_config is not None:
    if args.pixel_upsample:
        raise ValueError("--latent_upsampler_config and --pixel_upsample are mutually exclusive")
    lu_cfg = OmegaConf.load(args.latent_upsampler_config)
    config.latent_upsample_mode = lu_cfg.get("latent_upsample_mode", "bilinear")
    config.latent_upsampler_arch_config = lu_cfg.get("latent_upsampler_arch_config", None)
    config.latent_upsampler_precision = lu_cfg.get("latent_upsampler_precision", "bf16")
    if config.latent_upsample_mode == "bilinear" or config.latent_upsampler_arch_config is None:
        raise ValueError(
            f"{args.latent_upsampler_config} has no usable latent_upsampler_arch_config "
            f"(latent_upsample_mode={config.get('latent_upsample_mode')})")
if args.latent_upsampler_ckpt is not None:
    config.latent_upsampler_ckpt = args.latent_upsampler_ckpt

# The trained latent upsampler has a FIXED spatial scale (baked into the model):
# its output size = scale x LR-latent size, and it ignores target_size. So the LR
# must be encoded at exactly target_size / scale, otherwise the upsampler output
# won't match the HR noise tensor. Resolve the scale here so we can pre-resize LR
# pixels to the expected LR resolution before VAE encoding (matches training, where
# LR is a bilinear downsample of HR by `scale`).
using_latent_upsampler = getattr(config, "latent_upsample_mode", "bilinear") != "bilinear"
latent_upsample_scale = 1
if using_latent_upsampler:
    lu_arch = getattr(config, "latent_upsampler_arch_config", None)
    lu_params = lu_arch.get("params", {}) if lu_arch is not None else {}
    latent_upsample_scale = int(
        lu_params.get("upsample_scale", config.get("latent_upsample_scale", 2)))
    logging.info("Latent upsampler: mode=%s, scale=%dx, ckpt=%s",
                 config.latent_upsample_mode, latent_upsample_scale,
                 getattr(config, "latent_upsampler_ckpt", None))

# ── Initialize pipeline ──
is_causal = args.causal
if is_causal is None:
    dit_cfg = config.get("dit_arch_config", {})
    is_causal = dit_cfg.get("sparse_causal", False)
few_step = hasattr(config, 'denoising_step_list')
if not (is_causal and few_step):
    raise NotImplementedError(
        "This release supports the causal few-step pipeline only: the config needs "
        "denoising_step_list and a causal architecture (pass --causal or set "
        "dit_arch_config.sparse_causal).")

# Ensure config fields required by the causal pipeline
use_teacher_forcing = config.get("teacher_forcing", True)
model_kwargs = config.get("model_kwargs", {})
if not hasattr(config, "timestep_shift"):
    config.timestep_shift = model_kwargs.get("timestep_shift", 5.0)
if not hasattr(config, "independent_first_frame"):
    config.independent_first_frame = False
if not hasattr(config, "num_frame_per_block"):
    dit_cfg = config.get("dit_arch_config", {})
    config.num_frame_per_block = dit_cfg.get("stream_chunk_size", 2)
if args.kv_len is not None:
    config.stream_kv_len = args.kv_len
elif not hasattr(config, "stream_kv_len"):
    dit_cfg = config.get("dit_arch_config", {})
    config.stream_kv_len = dit_cfg.get("stream_kv_len", None)

from pipeline_sr.causal_inference import CausalInferencePipeline
pipeline = CausalInferencePipeline(config, device=device)

if args.use_lq_anchor and not use_teacher_forcing:
    logging.warning("--use_lq_anchor is only supported for the causal few-step sf path "
                    "(few_step + teacher_forcing=true); ignoring it for this configuration.")
    args.use_lq_anchor = False
if args.lq_anchor_native and not args.use_lq_anchor:
    # Silently inert would be the worst outcome here: the run looks like the
    # native A/B arm but is the plain baseline.
    logging.warning("--lq_anchor_native has no effect without --use_lq_anchor; "
                    "this run harvests no anchor at all.")

logging.info("Pipeline: causal few-step (%s), steps=%s, teacher_forcing=%s",
             "sf" if use_teacher_forcing else "df",
             list(config.denoising_step_list), use_teacher_forcing)

# ── Load checkpoint ──
if args.checkpoint_path:
    logging.info("Loading checkpoint from %s", args.checkpoint_path)
    state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "generator" in state_dict:
        gen_sd = state_dict["generator"]
    elif isinstance(state_dict, dict) and "generator_ema" in state_dict:
        gen_sd = state_dict["generator_ema"]
    elif isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
        gen_sd = state_dict["model"]
    else:
        gen_sd = state_dict

    fixed = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in gen_sd.items()}
    # If keys lack "model." prefix, add it (checkpoint from different framework)
    sample_key = next(iter(fixed), "")
    if not sample_key.startswith("model.") and hasattr(pipeline.generator, "model"):
        fixed = {"model." + k: v for k, v in fixed.items()}
    missing, unexpected = pipeline.generator.load_state_dict(fixed, strict=True)
    logging.info("Loaded: %d missing, %d unexpected keys", len(missing), len(unexpected))
    if missing:
        logging.warning("Missing (first 5): %s", missing[:5])
    if unexpected:
        logging.warning("Unexpected (first 5): %s", unexpected[:5])
    del state_dict, gen_sd, fixed

# ── Load LoRA adapter (optional; the released checkpoints are pre-merged) ──
if args.lora_checkpoint_path:
    logging.info("Loading LoRA checkpoint from %s", args.lora_checkpoint_path)

    # Infer rank from checkpoint if not specified
    inferred_rank = args.lora_rank
    if inferred_rank <= 0:
        probe = torch.load(args.lora_checkpoint_path, map_location="cpu",
                           weights_only=True, mmap=True)
        probe_sd = probe.get("generator_lora", probe) if isinstance(probe, dict) else probe
        for k, v in probe_sd.items():
            if "lora_A" in k and hasattr(v, "shape") and v.ndim == 2:
                inferred_rank = int(v.shape[0])
                break
        del probe
        if inferred_rank <= 0:
            raise ValueError(
                f"Could not infer LoRA rank from {args.lora_checkpoint_path}; "
                f"pass --lora_rank explicitly.")

    inferred_alpha = args.lora_alpha if args.lora_alpha > 0 else inferred_rank
    adapter_config = OmegaConf.create({
        "type": "lora", "rank": inferred_rank, "alpha": inferred_alpha,
        "dropout": 0.0,
    })
    logging.info("LoRA adapter: rank=%d, alpha=%d", inferred_rank, inferred_alpha)

    # Apply LoRA to the transformer (pipeline.generator.model)
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model, adapter_config, is_main_process=True)

    # Load LoRA weights
    lora_checkpoint = torch.load(args.lora_checkpoint_path, map_location="cpu",
                                 weights_only=False)
    if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
        intended_sd = lora_checkpoint["generator_lora"]
    else:
        intended_sd = lora_checkpoint

    peft.set_peft_model_state_dict(pipeline.generator.model, intended_sd)

    # Verify LoRA weights actually loaded (peft silently no-ops on mismatched keys)
    current = peft.get_peft_model_state_dict(pipeline.generator.model)
    matched = sum(1 for k, v in intended_sd.items()
                  if k in current and current[k].shape == v.shape
                  and torch.allclose(current[k].float().cpu(),
                                     v.float().cpu(), atol=1e-5))
    logging.info("LoRA loaded: %d/%d tensors matched", matched, len(intended_sd))
    if matched != len(intended_sd):
        mismatched = [k for k in intended_sd if k not in current
                      or current[k].shape != intended_sd[k].shape]
        raise RuntimeError(
            f"LoRA load verification FAILED: only {matched}/{len(intended_sd)} "
            f"tensors matched. Mismatched[:3]={mismatched[:3]}. "
            f"Check that the adapter config (rank/target modules) matches the checkpoint.")
    lora_b_sum = sum(current[k].abs().sum().item()
                     for k in current if "lora_B" in k)
    if lora_b_sum == 0.0:
        raise RuntimeError(
            "LoRA load verification FAILED: all lora_B weights are zero "
            "(adapter is a no-op). The trained weights did not take effect.")
    logging.info("LoRA weights verified OK (sum|lora_B|=%.4f)", lora_b_sum)

    # Load extra trainable (non-adapter) modules saved under "generator_extra",
    # e.g. concat_lora's expanded in_dim=96 patch_embedding. peft's state_dict
    # loader restores adapters only, so these must be applied separately.
    extra_sd = lora_checkpoint.get("generator_extra") if isinstance(lora_checkpoint, dict) else None
    if extra_sd:
        model_keys = set(pipeline.generator.model.state_dict().keys())
        present = [k for k in extra_sd if k in model_keys]
        if len(present) != len(extra_sd):
            missing_extra = [k for k in extra_sd if k not in model_keys]
            raise RuntimeError(
                f"generator_extra keys not found in model: {missing_extra[:5]} "
                f"({len(present)}/{len(extra_sd)} matched)")
        pipeline.generator.model.load_state_dict(extra_sd, strict=False)
        logging.info("Loaded %d generator_extra tensor(s): %s",
                     len(extra_sd), list(extra_sd.keys()))

    del lora_checkpoint, intended_sd, current

    # Optionally merge LoRA into base weights
    if args.merge_lora:
        logging.info("Merging LoRA weights into base model...")
        pipeline.generator.model = pipeline.generator.model.merge_and_unload(safe_merge=True)
        logging.info("LoRA merged into base model")

pipeline = pipeline.to(dtype=torch.bfloat16)
pipeline.text_encoder.to(device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)
if hasattr(pipeline, 'latent_upsampler') and pipeline.latent_upsampler is not None:
    pipeline.latent_upsampler.to(device=device)

# The student decoder (LightVAE-NU) is decode-only, but SR inference must
# first encode the LR video to latent. Build a full Wan VAE for the encode step;
# pipeline.vae still handles the fast final decode. With the full Wan decoder,
# encode & decode share one VAE.
if getattr(pipeline.vae, "decode_only", False):
    from utils.sr_dit_wrapper import WanVAEWrapper22
    from utils.wan_wrapper import WanVAEWrapper
    wan_model_dir = getattr(config, "wan_model_dir", "wan_models/Wan2.1-T2V-1.3B")
    vae_version = getattr(config, "vae_version", "2.2")
    vae_pth = getattr(config, "vae_pth", None)
    if vae_version == "2.2":
        encode_vae = WanVAEWrapper22(vae_pth=vae_pth or os.path.join(wan_model_dir, "Wan2.2_VAE.pth"))
    else:
        encode_vae = WanVAEWrapper(vae_pth=vae_pth or os.path.join(wan_model_dir, "Wan2.1_VAE.pth"))
    encode_vae = encode_vae.to(device=device, dtype=torch.bfloat16)
    logging.info("%s decode enabled; full Wan%s VAE loaded for LR encoding",
                 type(pipeline.vae).__name__, vae_version)
else:
    encode_vae = pipeline.vae

# ── Optional: mask-free causal block-grid window self-attention ──
# Route through WanModel.enable_window_attention() so the MODEL-level flags stay in
# sync with the per-block ones and so `window_causal_kv_len` is never left unset —
# a parallel forward with it unset is silently NON-causal.
# Robust to LoRA/PeftModel wrapping: find the WanModel via modules(); fall back to
# a per-block walk if the generator does not expose one.
if args.use_window_attn:
    from wan.modules.sr_dit.models import WanModel, WanSelfAttention
    _root = next((_m for _m in pipeline.generator.model.modules()
                  if isinstance(_m, WanModel)), None)
    # Anything the CLI does not override keeps the value baked into the model config.
    _dit_cfg = config.get("dit_arch_config", {}) if hasattr(config, "get") else {}
    _blk_hw = tuple(args.window_block_hw or getattr(_root, "window_block_hw", (4, 4)))
    _blk_r = tuple(args.window_block_radius_hw
                   or getattr(_root, "window_block_radius_hw", (2, 2)))
    _blk_t = (args.window_block_t if args.window_block_t is not None
              else getattr(_root, "window_block_t", None))
    _rad_t = (args.window_radius_t if args.window_radius_t is not None
              else getattr(_root, "window_radius_t", None))
    _impl = (args.window_attn_impl or getattr(_root, "window_attn_impl", "auto"))
    # Causal span for the PARALLEL forward: config value, else align with the
    # streaming cache length (see enable_window_attention docstring).
    _causal_kv = (getattr(_root, "window_causal_kv_len", None)
                  or _dit_cfg.get("window_causal_kv_len", None)
                  or getattr(config, "stream_kv_len", None))
    if _root is not None:
        _n = _root.enable_window_attention(
            win_chunk=args.window_chunk,
            causal_kv_len=_causal_kv,
            window_block_hw=_blk_hw,
            window_block_radius_hw=_blk_r,
            window_block_t=_blk_t,
            window_radius_t=_rad_t,
            window_attn_impl=_impl)
    else:
        _n = 0
        for _m in pipeline.generator.model.modules():
            # type(...) is WanSelfAttention EXCLUDES WanT2VCrossAttention (subclass),
            # so cross-attn stays untouched.
            if type(_m) is WanSelfAttention:
                _m.use_window_attn = True
                _m.window_chunk = args.window_chunk
                _m.window_causal_kv_len = _causal_kv
                _m.window_block_hw = _blk_hw
                _m.window_block_radius_hw = _blk_r
                _m.window_block_t = _blk_t
                _m.window_radius_t = _rad_t
                _m.window_attn_impl = _impl
                _n += 1
    logging.info("Window attention ENABLED on %d self-attn blocks "
                 "(block_hw=%s, radius=%s, block_t=%s, radius_t=%s, impl=%s, "
                 "causal_kv_len=%s, win_chunk=%s)",
                 _n, _blk_hw, _blk_r, _blk_t, _rad_t, _impl, _causal_kv,
                 args.window_chunk)

# ── Optional: fp8 e4m3 row-wise GEMMs in the DiT blocks ──
# Must run AFTER the LoRA merge and after the model is on-device in its final
# dtype: the weight is quantised once here, so an adapter merged afterwards would
# be applied to a weight this module no longer owns, and a later `.to(dtype)`
# would try to reinterpret the e4m3 code words (Fp8Linear._apply guards that).
if args.fp8_linear:
    from wan.modules.sr_dit import fp8_linear as _f8
    from wan.modules.sr_dit.models import WanModel
    if args.lora_checkpoint_path and not args.merge_lora:
        raise RuntimeError(
            "--fp8_linear needs --merge_lora when a LoRA is loaded: the base weight "
            "is quantised once at setup, so an unmerged adapter would either be "
            "silently skipped (peft wraps the Linear, so the swap finds nothing) or "
            "applied to a weight Fp8Linear no longer owns.")
    if not _f8.fp8_supported(device):
        raise RuntimeError(
            f"--fp8_linear needs fp8 tensor cores (sm89+); {device} reports "
            f"capability {torch.cuda.get_device_capability(device)}.")
    _fp8_root = next((_m for _m in pipeline.generator.model.modules()
                      if isinstance(_m, WanModel)), None)
    if _fp8_root is None:
        raise RuntimeError("--fp8_linear: no WanModel under pipeline.generator.model")
    _fp8_names = _f8.convert_linears_to_fp8(
        _fp8_root,
        targets=_f8.TARGETS_FFN if args.fp8_targets == "ffn" else _f8.TARGETS_ALL,
        skip_first=args.fp8_skip_first, skip_last=args.fp8_skip_last)
    if not _fp8_names:
        raise RuntimeError(
            "--fp8_linear converted 0 Linears -- the target names do not match this "
            "model (expected WanAttentionBlock's self_attn/cross_attn/ffn layout).")
    logging.info("fp8 e4m3 rowwise ENABLED on %d Linears (%d blocks x %s, "
                 "skip_first=%d skip_last=%d). Per-GEMM error is ~22x bf16's; "
                 "verify PSNR before trusting the output.",
                 len(_fp8_names), len(_fp8_root.blocks) - args.fp8_skip_first
                 - args.fp8_skip_last, args.fp8_targets,
                 args.fp8_skip_first, args.fp8_skip_last)

logging.info("Pipeline ready on %s", device)

# ── Optional DiT-only profiling (PROFILE_DIT=1, =2 also logs per-call shapes) ──
if os.environ.get("PROFILE_DIT", "").lower() in ("1", "2", "true", "yes"):
    import atexit
    import time as _time

    _gen = pipeline.generator
    _orig_fwd = _gen.forward
    _shapes = [] if os.environ.get("PROFILE_DIT") == "2" else None
    _dit = {"ms": 0.0, "n": 0, "ms_first": None}

    def _timed_fwd(*a, **kw):
        torch.cuda.synchronize()
        _t0 = _time.perf_counter()
        out = _orig_fwd(*a, **kw)
        torch.cuda.synchronize()
        _ms = (_time.perf_counter() - _t0) * 1e3
        if _dit["ms_first"] is None:
            _dit["ms_first"] = _ms        # includes warmup/JIT on first call
        _dit["ms"] += _ms
        _dit["n"] += 1
        if _shapes is not None:
            x = a[0] if a else kw.get("noisy_image_or_video")
            _shapes.append((
                tuple(x.shape) if torch.is_tensor(x) else None,
                bool(kw.get("is_stream")), kw.get("anchor_kv") is not None,
                round(_ms)))
        return out

    _gen.forward = _timed_fwd

    def _dit_report():
        n, ms = _dit["n"], _dit["ms"]
        if n:
            print(f"[DIT] fwd={n} total={ms / 1000:.2f}s "
                  f"mean={ms / n:.1f}ms "
                  f"(first={_dit['ms_first']:.0f}ms incl warmup)", flush=True)
            for row in _shapes or []:
                print(f"[DIT]   shape={row[0]} stream={row[1]} "
                      f"anchor={row[2]} ms={row[3]}", flush=True)
    atexit.register(_dit_report)
    logging.info("DiT profiling ON (PROFILE_DIT=%s)",
                 os.environ.get("PROFILE_DIT"))

# ── Per-layer LQ-anchor influence probe (diagnostic, default OFF) ──
# Answers "which blocks is the anchor actually driving?". The anchor and the rolling
# KV cache share one softmax, so the mass the anchor takes is mass the history lost;
# knowing where that concentrates is what picks the k for --anchor_keep_prefix.
# Measures a counterfactual per block (drop this block's anchor, hold its input fixed,
# re-run it). The probe is bit-exact for the host run (it restores RNG and returns the
# original output), so it does NOT change the videos and therefore does NOT go in the
# output path. Consequence: point --output_folder at a FRESH directory, or the
# pipeline skips every already-rendered clip and the probe records nothing.
if args.anchor_influence_probe:
    import atexit as _atexit_probe

    from utils.anchor_influence_probe import AnchorInfluenceProbe
    from wan.modules.sr_dit.models import WanModel as _ProbeWanModel

    _probe_root = next((_m for _m in pipeline.generator.model.modules()
                        if isinstance(_m, _ProbeWanModel)), None)
    if _probe_root is None:
        raise RuntimeError(
            "--anchor_influence_probe: no WanModel under pipeline.generator.model")
    _anchor_probe = AnchorInfluenceProbe(_probe_root.blocks).attach()

    def _dump_anchor_probe():
        _anchor_probe.detach()
        _summary = _anchor_probe.summary()
        _payload = {
            "metric": ("relative L2 change of a block's self-attention output when "
                       "that block alone loses its anchor, input held fixed"),
            "num_blocks": len(_probe_root.blocks),
            "anchor_layers": list(getattr(config, "anchor_layers", None) or [])
                             if getattr(config, "anchor_layers", None) is not None
                             else None,
            "anchor_keep_prefix": args.anchor_keep_prefix,
            "lq_guidance_mode": args.lq_guidance_mode,
            "lq_guidance_scale": args.lq_guidance_scale,
            "native": bool(args.lq_anchor_native),
            "native_scheme": args.lq_anchor_native_scheme,
            "per_block": _summary,
        }
        _dirname = os.path.dirname(os.path.abspath(args.anchor_influence_probe))
        os.makedirs(_dirname, exist_ok=True)
        with open(args.anchor_influence_probe, "w") as _f:
            json.dump(_payload, _f, indent=2)
        if _summary:
            _top = sorted(_summary.items(), key=lambda kv: -kv[1]["rel"])[:5]
            print("[anchor-probe] most anchor-dependent blocks: "
                  + ", ".join(f"L{k}={v['rel']:.3f}" for k, v in _top), flush=True)
        print(f"[anchor-probe] {len(_summary)} block(s) recorded -> "
              f"{args.anchor_influence_probe}", flush=True)

    _atexit_probe.register(_dump_anchor_probe)
    logging.info("[anchor-probe] attached to %d blocks -> %s",
                 len(_probe_root.blocks), args.anchor_influence_probe)

# ── Gather input videos ──
samples = []
if args.input_path.endswith(".json"):
    with open(args.input_path) as f:
        entries = json.load(f)
    for entry in entries:
        samples.append((entry["video_path"], entry.get("prompt", args.prompt)))
elif os.path.isdir(args.input_path):
    for vf in sorted(glob.glob(os.path.join(args.input_path, "*.mp4"))):
        samples.append((vf, args.prompt))
else:
    samples.append((args.input_path, args.prompt))

if not samples:
    raise FileNotFoundError(f"No videos found at {args.input_path}")

logging.info("Found %d input video(s)", len(samples))

# ── Pre-encode text & offload text encoder (saves ~26GB VRAM) ──
unique_prompts = list(set(prompt for _, prompt in samples))
neg_prompt = config.get("negative_prompt", "")
with torch.no_grad():
    _cached_cond = {}
    for p in unique_prompts:
        _cached_cond[p] = {k: v.to(device) for k, v in pipeline.text_encoder(text_prompts=[p]).items()}
    _cached_uncond = {k: v.to(device) for k, v in pipeline.text_encoder(text_prompts=[neg_prompt]).items()}

pipeline.text_encoder.to("cpu")
torch.cuda.empty_cache()
logging.info("Text encoder offloaded to CPU, %d prompts cached (%.1f GiB free)",
             len(unique_prompts), torch.cuda.mem_get_info()[0] / 2**30)
os.makedirs(args.output_folder, exist_ok=True)

default_target_h, default_target_w = args.target_size

# ── Inference loop ──
for video_path, prompt in tqdm(samples, desc="SR inference"):
    basename = os.path.splitext(os.path.basename(video_path))[0]
    # Embed the (sanitized, truncated) driving prompt in the output filename.
    output_path = os.path.join(args.output_folder, f"{basename}__{_prompt_slug(prompt)}_sr.mp4")
    if os.path.exists(output_path):
        logging.info("Skipping %s (already exists)", output_path)
        continue

    video_tensor, _, info = read_video(video_path, pts_unit="sec")
    fps = info.get("video_fps", 16)
    T_pixel = video_tensor.shape[0]

    if args.num_frames == -1:
        # Use every available frame; snap down to the nearest 4n+1 so the frame
        # count matches the VAE's temporal compression (Wan VAE compresses by 4).
        num_frames = max(1, ((T_pixel - 1) // 4) * 4 + 1)
        if num_frames != T_pixel:
            video_tensor = video_tensor[:num_frames]
            logging.info("[%s] num_frames=-1: using %d/%d frames (snapped to 4n+1)",
                         basename, num_frames, T_pixel)
    else:
        num_frames = args.num_frames
        if T_pixel > num_frames:
            video_tensor = video_tensor[:num_frames]
        elif T_pixel < num_frames:
            pad = num_frames - T_pixel
            video_tensor = torch.cat([
                video_tensor, video_tensor[-1:].expand(pad, -1, -1, -1)
            ], dim=0)

    # Compute target size: auto (native * sr_scale) or fixed.
    if args.auto_target_size:
        nat_h, nat_w = int(video_tensor.shape[1]), int(video_tensor.shape[2])
        _r32 = lambda x: max(32, int(round(x * args.sr_scale / 32.0)) * 32)
        target_h, target_w = _r32(nat_h), _r32(nat_w)
        logging.info("[%s] auto target_size: native %dx%d * %.4g -> %dx%d",
                     basename, nat_h, nat_w, args.sr_scale, target_h, target_w)
    else:
        target_h, target_w = default_target_h, default_target_w

    # [T, H, W, C] uint8 -> [1, C, T, H, W] float [-1, 1]
    lr_pixels = video_tensor.float().div(255.0).mul(2.0).sub(1.0)
    lr_pixels = rearrange(lr_pixels, "t h w c -> 1 c t h w").to(device=device, dtype=torch.bfloat16)

    # Trained latent upsampler expects LR at target_size / scale (fixed-scale model).
    # Resize LR pixels to that size before encoding so its output matches the HR noise.
    if using_latent_upsampler and not args.pixel_upsample:
        lr_h_expected = target_h // latent_upsample_scale
        lr_w_expected = target_w // latent_upsample_scale
        if lr_pixels.shape[-2:] != (lr_h_expected, lr_w_expected):
            lr_pixels = F.interpolate(
                rearrange(lr_pixels, "b c t h w -> (b t) c h w"),
                size=(lr_h_expected, lr_w_expected),
                mode="bilinear", align_corners=False,
            )
            lr_pixels = rearrange(lr_pixels, "(b t) c h w -> b c t h w", b=1)
            logging.info("[%s] Resized LR pixels to %dx%d (target %dx%d / %dx upsampler)",
                         basename, lr_h_expected, lr_w_expected,
                         target_h, target_w, latent_upsample_scale)

    t0 = time.time()
    if args.pixel_upsample:
        # Upsample in pixel space, then VAE encode
        lr_up_pixels = F.interpolate(
            rearrange(lr_pixels, "b c t h w -> (b t) c h w"),
            size=(target_h, target_w),
            mode=args.pixel_upsample_mode,
            align_corners=False if args.pixel_upsample_mode != "nearest" else None,
        )
        lr_up_pixels = rearrange(lr_up_pixels, "(b t) c h w -> b c t h w", b=1)
        lr_latent = encode_vae.encode_to_latent(lr_up_pixels)
        del lr_up_pixels
        logging.info("[%s] Pixel-upsampled (%s) & encoded: %s (%.1fs)",
                     basename, args.pixel_upsample_mode, list(lr_latent.shape), time.time() - t0)
    else:
        lr_latent = encode_vae.encode_to_latent(lr_pixels)
        logging.info("[%s] LR encoded: %s (%.1fs)", basename, list(lr_latent.shape), time.time() - t0)

    _, T_lat, C_lat, h_lat, w_lat = lr_latent.shape
    T_lat_orig = T_lat
    vae_spatial_ratio = 8 if config.get("vae_version", "2.2") == "2.1" else 16
    hr_h_lat = target_h // vae_spatial_ratio
    hr_w_lat = target_w // vae_spatial_ratio

    # Pad latent temporal dim to be divisible by stream_chunk_size
    chunk_size = config.num_frame_per_block
    if T_lat % chunk_size != 0:
        pad_f = chunk_size - (T_lat % chunk_size)
        lr_latent = torch.cat([lr_latent, lr_latent[:, -1:].expand(-1, pad_f, -1, -1, -1)], dim=1)
        T_lat = lr_latent.shape[1]
        logging.info("[%s] Padded latent temporal: %d -> %d (chunk_size=%d)",
                     basename, T_lat_orig, T_lat, chunk_size)

    noise = torch.randn(
        [1, T_lat, C_lat, hr_h_lat, hr_w_lat],
        device=device, dtype=torch.bfloat16,
    )

    conditional_dict = _cached_cond.get(prompt, _cached_cond[unique_prompts[0]])
    unconditional_dict = _cached_uncond

    pipeline.stage_timer.reset()
    t0 = time.time()
    if not use_teacher_forcing:
        sr_latent = pipeline.inference_sr_df(
            lr_latent=lr_latent,
            noise=noise,
            text_prompts=[prompt],
            sigma_start=sigma_start,
            return_latents=False,
            return_video=False,
            cond_noise_sigma=args.cond_noise_sigma,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            lq_cfg=args.lq_cfg,
            use_flex_attention=args.use_flex_attention,
        )
        sr_latent = sr_latent[:, :T_lat_orig]
        _maybe_save_latent(sr_latent, basename)
        video_out = _decode_timed(pipeline, sr_latent)
        video_out = _normalize_decode(video_out)
    else:
        sr_latent = pipeline.inference_sr(
            lr_latent=lr_latent,
            noise=noise,
            text_prompts=[prompt],
            sigma_start=sigma_start,
            return_latents=False,
            return_video=False,
            cond_noise_sigma=args.cond_noise_sigma,
            conditional_dict=conditional_dict,
            use_lq_anchor=args.use_lq_anchor,
            lq_guidance_scale=args.lq_guidance_scale,
            lq_guidance_mode=args.lq_guidance_mode,
            anchor_context_noise=getattr(config, "anchor_context_noise", None),
            anchor_layers=getattr(config, "anchor_layers", None),
            anchor_align=args.lq_anchor_align,
            anchor_window_scope=args.lq_anchor_window_scope,
            native_lq_anchor=args.lq_anchor_native,
            native_anchor_scheme=args.lq_anchor_native_scheme,
        )
        sr_latent = sr_latent[:, :T_lat_orig]
        _maybe_save_latent(sr_latent, basename)
        video_out = _decode_timed(pipeline, sr_latent)
        video_out = _normalize_decode(video_out)
    elapsed = time.time() - t0
    logging.info("[%s] SR done: %s (%.1fs)", basename, list(video_out.shape), elapsed)
    # The aggregate above cannot separate a DiT win from a decoder win, and the
    # speed knobs only ever move one of the two. Same stream as `SR done` so the
    # two lines stay adjacent in the bench log.
    logging.info("[%s] %s", basename, pipeline.stage_timer.report(total_s=elapsed))

    video_out = rearrange(video_out, "b t c h w -> b t h w c").cpu()
    video_out = (video_out[0] * 255.0).clamp(0, 255).to(torch.uint8)
    video_out = video_out[:T_pixel]

    pipeline.vae.model.clear_cache()
    if encode_vae is not pipeline.vae:
        encode_vae.model.clear_cache()

    write_video(output_path, video_out, fps=fps)
    if args.keep_audio:
        _mux_audio(output_path, video_path, basename)
    logging.info("[%s] Saved to %s", basename, output_path)

logging.info("All done. Results in %s", args.output_folder)
