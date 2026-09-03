# SR-DiT Inference

Causal few-step video super-resolution (2x) with a 5B diffusion transformer
(SR-DiT) built on the Wan2.2 stack. The pipeline streams chunk-by-chunk with a
truncated KV cache and causal block-grid window attention, decodes
with the Wan2.2 VAE, and supports an optional LQ-anchor (low-res K/V context)
and an optional distilled LightVAE-NU decoder. The refiner only synthesizes
video; the input clip's audio track is stream-copied back into the output
(`--keep_audio`, on by default; disable with `--no-keep_audio`).

## Setup

```bash
pip install -r requirements.txt
```

You need a [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI) model directory
(VAE + UMT5-xxl text encoder + tokenizer). This release defaults
`wan_model_dir` in `configs/sr_dit_5b.yaml` to `../checkpoints/wan2.2_ti2v_5b`;
if you download the model yourself, point `wan_model_dir` at your copy.

## Checkpoints

The refiner weights live under `../checkpoints/refiner/`:

```
../checkpoints/refiner/sr_dit_5b.pt               # SR-DiT model
../checkpoints/refiner/latent_upsampler_flash.pt  # FlashLatentUpsampler weights
../checkpoints/refiner/latent_upsampler_2d_causal.pt  # causal 2D latent upsampler weights (optional)
../checkpoints/refiner/lightvae_nu_scheme3.pt     # distilled fast decoder (optional, off by default)
```

## 1. Run inference

```bash
INPUT=/path/to/video.mp4 bash run_inference.sh
```

`INPUT` accepts an `.mp4` file, a folder of `.mp4` files, or a JSON list of
`{"video_path": ..., "prompt": ...}` entries. The main knobs (environment
variables, all documented in the script):

| Variable | Default | Meaning |
|---|---|---|
| `CHECKPOINT` | `../checkpoints/refiner/sr_dit_5b.pt` | merged SR-DiT model |
| `INPUT` | (required) | input video(s) |
| `SR_SCALE` | `2.0` | target = native resolution x scale |
| `NUM_FRAMES` | `-1` | frames per clip (-1 = all, snapped to 4n+1) |
| `KV_LEN` | `9` | latent frames of history in the streaming KV cache |
| `SIGMA_START` | `0.6251` | truncated flow-matching sigma start |
| `USE_WINDOW_ATTN` / `WINDOW_ATTN_IMPL` | `1` / `triton` | block-grid window attention and backend |
| `USE_LQ_ANCHOR` / `ANCHOR_MODE` / `ANCHOR_ALIGN` | `1` / `v` / `frame` | low-res K/V anchor |
| `LATENT_UPSAMPLER` | `flash` | LR->HR latent upsampler (`flash`, `2d_causal`, `none`) |
| `ENABLE_NU_LIGHTVAE` | `0` | distilled fast decoder (quality trade; off by default) |
| `ENABLE_FP8` | `0` | fp8 row-wise DiT GEMMs (quality trade; off by default) |

Every pixel-changing knob is embedded in the output directory name: the
pipeline skips clips whose output file already exists, so changing a knob
without changing the path would silently re-serve old videos.

## Speed / quality knobs

For faster inference (with some quality loss), enable the accelerated decoder
and fp8 DiT GEMMs — together they cut the two dominant stages (decode ~76%,
DiT ~23% of wall time). For higher fidelity, switch the LR->HR path to
pixel-space upsampling (`PIXEL_UPSAMPLE=1`, bicubic) instead of the learned
latent upsampler.

Benchmark: (1248x704 -> 2K output, 117 frames each),
default knobs, single H20, mean per clip:

| arm | command | total | DiT | decode | speedup | PSNR vs baseline |
|-----|---------|-------|-----|--------|---------|------------------|
| bf16 baseline (defaults) | `bash run_inference.sh` | 155.8 s | 35.8 s | 118.2 s | 1.0x | — |
| fp8 | `ENABLE_FP8=1 bash run_inference.sh` | 147.8 s | 28.2 s | 117.5 s | 1.05x | 38.59 dB |
| fp8 + lightvae | `ENABLE_NU_LIGHTVAE=1 ENABLE_FP8=1 bash run_inference.sh` | 46.2 s | 28.2 s | 16.9 s | 3.4x | 36.02 dB |

fp8 only accelerates the DiT stage (~1.27x per forward); LightVAE-NU only
accelerates the decode stage (~7x) — enabling both is what gives the 3.4x
end-to-end speedup, at a PSNR cost comparable to the VAE decoder's own
round-trip error.

## Repository layout

```
inference_sr.py      inference entry point (CLI)
run_inference.sh     launcher with documented env knobs
configs/             model / upsampler configs
pipeline_sr/         causal few-step pipeline (sf paths)
utils/               DiT / VAE / text-encoder wrappers, scheduler, LoRA utils
wan/                 Wan2.2 modules used at inference (VAE, T5, SR-DiT)
lightvae_nu/         distilled LightVAE-NU student decoder (optional)
```

## License

Apache License 2.0. See the repository's top-level [LICENSE](../LICENSE).
