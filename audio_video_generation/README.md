# DreamX-Creator 1.0 — Native Joint Audio-Video Generation

Single-image + prompt to synchronized audio-video. This directory contains the
Creator gating joint video/audio denoiser and the model/VAE/text-encoder code it
imports. Training, datasets, and evaluation are intentionally excluded.

The release checkpoints (see the repository's top-level `checkpoints/`
directory) are expected at:

```text
checkpoints/creator/video_model/config.json
checkpoints/creator/audio_model/config.json
checkpoints/creator/cross_attn_weights.safetensors
checkpoints/audio_vae/                     # CreatorDACVAE
checkpoints/wan2.2_ti2v_5b/                # video VAE + UMT5-xxl text encoder + tokenizer
```

All paths are overridable through environment variables and default to the
layout above:

```bash
cd audio_video_generation
./inference.sh
```

Override the input image / prompt / model directories with `INPUT_IMAGE`,
`PROMPT`, `MODEL_NAME`, `TRANSFORMER_PATH`, or `AUDIO_VAE_PATH`; extra options
are passed through to `inference.py`. Override the interpreter with
`PYTHON_BIN=/path/to/python` if needed. (`INPUT_IMAGE` — not `IMAGE_PATH` — is
the image override variable: some cluster shells predefine `IMAGE_PATH` as a
runtime search path.)

## Verse-Bench evaluation cases

`assets/` ships six official Verse-Bench evaluation cases (first frame +
prompt pairs, named `case<N>.jpg` / `case<N>.txt`):

| CASE | Content | Verse-Bench |
|---|---|---|
| `case1` | man on a yellow couch talking (single-speaker) | set2/303 |
| `case2` | humanoid robot cooking (non-speech) | set1/00192 |
| `case3` | night lightning over a city (non-speech) | set2/304 |
| `case4` | wolf howling in an autumn forest (non-speech) | set2/317 |
| `case5` | person typing on a keyboard (non-speech) | set2/329 |
| `case6` | man playing acoustic guitar (non-speech) | set2/427 |

`inference.sh` and `inference_sp.sh` default to `case1` and read the matching
prompt automatically; pick another with `CASE=...`:

```bash
CASE=case4 ./inference.sh
```

By default, the first frame keeps its original aspect ratio and is resized to
use between 95% and 100% of the 880 spatial-token budget. Use
`--target_spatial_tokens` and `--min_token_ratio` to change the budget bounds.

Install the pinned Python runtime dependencies with:

```bash
python -m pip install -r requirements.txt
```

The command writes `<output>.mp4`, `<output>.video.mp4`, `<output>.wav`, and the
resized first frame `<output>_first_frame.png`. A bare output stem automatically
gets the `.mp4` suffix. The default settings: 24 FPS, 5 seconds, 50 denoising
steps, and multimodal CFG with the audio-video bridge guidance at 3.5
(`--video_bridge_guidance_scale` / `--audio_bridge_guidance_scale`) and text
guidance at 5.0 (`--guidance_scale`).

For lower GPU memory usage, enable CPU offload independently for the text
encoder and both VAEs:

```bash
./inference.sh --text_encoder_cpu_offload \
  --vae_cpu_offload
```

Each option moves its module to GPU only for the required encode/decode phase.
Use `--video_vae_cpu_offload` or `--audio_vae_cpu_offload` when only one VAE
should be offloaded.
When `--GPU_memory_mode model_cpu_offload` is used, all three options default
to enabled; pass the corresponding `--no-*` option to override an individual
module.

`requirements.txt` contains only the direct inference dependencies, including
`tqdm` for denoising progress; it intentionally omits training/evaluation
packages, transitive CUDA wheels, and optional attention kernels. `ffmpeg` is
an external system executable required for final MP4 muxing. `soundfile` and
`torchaudio` are both supported for WAV output; the script also has a standard
library fallback.

## Multi-GPU sequence-parallel inference

`inference_sp.sh` runs the same single-sample pipeline split across GPUs with
Wan2.2-style Ulysses sequence parallelism: every rank keeps a full copy of the
weights and attention heads are sharded across ranks, so per-rank peak
activation memory drops while the sampled result matches single-GPU inference.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./inference_sp.sh
```

The GPU count is the sequence-parallel size and must divide the Creator head
counts (video 24 / audio 12): use 2, 3, 4, 6, or 12 GPUs. When
`CUDA_VISIBLE_DEVICES` is unset, the script defaults to every GPU on the node.
The same `INPUT_IMAGE` / `PROMPT` / `MODEL_NAME` / `TRANSFORMER_PATH` /
`AUDIO_VAE_PATH` overrides apply, plus `OUTPUT_PATH`, `SEED`,
`NUM_INFERENCE_STEPS`, and `DURATION`; `TORCHRUN_BIN` overrides the launcher
(default: the `torchrun` shipped next to `PYTHON_BIN`, then `torchrun` on
PATH). The noise latents and the encoded first frame are broadcast from
rank 0 so all ranks denoise identical inputs; rank 0 alone decodes and writes
`<output>.mp4`, `<output>.video.mp4`, and `<output>.wav`. Extra options are
passed through to `inference.py`.
