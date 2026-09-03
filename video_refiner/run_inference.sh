#!/bin/bash
# SR-DiT causal few-step inference launcher.
#
# The pipeline skips clips whose output file already exists, so every knob that
# changes pixels is embedded in the OUTPUT directory name. Change a knob ->
# new directory -> fresh videos; forget one -> the A/B silently re-serves the
# first arm's videos.

cd "$(dirname "$0")"

# ── Required paths (override per run / per environment) ──
# CHECKPOINT: merged SR-DiT model (base + LoRA folded in);
# loaded directly, no --lora_checkpoint_path involved.
CHECKPOINT="${CHECKPOINT:-../checkpoints/refiner/sr_dit_5b.pt}"
# INPUT (required): an .mp4 file, a folder of .mp4 files, or a JSON list of
# {"video_path": ..., "prompt": ...} entries.
INPUT="${INPUT:-}"
if [[ -z "${INPUT}" ]]; then
    echo "ERROR: set INPUT to an .mp4 file, a folder of .mp4 files, or a JSON" >&2
    echo "       list of {\"video_path\": ..., \"prompt\": ...} entries." >&2
    exit 1
fi

# ── Runtime ──
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/sr_dit_5b.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs}"
CUDA_DEV="${CUDA_DEV:-0}"
# Frames per clip; -1 = all frames (snapped to 4n+1 for the VAE).
NUM_FRAMES="${NUM_FRAMES:--1}"
SEED="${SEED:-42}"

# ── Super-resolution ──
# Target resolution per clip = native resolution * SR_SCALE (snapped to /32,
# aspect preserved).
SR_SCALE="${SR_SCALE:-2.0}"
# Truncated flow-matching sigma_start of the refiner.
SIGMA_START="${SIGMA_START:-0.6251}"
# Latent frames of history kept in the streaming KV cache.
KV_LEN="${KV_LEN:-9}"

# ── Latent upsampler (LR latent -> HR latent) ──
# flash          FlashLatentUpsampler (default)
# 2d_causal      2D backbone, causal (left-padded) TemporalConv
LATENT_UPSAMPLER="${LATENT_UPSAMPLER:-flash}"
LATENT_UPSAMPLER_CKPT="${LATENT_UPSAMPLER_CKPT:-}"
# PIXEL_UPSAMPLE=1 replaces the latent upsampler with pixel-space interpolation
# before VAE encoding (cheap baseline arm).
PIXEL_UPSAMPLE="${PIXEL_UPSAMPLE:-0}"
PIXEL_UPSAMPLE_MODE="${PIXEL_UPSAMPLE_MODE:-bicubic}"

# ── Window attention (block-grid geometry + backend) ──
USE_WINDOW_ATTN="${USE_WINDOW_ATTN:-1}"
# triton (default) | flash | sdpa | flex | magi | auto
WINDOW_ATTN_IMPL="${WINDOW_ATTN_IMPL:-triton}"
BLOCK_HW="${BLOCK_HW:-4 4}"
RADIUS_HW="${RADIUS_HW:-3 3}"

# ── LQ anchor (per-chunk low-res K/V context during denoising) ──
USE_LQ_ANCHOR="${USE_LQ_ANCHOR:-1}"
ANCHOR_MODE="${ANCHOR_MODE:-v}"
ANCHOR_SCALE="${ANCHOR_SCALE:-1.0}"
# frame = strict 1:1 (HR frame i attends only anchor frame i);
# chunk = whole-chunk anchor visibility.
ANCHOR_ALIGN="${ANCHOR_ALIGN:-frame}"

# ── LightVAE-NU fast decoder ──
# OFF by default (quality trade). When enabled, decodes with the distilled
# student instead of the full Wan2.2 VAE; the full VAE still does the LR encode.
ENABLE_NU_LIGHTVAE="${ENABLE_NU_LIGHTVAE:-0}"
NU_LIGHTVAE_TYPE="${NU_LIGHTVAE_TYPE:-scheme3}"
NU_LIGHTVAE_CKPT="${NU_LIGHTVAE_CKPT:-}"

# ── fp8 DiT GEMMs ──
# OFF by default (quality trade): quantises the transformer-block Linears
# (q/k/v/o, cross-attn, ffn) to fp8 e4m3 with row-wise dynamic scales; ~1.23x
# per block on sm89+. Per-GEMM relative error is ~22x bf16's, compounding over
# 30 blocks — check PSNR before trusting the output.
ENABLE_FP8="${ENABLE_FP8:-0}"
FP8_TARGETS="${FP8_TARGETS:-all}"   # all = every block Linear; ffn = ffn.0/ffn.2 only
FP8_SKIP_FIRST="${FP8_SKIP_FIRST:-0}"
FP8_SKIP_LAST="${FP8_SKIP_LAST:-0}"

# ── Optional diagnostics ──
SAVE_LATENT_DIR="${SAVE_LATENT_DIR:-}"

# ══ Assemble arguments ══════════════════════════════════════════════════════

UPSCALE_ARG=()
UPSCALE_TAG=""
if [ "${PIXEL_UPSAMPLE}" = "1" ]; then
    UPSCALE_ARG=(--pixel_upsample --pixel_upsample_mode "${PIXEL_UPSAMPLE_MODE}")
    UPSCALE_TAG="_pixelup_${PIXEL_UPSAMPLE_MODE}"
else
    case "${LATENT_UPSAMPLER}" in
        flash)
            UPSCALE_ARG=(--latent_upsampler_config configs/latent_upsampler_flash.yaml
                         --latent_upsampler_ckpt "${LATENT_UPSAMPLER_CKPT:-../checkpoints/refiner/latent_upsampler_flash.pt}")
            ;;
        2d_causal)
            UPSCALE_ARG=(--latent_upsampler_config configs/latent_upsampler_2d_causal.yaml
                         --latent_upsampler_ckpt "${LATENT_UPSAMPLER_CKPT:-../checkpoints/refiner/latent_upsampler_2d_causal.pt}")
            UPSCALE_TAG="_lu2dcausal"
            ;;
        none)
            ;;
        *)
            echo "Unknown LATENT_UPSAMPLER: ${LATENT_UPSAMPLER}" >&2
            exit 1
            ;;
    esac
fi

WINDOW_ARG=()
GEOM_TAG="nowindow"
if [ "${USE_WINDOW_ATTN}" = "1" ]; then
    _b=$(echo ${BLOCK_HW} | awk '{print ($1==$2)?$1:$1"x"$2}')
    _r=$(echo ${RADIUS_HW} | awk '{print ($1==$2)?$1:$1"x"$2}')
    WINDOW_ARG=(--use_window_attn
                --window_attn_impl "${WINDOW_ATTN_IMPL}"
                --window_block_hw ${BLOCK_HW}
                --window_block_radius_hw ${RADIUS_HW})
    GEOM_TAG="block_${_b}_radius_${_r}_impl_${WINDOW_ATTN_IMPL}"
fi

ANCHOR_ARG=()
ANCHOR_TAG="no_anchor"
if [ "${USE_LQ_ANCHOR}" = "1" ]; then
    ANCHOR_ARG=(--use_lq_anchor
                --lq_guidance_mode "${ANCHOR_MODE}"
                --lq_guidance_scale "${ANCHOR_SCALE}"
                --lq_anchor_align "${ANCHOR_ALIGN}")
    ANCHOR_TAG="anchor_${ANCHOR_MODE}_${ANCHOR_ALIGN}"
fi

LIGHTVAE_ARG=()
LIGHTVAE_TAG=""
if [ "${ENABLE_NU_LIGHTVAE}" = "1" ]; then
    LIGHTVAE_ARG=(--enable_nu_lightvae --nu_lightvae_type "${NU_LIGHTVAE_TYPE}")
    if [ -n "${NU_LIGHTVAE_CKPT}" ]; then
        LIGHTVAE_ARG+=(--nu_lightvae_ckpt "${NU_LIGHTVAE_CKPT}")
    fi
    LIGHTVAE_TAG="_lightvae_${NU_LIGHTVAE_TYPE}"
fi

# fp8 goes into the tag: it changes pixels, and the pipeline skips existing
# outputs, so a shared directory would re-serve the bf16 arm's videos.
FP8_ARG=()
FP8_TAG=""
if [ "${ENABLE_FP8}" = "1" ]; then
    FP8_ARG=(--fp8_linear
             --fp8_targets "${FP8_TARGETS}"
             --fp8_skip_first "${FP8_SKIP_FIRST}"
             --fp8_skip_last "${FP8_SKIP_LAST}")
    FP8_TAG="_fp8_${FP8_TARGETS}"
fi

SAVE_LATENT_ARG=()
if [ -n "${SAVE_LATENT_DIR}" ]; then
    SAVE_LATENT_ARG=(--save_latent_dir "${SAVE_LATENT_DIR}")
fi

SEED_TAG=""
if [ "${SEED}" != "42" ]; then
    SEED_TAG="_seed${SEED}"
fi

# The input basename goes into the tag: different eval sets can share sample
# ids, and a shared OUTPUT dir would re-serve stale videos.
INPUT_TAG="$(basename "${INPUT}")"
INPUT_TAG="${INPUT_TAG%.*}"

OUTPUT="${OUTPUT:-${OUTPUT_BASE}/ckpt_$(basename "${CHECKPOINT}" .pt)/sigma_${SIGMA_START}_${ANCHOR_TAG}_kv${KV_LEN}_scale${SR_SCALE}${LIGHTVAE_TAG}${FP8_TAG}_${GEOM_TAG}${UPSCALE_TAG}_${INPUT_TAG}${SEED_TAG}/}"

CUDA_VISIBLE_DEVICES="${CUDA_DEV}" "${PYTHON_BIN}" inference_sr.py \
    --config_path "${CONFIG}" \
    --checkpoint_path "${CHECKPOINT}" \
    --input_path "${INPUT}" \
    --output_folder "${OUTPUT}" \
    --prompt "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo-realistic maximum detail, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations" \
    --sigma_start "${SIGMA_START}" \
    --num_frames "${NUM_FRAMES}" \
    --auto_target_size \
    --sr_scale "${SR_SCALE}" \
    --causal \
    --seed "${SEED}" \
    --kv_len "${KV_LEN}" \
    "${UPSCALE_ARG[@]}" \
    "${WINDOW_ARG[@]}" \
    "${ANCHOR_ARG[@]}" \
    "${LIGHTVAE_ARG[@]}" \
    "${FP8_ARG[@]}" \
    "${SAVE_LATENT_ARG[@]}"
