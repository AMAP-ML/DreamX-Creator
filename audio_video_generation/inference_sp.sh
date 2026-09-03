#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# torchrun ships with the interpreter's environment: prefer the one next to
# PYTHON_BIN, then anything on PATH.
if [[ -z "${TORCHRUN_BIN:-}" ]]; then
    if [[ "${PYTHON_BIN}" == */* && -x "${PYTHON_BIN%/*}/torchrun" ]]; then
        TORCHRUN_BIN="${PYTHON_BIN%/*}/torchrun"
    else
        TORCHRUN_BIN="torchrun"
    fi
fi

# Official Verse-Bench evaluation cases shipped in assets/ (first frame .jpg +
# prompt .txt pairs). Pick one with CASE=<name>; INPUT_IMAGE / PROMPT still
# override both inputs directly.
#   case1  man on a yellow couch talking          (Verse-Bench set2/303, default)
#   case2  humanoid robot cooking                 (Verse-Bench set1/00192)
#   case3  night lightning over a city            (Verse-Bench set2/304)
#   case4  wolf howling in an autumn forest       (Verse-Bench set2/317)
#   case5  person typing on a keyboard            (Verse-Bench set2/329)
#   case6  man playing acoustic guitar            (Verse-Bench set2/427)
CASE="${CASE:-case1}"
CASE_IMAGE="${SCRIPT_DIR}/assets/${CASE}.jpg"
CASE_PROMPT_FILE="${SCRIPT_DIR}/assets/${CASE}.txt"
if [[ ! -f "${CASE_IMAGE}" || ! -f "${CASE_PROMPT_FILE}" ]]; then
    echo "ERROR: unknown CASE '${CASE}' (need ${CASE_IMAGE} and ${CASE_PROMPT_FILE})" >&2
    exit 1
fi

# INPUT_IMAGE is the override variable: some cluster shells predefine
# IMAGE_PATH as a runtime search path, so IMAGE_PATH cannot be trusted.
IMAGE_PATH="${IMAGE_PATH:-}"
if [[ -z "${IMAGE_PATH}" || "${IMAGE_PATH}" == *:* ]]; then
    IMAGE_PATH="${INPUT_IMAGE:-${CASE_IMAGE}}"
fi
PROMPT="${PROMPT:-$(cat "${CASE_PROMPT_FILE}")}"
# Shared Wan2.2-TI2V-5B dependency directory: video VAE, UMT5-xxl text
# encoder, and tokenizer.
MODEL_NAME="${MODEL_NAME:-${REPO_ROOT}/checkpoints/wan2.2_ti2v_5b}"
# DreamX-Creator 1.0 joint audio-video generator (merged checkpoint).
TRANSFORMER_PATH="${TRANSFORMER_PATH:-${REPO_ROOT}/checkpoints/creator}"
# Creator audio VAE (CreatorDACVAE) directory.
AUDIO_VAE_PATH="${AUDIO_VAE_PATH:-${REPO_ROOT}/checkpoints/audio_vae}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config/config.yaml}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/outputs/${CASE}.mp4}"
SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
DURATION="${DURATION:-5}"

# Sequence-parallel inference shards the attention heads, so the GPU count
# must divide the Creator head counts (video 24 / audio 12): 2, 3, 4, 6, or 12.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    gpu_count="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
    if (( gpu_count >= 2 )); then
        CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((gpu_count - 1)))"
    else
        echo "ERROR: set CUDA_VISIBLE_DEVICES to the GPUs to use (at least 2 for SP inference)" >&2
        exit 1
    fi
fi

IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
SP_SIZE="${SP_SIZE:-${#visible_gpus[@]}}"
if [[ "${#visible_gpus[@]}" -ne "${SP_SIZE}" ]]; then
    echo "ERROR: SP_SIZE=${SP_SIZE} but CUDA_VISIBLE_DEVICES exposes ${#visible_gpus[@]} GPUs" >&2
    exit 1
fi
case "${SP_SIZE}" in
    2|3|4|6|12) ;;
    *)
        echo "ERROR: SP_SIZE=${SP_SIZE} must divide the Creator head counts (24/12); use 2, 3, 4, 6, or 12 GPUs" >&2
        exit 1
        ;;
esac
if [[ "${TORCHRUN_BIN}" == */* ]]; then
    if [[ ! -x "${TORCHRUN_BIN}" ]]; then
        echo "ERROR: torchrun not found or not executable: ${TORCHRUN_BIN}" >&2
        exit 1
    fi
else
    if ! command -v "${TORCHRUN_BIN}" >/dev/null 2>&1; then
        echo "ERROR: torchrun not found on PATH: ${TORCHRUN_BIN}" >&2
        exit 1
    fi
fi
if [[ "${IMAGE_PATH}" == *:* ]]; then
    echo "ERROR: IMAGE_PATH must be one image file, not a PATH list: ${IMAGE_PATH}" >&2
    exit 1
fi
if [[ "${AUDIO_VAE_PATH}" == *:* ]]; then
    echo "ERROR: AUDIO_VAE_PATH must be one directory, not a PATH list: ${AUDIO_VAE_PATH}" >&2
    exit 1
fi
for required_path in "${IMAGE_PATH}" "${MODEL_NAME}" "${TRANSFORMER_PATH}" "${AUDIO_VAE_PATH}" "${CONFIG_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "ERROR: required path not found: ${required_path}" >&2
        exit 1
    fi
done

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

exec "${TORCHRUN_BIN}" --standalone --nproc_per_node="${SP_SIZE}" \
    "${SCRIPT_DIR}/inference_sp.py" \
    --sp_size "${SP_SIZE}" \
    --config_path "${CONFIG_PATH}" \
    --image "${IMAGE_PATH}" \
    --prompt "${PROMPT}" \
    --model_name "${MODEL_NAME}" \
    --transformer_path "${TRANSFORMER_PATH}" \
    --audio_vae_path "${AUDIO_VAE_PATH}" \
    --output "${OUTPUT_PATH}" \
    --duration "${DURATION}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --seed "${SEED}" \
    "$@"
