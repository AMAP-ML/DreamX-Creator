#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

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
IMAGE_PATH="${INPUT_IMAGE:-${CASE_IMAGE}}"
PROMPT="${PROMPT:-$(cat "${CASE_PROMPT_FILE}")}"
# Shared Wan2.2-TI2V-5B dependency directory: video VAE, UMT5-xxl text
# encoder, and tokenizer.
MODEL_NAME="${MODEL_NAME:-${REPO_ROOT}/checkpoints/wan2.2_ti2v_5b}"
# DreamX-Creator 1.0 joint audio-video generator (merged checkpoint).
TRANSFORMER_PATH="${TRANSFORMER_PATH:-${REPO_ROOT}/checkpoints/creator}"
# Creator audio VAE (CreatorDACVAE) directory.
AUDIO_VAE_PATH="${AUDIO_VAE_PATH:-${REPO_ROOT}/checkpoints/audio_vae}"

CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config/config.yaml}"

if [[ -z "${AUDIO_VAE_PATH}" ]]; then
    echo "ERROR: set AUDIO_VAE_PATH to the Creator audio VAE directory" >&2
    exit 1
fi
if [[ "${AUDIO_VAE_PATH}" == *:* ]]; then
    echo "ERROR: AUDIO_VAE_PATH must be one directory, not PATH/CUDA_VISIBLE_DEVICES; got: ${AUDIO_VAE_PATH}" >&2
    exit 1
fi

if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "ERROR: image not found: ${IMAGE_PATH}" >&2
    exit 1
fi

for required_path in "${MODEL_NAME}" "${TRANSFORMER_PATH}" "${AUDIO_VAE_PATH}" "${CONFIG_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "ERROR: required path not found: ${required_path}" >&2
        exit 1
    fi
done

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/inference.py" \
    --config_path "${CONFIG_PATH}" \
    --image "${IMAGE_PATH}" \
    --prompt "${PROMPT}" \
    --model_name "${MODEL_NAME}" \
    --transformer_path "${TRANSFORMER_PATH}" \
    --audio_vae_path "${AUDIO_VAE_PATH}" \
    --output "outputs/${CASE}.mp4" \
    --duration 5 \
    --seed 113 \
    "$@"
