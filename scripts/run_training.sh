#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SD_SCRIPTS_DIR="${REPO_ROOT}/external/sd-scripts"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/dataset/knightro}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/finetune/knightro}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/models}"
BASE_MODEL="${BASE_MODEL:-runwayml/stable-diffusion-v1-5}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
RESOLUTION="${RESOLUTION:-512,512}"
BUCKET_STEPS="${BUCKET_STEPS:-64}"
MIN_BUCKET="${MIN_BUCKET:-256}"
MAX_BUCKET="${MAX_BUCKET:-1024}"
SAVE_EVERY="${SAVE_EVERY:-200}"
CAPTION_EXTENSION="${CAPTION_EXTENSION:-.txt}"
NETWORK_MODULE="${NETWORK_MODULE:-networks.lora}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"

cd "${SD_SCRIPTS_DIR}"
mkdir -p "${MODELS_DIR}"

if [[ "${SKIP_TRAINING:-0}" != "1" ]]; then
  accelerate launch train_network.py \
    --pretrained_model_name_or_path="${BASE_MODEL}" \
    --train_data_dir="${DATASET_DIR}" \
    --resolution="${RESOLUTION}" \
    --output_dir="${OUTPUT_DIR}" \
    --logging_dir="${LOG_DIR}" \
    --network_module="${NETWORK_MODULE}" \
    --train_batch_size="${BATCH_SIZE}" \
    --max_train_steps="${TRAIN_STEPS}" \
    --learning_rate="${LEARNING_RATE}" \
    --lr_scheduler="${LR_SCHEDULER}" \
    --mixed_precision="${MIXED_PRECISION}" \
    --save_every_n_steps="${SAVE_EVERY}" \
    --caption_extension="${CAPTION_EXTENSION}" \
    --enable_bucket \
    --bucket_reso_steps="${BUCKET_STEPS}" \
    --min_bucket_reso="${MIN_BUCKET}" \
    --max_bucket_reso="${MAX_BUCKET}" \
    --bucket_no_upscale
else
  echo "[i] SKIP_TRAINING=1 so training stage is skipped."
fi

DEFAULT_MERGE_MODEL="${MODELS_DIR}/sd-v1-5-pruned.safetensors"
MERGE_SD_MODEL="${MERGE_SD_MODEL:-${DEFAULT_MERGE_MODEL}}"
MERGED_OUTPUT="${MERGED_OUTPUT:-${OUTPUT_DIR}/merged.safetensors}"
MERGE_SOURCE="${MERGE_SOURCE:-${OUTPUT_DIR}/last.safetensors}"
MERGE_PRECISION="${MERGE_PRECISION:-fp16}"
MERGE_SD_MODEL_URL="${MERGE_SD_MODEL_URL:-https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors?download=true}"

if [[ ! -f "${MERGE_SOURCE}" ]]; then
  echo "[!] Cannot find LoRA weights at ${MERGE_SOURCE}. Set MERGE_SOURCE to the desired file." >&2
  exit 1
fi

if [[ ! -f "${MERGE_SD_MODEL}" ]]; then
  if [[ -z "${MERGE_SD_MODEL_URL}" ]]; then
    echo "[!] Base model ${MERGE_SD_MODEL} not found and MERGE_SD_MODEL_URL is empty. Cannot proceed." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${MERGE_SD_MODEL}")"
  echo "[i] Downloading base model to ${MERGE_SD_MODEL}"
  curl -L -o "${MERGE_SD_MODEL}" "${MERGE_SD_MODEL_URL}"
fi

PYTHONPATH=. python3 networks/merge_lora.py \
  --sd_model "${MERGE_SD_MODEL}" \
  --save_to "${MERGED_OUTPUT}" \
  --models "${MERGE_SOURCE}" \
  --ratios 1.0 \
  --precision "${MERGE_PRECISION}" \
  --save_precision "${MERGE_PRECISION}"

echo "[+] Merged model written to ${MERGED_OUTPUT}"
