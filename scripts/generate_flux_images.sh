#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROMPT_FILE="${PROMPT_FILE:-${REPO_ROOT}/prompts/knightro_prompts.txt}"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/artifacts/finetune/knightro_flux/merged.safetensors}"
CLIP_PATH="${CLIP_PATH:-${REPO_ROOT}/models/black-forest-labs_FLUX.1-dev/clip_l.safetensors}"
T5_PATH="${T5_PATH:-${REPO_ROOT}/models/black-forest-labs_FLUX.1-dev/t5xxl_fp16.safetensors}"
AE_PATH="${AE_PATH:-${REPO_ROOT}/models/black-forest-labs_FLUX.1-dev/ae.safetensors}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/outputs/flux-merged}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[!] Prompt file not found at ${PROMPT_FILE}" >&2
  exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[!] FLUX checkpoint not found at ${CKPT_PATH}" >&2
  exit 1
fi

cd "${REPO_ROOT}/external/kohya_flux/sd-scripts"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "[!] Missing sd-scripts virtualenv at external/kohya_flux/sd-scripts/.venv" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

timestamp="$(date +%Y%m%d_%H%M%S)"
run_dir="${OUTPUT_ROOT}/${timestamp}"
mkdir -p "${run_dir}"

echo "[i] Saving images to ${run_dir}"

while IFS= read -r prompt || [[ -n "${prompt}" ]]; do
  prompt="$(echo "${prompt}" | tr -d '\r')"
  if [[ -z "${prompt}" ]]; then
    continue
  fi

  safe_name="$(echo "${prompt}" | tr ' /"' '_' | cut -c1-60)"
  out_path="${run_dir}/${safe_name}"
  mkdir -p "${out_path}"

  python flux_minimal_inference.py \
    --ckpt "${CKPT_PATH}" \
    --clip_l "${CLIP_PATH}" \
    --t5xxl "${T5_PATH}" \
    --ae "${AE_PATH}" \
    --prompt "${prompt}" \
    --out "${out_path}"
done < "${PROMPT_FILE}"

echo "[+] Done. Images saved under ${run_dir}"
