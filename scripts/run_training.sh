#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/.env"
  set +a
fi

# Default to standard sd-scripts if not set
SD_SCRIPTS_DIR="${SD_SCRIPTS_DIR:-external/sd-scripts}"

# Convert to absolute path if it's a relative path
if [[ "${SD_SCRIPTS_DIR}" != /* ]]; then
  SD_SCRIPTS_DIR="${REPO_ROOT}/${SD_SCRIPTS_DIR}"
fi
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/dataset/knightro}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/finetune/knightro}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/models}"
BASE_MODEL="${BASE_MODEL:-runwayml/stable-diffusion-v1-5}"
BASE_MODEL_PATH="${BASE_MODEL}"
FLUX_CLIP_L_PATH="${FLUX_CLIP_L_PATH:-}"
FLUX_T5XXL_PATH="${FLUX_T5XXL_PATH:-}"
FLUX_AE_PATH="${FLUX_AE_PATH:-}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_STATE="${RESUME_STATE:-}"
TRAIN_METHOD="${TRAIN_METHOD:-lora}"
MODEL_VARIANT="${MODEL_VARIANT:-auto}"
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
OPTIMIZER_ARGS="${OPTIMIZER_ARGS:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_MERGE_ARGS="${EXTRA_MERGE_ARGS:-}"

if [[ -n "${TRAIN_SCRIPT:-}" ]]; then
  echo "[!] TRAIN_SCRIPT is deprecated and ignored. Use TRAIN_ENTRYPOINT to override the trainer if needed." >&2
fi

detect_sdxl_variant() {
  local hint="${1:-}"
  hint="${hint,,}"
  [[ "${hint}" == *"sdxl"* || "${hint}" == *"stable-diffusion-xl"* || "${hint}" == *"sd_xl"* || "${hint}" == *"xl-base"* || "${hint}" == *"_xl"* || "${hint}" == *"xl_"* || "${hint}" == *"-xl"* || "${hint}" == *"xl-"* || "${hint}" == *"/xl"* || "${hint}" == *"xl/"* ]]
}

detect_flux_variant() {
  local hint="${1:-}"
  hint="${hint,,}"
  [[ "${hint}" == *"flux"* ]]
}

if [[ "${MODEL_VARIANT}" == "auto" ]]; then
  if detect_flux_variant "${BASE_MODEL}"; then
    MODEL_VARIANT="flux"
  fi
fi

if [[ "${MODEL_VARIANT}" == "auto" ]]; then
  if detect_sdxl_variant "${BASE_MODEL}"; then
    MODEL_VARIANT="sdxl"
  fi
fi

if [[ "${MODEL_VARIANT}" == "auto" ]]; then
  if detect_flux_variant "${OUTPUT_DIR}" || detect_flux_variant "${DATASET_DIR}" || detect_flux_variant "${MERGE_SD_MODEL:-}"; then
    MODEL_VARIANT="flux"
  fi
fi

if [[ "${MODEL_VARIANT}" == "auto" ]]; then
  if detect_sdxl_variant "${OUTPUT_DIR}" || detect_sdxl_variant "${DATASET_DIR}" || detect_sdxl_variant "${MERGE_SD_MODEL:-}"; then
    MODEL_VARIANT="sdxl"
  fi
fi

if [[ "${MODEL_VARIANT}" == "auto" ]]; then
  IFS=',' read -r RESO_WIDTH RESO_HEIGHT <<< "${RESOLUTION}"
  if [[ "${RESO_WIDTH}" =~ ^[0-9]+$ && "${RESO_HEIGHT}" =~ ^[0-9]+$ ]]; then
    if (( RESO_WIDTH >= 1024 || RESO_HEIGHT >= 1024 )); then
      MODEL_VARIANT="sdxl"
    fi
  fi
fi

if [[ "${MODEL_VARIANT}" == "auto" ]]; then
  MODEL_VARIANT="sd15"
fi

# FLUX base models cannot be trained with SD or SDXL scripts, so force the variant.
if detect_flux_variant "${BASE_MODEL}"; then
  if [[ "${MODEL_VARIANT}" != "flux" ]]; then
    echo "[i] Detected FLUX base model, overriding MODEL_VARIANT=${MODEL_VARIANT} with 'flux'."
    MODEL_VARIANT="flux"
  fi
fi

if [[ -z "${RESUME_STATE}" && "${AUTO_RESUME}" == "1" ]]; then
  if compgen -G "${OUTPUT_DIR}/"*"-state" >/dev/null; then
    RESUME_STATE="$(ls -1dt "${OUTPUT_DIR}/"*"-state" 2>/dev/null | head -n1)"
    if [[ -n "${RESUME_STATE}" ]]; then
      echo "[i] Auto-detected resume state at ${RESUME_STATE}"
    fi
  fi
fi

if [[ "${MODEL_VARIANT}" == "flux" && "${TRAIN_METHOD}" == "lora" ]]; then
  if [[ -z "${NETWORK_MODULE}" || "${NETWORK_MODULE}" == "networks.lora" ]]; then
    NETWORK_MODULE="networks.lora_flux"
  elif [[ "${NETWORK_MODULE}" != "networks.lora_flux" ]]; then
    echo "[!] FLUX LoRA training requires NETWORK_MODULE=networks.lora_flux (found ${NETWORK_MODULE})." >&2
    echo "[!] Update NETWORK_MODULE or unset it to use the default." >&2
    exit 1
  fi
fi

case "${MODEL_VARIANT}" in
  sd15|sdxl|flux)
    ;;
  *)
    echo "[!] MODEL_VARIANT=${MODEL_VARIANT} is not supported. Use 'sd15', 'sdxl', or 'flux'." >&2
    exit 1
    ;;
esac

case "${TRAIN_METHOD}" in
  lora|full)
    ;;
  *)
    echo "[!] Unknown TRAIN_METHOD=${TRAIN_METHOD}. Use 'lora' or 'full'." >&2
    exit 1
    ;;
esac

if [[ "${TRAIN_METHOD}" == "lora" ]]; then
  case "${MODEL_VARIANT}" in
    flux)
      DEFAULT_TRAIN_SCRIPT="flux_train_network.py"
      ;;
    sdxl)
      DEFAULT_TRAIN_SCRIPT="sdxl_train_network.py"
      ;;
    *)
      DEFAULT_TRAIN_SCRIPT="train_network.py"
      ;;
  esac
else
  case "${MODEL_VARIANT}" in
    flux)
      DEFAULT_TRAIN_SCRIPT="flux_train.py"
      ;;
    sdxl)
      DEFAULT_TRAIN_SCRIPT="sdxl_train.py"
      ;;
    *)
      DEFAULT_TRAIN_SCRIPT="train_db.py"
      ;;
  esac
fi

TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-}"
if [[ -n "${TRAIN_ENTRYPOINT}" ]]; then
  TRAIN_SCRIPT="${TRAIN_ENTRYPOINT}"
else
  TRAIN_SCRIPT="${DEFAULT_TRAIN_SCRIPT}"
fi

cd "${SD_SCRIPTS_DIR}"
mkdir -p "${MODELS_DIR}"

# For FLUX checkpoints coming from Hugging Face, download them locally because flux_utils expects files.
if [[ "${MODEL_VARIANT}" == "flux" ]]; then
  if [[ ! -e "${BASE_MODEL_PATH}" ]]; then
    SANITIZED_BASE_MODEL="${BASE_MODEL//[\/:]/_}"
    FLUX_MODEL_CACHE_DIR="${MODELS_DIR}/${SANITIZED_BASE_MODEL}"
    BASE_MODEL_PATH="${FLUX_MODEL_CACHE_DIR}"

    if [[ ! -d "${FLUX_MODEL_CACHE_DIR}" ]]; then
      echo "[i] Downloading FLUX base model ${BASE_MODEL} to ${FLUX_MODEL_CACHE_DIR}"
      mkdir -p "${FLUX_MODEL_CACHE_DIR}"
      HF_REPO_ID="${BASE_MODEL}" LOCAL_DIR="${FLUX_MODEL_CACHE_DIR}" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["HF_REPO_ID"]
local_dir = os.environ["LOCAL_DIR"]

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
    fi
  fi

  if [[ -z "${FLUX_CLIP_L_PATH}" ]]; then
    if [[ -e "${BASE_MODEL_PATH}/clip_l.safetensors" ]]; then
      FLUX_CLIP_L_PATH="${BASE_MODEL_PATH}/clip_l.safetensors"
    elif [[ -e "${BASE_MODEL_PATH}/text_encoder" ]]; then
      FLUX_CLIP_L_PATH="${BASE_MODEL_PATH}/text_encoder"
    fi
  fi

  if [[ -z "${FLUX_T5XXL_PATH}" ]]; then
    if [[ -e "${BASE_MODEL_PATH}/t5xxl_fp16.safetensors" ]]; then
      FLUX_T5XXL_PATH="${BASE_MODEL_PATH}/t5xxl_fp16.safetensors"
    elif [[ -e "${BASE_MODEL_PATH}/text_encoder_2" ]]; then
      FLUX_T5XXL_PATH="${BASE_MODEL_PATH}/text_encoder_2"
    fi
  fi

  if [[ -z "${FLUX_AE_PATH}" ]]; then
    if [[ -e "${BASE_MODEL_PATH}/ae.safetensors" ]]; then
      FLUX_AE_PATH="${BASE_MODEL_PATH}/ae.safetensors"
    fi
  fi
fi

# Activate the virtual environment from the sd-scripts directory
if [[ -f "${SD_SCRIPTS_DIR}/.venv/bin/activate" ]]; then
  echo "[i] Activating virtual environment from ${SD_SCRIPTS_DIR}/.venv"
  # shellcheck disable=SC1091
  source "${SD_SCRIPTS_DIR}/.venv/bin/activate"
else
  echo "[!] ERROR: No virtual environment found at ${SD_SCRIPTS_DIR}/.venv" >&2
  echo "[!] Run: SD_SCRIPTS_DIR=${SD_SCRIPTS_DIR} ./scripts/setup_venv.sh" >&2
  exit 1
fi

# Check if torchvision is installed, install it if missing
if ! python3 -c "import torchvision" 2>/dev/null; then
  echo "[i] torchvision not found, installing..."
  pip install torchvision
fi

echo "[i] Training method=${TRAIN_METHOD}, model_variant=${MODEL_VARIANT}, entrypoint=${TRAIN_SCRIPT}"

TRAIN_ARGS=(
  --pretrained_model_name_or_path="${BASE_MODEL_PATH}"
  --train_data_dir="${DATASET_DIR}"
  --resolution="${RESOLUTION}"
  --output_dir="${OUTPUT_DIR}"
  --logging_dir="${LOG_DIR}"
  --train_batch_size="${BATCH_SIZE}"
  --max_train_steps="${TRAIN_STEPS}"
  --learning_rate="${LEARNING_RATE}"
  --lr_scheduler="${LR_SCHEDULER}"
  --mixed_precision="${MIXED_PRECISION}"
  --save_every_n_steps="${SAVE_EVERY}"
  --caption_extension="${CAPTION_EXTENSION}"
  --enable_bucket
  --bucket_reso_steps="${BUCKET_STEPS}"
  --min_bucket_reso="${MIN_BUCKET}"
  --max_bucket_reso="${MAX_BUCKET}"
  --bucket_no_upscale
)

if [[ "${TRAIN_METHOD}" == "lora" ]]; then
  TRAIN_ARGS+=(--network_module="${NETWORK_MODULE}")
fi

if [[ -n "${OPTIMIZER_ARGS}" ]]; then
  # shellcheck disable=SC2206
  read -r -a OPT_ARGS_ARRAY <<< "${OPTIMIZER_ARGS}"
  TRAIN_ARGS+=("${OPT_ARGS_ARRAY[@]}")
fi

if [[ -n "${EXTRA_TRAIN_ARGS}" ]]; then
  # shellcheck disable=SC2206
  read -r -a EXTRA_ARGS_ARRAY <<< "${EXTRA_TRAIN_ARGS}"
  TRAIN_ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

if [[ "${MODEL_VARIANT}" == "flux" ]]; then
  flux_arg_present() {
    local flag="$1"
    shift
    for arg in "$@"; do
      if [[ "${arg}" == "${flag}" ]]; then
        return 0
      fi
    done
    return 1
  }

  if ! flux_arg_present "--clip_l" "${TRAIN_ARGS[@]}"; then
    if [[ -n "${FLUX_CLIP_L_PATH}" ]]; then
      TRAIN_ARGS+=(--clip_l "${FLUX_CLIP_L_PATH}")
    else
      echo "[!] FLUX training requires --clip_l. Set FLUX_CLIP_L_PATH or provide the argument via EXTRA_TRAIN_ARGS." >&2
      exit 1
    fi
  fi

  if ! flux_arg_present "--t5xxl" "${TRAIN_ARGS[@]}"; then
    if [[ -n "${FLUX_T5XXL_PATH}" ]]; then
      TRAIN_ARGS+=(--t5xxl "${FLUX_T5XXL_PATH}")
    else
      echo "[!] FLUX training requires --t5xxl. Set FLUX_T5XXL_PATH or provide the argument via EXTRA_TRAIN_ARGS." >&2
      exit 1
    fi
  fi

  if ! flux_arg_present "--ae" "${TRAIN_ARGS[@]}"; then
    if [[ -n "${FLUX_AE_PATH}" ]]; then
      TRAIN_ARGS+=(--ae "${FLUX_AE_PATH}")
    else
      echo "[!] FLUX training requires --ae. Set FLUX_AE_PATH or provide the argument via EXTRA_TRAIN_ARGS." >&2
      exit 1
    fi
  fi
fi

if [[ -n "${RESUME_STATE}" ]]; then
  if [[ -d "${RESUME_STATE}" ]]; then
    python3 - "$RESUME_STATE" <<'PY'
import glob
import json
import os
import sys
import torch

state_dir = sys.argv[1]
train_state_path = os.path.join(state_dir, "train_state.json")
if not os.path.isfile(train_state_path):
    sys.exit(0)

with open(train_state_path, "r", encoding="utf-8") as f:
    train_state = json.load(f)

step = train_state.get("current_step") or train_state.get("step")
if step is None:
    sys.exit(0)

updated_any = False
for rng_file in glob.glob(os.path.join(state_dir, "random_states_*.pkl")):
    try:
        data = torch.load(rng_file, weights_only=False)
    except Exception as ex:
        print(f"[!] Failed to read {rng_file}: {ex}")
        continue
    if data.get("step") == step:
        continue
    data["step"] = step
    numpy_state = data.get("numpy_random_seed")
    if isinstance(numpy_state, tuple) and len(numpy_state) > 1:
        second = numpy_state[1]
        if hasattr(second, "tolist"):
            numpy_state = list(numpy_state)
            numpy_state[1] = second.tolist()
            data["numpy_random_seed"] = tuple(numpy_state)
    torch.save(data, rng_file)
    updated_any = True

if updated_any:
    print(f"[i] Patched random state files in {state_dir} with step={step}")
PY
    TRAIN_ARGS+=(--resume "${RESUME_STATE}")
  else
    echo "[!] AUTO_RESUME detected ${RESUME_STATE} but directory is missing. Ignoring resume request." >&2
  fi
fi

if [[ "${SKIP_TRAINING:-0}" != "1" ]]; then
  accelerate launch "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"
else
  echo "[i] SKIP_TRAINING=1 so training stage is skipped."
fi

if [[ "${TRAIN_METHOD}" == "lora" ]]; then
  case "${MODEL_VARIANT}" in
    flux)
      DEFAULT_MERGE_MODEL="${MODELS_DIR}/flux1-dev.safetensors"
      DEFAULT_MERGE_SD_MODEL_URL="https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors?download=true"
      DEFAULT_MERGE_SCRIPT="networks/flux_merge_lora.py"
      DEFAULT_MERGE_PRECISION="bf16"
      ;;
    sdxl)
      DEFAULT_MERGE_MODEL="${MODELS_DIR}/sd_xl_base_1.0.safetensors"
      DEFAULT_MERGE_SD_MODEL_URL="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true"
      DEFAULT_MERGE_SCRIPT="networks/sdxl_merge_lora.py"
      DEFAULT_MERGE_PRECISION="bf16"
      ;;
    *)
      DEFAULT_MERGE_MODEL="${MODELS_DIR}/sd-v1-5-pruned.safetensors"
      DEFAULT_MERGE_SD_MODEL_URL="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors?download=true"
      DEFAULT_MERGE_SCRIPT="networks/merge_lora.py"
      DEFAULT_MERGE_PRECISION="fp16"
      ;;
  esac

  MERGE_SD_MODEL="${MERGE_SD_MODEL:-${DEFAULT_MERGE_MODEL}}"
  MERGED_OUTPUT="${MERGED_OUTPUT:-${OUTPUT_DIR}/merged.safetensors}"
  MERGE_SOURCE="${MERGE_SOURCE:-${OUTPUT_DIR}/last.safetensors}"
  MERGE_PRECISION="${MERGE_PRECISION:-${DEFAULT_MERGE_PRECISION}}"
  MERGE_SD_MODEL_URL="${MERGE_SD_MODEL_URL:-${DEFAULT_MERGE_SD_MODEL_URL}}"
  MERGE_SCRIPT="${MERGE_SCRIPT:-${DEFAULT_MERGE_SCRIPT}}"

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

  if [[ "${MODEL_VARIANT}" == "flux" ]]; then
    MERGE_ARGS=(
      --flux_model "${MERGE_SD_MODEL}"
      --save_to "${MERGED_OUTPUT}"
      --models "${MERGE_SOURCE}"
      --ratios 1.0
      --precision "${MERGE_PRECISION}"
      --save_precision "${MERGE_PRECISION}"
    )
  else
    MERGE_ARGS=(
      --sd_model "${MERGE_SD_MODEL}"
      --save_to "${MERGED_OUTPUT}"
      --models "${MERGE_SOURCE}"
      --ratios 1.0
      --precision "${MERGE_PRECISION}"
      --save_precision "${MERGE_PRECISION}"
    )
  fi

  if [[ -n "${EXTRA_MERGE_ARGS}" ]]; then
    # shellcheck disable=SC2206
    read -r -a EXTRA_MERGE_ARRAY <<< "${EXTRA_MERGE_ARGS}"
    MERGE_ARGS+=("${EXTRA_MERGE_ARRAY[@]}")
  fi

  PYTHONPATH=. python3 "${MERGE_SCRIPT}" "${MERGE_ARGS[@]}"

  echo "[+] Merged model written to ${MERGED_OUTPUT}"
else
  echo "[i] TRAIN_METHOD=${TRAIN_METHOD}, skipping LoRA merge step."
fi
