#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/.env"
  set +a
fi

# Function to get Python binary for a specific version
get_python_bin() {
  local version="$1"
  local python_bin="python3"

  if command -v pyenv &> /dev/null; then
    python_bin="$(pyenv root)/versions/${version}/bin/python"
    if [[ ! -x "${python_bin}" ]]; then
      echo "[!] ERROR: Python ${version} not found in pyenv" >&2
      echo "[!] Run: pyenv install ${version}" >&2
      return 1
    fi
    echo "[i] Using Python ${version} from pyenv: ${python_bin}" >&2
  else
    echo "[!] WARNING: pyenv not found, using default python3" >&2
  fi

  echo "${python_bin}"
}

# 1. Create project-level venv
echo "=== Setting up project-level virtual environment ==="

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -f "${REPO_ROOT}/.python-version" ]]; then
  PYTHON_VERSION="$(cat "${REPO_ROOT}/.python-version")"
  echo "[i] Found .python-version: ${PYTHON_VERSION}"
  PYTHON_BIN=$(get_python_bin "${PYTHON_VERSION}")
fi

VENV_PATH="${REPO_ROOT}/.venv"

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "[+] Creating virtual environment at ${VENV_PATH}"
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
else
  echo "[i] Virtual environment already exists at ${VENV_PATH}"
fi

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

pip install --upgrade pip

# Install project requirements
if [[ -f "${REPO_ROOT}/requirements.txt" ]]; then
  echo "[+] Installing project requirements from ${REPO_ROOT}/requirements.txt"
  pip install -r "${REPO_ROOT}/requirements.txt"
else
  echo "[!] WARNING: No requirements.txt found" >&2
fi

deactivate

echo "[+] Project environment ready at ${VENV_PATH}"

# 2. Create sd-scripts venv(s)
SD_SCRIPTS_DIR="${SD_SCRIPTS_DIR:-external/sd-scripts}"

# Convert to absolute path if it's a relative path
if [[ "${SD_SCRIPTS_DIR}" != /* ]]; then
  SD_SCRIPTS_DIR="${REPO_ROOT}/${SD_SCRIPTS_DIR}"
fi

if [[ -d "${SD_SCRIPTS_DIR}" ]]; then
  echo ""
  echo "=== Setting up sd-scripts virtual environment ==="
  echo "[i] SD_SCRIPTS_DIR: ${SD_SCRIPTS_DIR}"

  # Detect if this is FLUX (kohya_flux) - check for "flux" in path
  IS_FLUX=false
  if [[ "${SD_SCRIPTS_DIR}" == *"flux"* ]]; then
    IS_FLUX=true
  fi

  # Determine Python version for sd-scripts
  if [[ "${IS_FLUX}" == true ]]; then
    SD_PYTHON_VERSION="3.10.9"
    echo "[i] Detected FLUX installation, using Python ${SD_PYTHON_VERSION}"
  else
    SD_PYTHON_VERSION="${PYTHON_VERSION:-3.11.11}"
    echo "[i] Using Python ${SD_PYTHON_VERSION} for standard sd-scripts"
  fi

  SD_PYTHON_BIN=$(get_python_bin "${SD_PYTHON_VERSION}")
  SD_VENV_PATH="${SD_SCRIPTS_DIR}/.venv"

  if [[ ! -d "${SD_VENV_PATH}" ]]; then
    echo "[+] Creating virtual environment at ${SD_VENV_PATH}"
    "${SD_PYTHON_BIN}" -m venv "${SD_VENV_PATH}"
  else
    echo "[i] Virtual environment already exists at ${SD_VENV_PATH}"
  fi

  # shellcheck disable=SC1091
  source "${SD_VENV_PATH}/bin/activate"

  pip install --upgrade pip

  # Install sd-scripts requirements (cd to the directory first for relative paths)
  if [[ -f "${SD_SCRIPTS_DIR}/requirements.txt" ]]; then
    echo "[+] Installing sd-scripts requirements from ${SD_SCRIPTS_DIR}/requirements.txt"
    cd "${SD_SCRIPTS_DIR}"
    pip install -r requirements.txt
    cd "${REPO_ROOT}"
  else
    echo "[!] WARNING: No requirements.txt found in ${SD_SCRIPTS_DIR}" >&2
  fi

  deactivate

  echo "[+] sd-scripts environment ready at ${SD_VENV_PATH}"
else
  echo ""
  echo "[!] WARNING: SD_SCRIPTS_DIR not found at ${SD_SCRIPTS_DIR}" >&2
  echo "[!] Skipping sd-scripts venv setup" >&2
fi

echo ""
echo "[+] Setup complete!"
echo "[+] Project venv: source .venv/bin/activate"
if [[ -d "${SD_SCRIPTS_DIR}" ]]; then
  echo "[+] Training venv: Automatically activated by scripts/run_training.sh"
fi
