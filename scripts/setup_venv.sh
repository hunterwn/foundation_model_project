#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "[+] Creating virtual environment at ${VENV_PATH}"
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
else
  echo "[i] Virtual environment already exists at ${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"

pip install --upgrade pip
pip install -r "${REPO_ROOT}/requirements.txt"

echo "[+] Environment ready. Run 'source ${VENV_PATH}/bin/activate' before executing project scripts."
