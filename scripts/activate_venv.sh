#!/usr/bin/env bash
# Helper script to activate the project-level virtual environment
# Usage: source scripts/activate_venv.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/.env"
  set +a
fi

# Use project-level venv
VENV_PATH="${REPO_ROOT}/.venv"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "[!] ERROR: No virtual environment found at ${VENV_PATH}" >&2
  echo "[!] Run: ./scripts/setup_venv.sh" >&2
  return 1 2>/dev/null || exit 1
fi

echo "[i] Activating virtual environment from ${VENV_PATH}"
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

echo "[+] Virtual environment activated"
