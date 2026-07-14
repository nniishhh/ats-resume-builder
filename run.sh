#!/usr/bin/env bash
# One-command launcher for the Resume Builder UI.
# Usage: ./run.sh
set -euo pipefail

# Always run from the repo root (directory of this script).
cd "$(dirname "$0")"

# Load secrets/config from .env if present (OPENAI_API_KEY, etc.).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Activate the project virtualenv.
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Warning: OPENAI_API_KEY is not set. Add it to .env before generating bullets." >&2
fi

exec streamlit run main_code/app.py
