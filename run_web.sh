#!/usr/bin/env bash
# One-command launcher for the standalone (non-Streamlit) Resume Tailor web app.
# Usage: ./run_web.sh
set -euo pipefail

cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Warning: OPENAI_API_KEY is not set. Add it to .env before generating bullets." >&2
fi

exec uvicorn main_code.api_server:app --host 0.0.0.0 --port "${PORT:-8000}"
