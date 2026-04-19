#!/usr/bin/env bash
# Start the ROMP metrics frontend: FastAPI backend serving a single-page
# Plotly UI at http://127.0.0.1:8000.
#
# Requires the momp package (installed editable) plus fastapi + uvicorn.
# Install with e.g.:
#
#     uv pip install -e . fastapi uvicorn
#
# Usage:
#     ./frontend/run.sh
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${ROMP_FRONTEND_HOST:-127.0.0.1}"
PORT="${ROMP_FRONTEND_PORT:-8000}"

PY="${PY:-python3}"
exec "$PY" -m uvicorn frontend.api.app:app --host "$HOST" --port "$PORT" --reload
