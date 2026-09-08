#!/usr/bin/env bash
# Convenience runner for the Overcooked-AI Gameplay Renderer.
# Automatically uses the local Python 3.10 virtual environment if present.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/overcooked-agent-eval/.venv/bin/python"

if [ -f "${VENV_PYTHON}" ]; then
  exec "${VENV_PYTHON}" "${SCRIPT_DIR}/overcooked-agent-eval/experiments/render_gameplay.py" "$@"
else
  exec python "${SCRIPT_DIR}/overcooked-agent-eval/experiments/render_gameplay.py" "$@"
fi
