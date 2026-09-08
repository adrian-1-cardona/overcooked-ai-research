#!/usr/bin/env bash
# ==============================================================================
# Overcooked-AI Gameplay Renderer - Convenience Runner
# ==============================================================================
# Automatically locates and uses the project's Python 3.10 virtual environment.
#
# QUICK START COMMANDS (run from project root):
#
# 1. Watch AI agents play live in a desktop window (default: cramped_room):
#      ./render.sh
#
# 2. Jump in and play interactively as Chef 0 with your keyboard:
#      ./render.sh --agent-0 human
#      (Controls: WASD/Arrows to move, SPACE/ENTER/F to interact, P to pause)
#
# 3. Watch agents on a different layout at a custom speed:
#      ./render.sh --layout asymmetric_advantages --fps 15
#      ./render.sh --layout coordination_ring
#      ./render.sh --layout forced_coordination
#      ./render.sh --layout counter_circuit
#
# 4. Export gameplay to an MP4 video file (headless-safe):
#      ./render.sh --mode video --horizon 200 --output overcooked-agent-eval/results/gameplay.mp4
#
# 5. Replay an existing recorded telemetry CSV run:
#      ./render.sh --replay-csv overcooked-agent-eval/results/random_baseline_cramped_room.csv
#
# 6. Export individual PNG image frames:
#      ./render.sh --mode frames --horizon 50 --output overcooked-agent-eval/results/frames
#
# 7. View all CLI options:
#      ./render.sh --help
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/overcooked-agent-eval/.venv/bin/python"

if [ -f "${VENV_PYTHON}" ]; then
  exec "${VENV_PYTHON}" "${SCRIPT_DIR}/overcooked-agent-eval/experiments/render_gameplay.py" "$@"
else
  exec python "${SCRIPT_DIR}/overcooked-agent-eval/experiments/render_gameplay.py" "$@"
fi
