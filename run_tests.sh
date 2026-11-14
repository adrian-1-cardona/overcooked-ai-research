#!/usr/bin/env bash
# ==============================================================================
# Overcooked-AI Test Suite & Stateless Contract Verification Runner
# ==============================================================================
# Usage:
#   ./run_tests.sh
#
# Runs all 4 stages of tests in process-isolated steps to prevent Ray/TF/Pygame
# thread conflicts on macOS and Linux, ensuring all stateless encoding and
# agent behaviors are strictly verified.
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Resolve python and pytest executable
if [[ -f "overcooked-agent-eval/.venv/bin/pytest" ]]; then
    PYTEST="overcooked-agent-eval/.venv/bin/pytest"
    PYTHON="overcooked-agent-eval/.venv/bin/python"
elif command -v pytest &> /dev/null; then
    PYTEST="pytest"
    PYTHON="python"
else
    echo "Error: pytest not found. Please activate your environment or install pytest."
    exit 1
fi

export PYTHONPATH=".:$SCRIPT_DIR:$SCRIPT_DIR/overcooked-agent-eval:$SCRIPT_DIR/external/overcooked_ai/src"
export MPLBACKEND="Agg"
export SDL_VIDEODRIVER="dummy"
export PYGAME_HIDE_SUPPORT_PROMPT="1"
export TF_CPP_MIN_LOG_LEVEL="3"
export GRPC_ENABLE_FORK_SUPPORT=0

echo "======================================================================"
echo " Stage 1/5: Evaluation Framework Tests (overcooked-agent-eval)"
echo "======================================================================"
$PYTEST overcooked-agent-eval/tests/ -v

echo ""
echo "======================================================================"
echo " Stage 2/5: Stateless Encoding Contract & Core Agent Policy Tests"
echo "======================================================================"
$PYTEST \
    tests/test_stateless_encoding.py \
    tests/test_greedy_policy.py \
    tests/test_joint_step.py \
    tests/test_partner_env.py \
    tests/test_frame_stacked_agent.py \
    tests/test_timer_and_progress.py \
    -v

echo ""
echo "======================================================================"
echo " Stage 3/5: UI Dropdown, Model Presets & Headless GUI State Tests"
echo "======================================================================"
$PYTEST \
    tests/test_recipe_dropdown.py \
    tests/test_runover_gui.py \
    -v

echo ""
echo "======================================================================"
echo " Stage 4/5: Agent Training Pipeline & Database Lifecycle Tests"
echo "======================================================================"
$PYTEST \
    tests/test_train_mode_and_db.py \
    -v

echo ""
echo "======================================================================"
echo " Stage 5/5: Rollout Baseline Smoke Verification"
echo "======================================================================"
$PYTHON overcooked-agent-eval/experiments/run_random_baseline.py --horizon 20

echo ""
echo "======================================================================"
echo " ALL TESTS & STATELESS CONTRACT CHECKS PASSED SUCCESSFULLY!"
echo "======================================================================"
