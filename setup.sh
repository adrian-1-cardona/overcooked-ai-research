#!/usr/bin/env bash
# ==============================================================================
# Overcooked-AI Research Environment Setup Script
# ==============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Overcooked-AI Setup ==="

# Initialize submodules
if [ -d ".git" ]; then
    echo "Updating git submodules..."
    git submodule update --init --recursive
fi

# Detect or create virtual environment
VENV_DIR="$SCRIPT_DIR/overcooked-agent-eval/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with Python 3.10..."
    PYTHON_CMD=""
    if command -v python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo "Error: Python 3.10 is required."
        exit 1
    fi
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip and installing dependencies..."
pip install --upgrade pip
pip install -e ./external/overcooked_ai

echo "=== Setup complete! ==="
echo "Activate with: source overcooked-agent-eval/.venv/bin/activate"
echo "Launch GUI with: python runover.py"
