#!/usr/bin/env python3
"""Interactive GUI Launcher for Overcooked-AI.

Run from repository root:
    python runover.py

Optional CLI overrides:
    python runover.py --layout asymmetric_advantages
    python runover.py --agent-0 human --agent-1 ppo
    python runover.py --horizon 400 --fps 12
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
AGENT_EVAL_DIR = REPO_ROOT / "overcooked-agent-eval"
VENV_DIR = AGENT_EVAL_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"


def _restart_in_project_venv() -> None:
    """Ensure the project virtual environment is active even when run globally."""
    if Path(sys.prefix).resolve() == VENV_DIR.resolve():
        return

    if VENV_PYTHON.is_file():
        os.execv(
            str(VENV_PYTHON),
            [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        )


_restart_in_project_venv()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(AGENT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_EVAL_DIR))

try:
    from gui.app import OvercookedApp, DEFAULT_LAYOUT, DEFAULT_HORIZON, DEFAULT_FPS
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing dependency '{error.name}'. Please ensure project venv is set up:\n"
        "  python3.10 -m venv overcooked-agent-eval/.venv\n"
        "  overcooked-agent-eval/.venv/bin/pip install -e ./external/overcooked_ai\n"
    ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Overcooked-AI Interactive Runner GUI."
    )
    parser.add_argument(
        "--layout",
        type=str,
        default=DEFAULT_LAYOUT,
        help=f"Initial layout to load (default: {DEFAULT_LAYOUT})",
    )
    parser.add_argument(
        "--agent-0",
        type=str,
        default="human",
        help="Initial agent type for Agent 1 (Chef 0, red hat; default: human)",
    )
    parser.add_argument(
        "--agent-1",
        type=str,
        default="ppo",
        help="Initial agent type for Agent 2 (Chef 1, blue hat; default: ppo)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Simulation episode horizon in timesteps (default: {DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Framerate for gameplay simulation (default: {DEFAULT_FPS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = OvercookedApp(
        initial_layout=args.layout,
        horizon=args.horizon,
        fps=args.fps,
    )
    app.selected_agent_0_type = args.agent_0
    app.dropdown_agent_0.selected_value = args.agent_0
    app.selected_agent_1_type = args.agent_1
    app.dropdown_agent_1.selected_value = args.agent_1
    app.run()


if __name__ == "__main__":
    main()
