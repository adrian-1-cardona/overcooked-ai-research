#!/usr/bin/env python3
"""Convenience launcher for Overcooked-AI Gameplay Renderer.

Usage from project root:
    python render.py
    python render.py --agent-0 human
    python render.py --mode video --horizon 200
    python render.py --replay-csv overcooked-agent-eval/results/random_baseline_cramped_room.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
AGENT_EVAL_DIR = REPO_ROOT / "overcooked-agent-eval"

if str(AGENT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_EVAL_DIR))

from experiments.render_gameplay import main

if __name__ == "__main__":
    main()
