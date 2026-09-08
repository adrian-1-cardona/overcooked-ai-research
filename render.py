#!/usr/bin/env python3
"""Convenience launcher for Overcooked-AI Gameplay Renderer.

==============================================================================
COMMANDS TO RUN (from project root):
==============================================================================

1. Watch AI agents play live in a desktop window (default: cramped_room):
     python render.py

2. Play interactively as Chef 0 with keyboard controls:
     python render.py --agent-0 human

   Interactive Controls:
     [WASD / Arrow Keys] : Move North, South, West, East
     [SPACE / ENTER / F] : Interact (pick up items, drop, chop, cook)
     [P]                 : Pause / resume simulation
     [R]                 : Restart episode
     [ESC / Q]           : Quit window

3. Watch agents on a different layout:
     python render.py --layout asymmetric_advantages --fps 15
     python render.py --layout coordination_ring
     python render.py --layout forced_coordination
     python render.py --layout counter_circuit

4. Export gameplay to an MP4 video file (headless-safe):
     python render.py --mode video --horizon 200 --output overcooked-agent-eval/results/gameplay.mp4

5. Replay an existing recorded telemetry CSV run:
     python render.py --replay-csv overcooked-agent-eval/results/random_baseline_cramped_room.csv

6. Export step-by-step PNG image frames:
     python render.py --mode frames --horizon 50 --output overcooked-agent-eval/results/frames

7. View all CLI options:
     python render.py --help
==============================================================================
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
